import logging
import os
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel, Field
import requests
import aiohttp
from dotenv import load_dotenv
from functools import partial
import json
from json.decoder import JSONDecodeError
import base64
import csv
from typing import Optional, Dict, Any, List, Annotated, TypedDict, Sequence
import PyPDF2
from PIL import Image
import random
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, HttpUrl, ValidationError
import re
import aiofiles
from utils import get_llm_response
from services.llm import get_llm
import threading
import uuid
import contextvars
import gc
import weakref
import ssl
import jwt
import hashlib
from functools import lru_cache
from dataclasses import dataclass, field
from typing import Tuple

# Shared thread pool for blocking I/O operations (PDF parsing, etc.)
_IO_THREAD_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="langraph_io_")

# Semaphore to limit concurrent LLM calls (prevents API rate limiting)
_LLM_SEMAPHORE = asyncio.Semaphore(10)

# Connection pool for aiohttp (reused across requests)
_AIOHTTP_CONNECTOR = None
_CONNECTOR_LOCK = asyncio.Lock()

# ============================================================================
# CACHING INFRASTRUCTURE
# ============================================================================

@dataclass
class CacheEntry:
    """Cache entry with TTL support"""
    value: Any
    created_at: float
    ttl_seconds: float = 300  # 5 minute default TTL
    
    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_seconds

class KnowledgeCache:
    """
    Thread-safe cache for parsed knowledge sources.
    Caches by knowledge_base_id with file modification time validation.
    """
    def __init__(self, max_size: int = 50, default_ttl: float = 600):
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
    
    def _generate_cache_key(self, knowledge_base_id: str, sources_hash: str) -> str:
        """Generate cache key from knowledge base ID and sources hash"""
        return f"{knowledge_base_id}:{sources_hash}"
    
    async def get(self, knowledge_base_id: str, sources_hash: str) -> Optional[List[str]]:
        """Get cached knowledge sources if valid"""
        async with self._lock:
            key = self._generate_cache_key(knowledge_base_id, sources_hash)
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None
            
            self._hits += 1
            return entry.value
    
    async def set(self, knowledge_base_id: str, sources_hash: str, 
                  content: List[str], ttl: Optional[float] = None):
        """Cache knowledge sources"""
        async with self._lock:
            # Evict oldest entries if at capacity
            if len(self._cache) >= self._max_size:
                oldest_key = min(self._cache.keys(), 
                               key=lambda k: self._cache[k].created_at)
                del self._cache[oldest_key]
            
            key = self._generate_cache_key(knowledge_base_id, sources_hash)
            self._cache[key] = CacheEntry(
                value=content,
                created_at=time.time(),
                ttl_seconds=ttl or self._default_ttl
            )
    
    async def invalidate(self, knowledge_base_id: str = None):
        """Invalidate cache entries"""
        async with self._lock:
            if knowledge_base_id is None:
                self._cache.clear()
            else:
                keys_to_delete = [k for k in self._cache.keys() 
                                 if k.startswith(f"{knowledge_base_id}:")]
                for key in keys_to_delete:
                    del self._cache[key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
            "size": len(self._cache),
            "max_size": self._max_size
        }

class PayloadCache:
    """
    LRU cache for generated API payloads.
    Caches by hash of (schema + user_input) to avoid redundant LLM calls.
    """
    def __init__(self, max_size: int = 100, default_ttl: float = 300):
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hits = 0
        self._misses = 0
    
    def _generate_key(self, schema_hash: str, input_hash: str) -> str:
        return f"{schema_hash}:{input_hash}"
    
    @staticmethod
    def hash_content(content: Any) -> str:
        """Generate hash for cache key"""
        if isinstance(content, dict):
            content = json.dumps(content, sort_keys=True)
        return hashlib.md5(str(content).encode()).hexdigest()[:16]
    
    async def get(self, schema: Dict, user_input: str) -> Optional[Dict]:
        """Get cached payload if valid"""
        async with self._lock:
            key = self._generate_key(
                self.hash_content(schema),
                self.hash_content(user_input)
            )
            entry = self._cache.get(key)
            
            if entry is None or entry.is_expired():
                self._misses += 1
                if entry and entry.is_expired():
                    del self._cache[key]
                return None
            
            self._hits += 1
            return entry.value
    
    async def set(self, schema: Dict, user_input: str, payload: Dict, 
                  ttl: Optional[float] = None):
        """Cache generated payload"""
        async with self._lock:
            if len(self._cache) >= self._max_size:
                # Remove oldest entry
                oldest_key = min(self._cache.keys(),
                               key=lambda k: self._cache[k].created_at)
                del self._cache[oldest_key]
            
            key = self._generate_key(
                self.hash_content(schema),
                self.hash_content(user_input)
            )
            self._cache[key] = CacheEntry(
                value=payload,
                created_at=time.time(),
                ttl_seconds=ttl or self._default_ttl
            )
    
    def get_stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
            "size": len(self._cache)
        }

# Global cache instances
_KNOWLEDGE_CACHE = KnowledgeCache(max_size=50, default_ttl=600)  # 10 min TTL
_PAYLOAD_CACHE = PayloadCache(max_size=100, default_ttl=300)  # 5 min TTL

def get_cache_stats() -> Dict[str, Any]:
    """Get statistics for all caches"""
    return {
        "knowledge_cache": _KNOWLEDGE_CACHE.get_stats(),
        "payload_cache": _PAYLOAD_CACHE.get_stats()
    }

async def invalidate_knowledge_cache(knowledge_base_id: str = None):
    """Invalidate knowledge cache (call when knowledge base is updated)"""
    await _KNOWLEDGE_CACHE.invalidate(knowledge_base_id)

# ============================================================================

async def get_shared_connector() -> aiohttp.TCPConnector:
    """Get or create a shared aiohttp connector with connection pooling"""
    global _AIOHTTP_CONNECTOR
    async with _CONNECTOR_LOCK:
        if _AIOHTTP_CONNECTOR is None or _AIOHTTP_CONNECTOR.closed:
            _AIOHTTP_CONNECTOR = aiohttp.TCPConnector(
                limit=100,  # Total connection pool size
                limit_per_host=20,  # Per-host connection limit
                ttl_dns_cache=300,  # DNS cache TTL
                enable_cleanup_closed=True,
                keepalive_timeout=30
            )
        return _AIOHTTP_CONNECTOR

# LangGraph specific imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool, BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

ssl._create_default_https_context = ssl._create_unverified_context

load_dotenv()

# Constants
CURRENT_DATE_PLACEHOLDER = "{current_date}"
ENABLE_PAYLOAD_CACHE = os.getenv("ENABLE_PAYLOAD_CACHE", "false").lower() == "true"

# CRITICAL FIX: Add execution context isolation
_EXECUTION_CONTEXT = contextvars.ContextVar('execution_context', default=None)

def get_execution_context():
    """Get or create execution context for current asyncio task"""
    if _EXECUTION_CONTEXT.get() is None:
        _EXECUTION_CONTEXT.set({
            'execution_id': str(uuid.uuid4()),
            'user_input': None,
            'created_at': datetime.now(timezone.utc)
        })
    return _EXECUTION_CONTEXT.get()

def set_execution_context(execution_id: str, user_input: str, additional_context: dict = None):
    """Set execution context for current asyncio task with enhanced isolation"""
    context = {
        'execution_id': execution_id,
        'user_input': user_input,
        'created_at': datetime.now(timezone.utc),
        'thread_id': threading.get_ident(),
        'process_id': os.getpid(),
        'memory_path': f"temp/langraph_memory_{execution_id}",
        'cache_path': f"temp/langraph_cache_{execution_id}",
        **(additional_context or {})
    }
    _EXECUTION_CONTEXT.set(context)

def sanitize_for_logging(text: Any) -> str:
    try:
        text_str = str(text)
        return text_str.encode("utf-8", errors="replace").decode("utf-8")
    except Exception:
        return str(text).encode("utf-8", errors="replace").decode("utf-8")

def mask_sensitive_payload(payload: Any) -> Any:
    """Mask sensitive fields in payload for logging (similar to task_executor.py)"""
    if not isinstance(payload, dict):
        return payload
    
    masked = payload.copy()
    
    # Mask db_config
    if "db_config" in masked:
        masked["db_config"] = "XXXXXXXXXXXXXXXXXXXXXXX"
    
    # Mask common sensitive fields
    sensitive_keys = ["password", "token", "api_key", "secret", "auth", "authorization", "access_token", "refresh_token"]
    for key in sensitive_keys:
        if key in masked:
            masked[key] = "XXXXXXXXXXXXXXXXXXXXXXX"
    
    # Recursively mask nested dictionaries
    for k, v in masked.items():
        if isinstance(v, dict):
            masked[k] = mask_sensitive_payload(v)
        elif isinstance(v, list):
            masked[k] = [mask_sensitive_payload(item) if isinstance(item, dict) else item for item in v]
    
    return masked

def generate_token(email: str, expires_delta: timedelta = timedelta(minutes=20)) -> str:
    """
    Generate a JWT token for the given email
    
    Args:
        email: User's email address
        expires_delta: How long the token should be valid for (default: 20 minutes)
        
    Returns:
        str: JWT token
    """
    expire = datetime.utcnow() + expires_delta
    payload = {
        "email": email,
        "exp": expire,
        "iat": datetime.utcnow()
    }
    SECRET_KEY = "AUTHSECAJUN9544213YR2025"
    ALGORITHM = "HS256"
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token

# State definition for LangGraph
class AgentState(TypedDict):
    """State for the agent graph"""
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    user_input: str
    file_path: Optional[str]
    file_type: Optional[str]
    history: List[Dict[str, str]]
    intermediate_steps: List[tuple]
    final_output: Optional[str]

class LangGraphExecutor:
    def __init__(
        self,
        agent_config: Dict[str, str],
        tools_config: Optional[list] = None,
        log_file: Optional[str] = None,
        user_email: Optional[str] = None,
        enable_planning: bool = False,
        enable_memory: bool = False,
        verbose: bool = True,
        raw_output: bool = False,
        knowledge_sources: List[Any] = None,
        execution_id: Optional[str] = None,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
    ):
        # CRITICAL FIX: Store execution_id for isolation
        self.execution_id = execution_id or str(uuid.uuid4())
        self.expected_user_input = None
        self.mcp_servers_config = mcp_servers or []
        self._mcp_manager = None  # Set later via _start_mcp() in create()
        
        self.suggestive_questions = []  # Side-channel: captured from tool API responses
        self.display_metadata = None  # Side-channel: captured from image generation tool responses

        self.logger = logging.getLogger(f"langraph_executor_{self.execution_id}")
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(f"[{self.execution_id}] %(asctime)s [%(levelname)s] %(message)s")
        
        if log_file:
            try:
                file_handler = logging.FileHandler(log_file, encoding="utf-8")
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.error(f"Failed to create log file handler for {log_file}: {sanitize_for_logging(e)}")
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Initialize LangGraph with Vertex AI (primary) + OpenAI (fallback)
        self.llm = get_llm(
            temperature=0.4,
            max_output_tokens=65536,
        )
        
        # Internal LLM for payload generation (JSON mode)
        self.internal_llm = get_llm(
            temperature=0.2,
            max_output_tokens=32768,
        )
        
        self.enable_planning = enable_planning
        self.enable_memory = enable_memory
        self.verbose = verbose
        self.raw_output = raw_output
        self.agent_id = agent_config["id"]
        self.agent_name = agent_config["name"]
        self.agent_role = agent_config.get("role", "AI Assistant")
        self.agent_goal = agent_config.get("goal", "Help the user with their request")
        self.agent_backstory = agent_config.get("backstory", "")
        self.knowledge_base_id = agent_config.get("knowledge_base_id")
        
        # Store knowledge sources and combine them into a single context string
        self.knowledge_sources = knowledge_sources or []
        if self.knowledge_sources:
            self.knowledge_content = "\n\n".join(self.knowledge_sources)
            self.logger.info(f"Loaded {len(self.knowledge_sources)} knowledge sources with {len(self.knowledge_content)} total characters")
        else:
            self.knowledge_content = ""
        
        if user_email:
            self.token = generate_token(user_email)
            self.logger.debug(f"Generated token for user: {user_email}")
        else:
            self.token = None
            self.logger.debug("No user email provided, token not generated")
        
        # Initialize tools
        self.tools = []
        self.tools_config = tools_config or []
        self.status_queue = None  # Will be set during execution
        
        # Create LangChain-compatible tools from tools_config
        if tools_config:
            self._create_tools(tools_config)
        
        # Build the agent graph
        self.graph = None
        self._build_graph()
        
        self.logger.info(f"LangGraph executor initialized for agent: {self.agent_name}")
    
    def _create_tools(self, tools_config: List[Dict]) -> None:
        """Create LangChain-compatible tools from tools_config"""
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field, create_model
        
        for tool_config in tools_config:
            tool_schema = tool_config["schema"]
            tool_headers = tool_config.get("auth", {}).get("headers", {})
            tool_params = tool_config.get("auth", {}).get("params", {})
            tool_data_connector = tool_config.get("data_connector", None)
            
            tool_name = tool_schema["info"]["title"].lower().replace(" ", "_")
            tool_description = f"Calls API: {tool_schema['info']['title']}. {tool_schema['info'].get('description', '')}"
            
            # Build a richer Pydantic schema from the OpenAPI spec so Vertex AI
            # gets properly typed parameters instead of a single opaque string.
            args_schema = self._build_tool_args_schema(tool_name, tool_schema)
            
            # Create a tool function.
            # input_data defaults to "" because the rich schema may not include it —
            # the LLM sends typed fields (natural_language_query, etc.) via **kwargs instead.
            async def tool_func(input_data: str = "", _tool_schema=tool_schema, _tool_headers=tool_headers, 
                              _tool_params=tool_params, _tool_data_connector=tool_data_connector,
                              **kwargs):
                """Execute the API tool"""
                try:
                    # Merge all kwargs into a single input_data JSON string.
                    # When Vertex AI uses the rich schema, it sends typed fields
                    # (e.g. natural_language_query='...') which arrive as kwargs.
                    if kwargs:
                        # Filter out empty/default values from kwargs
                        non_empty_kwargs = {k: v for k, v in kwargs.items() 
                                           if v is not None and v != "" and v != [] and v != {}}
                        if non_empty_kwargs:
                            if input_data and input_data.strip():
                                # Merge with existing input_data
                                try:
                                    existing = json.loads(input_data)
                                    if isinstance(existing, dict):
                                        existing.update(non_empty_kwargs)
                                        input_data = json.dumps(existing)
                                except (json.JSONDecodeError, TypeError):
                                    input_data = json.dumps(non_empty_kwargs)
                            else:
                                input_data = json.dumps(non_empty_kwargs)
                    
                    result = await self._generate_and_call_api(
                        input_data, _tool_schema, _tool_headers, _tool_params, _tool_data_connector
                    )
                    # Capture suggestive_questions from tool response (side-channel)
                    # and remove from result so the LLM doesn't duplicate them in its text output
                    if isinstance(result, dict):
                        if "suggestive_questions" in result:
                            sq = result.pop("suggestive_questions", None)
                            if isinstance(sq, list) and sq:
                                self.suggestive_questions = sq

                        # Capture display_metadata from image generation tool response (side-channel)
                        if "display_metadata" in result:
                            dm = result.pop("display_metadata", None)
                            if isinstance(dm, dict) and dm:
                                self.display_metadata = dm

                    return json.dumps(result)
                except Exception as e:
                    self.logger.error(f"Tool execution error: {str(e)}")
                    return json.dumps({"error": str(e)})
            
            # Create LangChain structured tool with richer schema
            lang_tool = StructuredTool.from_function(
                func=tool_func,
                name=tool_name,
                description=tool_description,
                args_schema=args_schema,
                coroutine=tool_func  # async version
            )
            
            self.tools.append(lang_tool)
            self.logger.debug(f"Created LangChain tool: {tool_name}")
    
    def _build_tool_args_schema(self, tool_name: str, tool_schema: Dict) -> type:
        """Build a clean Pydantic model from the OpenAPI spec for Vertex AI tool binding.
        
        Key design decisions for Vertex AI compatibility:
        - Use plain types (str, int) NOT Optional — Vertex AI's function calling
          protocol handles optionality itself; Optional wrappers cause schema 
          conversion issues that lead to MALFORMED_FUNCTION_CALL errors.
        - Do NOT mix a catch-all `input_data: str` with typed fields — having both
          confuses the model into generating Python-style code calls.
        - All fields get defaults so the model can omit any it doesn't need.
        """
        from pydantic import BaseModel, Field, create_model
        
        try:
            # Extract request body schema from OpenAPI spec
            paths = tool_schema.get("paths", {})
            body_props = {}
            required_fields = []
            for path_key, path_val in paths.items():
                for method in ["post", "put", "patch"]:
                    method_spec = path_val.get(method, {})
                    req_body = method_spec.get("requestBody", {})
                    content = req_body.get("content", {})
                    json_content = content.get("application/json", {})
                    schema = json_content.get("schema", {})
                    body_props = schema.get("properties", {})
                    required_fields = schema.get("required", [])
                    if body_props:
                        break
                if body_props:
                    break
            
            if body_props:
                field_definitions = {}
                for prop_name, prop_spec in body_props.items():
                    prop_type = prop_spec.get("type", "string")
                    prop_desc = prop_spec.get("description", f"Parameter: {prop_name}")
                    
                    # Use plain types — NOT Optional — for Vertex AI compatibility.
                    # Vertex AI handles optionality via the function calling protocol;
                    # wrapping in Optional causes schema issues.
                    type_map = {
                        "string": str,
                        "integer": int,
                        "number": float,
                        "boolean": bool,
                        "array": list,
                        "object": dict,
                    }
                    python_type = type_map.get(prop_type, str)
                    
                    # Use appropriate defaults for each type
                    default_map = {
                        str: "",
                        int: 0,
                        float: 0.0,
                        bool: False,
                        list: [],
                        dict: {},
                    }
                    default_val = default_map.get(python_type, "")
                    
                    field_definitions[prop_name] = (
                        python_type,
                        Field(default=default_val, description=prop_desc)
                    )
                
                # Create a dynamic Pydantic model
                model_name = "".join(word.capitalize() for word in tool_name.split("_")) + "Input"
                schema_model = create_model(model_name, **field_definitions)
                self.logger.debug(f"[Tool Schema] Built rich schema for '{tool_name}' with fields: {list(field_definitions.keys())}")
                return schema_model
        except Exception as e:
            self.logger.warning(f"[Tool Schema] Failed to build rich schema for '{tool_name}': {e}, falling back to simple schema")
        
        # Fallback: simple input_data schema
        class SimpleToolInput(BaseModel):
            input_data: str = Field(default="", description="The user query or request data for the API call")
        
        return SimpleToolInput
    
    def _bind_tools_with_fallbacks(self, tools: list):
        """Bind tools to the LLM (primary + fallback).
        
        The _LoggingFallbackLLM returned by get_llm() already delegates
        bind_tools() to both the primary and fallback LLMs internally,
        stripping Vertex-specific kwargs (e.g. tool_config) for OpenAI.
        """
        self.logger.debug(f"[Tool Binding] Binding {len(tools)} tools to LLM")
        llm_with_tools = self.llm.bind_tools(
            tools,
            tool_config={"function_calling_config": {"mode": "AUTO"}},
        )
        self.logger.info(f"[Tool Binding] Bound {len(tools)} tools to LLM (primary + fallback)")
        return llm_with_tools
    
    def _extract_final_json_object(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Extract the final valid JSON object from LLM response with improved robustness"""
        if not response_text or not response_text.strip():
            self.logger.warning("[JSON Extract] Empty response text")
            return None
        
        # First, try to clean up common LLM response patterns
        cleaned_text = response_text.strip()
        
        # Remove common prefixes like "Here's the JSON:", "```json", etc.
        prefixes_to_remove = [
            r'^```json\s*', r'^```\s*', r'^\s*json\s*:\s*', 
            r'^here\'?s?\s+the\s+json\s*:?\s*', r'^the\s+json\s*:?\s*',
            r'^response\s*:?\s*', r'^output\s*:?\s*'
        ]
        for prefix in prefixes_to_remove:
            cleaned_text = re.sub(prefix, '', cleaned_text, flags=re.IGNORECASE)
        
        # Remove all markdown code fences more thoroughly
        cleaned_text = re.sub(r'```\w*\n?', '', cleaned_text)  # Opening fences like ```json, ```
        cleaned_text = re.sub(r'\n?```', '', cleaned_text)      # Closing fences
        cleaned_text = cleaned_text.strip()
        
        # Method 1: Try direct JSON parse first (fastest path)
        try:
            direct_parse = json.loads(cleaned_text)
            if isinstance(direct_parse, dict):
                self.logger.debug("[JSON Extract] Direct parse successful")
                return direct_parse
        except json.JSONDecodeError as e:
            self.logger.debug(f"[JSON Extract] Direct parse failed: {str(e)[:100]}")
        
        # Method 1.5: Extract from first { to last } and parse
        # This works better than brace counting because json.loads() handles strings properly
        first_brace = cleaned_text.find('{')
        last_brace = cleaned_text.rfind('}')
        if first_brace != -1 and last_brace > first_brace:
            potential_json = cleaned_text[first_brace:last_brace + 1]
            try:
                parsed = json.loads(potential_json)
                if isinstance(parsed, dict):
                    self.logger.debug("[JSON Extract] First-to-last brace extraction successful")
                    return parsed
            except json.JSONDecodeError as e:
                self.logger.debug(f"[JSON Extract] First-to-last brace parse failed: {str(e)[:100]}")
        
        # Method 2: Find JSON objects using brace matching (for edge cases)
        brace_count = 0
        start_pos = -1
        json_candidates = []
        
        for i, char in enumerate(cleaned_text):
            if char == '{':
                if brace_count == 0:
                    start_pos = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_pos != -1:
                    json_candidates.append(cleaned_text[start_pos:i+1])
                    start_pos = -1
        
        valid_jsons = []
        for json_str in json_candidates:
            try:
                # Try to fix common JSON issues before parsing
                fixed_json = self._fix_json_string(json_str)
                parsed = json.loads(fixed_json)
                if isinstance(parsed, dict):
                    valid_jsons.append(parsed)
            except json.JSONDecodeError:
                continue
        
        if valid_jsons:
            # Prefer JSON with payload+endpoint_url or error key
            for json_obj in reversed(valid_jsons):
                if ('payload' in json_obj and 'endpoint_url' in json_obj) or 'error' in json_obj:
                    self.logger.debug("[JSON Extract] Found valid payload JSON via brace matching")
                    return json_obj
            self.logger.debug("[JSON Extract] Returning last valid JSON from brace matching")
            return valid_jsons[-1]
        
        # Method 3: Try regex patterns as fallback
        json_patterns = [
            r'\{[^{}]*"payload"[^{}]*\{.*?\}[^{}]*"endpoint_url"[^{}]*\}',  # Basic payload pattern
            r'\{[^{}]*"error"[^{}]*\}',  # Error pattern
            r'\{[\s\S]*?\}',  # Any JSON object
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, cleaned_text, re.DOTALL)
            for match in reversed(matches):
                try:
                    fixed_json = self._fix_json_string(match)
                    parsed = json.loads(fixed_json)
                    if isinstance(parsed, dict):
                        self.logger.debug(f"[JSON Extract] Found valid JSON via regex pattern")
                        return parsed
                except json.JSONDecodeError:
                    continue
        
        self.logger.warning(f"[JSON Extract] No valid JSON found in response (length: {len(response_text)})")
        self.logger.debug(f"[JSON Extract] First 500 chars: {sanitize_for_logging(response_text[:500])}")
        return None
    
    def _fix_json_string(self, json_str: str) -> str:
        """Attempt to fix common JSON formatting issues"""
        fixed = json_str.strip()
        
        # Fix trailing commas before closing braces/brackets
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        # Fix single quotes to double quotes (common LLM mistake)
        # Only do this if there are no double quotes (to avoid breaking valid JSON)
        if '"' not in fixed and "'" in fixed:
            fixed = fixed.replace("'", '"')
        
        # Fix unquoted keys (simple cases)
        fixed = re.sub(r'(\{|\,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed)
        
        return fixed

    def _construct_fallback_payload(self, schema: Dict, user_input: Union[str, dict]) -> Optional[Dict[str, Any]]:
        """
        Construct a basic payload from schema when LLM fails to generate valid JSON.
        This is a last resort fallback.
        """
        try:
            paths = schema.get("paths", {})
            if not paths:
                return None
            
            path_key = next(iter(paths))
            servers = schema.get("servers", [])
            base_url = servers[0].get("url", "") if servers else ""
            
            endpoint_url = f"{base_url}{path_key}" if base_url else path_key
            
            # Build basic payload from user input
            if isinstance(user_input, dict):
                payload = user_input
            else:
                # Try to extract key-value from text input
                payload = {"query": str(user_input)}
            
            self.logger.info(f"[Fallback] Constructed fallback payload for {endpoint_url}")
            return {"payload": payload, "endpoint_url": endpoint_url}
        except Exception as e:
            self.logger.error(f"[Fallback] Failed to construct fallback payload: {e}")
            return None

    def _build_db_config(self, tool_data_connector: Dict) -> Optional[Dict]:
        """
        Programmatically build db_config from tool_data_connector based on connectorType.
        This ensures db_config is always correctly constructed regardless of LLM output.
        """
        if not tool_data_connector:
            return None
        
        connector_type = tool_data_connector.get("connectorType")
        if not connector_type:
            self.logger.warning("[DB Config] No connectorType found in data_connector")
            return None
        
        db_config = None
        
        if connector_type == "postgres":
            db_config = {
                "db_type": "postgres",
                "host": tool_data_connector.get("vectorStoreHost", ""),
                "database": tool_data_connector.get("vectorStoreDBName", ""),
                "user": tool_data_connector.get("vectorStoreUser", ""),
                "password": tool_data_connector.get("vectorStorePassword", ""),
                "port": tool_data_connector.get("vectorStorePort", "5432")
            }
            self.logger.debug(f"[DB Config] Built postgres db_config for host: {db_config['host']}, database: {db_config['database']}")
        
        elif connector_type == "bigquery":
            db_config = {
                "db_type": "bigquery",
                "project_id": tool_data_connector.get("projectId", ""),
                "database": tool_data_connector.get("datasetId", ""),
            }
            service_account_key = tool_data_connector.get("serviceAccountKey")
            # Only include credentials_json when a service account key is actually provided.
            # If omitted, downstream BigQuery client uses Application Default Credentials
            # (compute engine / GCE / Cloud Run attached identity).
            if service_account_key:
                db_config["credentials_json"] = service_account_key
                self.logger.debug(f"[DB Config] Built bigquery db_config with explicit credentials for project: {db_config['project_id']}, dataset: {db_config['database']}")
            else:
                self.logger.debug(f"[DB Config] Built bigquery db_config using ADC (no serviceAccountKey) for project: {db_config['project_id']}, dataset: {db_config['database']}")
        
        else:
            self.logger.warning(f"[DB Config] Unknown connectorType: {connector_type}")
            return None
        
        return db_config

    async def _generate_payload(self, user_input: Union[str, dict], schema: Dict, 
                               tool_data_connector: Optional[Dict] = None, 
                               tool_name: Optional[str] = None,
                               use_cache: bool = True) -> Dict[str, Any]:
        """Generate API payload using LLM with optional caching"""
        self.logger.debug(f"[Payload Gen] Starting payload generation for: {tool_name or 'API Tool'}")
        
        # Check payload cache first (skip db_config in cache key - it's added programmatically)
        cache_user_input = json.dumps(user_input) if isinstance(user_input, dict) else str(user_input)
        if use_cache and ENABLE_PAYLOAD_CACHE:
            cached_payload = await _PAYLOAD_CACHE.get(schema, cache_user_input)
            if cached_payload is not None:
                self.logger.info(f"[Payload Gen] Cache HIT for {tool_name or 'API Tool'}")
                # Inject db_config if needed (since it's not cached)
                db_config = self._build_db_config(tool_data_connector) if tool_data_connector else None
                if db_config and cached_payload.get("payload"):
                    cached_payload["payload"]["db_config"] = db_config
                return cached_payload
            self.logger.debug(f"[Payload Gen] Cache MISS for {tool_name or 'API Tool'}")
        
        try:
            if self.status_queue:
                await self.status_queue.put(f"Preparing payload for \"{tool_name or 'API Tool'}\"")
        except:
            pass
        
        paths = schema.get("paths", {})
        if not paths:
            self.logger.error(f"[Payload Gen] No paths found in schema")
            return {"error": "No paths in schema"}

        path_key = next(iter(paths))
        method = next((m for m in paths[path_key] if m.lower() in ["get", "post"]), "post")
        
        self.logger.debug(f"[Payload Gen] Path: {path_key}, Method: {method.upper()}")

        request_schema = {}
        if method == "get":
            parameters = paths[path_key].get(method, {}).get("parameters", [])
            if parameters:
                request_schema = {"parameters": parameters}
                self.logger.debug(f"[Payload Gen] GET request with {len(parameters)} parameters")
        else:
            request_body = paths[path_key].get(method, {}).get("requestBody", {})
            if request_body:
                request_schema = request_body.get("content", {}).get("application/json", {}).get("schema", {})
                self.logger.debug(f"[Payload Gen] POST request with JSON body")

        # Build db_config programmatically from data connector
        db_config = self._build_db_config(tool_data_connector) if tool_data_connector else None
        
        user_input_str = f"User Input (JSON): {json.dumps(user_input, indent=2)}" if isinstance(user_input, dict) else f"User Input: '{user_input}'"
        
        if tool_data_connector:
            connector_type = tool_data_connector.get('connectorType', 'unknown')
            self.logger.debug(f"[Payload Gen] Data connector present (type: {connector_type}) — db_config will be injected programmatically, not passed to LLM")

        payload_prompt = f"""{user_input_str}
Request Schema: {json.dumps(request_schema, indent=2)}
Full Schema: {json.dumps(schema, indent=2)}

Generate a valid JSON payload from the user input based on the OpenAPI schema.
Construct the endpoint URL from the schema's "servers" and "paths" fields.
If schema has a db_config field, set it to null.
PRESERVE all field values from user input EXACTLY - do NOT summarize or truncate.

Return: {{"payload": {{...}}, "endpoint_url": "https://..."}}
Or: {{"error": "brief error message"}}"""

        # Stricter retry prompt for second attempt - includes schema to ensure correct field names
        server_url = schema.get('servers', [{}])[0].get('url', 'http://localhost')
        retry_prompt = f"""{user_input_str}

REQUEST SCHEMA (use these exact field names):
{json.dumps(request_schema, indent=2)}

Use EXACTLY these field names. PRESERVE all values VERBATIM. If db_config field exists, set to null.

Return: {{"payload": {{...}}, "endpoint_url": "{server_url}{path_key}"}}"""

        for attempt in range(3):  # Increased to 3 attempts
            try:
                current_prompt = payload_prompt if attempt == 0 else retry_prompt
                self.logger.debug(f"[Payload Gen] LLM attempt {attempt + 1}/3")
                messages = [HumanMessage(content=current_prompt)]
                # Use semaphore to limit concurrent LLM calls
                async with _LLM_SEMAPHORE:
                    response = await self.internal_llm.ainvoke(messages)
                result_str = response.content
                
                self.logger.debug(f"[Payload Gen] LLM response length: {len(str(result_str))} chars")
                # Log raw LLM response for debugging
                self.logger.debug(f"[Payload Gen] Raw LLM response: {sanitize_for_logging(result_str[:1000])}")

                # With response_mime_type="application/json", try direct parse first (fast path)
                result_json = None
                try:
                    result_json = json.loads(result_str)
                except (json.JSONDecodeError, TypeError):
                    # Fallback to robust extraction if direct parse fails
                    if "```" in result_str:
                        result_str = re.sub(r"```(json)?", "", result_str).strip()
                    result_json = self._extract_final_json_object(result_str)
                if not result_json:
                    self.logger.warning(f"[Payload Gen] No valid JSON found in LLM response (attempt {attempt + 1}). Response was: {sanitize_for_logging(result_str[:500])}")
                    if attempt < 2:
                        await asyncio.sleep(0.2)  # Brief delay before retry (reduced from 0.5s)
                        continue
                    
                    # Final fallback: try to construct a basic payload from schema
                    self.logger.warning("[Payload Gen] Attempting fallback payload construction")
                    fallback_result = self._construct_fallback_payload(schema, user_input)
                    if fallback_result:
                        return fallback_result
                    
                    self.logger.error("[Payload Gen] Failed to extract JSON after 3 attempts")
                    return {"error": "No valid JSON in LLM response"}

                if "error" in result_json:
                    self.logger.warning(f"[Payload Gen] LLM returned error: {result_json['error']}")
                    return result_json

                payload = result_json.get("payload")
                endpoint_url = result_json.get("endpoint_url")
                
                if not endpoint_url:
                    self.logger.error(f"[Payload Gen] Missing endpoint_url in LLM response")
                    return {"error": "Missing endpoint_url"}

                # Validate that required fields from the request schema are present.
                # This rejects hallucinated/degraded schemas (e.g. the LLM inventing
                # "image_urls" instead of "products"/"user_input") before they ship.
                required_fields = request_schema.get("required", []) if isinstance(request_schema, dict) else []
                if payload and isinstance(payload, dict) and required_fields:
                    missing = [f for f in required_fields
                               if f not in payload or payload[f] in (None, "", [], {})]
                    if missing:
                        self.logger.warning(f"[Payload Gen] Payload missing required fields {missing} (attempt {attempt + 1})")
                        if attempt < 2:
                            await asyncio.sleep(0.2)
                            continue
                        return {"error": f"Payload missing required fields: {missing}"}

                # CRITICAL: Inject the programmatically-built db_config into the payload.
                # This guarantees db_config is always correct regardless of LLM output.
                if db_config and payload and isinstance(payload, dict):
                    payload["db_config"] = db_config
                    self.logger.info(f"[Payload Gen] Injected programmatic db_config (type: {db_config['db_type']}) into payload")
                elif db_config and payload is None:
                    payload = {"db_config": db_config}
                    self.logger.info(f"[Payload Gen] Created payload with db_config (type: {db_config['db_type']})")

                self.logger.info(f"[Payload Gen] Successfully generated payload")
                
                # Mask sensitive values for logging
                masked_payload = mask_sensitive_payload(payload) if payload else None
                self.logger.debug(f"[Payload Gen] Generated payload: {json.dumps(masked_payload, ensure_ascii=False) if masked_payload else 'null'}")
                self.logger.debug(f"[Payload Gen] Generated endpoint URL: {endpoint_url}")

                try:
                    if self.status_queue:
                        await self.status_queue.put(f"Payload prepared for \"{tool_name or 'API Tool'}\"")
                except:
                    pass

                # Cache the successful result (without db_config - it's injected separately)
                if use_cache and ENABLE_PAYLOAD_CACHE:
                    cache_result = {"payload": result_json.get("payload"), "endpoint_url": endpoint_url}
                    await _PAYLOAD_CACHE.set(schema, cache_user_input, cache_result)
                    self.logger.debug(f"[Payload Gen] Cached payload for {tool_name or 'API Tool'}")

                return {"payload": payload, "endpoint_url": endpoint_url}

            except Exception as e:
                self.logger.error(f"[Payload Gen] Exception on attempt {attempt + 1}: {sanitize_for_logging(str(e))}")
                if attempt == 0:
                    continue
                return {"error": f"Payload generation error: {e}"}

    async def _generate_and_call_api(self, input_data: str, tool_schema: Dict, 
                                    tool_headers: Dict, tool_params: Dict, 
                                    tool_data_connector: Optional[Dict]) -> Dict:
        """Generate payload using LLM and call API - full TaskExecutor logic"""
        tool_name = tool_schema.get("info", {}).get("title", "Unknown Tool")
        self.logger.info(f"[API Call] Starting API call for: {tool_name}")
        
        try:
            # Generate payload using LLM with retry logic for invalid responses
            max_payload_retries = 3
            payload_retry_delay = 0.5
            payload = None
            endpoint_url = None
            last_payload_error = None
            
            for payload_attempt in range(max_payload_retries):
                try:
                    # Skip cache on retry to force fresh LLM generation
                    use_cache = (payload_attempt == 0)
                    result = await self._generate_payload(input_data, tool_schema, tool_data_connector, tool_name, use_cache=use_cache)
                    
                    if not result or "error" in result:
                        last_payload_error = result.get('error', 'Unknown error') if result else 'Empty result'
                        self.logger.warning(f"[API Call] Payload generation failed (attempt {payload_attempt + 1}/{max_payload_retries}): {last_payload_error}")
                        if payload_attempt < max_payload_retries - 1:
                            await asyncio.sleep(payload_retry_delay)
                            payload_retry_delay *= 2
                            continue
                        else:
                            self.logger.error(f"[API Call] Payload generation failed after {max_payload_retries} attempts")
                            return result or {"error": "Payload generation failed"}

                    payload = result.get("payload")
                    endpoint_url = result.get("endpoint_url")
                    
                    if not endpoint_url:
                        last_payload_error = "Missing endpoint URL"
                        self.logger.warning(f"[API Call] Missing endpoint URL in LLM response (attempt {payload_attempt + 1}/{max_payload_retries})")
                        if payload_attempt < max_payload_retries - 1:
                            await asyncio.sleep(payload_retry_delay)
                            payload_retry_delay *= 2
                            continue
                        else:
                            self.logger.error(f"[API Call] Missing endpoint URL after {max_payload_retries} attempts")
                            return {"error": "Missing endpoint URL after multiple LLM attempts"}

                    # Validate endpoint URL is a full URL (has scheme)
                    if not endpoint_url.startswith(('http://', 'https://')):
                        # Try to construct full URL from schema servers
                        servers = tool_schema.get("servers", [])
                        if servers and servers[0].get("url"):
                            base_url = servers[0]["url"].rstrip('/')
                            endpoint_url = f"{base_url}{endpoint_url}" if endpoint_url.startswith('/') else f"{base_url}/{endpoint_url}"
                            self.logger.info(f"[API Call] Constructed full URL from schema servers: {endpoint_url}")
                        else:
                            # No server URL in schema - retry LLM to get full URL
                            last_payload_error = f"Invalid endpoint URL (no scheme): {endpoint_url}"
                            self.logger.warning(f"[API Call] {last_payload_error} (attempt {payload_attempt + 1}/{max_payload_retries})")
                            if payload_attempt < max_payload_retries - 1:
                                await asyncio.sleep(payload_retry_delay)
                                payload_retry_delay *= 2
                                continue
                            else:
                                self.logger.error(f"[API Call] Invalid endpoint URL after {max_payload_retries} attempts. Schema missing server URL.")
                                return {"error": f"Invalid endpoint URL: {endpoint_url}. Schema is missing server URL configuration."}
                    
                    # Valid payload and endpoint URL obtained
                    self.logger.info(f"[API Call] Valid payload obtained on attempt {payload_attempt + 1}")
                    break
                    
                except Exception as e:
                    last_payload_error = str(e)
                    self.logger.warning(f"[API Call] Exception during payload generation (attempt {payload_attempt + 1}/{max_payload_retries}): {sanitize_for_logging(last_payload_error)}")
                    if payload_attempt < max_payload_retries - 1:
                        await asyncio.sleep(payload_retry_delay)
                        payload_retry_delay *= 2
                        continue
                    else:
                        self.logger.error(f"[API Call] Payload generation exception after {max_payload_retries} attempts")
                        return {"error": f"Payload generation failed: {last_payload_error}"}

            # Determine HTTP method
            paths = tool_schema.get("paths", {})
            path_key = next(iter(paths))
            method = next((m for m in paths[path_key] if m.lower() in ["get", "post"]), None)
            
            if not method:
                self.logger.error(f"[API Call] No supported HTTP method found in schema")
                return {"error": "No supported HTTP method found"}
            
            self.logger.info(f"[API Call] Method: {method.upper()}, Endpoint: {endpoint_url}")
            
            # Log payload (masked) before sending
            if payload:
                masked_payload = mask_sensitive_payload(payload)
                self.logger.debug(f"[API Call] Payload: {json.dumps(masked_payload, ensure_ascii=False) if isinstance(masked_payload, dict) else str(masked_payload)}")

            # Prepare headers and params
            headers = tool_headers.copy() if tool_headers else {}
            params = tool_params.copy() if tool_params else {}
            
            self.logger.debug(f"[API Call] Initial headers: {len(headers)} items")
            self.logger.debug(f"[API Call] Initial params: {len(params)} items")

            if self.token:
                headers["Authorization"] = f"Bearer {self.token}"
                app_url = endpoint_url.lower()
                app_host = app_url.split("://")[-1].split("/")[0]
                
                self.logger.debug(f"[API Call] Added auth token, host: {app_host}")
                
                if (app_host == "impact-agents.ai" or 
                    app_host.endswith(".impact-agents.ai") or 
                    app_host == "localhost:8000"):
                    headers["AGENT-API-KEY"] = os.getenv("AGENT_API_KEY", "")
                    self.logger.debug(f"[API Call] Added AGENT-API-KEY header")

            if method == "get" and payload:
                params.update(payload)
                masked_params = mask_sensitive_payload(params) if params else None
                self.logger.debug(f"[API Call] Query params: {json.dumps(masked_params, ensure_ascii=False) if masked_params else 'None'}")

            # Make the API request with retry logic for HTTP errors
            max_retries = 3
            retry_delay = 1.0  # Initial delay in seconds
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    self.logger.info(f"[API Call] Sending {method.upper()} request to {endpoint_url} (attempt {attempt + 1}/{max_retries})")
                    call_start = time.time()
                    
                    # Status update: Invoking API
                    try:
                        if self.status_queue:
                            status_msg = f"Executing \"{tool_name}\" ({method.upper()})"
                            if attempt > 0:
                                status_msg += f" - Retry {attempt}"
                            await self.status_queue.put(status_msg)
                    except:
                        pass
                    
                    timeout = aiohttp.ClientTimeout(total=1200, connect=30, sock_read=900)
                    connector = await get_shared_connector()
                    async with aiohttp.ClientSession(timeout=timeout, connector=connector, connector_owner=False) as session:
                        if method == "get":
                            self.logger.debug(f"[API Call] GET with params: {bool(params)}")
                            async with session.get(endpoint_url, headers=headers, params=params if params else None) as response:
                                call_duration = time.time() - call_start
                                self.logger.info(f"[API Call] Response received: {response.status} in {call_duration:.2f}s")
                                
                                if response.status == 200:
                                    try:
                                        result = await response.json()
                                        self.logger.info(f"[API Call] Success - JSON response")
                                        # Log response payload (masked) similar to task_executor.py
                                        masked_response = mask_sensitive_payload(result) if isinstance(result, dict) else result
                                        self.logger.debug(f"[API Call] Response data: {json.dumps(masked_response, ensure_ascii=False) if isinstance(masked_response, dict) else str(masked_response)}")
                                        
                                        # Status update: Completed API call
                                        try:
                                            if self.status_queue:
                                                await self.status_queue.put(f"Completed \"{tool_name}\"")
                                        except:
                                            pass
                                        
                                        return result
                                    except:
                                        result = {"result": await response.text()}
                                        self.logger.info(f"[API Call] Success - Text response")
                                        self.logger.debug(f"[API Call] Response text: {sanitize_for_logging(str(result.get('result', ''))[:500])}")
                                        
                                        # Status update: Completed API call
                                        try:
                                            if self.status_queue:
                                                await self.status_queue.put(f"Completed \"{tool_name}\"")
                                        except:
                                            pass
                                        
                                        return result
                                elif response.status >= 500:
                                    # Server errors are retryable
                                    error_text = await response.text() or str(response.status)
                                    self.logger.warning(f"[API Call] Server error {response.status}, will retry: {sanitize_for_logging(error_text[:200])}")
                                    last_error = f"HTTP {response.status}: {error_text}"
                                    # Don't return, continue to retry
                                else:
                                    # Client errors (4xx) are not retryable
                                    self.logger.error(f"[API Call] HTTP error: {response.status}")
                                    try:
                                        error_data = await response.json()
                                        error_msg = error_data.get('detail', str(response.status))
                                        
                                        # Log full error details for debugging (similar to task_executor.py)
                                        self.logger.error(f"[API Call] Error data: {json.dumps(error_data, ensure_ascii=False)}")
                                        self.logger.error(f"[API Call] Error detail: {sanitize_for_logging(str(error_msg))}")
                                        
                                        return {"error": error_msg}
                                    except:
                                        error_text = await response.text() or str(response.status)
                                        self.logger.error(f"[API Call] Error response text: {sanitize_for_logging(error_text)}")
                                        return {"error": error_text}
                        else:  # POST
                            self.logger.debug(f"[API Call] POST with JSON payload: {bool(payload)}")
                            async with session.post(endpoint_url, headers=headers, params=params if params else None, json=payload if payload else None) as response:
                                call_duration = time.time() - call_start
                                self.logger.info(f"[API Call] Response received: {response.status} in {call_duration:.2f}s")
                                
                                if response.status == 200:
                                    try:
                                        result = await response.json()
                                        self.logger.info(f"[API Call] Success - JSON response")
                                        # Log response payload (masked) similar to task_executor.py
                                        masked_response = mask_sensitive_payload(result) if isinstance(result, dict) else result
                                        self.logger.debug(f"[API Call] Response data: {json.dumps(masked_response, ensure_ascii=False) if isinstance(masked_response, dict) else str(masked_response)}")
                                        
                                        # Status update: Completed API call
                                        try:
                                            if self.status_queue:
                                                await self.status_queue.put(f"Completed \"{tool_name}\"")
                                        except:
                                            pass
                                        
                                        return result
                                    except:
                                        result = {"result": await response.text()}
                                        self.logger.info(f"[API Call] Success - Text response")
                                        self.logger.debug(f"[API Call] Response text: {sanitize_for_logging(str(result.get('result', ''))[:500])}")
                                        
                                        # Status update: Completed API call
                                        try:
                                            if self.status_queue:
                                                await self.status_queue.put(f"Completed \"{tool_name}\"")
                                        except:
                                            pass
                                        
                                        return result
                                elif response.status >= 500:
                                    # Server errors are retryable
                                    error_text = await response.text() or str(response.status)
                                    self.logger.warning(f"[API Call] Server error {response.status}, will retry: {sanitize_for_logging(error_text[:200])}")
                                    last_error = f"HTTP {response.status}: {error_text}"
                                    # Don't return, continue to retry
                                else:
                                    # Client errors (4xx) are not retryable
                                    self.logger.error(f"[API Call] HTTP error: {response.status}")
                                    try:
                                        error_data = await response.json()
                                        error_msg = error_data.get('detail', str(response.status))
                                        
                                        # Log full error details for debugging (similar to task_executor.py)
                                        self.logger.error(f"[API Call] Error data: {json.dumps(error_data, ensure_ascii=False)}")
                                        self.logger.error(f"[API Call] Error detail: {sanitize_for_logging(str(error_msg))}")
                                        
                                        return {"error": error_msg}
                                    except:
                                        error_text = await response.text() or str(response.status)
                                        self.logger.error(f"[API Call] Error response text: {sanitize_for_logging(error_text)}")
                                        return {"error": error_text}
                                
                except Exception as e:
                    last_error = str(e)
                    self.logger.warning(f"[API Call] Exception on attempt {attempt + 1}/{max_retries}: {sanitize_for_logging(last_error)}")
                    import traceback
                    self.logger.debug(f"[API Call] Traceback:\n{traceback.format_exc()}")
                
                # If we get here, need to retry (didn't return successfully or with non-retryable error)
                if attempt < max_retries - 1:
                    self.logger.info(f"[API Call] Retrying in {retry_delay:.1f}s...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
            
            # All retries exhausted
            self.logger.error(f"[API Call] All {max_retries} attempts failed. Last error: {sanitize_for_logging(str(last_error))}")
            return {"error": f"API call failed after {max_retries} attempts: {last_error}"}
                            
        except Exception as e:
            self.logger.error(f"[API Call] Exception occurred: {sanitize_for_logging(str(e))}")
            import traceback
            self.logger.error(f"[API Call] Traceback:\n{traceback.format_exc()}")
            return {"error": str(e)}

    @classmethod
    async def create(
        cls,
        agent_config: Dict[str, str],
        tools_config: Optional[list] = None,
        log_file: Optional[str] = None,
        user_email: Optional[str] = None,
        enable_planning: bool = False,
        enable_memory: bool = False,
        verbose: bool = True,
        raw_output: bool = False,
        execution_id: Optional[str] = None,
        mcp_servers: Optional[List[Dict[str, Any]]] = None,
    ):
        """Factory method to create LangGraphExecutor instance"""
        # Create temporary logger for knowledge loading
        temp_logger = logging.getLogger(f"langraph_executor_loader_{execution_id}")
        temp_formatter = logging.Formatter(f"[{execution_id}] %(asctime)s [%(levelname)s] %(message)s")
        temp_handler = logging.StreamHandler()
        temp_handler.setFormatter(temp_formatter)
        temp_logger.addHandler(temp_handler)
        temp_logger.setLevel(logging.DEBUG)
        
        # Load knowledge sources
        knowledge_sources = await cls._load_knowledge_sources(agent_config.get("knowledge_base_id"), temp_logger)
        
        instance = cls(
            agent_config=agent_config,
            tools_config=tools_config,
            log_file=log_file,
            user_email=user_email,
            enable_planning=enable_planning,
            enable_memory=enable_memory,
            verbose=verbose,
            raw_output=raw_output,
            knowledge_sources=knowledge_sources,
            execution_id=execution_id,
            mcp_servers=mcp_servers,
        )
        # Start MCP sessions and merge their tools into the executor's tool set
        await instance._start_mcp_sessions()
        return instance

    async def _start_mcp_sessions(self) -> None:
        """Connect to any configured MCP servers and append their tools to self.tools."""
        if not self.mcp_servers_config:
            return
        try:
            from services.mcp_service import MCPClientManager
        except ImportError as e:
            self.logger.error(f"[MCP] mcp_service unavailable: {e}")
            return
        try:
            manager = MCPClientManager(self.mcp_servers_config, logger_=self.logger)
            await manager.start()
            self._mcp_manager = manager
            if manager.tools:
                self.tools.extend(manager.tools)
                self.logger.info(
                    f"[MCP] Loaded {len(manager.tools)} MCP tool(s) from {len(self.mcp_servers_config)} server(s)"
                )
            if manager.connection_errors:
                for err in manager.connection_errors:
                    self.logger.warning(
                        f"[MCP] Connection error for '{err.get('name')}' ({err.get('url')}): {err.get('error')}"
                    )
        except Exception as e:
            self.logger.error(f"[MCP] Failed to initialize MCP sessions: {sanitize_for_logging(str(e))}")
    
    @staticmethod
    def _parse_pdf_sync(pdf_bytes: bytes, filename: str) -> str:
        """Parse PDF synchronously - to be run in thread pool"""
        from io import BytesIO
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
        pdf_text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                pdf_text += page_text
        return f"=== {filename} ===\n{pdf_text}"

    @staticmethod
    async def _load_single_source(source: Dict, logger: logging.Logger) -> Optional[str]:
        """Load a single knowledge source asynchronously"""
        source_type = source["type"]
        source_id = source["id"]
        source_content = source["content"]
        
        try:
            if source_type == "text":
                logger.debug(f"Loaded text knowledge source: {source_id}")
                return source_content

            elif source_type == "file":
                file_path = f"{source.get('filename')}"
                actual_file_path = f"knowledge/{file_path}"
                
                # Use async path check
                exists = await asyncio.get_event_loop().run_in_executor(
                    _IO_THREAD_POOL, os.path.exists, actual_file_path
                )
                if not actual_file_path or not exists:
                    logger.error(f"File path missing or invalid for source: {source_id} ({source_content})")
                    return None

                file_extension = os.path.splitext(file_path)[1].lower()
                filename = source.get('filename', file_path)
                
                if file_extension == ".txt":
                    async with aiofiles.open(actual_file_path, "r", encoding="utf-8") as f:
                        file_content = await f.read()
                        logger.debug(f"Loaded text file knowledge source: {source_id} ({file_path})")
                        return f"=== {filename} ===\n{file_content}"
                        
                elif file_extension == ".pdf":
                    try:
                        # Read file content asynchronously
                        async with aiofiles.open(actual_file_path, "rb") as pdf_file:
                            pdf_bytes = await pdf_file.read()
                        # Parse PDF in thread pool (CPU-bound operation)
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            _IO_THREAD_POOL,
                            LangGraphExecutor._parse_pdf_sync,
                            pdf_bytes,
                            filename
                        )
                        logger.debug(f"Loaded PDF knowledge source: {source_id} ({file_path})")
                        return result
                    except Exception as e:
                        logger.error(f"Error reading PDF {file_path}: {str(e)}")
                        return None
                        
                elif file_extension == ".csv":
                    async with aiofiles.open(actual_file_path, "r", encoding="utf-8") as f:
                        csv_content = await f.read()
                        logger.debug(f"Loaded CSV knowledge source: {source_id} ({file_path})")
                        return f"=== {filename} ===\n{csv_content}"
                else:
                    logger.warning(f"Unsupported file extension: {file_extension} for source: {source_id}")
                    return None
            else:
                logger.warning(f"Unsupported knowledge source type: {source_type} for source: {source_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading knowledge source {source_id}: {sanitize_for_logging(str(e))}")
            return None

    @staticmethod
    def _compute_sources_hash(sources: List[Dict]) -> str:
        """Compute hash of knowledge sources for cache validation"""
        # Include source IDs, types, and file modification times
        hash_parts = []
        for source in sources:
            source_id = source.get("id", "")
            source_type = source.get("type", "")
            
            if source_type == "file":
                filename = source.get("filename", "")
                file_path = f"knowledge/{filename}"
                try:
                    mtime = os.path.getmtime(file_path) if os.path.exists(file_path) else 0
                except:
                    mtime = 0
                hash_parts.append(f"{source_id}:{source_type}:{filename}:{mtime}")
            else:
                # For text sources, hash the content
                content_hash = hashlib.md5(str(source.get("content", "")).encode()).hexdigest()[:8]
                hash_parts.append(f"{source_id}:{source_type}:{content_hash}")
        
        combined = "|".join(sorted(hash_parts))
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    @staticmethod
    async def _load_knowledge_sources(knowledge_base_id: Optional[str] = None, executor_logger: Optional[logging.Logger] = None) -> List[str]:
        """
        Load knowledge sources from knowledge_base.json based on knowledge_base_id.
        Returns a list of knowledge content strings.
        
        OPTIMIZED: 
        - Uses caching to avoid re-parsing unchanged files
        - Loads all sources in parallel using asyncio.gather()
        """
        logger = executor_logger or logging.getLogger("langraph_executor_loader")
        if not knowledge_base_id:
            logger.debug("No knowledge_base_id provided in agent config")
            return []

        try:
            async with aiofiles.open("knowledge_base.json", "r", encoding="utf-8") as f:
                content = await f.read()
                knowledge_data = json.loads(content)
            
            kb_entry = next((entry for entry in knowledge_data if entry["id"] == knowledge_base_id), None)
            if not kb_entry:
                logger.error(f"No knowledge base found for ID: {knowledge_base_id}")
                return []

            # Compute hash for cache validation (includes file modification times)
            sources_hash = LangGraphExecutor._compute_sources_hash(kb_entry["sources"])
            
            # Check cache first
            cached_content = await _KNOWLEDGE_CACHE.get(knowledge_base_id, sources_hash)
            if cached_content is not None:
                logger.info(f"Knowledge cache HIT for {knowledge_base_id} ({len(cached_content)} sources)")
                return cached_content

            logger.info(f"Knowledge cache MISS - Loading sources for: {kb_entry['title']} (ID: {knowledge_base_id})")
            
            # Load all sources in parallel
            load_tasks = [
                LangGraphExecutor._load_single_source(source, logger)
                for source in kb_entry["sources"]
            ]
            
            # Execute all loads concurrently
            results = await asyncio.gather(*load_tasks, return_exceptions=True)
            
            # Filter out None values and exceptions
            knowledge_contents = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    source_id = kb_entry["sources"][i].get("id", f"index_{i}")
                    logger.error(f"Exception loading source {source_id}: {sanitize_for_logging(str(result))}")
                elif result is not None:
                    knowledge_contents.append(result)

            # Cache the results
            await _KNOWLEDGE_CACHE.set(knowledge_base_id, sources_hash, knowledge_contents)
            logger.info(f"Cached {len(knowledge_contents)} knowledge sources for knowledge base ID: {knowledge_base_id}")
            
            return knowledge_contents

        except Exception as e:
            logger.error(f"Error loading knowledge_base.json: {sanitize_for_logging(str(e))}")
            return []

    def _build_graph(self):
        """Build the LangGraph execution graph"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("action", self._action_node)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "action",
                "end": END
            }
        )
        
        # Add edge from action back to agent
        workflow.add_edge("action", "agent")
        
        # Compile the graph
        self.graph = workflow.compile()
        self.logger.info("LangGraph workflow compiled successfully")

    async def _agent_node(self, state: AgentState) -> AgentState:
        """Agent reasoning node"""
        self.logger.debug(f"[Agent Node] Starting reasoning cycle")
        messages = state["messages"]
        self.logger.debug(f"[Agent Node] Processing {len(messages)} messages in state")
        
        # Get current date for prompt
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Process agent configuration texts to handle special characters:
        # The agent config may contain {current_date} which we want to replace,
        # but may also contain other { } characters (JSON, special formatting) 
        # that should be treated as literal text, not template variables.
        def process_config_text(text: str) -> str:
            if not text:
                return text
            # Step 1: Replace {current_date} placeholder with actual date
            # After this, {current_date} is gone and replaced with "2026-02-24"
            processed = text.replace(CURRENT_DATE_PLACEHOLDER, current_date)
            # Step 2: Escape remaining { and } by doubling them
            # LangChain templates treat {{ as literal {, and }} as literal }
            # This prevents errors like "missing variable" for things like {some_json_key}
            processed = processed.replace("{", "{{").replace("}", "}}")
            return processed
        
        agent_role = process_config_text(self.agent_role) if self.agent_role else "AI Assistant"
        agent_goal = process_config_text(self.agent_goal) if self.agent_goal else "Help the user with their request"
        agent_backstory = process_config_text(self.agent_backstory) if self.agent_backstory else ""
        
        # Create system message with agent configuration
        # Now all curly braces are escaped, so LangChain won't try to interpret them as variables
        system_prompt = f"""You are {agent_role}.
Goal: {agent_goal}
Backstory: {agent_backstory}"""

        # Add task instructions if available (from agent's instructions field)
        if hasattr(self, 'task_description') and self.task_description:
            # Process the task description to escape curly braces
            task_desc = process_config_text(self.task_description)
            system_prompt += f"""

Instructions:
{task_desc}"""

        # Add expected output format if available
        if hasattr(self, 'task_expected_output') and self.task_expected_output:
            expected_out = process_config_text(self.task_expected_output)
            system_prompt += f"""

Expected Output Format:
{expected_out}"""

        # Add knowledge base content if available
        if self.knowledge_content:
            # CRITICAL FIX: Escape curly braces in knowledge content.
            # Knowledge sources (PDFs, text files, guidelines) may contain JSON-like
            # strings such as {"brand_name"} which LangChain interprets as template
            # variables, causing "missing variable" errors at prompt invocation time.
            escaped_knowledge = process_config_text(self.knowledge_content)
            system_prompt += f"""

Knowledge Base:
{escaped_knowledge}

Use the information from the knowledge base above to answer the user's questions accurately."""

        system_prompt += "\n\nRespond to the user's request thoughtfully and thoroughly. If you have used tools to gather information, you MUST use the data returned by the tools to provide a clear, comprehensive final answer. Do NOT provide an empty response."
        system_prompt += "\n\nIMPORTANT — When passing data from one tool's response as input to another tool, follow your task instructions carefully. If your instructions say to filter, transform, or select specific items, do so. Otherwise, pass the complete data as received — do NOT arbitrarily truncate or select only a few items without a clear reason."

        
        # Bind tools to LLM if available.
        # _LoggingFallbackLLM.bind_tools() handles both primary + fallback internally.
        if self.tools:
            llm_with_tools = self._bind_tools_with_fallbacks(self.tools)
            self.logger.debug(f"[Agent Node] Bound {len(self.tools)} tools to LLM")
        else:
            llm_with_tools = self.llm
            self.logger.debug(f"[Agent Node] No tools available, using LLM directly")
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        # Invoke the LLM (with semaphore to limit concurrent requests)
        self.logger.debug(f"[Agent Node] Invoking LLM for reasoning")
        
        # Log LLM provider info for debugging
        try:
            llm_base = self.llm
            # _LoggingFallbackLLM wraps primary in _primary, fallback in _fallback
            if hasattr(llm_base, '_primary'):
                primary_llm = llm_base._primary
                self.logger.info(f"[Agent Node] LLM primary class: {type(primary_llm).__name__}, model: {getattr(primary_llm, 'model_name', getattr(primary_llm, 'model', 'unknown'))}")
                if hasattr(llm_base, '_fallback'):
                    fb = llm_base._fallback
                    self.logger.info(f"[Agent Node] LLM fallback class: {type(fb).__name__}, model: {getattr(fb, 'model_name', getattr(fb, 'model', 'unknown'))}")
            else:
                self.logger.info(f"[Agent Node] LLM provider class: {type(llm_base).__name__}, model: {getattr(llm_base, 'model_name', getattr(llm_base, 'model', 'unknown'))}")
        except Exception:
            pass
        
        chain = prompt | llm_with_tools
        async with _LLM_SEMAPHORE:
            response = await chain.ainvoke({"messages": messages})
        
        # --- Fix B: Detect and handle MALFORMED_FUNCTION_CALL from Vertex AI ---
        response_metadata = getattr(response, 'response_metadata', {})
        finish_reason = response_metadata.get('finish_reason', '')
        
        if finish_reason == 'MALFORMED_FUNCTION_CALL':
            finish_message = response_metadata.get('finish_message', '')
            self.logger.warning(f"[Agent Node] MALFORMED_FUNCTION_CALL detected from Vertex AI")
            self.logger.warning(f"[Agent Node] Malformed call details: {finish_message[:500]}")
            
            # Extract the original user query from messages for fallback
            original_user_query = None
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    original_user_query = msg.content
                    break
            
            # Attempt to recover: parse the malformed function call and reconstruct tool calls
            recovered_tool_call = self._recover_malformed_tool_call(finish_message, original_user_query)
            if recovered_tool_call:
                self.logger.info(f"[Agent Node] Successfully recovered tool call: {recovered_tool_call['name']}")
                # Create a new AIMessage with the recovered tool call
                import uuid
                tool_call_id = str(uuid.uuid4())
                response = AIMessage(
                    content="",
                    tool_calls=[{
                        "id": tool_call_id,
                        "name": recovered_tool_call["name"],
                        "args": recovered_tool_call["args"]
                    }]
                )
            else:
                # Fix C: Recovery failed — raise so the retry loop (and with_fallbacks) can kick in
                self.logger.error(f"[Agent Node] Could not recover malformed tool call, raising for retry/fallback")
                raise ValueError(
                    f"Vertex AI MALFORMED_FUNCTION_CALL: model produced an unparseable tool call. "
                    f"Details: {finish_message[:200]}"
                )
        
        # Log response metadata for observability
        if response_metadata:
            model_name = response_metadata.get('model_name', 'unknown')
            usage = response_metadata.get('usage_metadata', {})
            self.logger.info(f"[Agent Node] Response metadata: model={model_name}, finish_reason={finish_reason}, token_usage={usage}")
        
        # Log response details
        self.logger.info(f"[Agent Node] LLM responded with message type: {type(response).__name__}")
        if hasattr(response, "tool_calls") and response.tool_calls:
            self.logger.info(f"[Agent Node] LLM requested {len(response.tool_calls)} tool call(s)")
            for tc in response.tool_calls:
                self.logger.debug(f"[Agent Node] Tool call: {tc.get('name', 'unknown')}")
        else:
            self.logger.info(f"[Agent Node] LLM provided final response (no tool calls)")
            content_snippet = str(response.content)[:100] if response.content else "EMPTY"
            self.logger.debug(f"[Agent Node] Final response content snippet: {content_snippet}")
            
            # Detect empty responses early for better diagnostics
            if not response.content or (isinstance(response.content, str) and not response.content.strip()):
                self.logger.warning(f"[Agent Node] EMPTY RESPONSE DETECTED - Dumping full diagnostics")
                self.logger.warning(f"[Agent Node] response.content repr: {repr(response.content)}")
                self.logger.warning(f"[Agent Node] response.type: {getattr(response, 'type', 'unknown')}")
                self.logger.warning(f"[Agent Node] finish_reason: {finish_reason}")
                if response_metadata:
                    self.logger.warning(f"[Agent Node] Full response_metadata: {response_metadata}")
        
        # Update state
        new_messages = list(messages)
        new_messages.append(response)
        
        self.logger.debug(f"[Agent Node] Updated state with {len(new_messages)} total messages")
        
        return {
            **state,
            "messages": new_messages
        }
    
    def _recover_malformed_tool_call(self, finish_message: str, original_user_query: Optional[str] = None) -> Optional[Dict]:
        """Attempt to parse a malformed Vertex AI function call and extract the tool name + args.
        
        Vertex AI sometimes produces calls like:
          print(default_api.text_to_sql_api(natural_language_query='...', sub_queries=[...]))
        
        We try to extract the tool name and arguments from this pattern.
        When the finish_message is truncated (common), we fall back to using the
        original user query as input_data so the downstream payload generator
        gets a complete, meaningful query.
        """
        if not finish_message:
            return None
        
        try:
            import re
            
            # Pattern: Malformed function call: print(default_api.TOOL_NAME(args...))
            # or:      Malformed function call: default_api.TOOL_NAME(args...)
            # or:      Malformed function call: TOOL_NAME(args...)
            match = re.search(
                r'(?:print\()?(?:default_api\.)?([a-zA-Z_][a-zA-Z0-9_]*)\((.+)',
                finish_message
            )
            if not match:
                self.logger.debug(f"[Recovery] No function call pattern found in finish_message")
                # Even without a pattern match, if we have exactly one tool, use it with the original query
                if original_user_query and len(self.tools) == 1:
                    tool_name = self.tools[0].name
                    self.logger.info(f"[Recovery] No pattern match but single tool available, using '{tool_name}' with original user query")
                    return {"name": tool_name, "args": {"input_data": original_user_query}}
                return None
            
            tool_name = match.group(1)
            args_str = match.group(2)
            
            # Verify this is one of our registered tools
            tool_names = [t.name for t in self.tools]
            if tool_name not in tool_names:
                self.logger.debug(f"[Recovery] Extracted tool '{tool_name}' not in registered tools: {tool_names}")
                return None
            
            # Try to extract the natural_language_query if present
            nlq_match = re.search(r"natural_language_query=['\"](.+?)['\"]", args_str)
            extracted_nlq = nlq_match.group(1) if nlq_match else args_str
            
            # Detect truncation: Vertex AI's finish_message is often cut off.
            # Signs of truncation: doesn't end with closing quote/paren, or ends abruptly.
            is_truncated = (
                not extracted_nlq.rstrip().endswith(('.', '?', '!', ')', "'", '"'))
                or len(extracted_nlq) < 20
            )
            
            if is_truncated and original_user_query:
                self.logger.info(
                    f"[Recovery] Detected truncated query from finish_message "
                    f"('{extracted_nlq[:50]}...'), using original user query instead: "
                    f"'{original_user_query[:100]}'"
                )
                recovered_args = {"input_data": original_user_query}
            else:
                recovered_args = {"input_data": extracted_nlq}
            
            self.logger.info(f"[Recovery] Recovered tool call: name={tool_name}, input_data_length={len(recovered_args['input_data'])}")
            return {"name": tool_name, "args": recovered_args}
            
        except Exception as e:
            self.logger.warning(f"[Recovery] Failed to parse malformed tool call: {e}")
            return None

    async def _action_node(self, state: AgentState) -> AgentState:
        """Tool execution node"""
        self.logger.debug(f"[Action Node] Starting tool execution")
        messages = state["messages"]
        last_message = messages[-1]
        
        # Execute tools if there are tool calls
        tool_messages = []
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            self.logger.info(f"[Action Node] Processing {len(last_message.tool_calls)} tool call(s)")
            
            for idx, tool_call in enumerate(last_message.tool_calls, 1):
                # Find and execute the tool
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # Pretty tool name for status updates
                pretty_tool_name = tool_name.replace("_", " ").title()
                
                self.logger.info(f"[Action Node] [{idx}/{len(last_message.tool_calls)}] Executing tool: {tool_name}")
                
                # Status update: Processing tool
                try:
                    if self.status_queue:
                        await self.status_queue.put(f"Processing Tool \"{pretty_tool_name}\"")
                except:
                    pass
                
                # Normalize tool args for the tool function.
                # The tool function accepts input_data="" + **kwargs, so typed fields
                # from the rich schema (e.g. natural_language_query) can be passed directly.
                # Only wrap into input_data if the tool uses the simple schema.
                if "input_data" not in tool_args and tool_args:
                    # Check if any of the args match the tool's schema fields (rich schema)
                    tool_obj = next((t for t in self.tools if t.name == tool_name), None)
                    if tool_obj and hasattr(tool_obj, 'args_schema'):
                        schema_fields = set(tool_obj.args_schema.model_fields.keys()) if hasattr(tool_obj.args_schema, 'model_fields') else set()
                        has_typed_fields = bool(schema_fields - {'input_data'})
                        args_match_schema = any(k in schema_fields for k in tool_args.keys())
                        
                        if has_typed_fields and args_match_schema:
                            # Args match the rich schema — pass them through as-is.
                            # The tool function's **kwargs will handle them.
                            self.logger.info(f"[Action Node] Tool args match rich schema fields — passing through directly")
                        else:
                            self.logger.info(f"[Action Node] Tool args missing 'input_data' field — wrapping LLM args as input_data string")
                            tool_args = {"input_data": json.dumps(tool_args)}
                    else:
                        self.logger.info(f"[Action Node] Tool args missing 'input_data' field — wrapping LLM args as input_data string")
                        tool_args = {"input_data": json.dumps(tool_args)}
                elif "input_data" in tool_args and isinstance(tool_args["input_data"], dict):
                    # Fix: LLM passed input_data as a dict instead of a JSON string
                    self.logger.info(f"[Action Node] Tool 'input_data' is a dict — serializing to JSON string")
                    tool_args = {"input_data": json.dumps(tool_args["input_data"])}
                
                # Find the tool in our tools list
                tool_result = None
                tool_found = False
                for tool in self.tools:
                    if tool.name == tool_name:
                        tool_found = True
                        try:
                            start_time = time.time()
                            
                            # Status update: Invoking tool
                            try:
                                if self.status_queue:
                                    await self.status_queue.put(f"Invoking \"{pretty_tool_name}\"")
                            except:
                                pass
                            
                            # Execute the tool using ainvoke (async) or invoke (sync)
                            # StructuredTool has ainvoke for async and invoke for sync
                            if hasattr(tool, 'ainvoke'):
                                tool_result = await tool.ainvoke(tool_args)
                            elif hasattr(tool, 'invoke'):
                                tool_result = tool.invoke(tool_args)
                            else:
                                # Fallback to direct function call
                                tool_result = await tool.coroutine(**tool_args) if tool.coroutine else tool.func(**tool_args)
                            
                            execution_time = time.time() - start_time
                            self.logger.info(f"[Action Node] Tool {tool_name} executed successfully in {execution_time:.2f}s")
                            self.logger.debug(f"[Action Node] Tool result type: {type(tool_result).__name__}")
                            
                            # Status update: Completed tool
                            try:
                                if self.status_queue:
                                    await self.status_queue.put(f"Completed Tool \"{pretty_tool_name}\"")
                            except:
                                pass
                            
                        except Exception as e:
                            self.logger.error(f"[Action Node] Tool {tool_name} execution error: {sanitize_for_logging(str(e))}")
                            import traceback
                            self.logger.error(f"[Action Node] Traceback:\n{traceback.format_exc()}")
                            tool_result = json.dumps({"error": str(e)})
                        break
                
                if not tool_found:
                    self.logger.error(f"[Action Node] Tool {tool_name} not found in available tools")
                    self.logger.debug(f"[Action Node] Available tools: {[t.name for t in self.tools]}")
                    tool_result = json.dumps({"error": f"Tool {tool_name} not found"})
                
                # Convert tool result to string for ToolMessage
                if isinstance(tool_result, (dict, list)):
                    tool_result_str = json.dumps(tool_result)
                    self.logger.debug(f"[Action Node] Converted dict/list result to JSON string")
                else:
                    tool_result_str = str(tool_result)
                
                # Truncate oversized tool responses to prevent MAX_TOKENS on the next LLM call.
                # Gemini 2.5 Flash has a ~1M token input window; 200K chars (~50K tokens)
                # leaves plenty of headroom for system prompt, knowledge base, history, and
                # subsequent tool calls. Configurable via env var.
                MAX_TOOL_RESPONSE_CHARS = int(os.getenv("MAX_TOOL_RESPONSE_CHARS", "200000"))
                if len(tool_result_str) > MAX_TOOL_RESPONSE_CHARS:
                    self.logger.warning(
                        f"[Action Node] Tool response too large ({len(tool_result_str)} chars), truncating to {MAX_TOOL_RESPONSE_CHARS} chars"
                    )
                    tool_result_str = tool_result_str[:MAX_TOOL_RESPONSE_CHARS] + "\n\n...[TRUNCATED — full response was " + str(len(tool_result_str)) + " chars]"
                
                tool_message = ToolMessage(
                    content=tool_result_str,
                    tool_call_id=tool_call["id"]
                )
                tool_messages.append(tool_message)
                self.logger.debug(f"[Action Node] Created ToolMessage with {len(tool_result_str)} characters")
        else:
            self.logger.warning(f"[Action Node] Called but no tool_calls found in last message")
        
        new_messages = list(messages)
        new_messages.extend(tool_messages)
        
        self.logger.info(f"[Action Node] Completed, added {len(tool_messages)} tool message(s) to state")
        self.logger.debug(f"[Action Node] Total messages in state: {len(new_messages)}")
        
        return {
            **state,
            "messages": new_messages
        }

    def _should_continue(self, state: AgentState) -> str:
        """Determine if we should continue or end"""
        messages = state["messages"]
        last_message = messages[-1]
        
        # If there are tool calls, continue
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            self.logger.debug(f"[Router] Routing to ACTION node (tool calls detected)")
            return "continue"
        
        # Otherwise end
        self.logger.debug(f"[Router] Routing to END (no tool calls)")
        return "end"

    async def execute_task(
        self,
        description: str,
        expected_output: str,
        task_name: Optional[str] = None,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        input: Optional[str] = None,
        user_email: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        status_queue: Optional[asyncio.Queue] = None,
        **kwargs
    ) -> str:
        """Execute a task using LangGraph
        
        Args:
            description: Agent's instructions (from agent.instructions field)
            expected_output: Expected output format
            input: User's actual input/query
        """
        
        # Store description (instructions) for use in the agent node
        self.task_description = description
        self.task_expected_output = expected_output
        
        self.logger.info(f"=== Starting Task Execution ===")
        self.logger.info(f"Task: {task_name or 'Unnamed'}")
        
        # Set execution context
        if input:
            short_id = str(uuid.uuid4())[:6]
            temp_memory_dir = f"temp/lm_{short_id}"
            temp_cache_dir = f"temp/lc_{short_id}"
            
            # Run directory creation in thread pool (non-blocking)
            loop = asyncio.get_event_loop()
            await asyncio.gather(
                loop.run_in_executor(_IO_THREAD_POOL, lambda: os.makedirs(temp_memory_dir, exist_ok=True)),
                loop.run_in_executor(_IO_THREAD_POOL, lambda: os.makedirs(temp_cache_dir, exist_ok=True))
            )
            
            set_execution_context(short_id, input, {
                'temp_memory_dir': temp_memory_dir,
                'temp_cache_dir': temp_cache_dir,
                'execution_id': self.execution_id
            })
            self.expected_user_input = input
            self.logger.info(f"Execution context set - ID: {short_id}")
            self.logger.debug(f"Temp dirs created: {temp_memory_dir}, {temp_cache_dir}")
        
        # Store status_queue for tool access
        self.status_queue = status_queue
        self.logger.debug(f"Status queue: {'Enabled' if status_queue else 'Disabled'}")
        
        max_retries = 3
        retry_count = 0
        task_start_time = time.time()
        
        while retry_count < max_retries:
            try:
                if status_queue:
                    await status_queue.put("Processing with LangGraph...")
                
                attempt_start = time.time()
                self.logger.info(f"Starting LangGraph execution - Attempt {retry_count + 1}/{max_retries}")
                
                # Prepare messages
                messages = []
                
                # Add history if available
                if history:
                    self.logger.debug(f"Loading {len(history)} history entries (using last 5)")
                    for idx, entry in enumerate(history[-5:], 1):  # Last 5 entries
                        messages.append(HumanMessage(content=entry.get("input", "")))
                        messages.append(AIMessage(content=entry.get("response", "")))
                    self.logger.debug(f"Added {len(history[-5:]) * 2} messages from history")
                
                # Add current input
                user_message = input or description
                if file_path and file_type:
                    user_message = f"{user_message}\n\n[File attached: {file_path} (type: {file_type})]"
                    self.logger.debug(f"File attached: {file_type}")
                
                messages.append(HumanMessage(content=user_message))
                self.logger.debug(f"Total messages prepared: {len(messages)}")
                
                # Create initial state
                initial_state = {
                    "messages": messages,
                    "user_input": input or description,
                    "file_path": file_path,
                    "file_type": file_type,
                    "history": history or [],
                    "intermediate_steps": [],
                    "final_output": None
                }
                
                self.logger.debug(f"Initial state created with {len(messages)} messages")
                self.logger.debug(f"File info: {file_path or 'None'}")
                
                # Execute the graph
                self.logger.info(">>> Executing LangGraph workflow <<<")
                graph_start = time.time()
                final_state = await self.graph.ainvoke(initial_state)
                graph_duration = time.time() - graph_start
                self.logger.info(f">>> Graph execution completed in {graph_duration:.2f}s <<<")
                
                # Extract final response
                final_messages = final_state["messages"]
                self.logger.debug(f"Graph execution complete, total messages: {len(final_messages)}")
                
                if final_messages:
                    last_message = final_messages[-1]
                    self.logger.debug(f"Last message type: {type(last_message).__name__}")
                    
                    if isinstance(last_message, AIMessage):
                        # Handle different content types from Gemini
                        content = last_message.content
                        self.logger.debug(f"AIMessage content type: {type(content).__name__}")
                        
                        if isinstance(content, list):
                            # Gemini sometimes returns list of content parts
                            self.logger.info("Processing list-based AIMessage content from Gemini")
                            result_parts = []
                            for part in content:
                                if isinstance(part, dict):
                                    if 'text' in part:
                                        result_parts.append(part['text'])
                                    elif 'type' in part and part['type'] == 'text':
                                        result_parts.append(part.get('text', ''))
                                else:
                                    result_parts.append(str(part))
                            result = '\n'.join(result_parts) if result_parts else str(content)
                            self.logger.debug(f"Extracted {len(result_parts)} text parts from list")
                        elif isinstance(content, str):
                            result = content
                        else:
                            # Fallback for other types
                            self.logger.warning(f"Unexpected content type: {type(content).__name__}, converting to string")
                            result = str(content)
                    else:
                        result = str(last_message.content)
                else:
                    result = "No response generated"
                    self.logger.warning("No messages in final state")
                
                # CRITICAL FIX: If result is empty, raise an error to trigger a retry
                if not result or (isinstance(result, str) and not result.strip()):
                    self.logger.warning(f"Empty result generated on attempt {retry_count + 1}")
                    raise ValueError("LLM generated an empty response content")
                
                attempt_duration = time.time() - attempt_start
                total_duration = time.time() - task_start_time
                
                self.logger.info(f"=== Task Execution Successful ===")
                self.logger.info(f"Attempt duration: {attempt_duration:.2f}s")
                self.logger.info(f"Total duration: {total_duration:.2f}s")
                self.logger.info(f"Result length: {len(result)} characters")
                
                if status_queue:
                    await status_queue.put("Task completed successfully")
                
                return result
                
            except Exception as e:
                retry_count += 1
                attempt_duration = time.time() - attempt_start if 'attempt_start' in locals() else 0
                
                self.logger.error(f"!!! Error during LangGraph execution !!!")
                self.logger.error(f"Attempt {retry_count}/{max_retries} failed after {attempt_duration:.2f}s")
                self.logger.error(f"Error: {sanitize_for_logging(str(e))}")
                
                # Log full traceback
                import traceback
                self.logger.error(f"Traceback:\n{traceback.format_exc()}")
                
                if retry_count >= max_retries:
                    total_duration = time.time() - task_start_time
                    error_msg = f"LangGraph execution failed after {max_retries} attempts: {str(e)}"
                    self.logger.error(f"=== Task Execution Failed ===")
                    self.logger.error(f"Total duration: {total_duration:.2f}s")
                    self.logger.error(error_msg)
                    
                    if status_queue:
                        await status_queue.put(f"Error: {error_msg}")
                    
                    return error_msg
                
                # Wait before retry with exponential backoff
                wait_time = 2 ** retry_count
                self.logger.info(f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

    async def cleanup(self):
        """Cleanup resources asynchronously"""
        self.logger.info(f"Cleaning up LangGraph executor {self.execution_id}")
        
        # Close MCP sessions if any
        if self._mcp_manager is not None:
            try:
                await self._mcp_manager.close()
            except Exception as e:
                self.logger.warning(f"[MCP] Error during MCP cleanup: {sanitize_for_logging(str(e))}")
            finally:
                self._mcp_manager = None
        
        # Clear execution context
        _EXECUTION_CONTEXT.set(None)
        
        # Cleanup temporary directories asynchronously
        context = get_execution_context()
        if context:
            temp_memory_dir = context.get('temp_memory_dir')
            temp_cache_dir = context.get('temp_cache_dir')
            
            async def remove_dir(temp_dir: str):
                if temp_dir:
                    loop = asyncio.get_event_loop()
                    exists = await loop.run_in_executor(_IO_THREAD_POOL, os.path.exists, temp_dir)
                    if exists:
                        try:
                            import shutil
                            await loop.run_in_executor(_IO_THREAD_POOL, shutil.rmtree, temp_dir)
                            self.logger.debug(f"Removed temporary directory: {temp_dir}")
                        except Exception as e:
                            self.logger.warning(f"Failed to remove temporary directory {temp_dir}: {e}")
            
            # Remove directories in parallel
            await asyncio.gather(
                remove_dir(temp_memory_dir),
                remove_dir(temp_cache_dir),
                return_exceptions=True
            )
        
        self.logger.info("LangGraph executor cleanup completed")

# Cleanup function for old temporary directories
def cleanup_old_tmp_directories(max_age_hours: int = 24, dry_run: bool = False) -> Dict[str, Any]:
    """
    Clean up old LangGraph temporary directories
    
    Args:
        max_age_hours: Maximum age in hours for temporary directories
        dry_run: If True, only report what would be deleted without actually deleting
        
    Returns:
        Dictionary with cleanup statistics
    """
    import shutil
    from pathlib import Path
    
    stats = {
        "total_found": 0,
        "deleted": 0,
        "failed": 0,
        "skipped": 0,
        "size_freed_mb": 0
    }
    
    temp_dir = Path("temp")
    if not temp_dir.exists():
        return stats
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    # Look for LangGraph temp directories
    patterns = ["lm_*", "lc_*", "langraph_*"]
    
    for pattern in patterns:
        for path in temp_dir.glob(pattern):
            if path.is_dir():
                stats["total_found"] += 1
                
                try:
                    # Check age
                    dir_age = current_time - path.stat().st_mtime
                    
                    if dir_age > max_age_seconds:
                        # Calculate size
                        dir_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
                        
                        if not dry_run:
                            shutil.rmtree(path)
                            stats["deleted"] += 1
                            stats["size_freed_mb"] += dir_size / (1024 * 1024)
                        else:
                            stats["skipped"] += 1
                    else:
                        stats["skipped"] += 1
                        
                except Exception as e:
                    stats["failed"] += 1
                    logging.error(f"Failed to process directory {path}: {e}")
    
    return stats


# ============================================================================
# CLEANUP AND SHUTDOWN UTILITIES
# ============================================================================

async def cleanup_shared_resources():
    """
    Cleanup shared resources on application shutdown.
    Call this when the application is shutting down.
    """
    global _AIOHTTP_CONNECTOR
    
    # Close the shared aiohttp connector
    if _AIOHTTP_CONNECTOR is not None and not _AIOHTTP_CONNECTOR.closed:
        await _AIOHTTP_CONNECTOR.close()
        _AIOHTTP_CONNECTOR = None
    
    # Shutdown the thread pool
    _IO_THREAD_POOL.shutdown(wait=False)
    
    logging.info("LangGraph shared resources cleaned up")


async def clear_all_caches():
    """Clear all caches - useful for testing or memory pressure"""
    await _KNOWLEDGE_CACHE.invalidate()
    await _PAYLOAD_CACHE._lock.acquire()
    try:
        _PAYLOAD_CACHE._cache.clear()
    finally:
        _PAYLOAD_CACHE._lock.release()
    logging.info("All LangGraph caches cleared")