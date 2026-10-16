import logging
import os
# # ####
# from ddtrace import patch_all, tracer
# from ddtrace.profiling import Profiler
# from ddtrace.runtime import RuntimeMetrics
# from ddtrace.contrib.asgi import TraceMiddleware
# patch_all()
# Profiler().start()
# RuntimeMetrics.enable()
###
from datetime import datetime, timezone
from crewai import Crew, Process, Task, Agent as CrewAgent, LLM
from crewai.tools import BaseTool
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.source.crew_docling_source import CrewDoclingSource
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from crewai.knowledge.source.excel_knowledge_source import ExcelKnowledgeSource
from crewai.knowledge.knowledge_config import KnowledgeConfig

from pydantic import BaseModel, Field
import requests
import aiohttp
from dotenv import load_dotenv
from functools import partial
import json
from json.decoder import JSONDecodeError
import base64
import csv
from typing import Optional, Dict, Any, List
import PyPDF2
from PIL import Image
import random
import time
import asyncio
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, HttpUrl, ValidationError
import re
import aiofiles
from services.crewai_llm import get_crewai_llm, get_crewai_embedder
import threading
import uuid
import contextvars
import gc
import weakref
# import ssl

from datetime import datetime, timedelta
import jwt

# ssl._create_default_https_context = ssl._create_unverified_context


load_dotenv()

# CRITICAL FIX: Add execution context isolation
_EXECUTION_CONTEXT = contextvars.ContextVar('execution_context', default=None)



# Vertex AI embeddings for knowledge — local pulls the API key from Secret
# Manager (settings.VERTEX_API_KEY); production mints a short-lived ADC access
# token. Resolved per-execution (see get_crewai_embedder call at Crew creation)
# so the production token never goes stale.

# Retrieval tuning only (supported fields)
knowledge_config = KnowledgeConfig(
        results_limit=5,
        score_threshold=0.4
    )

def validate_path_length(path: str, max_length: int = 200) -> str:
    """
    Validate and truncate paths to prevent filesystem length issues.
    Returns a safe path that won't exceed filesystem limits.
    """
    if len(path) <= max_length:
        return path
    
    # If path is too long, create a shorter version with hash
    import hashlib
    path_hash = hashlib.md5(path.encode()).hexdigest()[:8]
    safe_path = f"temp/crew_safe_{path_hash}"
    return safe_path

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
        'memory_path': f"temp/crewai_memory_{execution_id}",
        'cache_path': f"temp/crewai_cache_{execution_id}",
        **(additional_context or {})
    }
    _EXECUTION_CONTEXT.set(context)

# CRITICAL FIX: Force disable all CrewAI caching and shared state
os.environ['CREWAI_DISABLE_CACHE'] = 'true'
os.environ['CREWAI_DISABLE_TELEMETRY'] = 'true'
os.environ['CREWAI_DISABLE_SHARED_STATE'] = 'true'
# Additional isolation based on research findings
os.environ['CREWAI_DISABLE_MEMORY'] = 'true'
os.environ['CREWAI_DISABLE_PLANNING_CACHE'] = 'true'
os.environ['CREWAI_DISABLE_AGENTS_CACHE'] = 'true'
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# CRITICAL FIX: Control embedding storage to prevent long paths
os.environ['CREWAI_EMBEDDING_CACHE_DIR'] = 'temp/embed_cache'
os.environ['CREWAI_VECTOR_STORE_DIR'] = 'temp/vector_store'
os.environ['CREWAI_DISABLE_KICKOFF_PERSISTENCE'] = 'true'

# ULTIMATE FIX: Override CrewAI's internal path generation to prevent long paths
import sys
from pathlib import Path

# Global variable to track current knowledge base storage paths
_CURRENT_KB_STORAGE = {}

def safe_db_storage_path():
    """Override for CrewAI's db_storage_path function to return safe paths"""
    return Path("temp/crewai_db")

def get_or_create_kb_storage(kb_id: str) -> str:
    """
    Get existing storage path for knowledge base or create a new one.
    Reuses directories for the same knowledge base ID.
    """
    if kb_id in _CURRENT_KB_STORAGE:
        existing_path = _CURRENT_KB_STORAGE[kb_id]
        if os.path.exists(existing_path):
            return existing_path
    
    # Create new storage path for this knowledge base
    kb_short_id = kb_id[:8] if len(kb_id) > 8 else kb_id
    base_storage_dir = os.environ.get("CREWAI_BASE_STORAGE_DIR", "temp/crewai_storage")
    storage_path = f"{base_storage_dir}_kb_{kb_short_id}"
    
    # Validate path length
    safe_path = validate_path_length(storage_path, max_length=180)
    
    # Create directory if it doesn't exist
    os.makedirs(safe_path, exist_ok=True)
    
    # Cache the path for reuse
    _CURRENT_KB_STORAGE[kb_id] = safe_path
    
    return safe_path

# Monkey patch CrewAI's path utilities to prevent long path issues
try:
    from crewai.utilities import paths
    if hasattr(paths, 'db_storage_path'):
        paths.db_storage_path = safe_db_storage_path
        print("Successfully overrode CrewAI db_storage_path function")
except ImportError:
    pass  # CrewAI utilities may not be available yet

# Force override via environment variables that CrewAI checks
os.environ['CREWAI_DB_PATH'] = 'temp/crewai_db'
os.environ['CREWAI_CACHE_PATH'] = 'temp/crewai_cache'

# os.environ['CREWAI_STRICT_JSON'] = 'true'

ALLOWED_FILE_TYPES = {
    "image/jpeg": "image",
    "image/png": "image",
    "image/gif": "image",
    "text/csv": "csv",
    "application/json": "json",
    "text/plain": "text",
    "application/pdf": "pdf"
}

EXTENSION_TO_MIME = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "csv": "text/csv",
    "json": "application/json",
    "txt": "text/plain",
    "pdf": "application/pdf"
}

def sanitize_for_logging(text: Any) -> str:
    try:
        text_str = str(text)
        return text_str.encode("utf-8", errors="replace").decode("utf-8")
    except Exception:
        return str(text).encode("utf-8", errors="replace").decode("utf-8")


def generate_token(email: str, expires_delta: timedelta = timedelta(minutes=20)) -> str:
    """
    Generate a JWT token for the given email
    
    Args:
        email: User's email address
        expires_delta: How long the token should be valid for (default: 20 minutes)
        
    Returns:
        str: JWT token
    """
    # Define when the token expires
    expire = datetime.utcnow() + expires_delta
    
    # Create the JWT payload
    payload = {
        "email": email,
        "exp": expire,  # Expiration time
        "iat": datetime.utcnow()  # Issued at time
    }
    
    # Generate the token using the same SECRET_KEY and ALGORITHM from application.py
    SECRET_KEY = "AUTHSECAJUN9544213YR2025"
    ALGORITHM = "HS256"
    
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token

class GeminiFailoverLLM:
    def __init__(self, model: str, api_keys: List[str], temperature: float = 1, executor_logger: logging.Logger = None):
        self.model = model
        self.api_keys = api_keys
        self.temperature = temperature
        self.logger = executor_logger or logging.getLogger("gemini_failover")

    def _get_llm(self, api_key):
        return LLM(
            model=self.model,
            api_key=api_key,
            temperature=self.temperature
        )

    def _mask_key(self, key: str) -> str:
        """Utility to mask API key for logging (show prefix only)."""
        return f"{key[:6]}...{key[-4:]}" if len(key) > 10 else key

    def run(self, *args, **kwargs):
        shuffled_keys = self.api_keys[:]
        random.shuffle(shuffled_keys)

        for key in shuffled_keys:
            try:
                self.logger.info(f"[Gemini LLM] Trying API key: {self._mask_key(key)}")
                llm = self._get_llm(key)
                return llm.run(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"[LLM Failover] Failed with key {self._mask_key(key)}: {e}")
                continue
        raise RuntimeError("All Gemini API keys failed.")

    async def arun(self, *args, **kwargs):
        shuffled_keys = self.api_keys[:]
        random.shuffle(shuffled_keys)

        for key in shuffled_keys:
            try:
                self.logger.info(f"[Gemini LLM] Trying API key: {self._mask_key(key)}")
                llm = self._get_llm(key)
                return await llm.arun(*args, **kwargs)
            except Exception as e:
                self.logger.warning(f"[LLM Failover] Async failed with key {self._mask_key(key)}: {e}")
                continue
        raise RuntimeError("All Gemini API keys failed.")

class TaskExecutor:
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
        execution_id: Optional[str] = None
    ):
        # CRITICAL FIX: Store execution_id for isolation
        self.execution_id = execution_id or str(uuid.uuid4())
        self.expected_user_input = None  # Will be set when executing
        
        self.suggestive_questions = []  # Side-channel: captured from tool API responses
        self.display_metadata = None  # Side-channel: captured from image generation tool responses

        self.logger = logging.getLogger(f"task_executor_{self.execution_id}")
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
     

        # Vertex AI LLM clients — credentials handled centrally:
        #   * local      → Vertex REST + API key from Secret Manager
        #   * production → Vertex AI via ADC on Compute Engine
        self.llm_client = get_crewai_llm(temperature=0.9)
        self.internal_llm_client = get_crewai_llm(temperature=0.0)

        # Get comma-separated API keys from environment
        # api_keys_string = os.getenv("RANDOM_GEMINI_API_KEYS", "")
        # gemini_keys = [key.strip() for key in api_keys_string.split(",") if key.strip()]
        # self.llm_client = GeminiFailoverLLM(
        #     model="gemini/gemini-2.5-flash",
        #     api_keys=gemini_keys,
        #     temperature=0.9,
        #     executor_logger=self.logger
        # )

        # self.internal_llm_client = LLM(model="gemini/gemini-2.5-flash", api_key=API_KEY)

        self.enable_planning = enable_planning
        self.enable_memory = enable_memory
        self.verbose = verbose
        self.raw_output = raw_output
        self.agent_id = agent_config["id"]
        self.agent_name = agent_config["name"]

        # OPTIMIZED FIX: Reuse storage directories for same knowledge base to avoid unnecessary path creation
        if "knowledge_base_id" in agent_config and agent_config["knowledge_base_id"]:
            kb_id = agent_config["knowledge_base_id"]
            
            # Use the optimized function that reuses existing directories for same KB
            safe_storage_path = get_or_create_kb_storage(kb_id)
            os.environ["CREWAI_STORAGE_DIR"] = safe_storage_path
            
            self.logger.debug(f"Using storage dir for KB {kb_id[:8]}: {safe_storage_path}")
            
            # Log whether this is a reused or new directory
            if safe_storage_path == _CURRENT_KB_STORAGE.get(kb_id):
                self.logger.info(f"Reusing existing storage directory for knowledge base: {kb_id[:8]}")
            else:
                self.logger.info(f"Created new storage directory for knowledge base: {kb_id[:8]}")
        else:
            # Ensure we have a clean base path even without knowledge base
            base_path = "temp/crewai_base"
            os.environ["CREWAI_STORAGE_DIR"] = base_path
            self.logger.debug(f"Using base storage dir: {base_path}")

        self.logger.debug(f"Choosed LLM Client: {getattr(self.llm_client, 'model', 'vertex_ai')}")

        if user_email:
            self.token = generate_token(user_email)
            self.logger.debug(f"Generated token for user: {user_email}")
        else:
            self.token = None
            self.logger.debug("No user email provided, token not generated")

        self.schema_agent = CrewAgent(
            role="Schema Analyzer",
            goal="Analyze OpenAPI schemas and extract key information",
            backstory="Expert at analyzing OpenAPI schemas.",
            verbose=False,
            allow_delegation=False,
            llm=self.internal_llm_client
        )

        self.payload_agent = CrewAgent(
            role="Payload Generator",
            goal="Generate accurate payloads for API tools",
            backstory="Expert at creating valid API payloads.",
            verbose=False,
            allow_delegation=False,
            llm=self.internal_llm_client
        )

        self.tools = []
        if tools_config:
            for tool_config in tools_config:
                tool_schema = tool_config["schema"]
                tool_headers = tool_config.get("auth", {}).get("headers", {})
                tool_params = tool_config.get("auth", {}).get("params", {})
                tool_data_connector = tool_config.get("data_connector", None)

                tool_name = tool_schema["info"]["title"].lower().replace(" ", "_")
                tool_description = f"Calls API: {tool_schema['info']['title']}."
                
                class ApiCallerToolArgsSchema(BaseModel):
                    input_data: Union[str, dict] = Field(..., description="The input data to process with the API tool. Can be a string (legacy format) or a JSON object with structured fields like {'user_input': '<INPUT>', 'meta_instruction': 'instructions', ...}")

                class ApiCallerTool(BaseTool):
                    name: str = tool_name
                    description: str = tool_description
                    args_schema: type[BaseModel] = ApiCallerToolArgsSchema
                    
                    def __init__(self, _tool_schema=tool_schema, _tool_headers=tool_headers, 
                                 _tool_params=tool_params, _tool_data_connector=tool_data_connector, 
                                 _executor=self):
                        super().__init__()
                        self._tool_schema = _tool_schema
                        self._tool_headers = _tool_headers
                        self._tool_params = _tool_params
                        self._tool_data_connector = _tool_data_connector
                        self._executor = _executor

                        
                    
                    def _parse_multiple_json_objects(self, line: str) -> List[dict]:
                        """
                        Parse multiple JSON objects from a single line (JSONL format).
                        
                        Args:
                            line: A string that may contain multiple JSON objects
                            
                        Returns:
                            List of parsed JSON objects
                        """
                        json_objects = []
                        decoder = json.JSONDecoder()
                        idx = 0
                        
                        while idx < len(line):
                            line = line[idx:].lstrip()  # Remove leading whitespace
                            if not line:
                                break
                                
                            try:
                                obj, end_idx = decoder.raw_decode(line)
                                json_objects.append(obj)
                                idx += end_idx
                            except JSONDecodeError as e:
                                self._executor.logger.error(f"Failed to parse JSON at position {idx}: {e}")
                                # Try to find the next potential JSON object start
                                next_brace = line.find('{', 1)
                                if next_brace == -1:
                                    break
                                idx += next_brace
                                continue
                        
                        return json_objects

                    def clean_and_capitalize(self, input_string):
                        remove_items = ['-', '_', 'api', 'API', 'Api', 'stream', 'generic']
                        for item in remove_items:
                            input_string = input_string.replace(item, ' ')
                        cleaned = ' '.join(input_string.strip().split())  # remove extra spaces
                        result = cleaned.capitalize()
                        return result

                    async def _make_request(self, method: str, url: str, headers: dict, params: dict = None, request_json: dict = None, status_queue: Optional[asyncio.Queue] = None) -> dict:
                        # Configure timeout for 20 minutes total with appropriate socket timeouts for high concurrent load
                        timeout = aiohttp.ClientTimeout(
                            total=1200,      # 20 minutes total timeout
                            connect=30,      # 30 seconds to establish connection
                            sock_connect=30, # 30 seconds for socket connection
                            sock_read=900    # 5 minutes for reading data - increased for large responses
                        )

                        try:
                            if status_queue:
                                ab_tool_name = self.clean_and_capitalize(self.name)
                                await status_queue.put(f"Process: {ab_tool_name}")
                        except Exception as e:
                            self._executor.logger.error(f"Failed to clean and capitalize tool name: {e}")

                        async with aiohttp.ClientSession(timeout=timeout) as session:
                            max_retries = 2
                            retry_count = 0
                            while True:
                                try:
                                    self._executor.logger.debug(f"Making {method.upper()} request to {url}")
                                    if params:
                                        self._executor.logger.debug(f"Query params: {params}")
                                    if request_json:
                                        masked_request_json = request_json.copy()
                                        if "db_config" in masked_request_json:
                                            masked_request_json["db_config"] = "XXXXXXXXXXXXXXXXXXXXXXX"
                                        self._executor.logger.debug(f"Request body: {masked_request_json}")

                                    if method.lower() == "get":
                                        async with session.get(url, headers=headers, params=params) as response:
                                            self._executor.logger.debug(f"Response status code: {response.status}")
                                            if response.status == 200:
                                                # Check if response is a streaming response
                                                content_type = response.headers.get('content-type', '')
                                                if 'application/x-ndjson' in content_type:
                                                    # Handle NDJSON streaming response
                                                    result = []
                                                    async for line in response.content:
                                                        decoded_line = line.decode('utf-8').strip()
                                                        if decoded_line:
                                                            try:
                                                                json_data = json.loads(decoded_line)
                                                                if isinstance(json_data, dict):
                                                                    # Handle status messages separately
                                                                    if json_data.get("type") == "status":
                                                                        if status_queue:
                                                                            await status_queue.put(json_data.get("message", "Processing..."))
                                                                        # Don't include status messages in the final result
                                                                        continue
                                                                    elif json_data.get("type") == "error":
                                                                        # Handle error messages
                                                                        self._executor.logger.error(f"API returned error: {json_data.get('error', 'Unknown error')}")
                                                                        return {"error": json_data.get('error', 'Unknown error')}
                                                                    # Include result and other non-status messages
                                                                    result.append(json_data)
                                                                else:
                                                                    result.append(json_data)
                                                            except json.JSONDecodeError as e:
                                                                self._executor.logger.error(f"Failed to parse NDJSON line: {decoded_line}, error: {e}")
                                                                continue
                                                    return {"stream_result": result}
                                                else:
                                                    # Handle regular JSON response
                                                    try:
                                                        response_data = await response.json()
                                                        self._executor.logger.debug(f"Response data: {response_data}")
                                                        return response_data
                                                    except ValueError:
                                                        response_text = await response.text()
                                                        self._executor.logger.debug(f"Response text: {response_text}")
                                                        return {"result": response_text}
                                            elif response.status == 400 and retry_count < max_retries:
                                                retry_count += 1
                                                self._executor.logger.warning(f"Received 400 status code. Retrying request (attempt {retry_count}/{max_retries})")
                                                await asyncio.sleep(1)
                                                continue
                                            else:
                                                try:
                                                    error_data = await response.json()
                                                    self._executor.logger.error(f"Request failed with status {response.status}. Error data: {error_data}")
                                                    return {"error": error_data.get('detail', str(response.status))}
                                                except ValueError:
                                                    error_text = await response.text()
                                                    self._executor.logger.error(f"Request failed with status {response.status}. Error text: {error_text}")
                                                    return {"error": error_text or str(response.status)}
                                    else:  # POST
                                        async with session.post(url, headers=headers, params=params, json=request_json) as response:
                                            self._executor.logger.debug(f"Response status code: {response.status}")
                                            if response.status == 200:
                                                # Check if response is a streaming response
                                                content_type = response.headers.get('content-type', '')
                                                if 'application/x-ndjson' in content_type:
                                                    # Handle NDJSON streaming response
                                                    result = []
                                                    async for line in response.content:
                                                        decoded_line = line.decode('utf-8').strip()
                                                        if decoded_line:
                                                            try:
                                                                json_data = json.loads(decoded_line)
                                                                if isinstance(json_data, dict):
                                                                    # Handle status messages separately
                                                                    if json_data.get("type") == "status":
                                                                        if status_queue:
                                                                            await status_queue.put(json_data.get("message", "Processing the results.."))
                                                                        # Don't include status messages in the final result
                                                                        continue
                                                                    elif json_data.get("type") == "error":
                                                                        # Handle error messages
                                                                        self._executor.logger.error(f"API returned error: {json_data.get('error', 'Unknown error')}")
                                                                        return {"error": json_data.get('error', 'Unknown error')}
                                                                    # Include result and other non-status messages
                                                                    result.append(json_data)
                                                                else:
                                                                    result.append(json_data)
                                                            except json.JSONDecodeError as e:
                                                                self._executor.logger.error(f"Failed to parse NDJSON line: {decoded_line}, error: {e}")
                                                                continue
                                                    return {"stream_result": result}
                                                else:
                                                    # Handle regular JSON response
                                                    try:
                                                        response_data = await response.json()
                                                        self._executor.logger.debug(f"Response data: {response_data}")
                                                        return response_data
                                                    except ValueError:
                                                        response_text = await response.text()
                                                        self._executor.logger.debug(f"Response text: {response_text}")
                                                        return {"result": response_text}
                                            elif response.status == 400 and retry_count < max_retries:
                                                retry_count += 1
                                                self._executor.logger.warning(f"Received 400 status code. Retrying request (attempt {retry_count}/{max_retries})")
                                                await asyncio.sleep(1)
                                                continue
                                            else:
                                                try:
                                                    error_data = await response.json()
                                                    self._executor.logger.error(f"Request failed with status {response.status}. Error data: {error_data}")
                                                    return {"error": error_data.get('detail', str(response.status))}
                                                except ValueError:
                                                    error_text = await response.text()
                                                    self._executor.logger.error(f"Request failed with status {response.status}. Error text: {error_text}")
                                                    return {"error": error_text or str(response.status)}
                                except aiohttp.ClientError as e:
                                    self._executor.logger.error(f"HTTP request failed with client error: {str(e)}")
                                    return {"error": f"HTTP request failed: {str(e)}"}
                                except Exception as e:
                                    self._executor.logger.error(f"Unexpected error during request: {str(e)}")
                                    return {"error": f"Unexpected error: {str(e)}"}
                                break
                            self._executor.logger.warning("Max retries exceeded")
                            return {"error": "Max retries exceeded"}

                    def _run(self, input_data: Union[str, dict]) -> str:
                        try:
                            # Use asyncio.run to run the async function in a synchronous context
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                result = loop.run_until_complete(self._async_run(input_data))
                            finally:
                                loop.close()
                            return json.dumps(result)
                        except Exception as e:
                            self._executor.logger.error(f"Tool returned error: {sanitize_for_logging(e)}")
                            return json.dumps({"error": sanitize_for_logging(str(e))})

                    async def _async_run(self, input_data: Union[str, dict]) -> dict:
                        try:
                            # RELAXED FIX: Log tool input but don't block execution unless there's clear contamination
                            expected_input = self._executor.expected_user_input
                            
                            # Log the tool execution for monitoring
                            # self._executor.logger.info(f"[{self._executor.execution_id}] Tool processing input: '{sanitize_for_logging(input_data)}'")
                            
                            # Handle input validation for both string and dict formats
                            if isinstance(input_data, dict):
                                # For dict input, convert to string representation for validation
                                validation_text = str(input_data)
                            else:
                                validation_text = input_data
                            
                            # Only validate if inputs are completely different (possible contamination)
                            # Allow for normal text processing and minor variations
                            if expected_input and validation_text and len(validation_text) > 10:
                                # Check for major differences that indicate contamination
                                # This is less strict - only alerts on significant mismatches
                                input_words = set(validation_text.lower().split())
                                expected_words = set(expected_input.lower().split())
                                
                                # If there's less than 20% word overlap and completely different content, it might be contamination
                                if len(input_words.intersection(expected_words)) / max(len(expected_words), 1) < 0.2:
                                    # Log warning but don't block - let execution continue
                                    self._executor.logger.warning(f"[{self._executor.execution_id}] Possible input variation detected. Expected: '{expected_input[:50]}...', Got: '{validation_text[:50]}...'")
                            
                            # Announce tool processing start
                            try:
                                if self._executor.status_queue:
                                    await self._executor.status_queue.put(f"Processing Tool \"{self.name}\"")
                            except Exception:
                                pass

                            result = await self._executor.generate_payload(input_data, self._tool_schema, self._tool_data_connector, tool_name=self.name)
                            
                            if not result or "error" in result:
                                return {"error": result.get("error", "Failed to generate payload or endpoint URL")}

                            payload = result.get("payload")
                            endpoint_url = result.get("endpoint_url")
                            if not endpoint_url:
                                return {"error": "Missing endpoint URL"}

                            paths = self._tool_schema.get("paths", {})
                            if not paths:
                                return {"error": "No paths found in schema"}

                            path_key = next(iter(paths))
                            method = None
                            for m in paths[path_key]:
                                if m.lower() in ["get", "post"]:
                                    method = m.lower()
                                    break
                            if not method:
                                return {"error": "No supported HTTP method (GET/POST) found in schema"}

                            headers = self._tool_headers.copy() if self._tool_headers else {}
                            params = self._tool_params.copy() if self._tool_params else {}

                            if self._executor.token:
                                headers["Authorization"] = f"Bearer {self._executor.token}"
                                app_url = endpoint_url.lower()

                                # Remove protocol
                                app_host = app_url.split("://")[-1].split("/")[0]

                                if (
                                    app_host == "impact-agents.ai"
                                    or app_host.endswith(".impact-agents.ai")
                                    or app_host == "localhost:8000"
                                ):
                                    headers["AGENT-API-KEY"] = os.getenv("AGENT_API_KEY", "")

                            if method == "get" and payload:
                                params.update(payload)

                            self._executor.logger.debug(f"Calling {method.upper()} {endpoint_url}")
                            # Stream status: invoking tool endpoint
                            try:
                                if self._executor.status_queue:
                                    await self._executor.status_queue.put(f"Invoking \"{self.name}\" ({method.upper()})")
                            except Exception:
                                pass
                            
                            collected_results = await self._make_request(
                                method=method,
                                url=endpoint_url,
                                headers=headers,
                                params=params if params else None,
                                request_json=payload if method == "post" and payload else None,
                                status_queue=self._executor.status_queue if hasattr(self._executor, 'status_queue') else None
                            )

                            # Stream status: tool completed
                            try:
                                if self._executor.status_queue:
                                    await self._executor.status_queue.put(f"Completed Tool \"{self.name}\"")
                            except Exception:
                                pass

                            if isinstance(collected_results, dict) and "suggestive_questions" in collected_results:
                                sq = collected_results.pop("suggestive_questions", None)
                                if isinstance(sq, list) and sq:
                                    self._executor.suggestive_questions = sq

                            # Capture display_metadata from image generation tool response (side-channel)
                            if isinstance(collected_results, dict) and "display_metadata" in collected_results:
                                dm = collected_results.pop("display_metadata", None)
                                if isinstance(dm, dict) and dm:
                                    self._executor.display_metadata = dm

                            return collected_results


                        except Exception as e:
                            self._executor.logger.error(f"Tool returned error: {sanitize_for_logging(e)}")
                            return {"error": sanitize_for_logging(str(e))}
                
                api_tool = ApiCallerTool()
                self.tools.append(api_tool)


        

        self.knowledge_sources = knowledge_sources

        self.agent = CrewAgent(
            role=agent_config["role"],
            goal=agent_config["goal"],
            backstory=agent_config["backstory"],
            llm=self.llm_client,
            tools=self.tools,
            verbose=self.verbose,  # Enable verbose for detailed logging
            knowledge_sources=self.knowledge_sources,
            knowledge_config=knowledge_config,
            cache=False
            # max_iter=2
        )

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
        execution_id: Optional[str] = None
    ):
        # Create a temporary logger for knowledge source loading since we don't have the instance yet
        temp_logger = logging.getLogger(f"task_executor_{execution_id or 'temp'}")
        temp_formatter = logging.Formatter(f"[{execution_id or 'temp'}] %(asctime)s [%(levelname)s] %(message)s")
        
        # Set up temporary logger with same formatting as instance logger
        if not temp_logger.handlers:
            temp_handler = logging.StreamHandler()
            temp_handler.setFormatter(temp_formatter)
            temp_logger.addHandler(temp_handler)
            temp_logger.setLevel(logging.DEBUG)
        
        knowledge_sources = await cls._load_knowledge_sources(agent_config.get("knowledge_base_id"), temp_logger)
        return cls(
            agent_config=agent_config,
            tools_config=tools_config,
            log_file=log_file,
            user_email=user_email,
            enable_planning=enable_planning,
            enable_memory=enable_memory,
            verbose=verbose,
            raw_output=raw_output,
            knowledge_sources=knowledge_sources,
            execution_id=execution_id
        )

    @staticmethod
    async def _load_knowledge_sources(knowledge_base_id: Optional[str] = None, executor_logger: Optional[logging.Logger] = None) -> List[Any]:
        """
        Load knowledge sources from knowledge_base.json based on knowledge_base_id.
        Returns a list of CrewAI knowledge source objects.
        """
        # Use the provided executor logger or create a temporary one
        logger = executor_logger or logging.getLogger(f"task_executor_loader")
        if not knowledge_base_id:
            logger.debug("No knowledge_base_id provided in agent config")
            return []

        knowledge_sources = []
        try:
            async with aiofiles.open("knowledge_base.json", "r", encoding="utf-8") as f:
                content = await f.read()
                knowledge_data = json.loads(content)
            
            kb_entry = next((entry for entry in knowledge_data if entry["id"] == knowledge_base_id), None)
            if not kb_entry:
                logger.error(f"No knowledge base found for ID: {knowledge_base_id}")
                return []

            logger.info(f"Loading knowledge sources for knowledge base: {kb_entry['title']} (ID: {knowledge_base_id})")
            for source in kb_entry["sources"]:
                source_type = source["type"]
                source_id = source["id"]
                source_content = source["content"]
                
                try:
                    if source_type == "text":
                        knowledge_source = StringKnowledgeSource(
                            content=source_content,
                            metadata={"id": source_id, "title": kb_entry["title"]}
                        )
                        logger.debug(f"Loaded text knowledge source: {source_id}")
                        knowledge_sources.append(knowledge_source)

                    elif source_type == "url":
                        knowledge_source = CrewDoclingSource(
                            file_paths=[source_content],
                            metadata={"id": source_id, "title": kb_entry["title"]}
                        )
                        logger.debug(f"Loaded URL knowledge source: {source_id} ({source_content})")
                        knowledge_sources.append(knowledge_source)

                    elif source_type == "file":
                        file_path = f"{source.get('filename')}"
                        actual_file_path = f"knowledge/{file_path}"
                        if not actual_file_path or not os.path.exists(actual_file_path):
                            logger.error(f"File path missing or invalid for source: {source_id} ({source_content})")
                            continue

                        file_extension = os.path.splitext(file_path)[1].lower()
                        if file_extension == ".txt":
                            knowledge_source = TextFileKnowledgeSource(
                                file_paths=[file_path],
                                metadata={"id": source_id, "title": kb_entry["title"], "filename": source.get("filename")}
                            )
                        elif file_extension == ".pdf":
                            knowledge_source = PDFKnowledgeSource(
                                file_paths=[file_path],
                                metadata={"id": source_id, "title": kb_entry["title"], "filename": source.get("filename")}
                            )
                        elif file_extension == ".csv":
                            knowledge_source = CSVKnowledgeSource(
                                file_paths=[file_path],
                                metadata={"id": source_id, "title": kb_entry["title"], "filename": source.get("filename")}
                            )
                        elif file_extension == ".xlsx":
                            knowledge_source = ExcelKnowledgeSource(
                                file_paths=[file_path],
                                metadata={"id": source_id, "title": kb_entry["title"], "filename": source.get("filename")}
                            )
                        else:
                            logger.error(f"Unsupported file extension: {file_extension} for source: {source_id}")
                            continue
                        
                        logger.debug(f"Loaded file knowledge source: {source_id} ({source_content})")
                        knowledge_sources.append(knowledge_source)

                    else:
                        logger.warning(f"Unsupported knowledge source type: {source_type} for source: {source_id}")
                except Exception as e:
                    logger.error(f"Error loading knowledge source {source_id}: {sanitize_for_logging(e)}")
                    continue

            logger.info(f"Loaded {len(knowledge_sources)} knowledge sources for knowledge base ID: {knowledge_base_id}")
            return knowledge_sources

        except Exception as e:
            logger.error(f"Error loading knowledge_base.json: {sanitize_for_logging(e)}")
            return []
	    

    async def analyze_schema(self, schema: dict) -> str:
        self.logger.debug("Analyzing schema")
        schema_str = json.dumps(schema, indent=2, ensure_ascii=False)
        analysis_task = Task(
            description=f"Analyze OpenAPI schema:\n{schema_str}\nFocus on required fields, types, constraints, parameters, and response structure.",
            expected_output="Summary of schema requirements including HTTP method, parameters, and payload requirements",
            agent=self.schema_agent
        )
        crew = Crew(agents=[self.schema_agent], tasks=[analysis_task], process=Process.sequential)
        result = await crew.kickoff_async()
        sanitized_result = sanitize_for_logging(result)
        return str(result)

    def _extract_final_json_object(self, response_text: str) -> Optional[Dict[str, Any]]:
        """
        Extract the final valid JSON object from agent response that contains the expected structure.
        Handles cases where agent returns multiple JSON objects mixed with explanatory text.
        Prioritizes JSON objects with 'payload' and 'endpoint_url' keys or 'error' key.
        """
        import json
        
        # Use brace counting to find all complete JSON objects
        brace_count = 0
        start_pos = -1
        json_candidates = []
        
        for i, char in enumerate(response_text):
            if char == '{':
                if brace_count == 0:
                    start_pos = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_pos != -1:
                    json_candidates.append(response_text[start_pos:i+1])
                    start_pos = -1
        
        # Try to parse each potential JSON object
        valid_jsons = []
        for json_str in json_candidates:
            try:
                parsed = json.loads(json_str.strip())
                if isinstance(parsed, dict):
                    valid_jsons.append(parsed)
            except json.JSONDecodeError:
                continue
        
        if not valid_jsons:
            return None
        
        # Prioritize JSON objects with the expected structure
        # Look for objects with 'payload' and 'endpoint_url' (final result) or 'error'
        for json_obj in reversed(valid_jsons):  # Check from last to first
            if ('payload' in json_obj and 'endpoint_url' in json_obj) or 'error' in json_obj:
                return json_obj
        
        # If no perfect match found, return the last valid JSON object
        # (assuming the agent puts the final result at the end)
        return valid_jsons[-1] if valid_jsons else None

    async def generate_payload(self, user_input: Union[str, dict], schema: dict, tool_data_connector: Optional[dict] = None, tool_name: Optional[str] = None) -> Dict[str, Any]:
        # RELAXED FIX: Log and monitor but don't block unless there's clear contamination
        try:
            if hasattr(self, 'status_queue') and self.status_queue:
                pretty_tool = tool_name or 'API Tool'
                await self.status_queue.put(f"Preparing payload for Tool \"{pretty_tool}\"")
        except Exception:
            pass
        
        # Handle both string and dict input formats
        if isinstance(user_input, dict):
            # For dict input, convert to string representation for validation
            input_for_validation = str(user_input)
        else:
            # Backward compatibility: treat as string
            input_for_validation = user_input
        
        # Only validate for major differences that indicate clear contamination
        if self.expected_user_input and input_for_validation and len(input_for_validation) > 10:
            if (self.expected_user_input.lower() not in input_for_validation.lower() and 
                input_for_validation.lower() not in self.expected_user_input.lower() and
                len(set(input_for_validation.lower().split()).intersection(set(self.expected_user_input.lower().split()))) < 2):
                self.logger.warning(
                    f"[{self.execution_id}] Input variation in payload generation. "
                    f"Expected context: '{self.expected_user_input[:50]}...', "
                    f"Processing: '{input_for_validation[:50]}...'"
                )
        
        paths = schema.get("paths", {})
        if not paths:
            self.logger.error("No paths in schema")
            return {"error": "No paths in schema"}

        path_key = next(iter(paths))
        method = next((m for m in paths[path_key] if m.lower() in ["get", "post"]), "post")

        request_schema = {}
        if method == "get":
            parameters = paths[path_key].get(method, {}).get("parameters", [])
            if parameters:
                request_schema = {"parameters": parameters}
        else:
            request_body = paths[path_key].get(method, {}).get("requestBody", {})
            if request_body:
                request_schema = request_body.get("content", {}).get("application/json", {}).get("schema", {})

        # Strip empty serviceAccountKey from the connector info sent to the LLM so it
        # doesn't emit `credentials_json: {}` — downstream BigQuery client should use
        # Application Default Credentials (ADC) when no explicit key is provided.
        sanitized_connector = tool_data_connector
        if tool_data_connector and not tool_data_connector.get("serviceAccountKey"):
            sanitized_connector = {k: v for k, v in tool_data_connector.items() if k != "serviceAccountKey"}
        connector_info = f"db_informations:\n{json.dumps(sanitized_connector, indent=2, ensure_ascii=False)}\n" if sanitized_connector else ""
        
        # Format user input - handle both string and dict formats
        if isinstance(user_input, dict):
            user_input_str = f"User Input (JSON): {json.dumps(user_input, indent=2, ensure_ascii=False)}"
        else:
            user_input_str = f"User Input: '{user_input}'"
        
        payload_task = Task(
            description=f"""
            {user_input_str}
            Request Schema: {json.dumps(request_schema, indent=2, ensure_ascii=False)}
            Full Schema: {json.dumps(schema, indent=2, ensure_ascii=False)}
            {connector_info}
            Perform the following steps:
            1. Analyze the provided OpenAPI schema, focusing on required fields, types, constraints, parameters, and response structure.
            2. Validate that the user input contains all necessary information to generate a valid JSON payload (for POST) or query parameters (for GET) based on the schema's required fields.
            - If any required fields are missing in the user input, return an error JSON object with the key "error" and a message listing the missing required fields. Do NOT use example values or defaults from the schema.
            - For GET requests, check if required query parameters are provided in the input.
            - For POST requests, check if required properties in the requestBody schema are provided in the input.
            3. If validation passes, generate a valid JSON payload (if required) and the endpoint URL for the API call.
            - For GET requests, return a dictionary of query parameters if parameters are defined in the schema, otherwise return null.
            - For POST requests, generate a JSON payload if a requestBody is defined in the schema, otherwise return null.
            4. Construct the endpoint URL using the appropriate server URL from the schema's "servers" field and the correct path from the "paths" field.
            5. If schema has db_config, you can refer db_informations to get those informations. Also sometimes required values will be mentioned in the user input. So check the user input for required values. But if it not clear don't assume anything.
            
            Return ONLY a single JSON object as your response (no explanatory text, no multiple JSON objects):
            - If successful: {{"payload": <payload_data>, "endpoint_url": "<url>"}}
            - If error: {{"error": "<error_message>"}}
            
            Do NOT include any explanatory text, intermediate JSON objects, or reasoning. 
            Return ONLY the final JSON object that contains either 'payload' and 'endpoint_url' or 'error'.
            Ensure the endpoint_url is valid and matches the schema. Do NOT use example values from the schema unless explicitly provided in the user input.
            """,
            expected_output="JSON object with 'payload' and 'endpoint_url', or 'error' if required fields are missing",
            agent=self.payload_agent
        )
        crew = Crew(agents=[self.payload_agent], tasks=[payload_task], process=Process.sequential)

        # Retry mechanism
        for attempt in range(2):  # one retry
            self.logger.info(f"Generating payload (Attempt {attempt + 1}/2)")
            try:
                result = await crew.kickoff_async()

                result_str = str(getattr(result, "raw", result))

                # Clean up fenced code blocks if present
                if "```" in result_str:
                    result_str = re.sub(r"```(json)?", "", result_str).strip()

                # Extract the correct JSON object using improved parsing
                result_json = self._extract_final_json_object(result_str)
                if result_json is None:
                    if attempt == 0:
                        self.logger.warning("No valid JSON found in response, retrying...")
                        continue
                    self.logger.error(f"Failed to extract valid JSON after retry: {result_str[:500]}...")
                    return {"error": "No valid JSON object found in agent response"}

                # If error in response
                if "error" in result_json:
                    self.logger.error(f"Payload generation failed: {result_json['error']}")
                    return {"error": result_json["error"]}

                payload = result_json.get("payload")
                endpoint_url = result_json.get("endpoint_url")

                if not endpoint_url:
                    self.logger.error("Missing endpoint_url in agent response")
                    return {"error": "Missing endpoint_url in agent response"}

                # Safety net: strip empty credentials_json from bigquery db_config so the
                # downstream client falls back to Application Default Credentials (ADC).
                if isinstance(payload, dict):
                    db_cfg = payload.get("db_config")
                    if isinstance(db_cfg, dict) and "credentials_json" in db_cfg and not db_cfg["credentials_json"]:
                        db_cfg.pop("credentials_json", None)
                        self.logger.debug("Removed empty credentials_json from db_config (using ADC)")

                # Mask sensitive values safely
                masked_payload = {}
                if isinstance(payload, dict):
                    masked_payload = payload.copy()
                    if "db_config" in masked_payload:
                        masked_payload["db_config"] = "XXXXXXXXXXXXXXXXXXXXXXX"

                self.logger.debug(f"Generated payload: {json.dumps(masked_payload, ensure_ascii=False) if masked_payload else 'null'}")
                self.logger.debug(f"Generated endpoint URL: {endpoint_url}")

                try:
                    if hasattr(self, 'status_queue') and self.status_queue:
                        pretty_tool = tool_name or 'API Tool'
                        await self.status_queue.put(f"Payload prepared for \"{pretty_tool}\"")
                except Exception:
                    pass

                return {"payload": payload, "endpoint_url": endpoint_url}

            except Exception as e:
                if attempt == 0:
                    self.logger.warning(f"Error during payload generation attempt {attempt+1}: {e}, retrying...")
                    continue
                self.logger.error(f"Unexpected error during payload generation: {e}")
                return {"error": f"Unexpected error during payload generation: {e}"}


    def get_task_descriptions(self, description: str, expected_output: str, task_name: Optional[str] = None, **kwargs) -> Dict[str, str]:
        self.logger.debug(f"Processing task descriptions for: {sanitize_for_logging(task_name)}")
        try:
            if kwargs:
                for key, value in kwargs.items():
                    placeholder = "{{" + key + "}}"
                    description = description.replace(placeholder, f"'{str(value)}'" if key == "input" else str(value))
                    expected_output = expected_output.replace(placeholder, str(value))
            return {"description": description, "expected_output": expected_output}
        except Exception as e:
            self.logger.error(f"Error processing task '{sanitize_for_logging(task_name)}': {sanitize_for_logging(e)}")
            return {"description": description, "expected_output": expected_output}

    def encode_image_to_base64(self, image_path: str) -> str:
        self.logger.debug(f"Encoding image: {sanitize_for_logging(image_path)}")
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            self.logger.error(f"Error encoding image: {sanitize_for_logging(e)}")
            return f"Error encoding image: {sanitize_for_logging(e)}"

    def read_csv_as_text(self, csv_path: str) -> str:
        self.logger.debug(f"Reading CSV: {sanitize_for_logging(csv_path)}")
        try:
            with open(csv_path, "r", encoding="utf-8", errors="replace") as csv_file:
                content = csv_file.read()
                return content
        except Exception as e:
            self.logger.error(f"Error reading CSV: {sanitize_for_logging(e)}")
            return f"Error reading CSV: {sanitize_for_logging(e)}"

    def read_json_as_text(self, json_path: str) -> str:
        self.logger.debug(f"Reading JSON: {sanitize_for_logging(json_path)}")
        try:
            with open(json_path, "r", encoding="utf-8", errors="replace") as json_file:
                data = json.load(json_file)
                return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error reading JSON: {sanitize_for_logging(e)}")
            return f"Error reading JSON: {sanitize_for_logging(e)}"

    def read_txt_as_text(self, txt_path: str) -> str:
        self.logger.debug(f"Reading TXT: {sanitize_for_logging(txt_path)}")
        try:
            with open(txt_path, "r", encoding="utf-8", errors="replace") as txt_file:
                return txt_file.read()
        except Exception as e:
            self.logger.error(f"Error reading TXT: {sanitize_for_logging(e)}")
            return f"Error reading TXT: {sanitize_for_logging(e)}"

    def read_pdf_as_text(self, pdf_path: str) -> str:
        self.logger.debug(f"Reading PDF: {sanitize_for_logging(pdf_path)}")
        try:
            with open(pdf_path, "rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = "".join(page.extract_text() or "" for page in reader.pages)
                return text.strip() or "No readable text in PDF"
        except Exception as e:
            self.logger.error(f"Error reading PDF: {sanitize_for_logging(e)}")
            return f"Error reading PDF: {sanitize_for_logging(e)}"

    def process_file_content(self, file_path: str, file_type: str) -> str:
        self.logger.debug(f"Processing file: {sanitize_for_logging(file_path)}, type: {file_type}")
        normalized_type = file_type.lower()
        if normalized_type.startswith("image/"):
            return ""
        elif normalized_type == "text/csv":
            content = self.read_csv_as_text(file_path)
            return f"\n\nCSV content:\n{content}"
        elif normalized_type == "application/json":
            content = self.read_json_as_text(file_path)
            return f"\n\nJSON content:\n{content}"
        elif normalized_type == "text/plain":
            content = self.read_txt_as_text(file_path)
            return f"\n\nText content:\n{content}"
        elif normalized_type == "application/pdf":
            content = self.read_pdf_as_text(file_path)
            return f"\n\nPDF content:\n{content}"
        return ""

    def _build_context_from_history(self, history: List[Dict[str, str]]) -> str:
        """
        Build context from chat history, limiting responses to 3-4 sentences to avoid overwhelming the agent.
        """
        context_lines = []
        for entry in history:
            user_input = entry.get("input", "")
            response = entry.get("response", "")
            timestamp = entry.get("timestamp", datetime.now(timezone.utc).isoformat())
            
            # Limit response to first 3-4 sentences to keep context concise
            if response:
                sentences = self._extract_first_sentences(response, max_sentences=4)
                response = sentences
            
            context_lines.append(f"[{timestamp}] User: {user_input}")
            if response:
                context_lines.append(f"[{timestamp}] Agent: {response}")
        
        return "\n".join(context_lines)
    
    def _extract_first_sentences(self, text: str, max_sentences: int = 4) -> str:
        """
        Extract the first few sentences from a text, handling various sentence endings.
        """
        if not text:
            return ""
        
        # Clean the text first
        text = text.strip()
        
        # Split by common sentence endings
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Take only the first max_sentences
        if len(sentences) <= max_sentences:
            return text
        
        # Join the first sentences and ensure proper ending
        result = ' '.join(sentences[:max_sentences])
        
        # If the result doesn't end with punctuation, add ellipsis
        if not result.endswith(('.', '!', '?')):
            result += '...'
        
        return result

    def _summarize_knowledge_sources(self) -> str:
        """
        Generate a summary of loaded knowledge sources for inclusion in task description.
        """
        summary_lines = []
        for source in self.knowledge_sources:
            metadata = source.metadata or {}
            source_id = metadata.get("id", "unknown")
            title = metadata.get("title", "Untitled")
            if isinstance(source, StringKnowledgeSource):
                content_snippet = source.content[:50].replace("\n", " ") + "..." if len(source.content) > 50 else source.content
                summary_lines.append(f"- Text Source (ID: {source_id}, Title: {title}): {content_snippet}")
            elif isinstance(source, CrewDoclingSource):
                summary_lines.append(f"- URL Source (ID: {source_id}, Title: {title}): {source.file_paths[0]}")
            elif isinstance(source, (TextFileKnowledgeSource, PDFKnowledgeSource, CSVKnowledgeSource, ExcelKnowledgeSource)):
                filename = metadata.get("filename", "unknown")
                summary_lines.append(f"- File Source (ID: {source_id}, Title: {title}, File: {filename})")
        return "\n".join(summary_lines) if summary_lines else "No knowledge sources available."

    async def clean_agent_output_with_llm(self, expected_output: str, user_input: str, final_output: str, api_key: str = None) -> str:
        """
        Uses LLM to clean agent output that contains unwanted thinking patterns.
        
        Args:
            expected_output: The expected format/content of the output
            user_input: The original user request/input
            final_output: The raw output from the agent that needs cleaning
            api_key: Gemini API key (optional, will use env var if not provided)
        
        Returns:
            Cleaned output string
        """
        
        # Check if cleaning is needed
        thinking_patterns = ['thought:', 'thoughts:']
        needs_cleaning = any(final_output.lower().strip().startswith(pattern) for pattern in thinking_patterns)
        
        if not needs_cleaning:
            return final_output

        self.logger.info(f"Cleaning agent 'Thought' output with LLM. Agent Output: {final_output}")

        # `api_key` is kept for backwards compatibility but is unused — the
        # internal Vertex AI client (Secret Manager / ADC) is used instead.

        cleaning_prompt = f"""
            You are tasked with cleaning an AI agent's output by removing unwanted thinking processes and formatting issues.

            USER INPUT:
            {user_input}

            EXPECTED OUTPUT FORMAT:
            {expected_output}

            RAW AGENT OUTPUT (needs cleaning):
            {final_output}

            INSTRUCTIONS:
            1. Remove any lines that start with "Thought:", "Thoughts:", "Action:", "Observation:", "Final Answer:", etc.
            2. Remove any reasoning steps or thinking processes
            3. Extract only the actual answer/content that directly addresses the user's request
            4. Format the output according to the expected output format
            5. Ensure the response is clean, direct, and professional
            6. Do not add any new information - only clean and reformat existing content

            Return only the cleaned output without any explanations or meta-commentary.
        """
        
        try:
            cleaned_output = await asyncio.to_thread(
                self.internal_llm_client.call,
                [{"role": "user", "content": cleaning_prompt}],
            )
            return (cleaned_output or "").strip()
        except Exception as e:
            self.logger.error(f"Error cleaning agent 'Thought' output with LLM: {e}")
            # Fallback to basic regex cleaning if LLM fails
            return self.basic_regex_clean(final_output)

    def basic_regex_clean(self, text: str) -> str:
        """Fallback cleaning function using regex"""
        lines = text.split('\n')
        cleaned_lines = []
        
        thinking_patterns = ['thought:', 'thoughts:']
        
        for line in lines:
            stripped_line = line.strip().lower()
            if not any(stripped_line.startswith(pattern) for pattern in thinking_patterns):
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
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
        # Initialize temp directories to avoid UnboundLocalError
        temp_memory_dir = None
        temp_cache_dir = None
        
        # ENHANCED FIX: Set execution context with request-specific isolation and path length safeguards
        if input:
            # Create request-specific temporary directories with ultra-short IDs
            short_id = str(uuid.uuid4())[:6]  # Use only first 6 chars of UUID for shorter paths
            temp_memory_dir = f"temp/cm_{short_id}"  # Ultra-short prefix
            temp_cache_dir = f"temp/cc_{short_id}"   # Ultra-short prefix
            
            # CRITICAL FIX: Clear any existing CrewAI storage environment variables to prevent accumulation
            crew_env_vars = [key for key in os.environ.keys() if key.startswith(('CREWAI_', 'CREW_'))]
            for var in crew_env_vars:
                if var not in ['CREWAI_DISABLE_CACHE', 'CREWAI_DISABLE_TELEMETRY', 'CREWAI_DISABLE_SHARED_STATE', 
                              'CREWAI_DISABLE_MEMORY', 'CREWAI_DISABLE_PLANNING_CACHE', 'CREWAI_DISABLE_AGENTS_CACHE']:
                    del os.environ[var]
            
            # Set only essential environment variables with controlled paths
            os.environ['CREWAI_STORAGE_DIR'] = f"temp/cs_{short_id}"  # Ultra-short storage dir
            
            # Don't set additional CREW_* variables that could interfere with CrewAI's path generation
            
            # Create directories if they don't exist
            os.makedirs(temp_memory_dir, exist_ok=True)
            os.makedirs(temp_cache_dir, exist_ok=True)
            
            set_execution_context(short_id, input, {
                'temp_memory_dir': temp_memory_dir,
                'temp_cache_dir': temp_cache_dir,
                'agent_instance_id': id(self.agent),
                'task_executor_id': id(self),
                'execution_id': self.execution_id  # Keep full ID for logging
            })
            self.expected_user_input = input
            self.logger.info(f"[{self.execution_id}] EXECUTION CONTEXT SET - Input: '{sanitize_for_logging(input)}', Thread: {threading.get_ident()}, Agent ID: {id(self.agent)}")
        
        max_retries = 3
        retry_delay = 5
        attempt = 1

        class AgentOutput(BaseModel):
            answer: str
            
            class Config:
                extra = "forbid"  # Prevents additional fields

        while attempt <= max_retries:
            try:
                self.logger.info(f"[{self.execution_id}] Starting task execution: {sanitize_for_logging(task_name or 'Unnamed Task')} (Attempt: {attempt})")
                self.logger.debug(f"[{self.execution_id}] Input kwargs: {sanitize_for_logging(kwargs)}")
                
                # RELAXED FIX: Check execution context but don't block execution
                context = get_execution_context()
                if input and context.get('user_input') != input:
                    # Log warning but continue - context might be legitimately different
                    self.logger.warning(f"[{self.execution_id}] Context variation detected. Input: '{input}', Context: '{context.get('user_input')}'")
                    # Don't raise exception - let execution continue
                
                # Store status queue for use by tools
                self.status_queue = status_queue if status_queue else None
                
                # Send status update if queue is available
                if status_queue:
                    await status_queue.put(f"Starting task execution: {task_name or 'Unnamed Task'}")
                    
                    # Add more detailed status updates about the task
                    if self.tools:
                        await status_queue.put(f"Preparing {len(self.tools)} tools for your request")
                    
                    # Add info about the model being used
                    if hasattr(self.llm_client, 'model'):
                        # model_name = self.llm_client.model.split('/')[-1] if '/' in self.llm_client.model else self.llm_client.model
                        # await status_queue.put(f"Using {model_name} to process your request")
                        await status_queue.put(f"Preprocessing Tools..")
                
                if file_path:
                    self.logger.info(f"Processing file: {sanitize_for_logging(file_path)} (Attempt: {attempt})")
                    if status_queue:
                        await status_queue.put(f"Processing file: {os.path.basename(file_path)}")
                        
                if user_email:
                    self.logger.info(f"Task requested by user: {user_email} (Attempt: {attempt})")
                if history:
                    self.logger.info(f"Received chat history with {len(history)} entries (Attempt: {attempt})")
                    if status_queue and len(history) > 0:
                        await status_queue.put(f"Using conversation history with {len(history)} messages")
                else:
                    self.logger.info(f"No chat history provided (Attempt: {attempt})")
                if self.knowledge_sources:
                    self.logger.info(f"Using {len(self.knowledge_sources)} knowledge sources (Attempt: {attempt})")
                    if status_queue:
                        await status_queue.put(f"Loading {len(self.knowledge_sources)} knowledge sources")

                if history:
                    context = self._build_context_from_history(history)
                    self.logger.debug(f"Generated context from history: {sanitize_for_logging(context)}")
                    history_instruction = (
                        "CHAT HISTORY FOR CONTEXT:\n"
                        f"{context}\n\n"
                        "IMPORTANT: Use the above chat history to understand context and enhance the current user input below. "
                        "Before processing the user's request, analyze if the input contains pronouns or references like "
                        "'this', 'it', 'he', 'she', 'his', 'her', 'that', 'those', 'them', etc. that need clarification. "
                        "If so, modify the user input to be more specific and contextual based on the chat history. "
                        "For example:\n"
                        "- 'What about it?' â†’ 'What about [specific topic from history]?'\n"
                        "- 'How does this work?' â†’ 'How does [specific item from history] work?'\n"
                        "- 'Tell me more about her' â†’ 'Tell me more about [specific person from history]'\n"
                        "After enhancing the input, proceed with your response using the enhanced version.\n\n"
                    )
                else:
                    context = ""
                    history_instruction = ""

                task_info = self.get_task_descriptions(description, expected_output, task_name, **kwargs)
                processed_description = task_info["description"]
                processed_expected_output = task_info["expected_output"]

                # Structure the input to encourage contextual enhancement
                if input:
                    if history_instruction:
                        # Provide history context and current input for enhancement
                        current_input_section = f"CURRENT USER INPUT TO ENHANCE:\n{input}\n\n"
                        
                        if "{{input}}" in processed_description:
                            processed_description = processed_description.replace("{{input}}", current_input_section)
                        else:
                            processed_description = f"{processed_description}\n\n{current_input_section}"
                        
                        # Add history instruction to guide enhancement
                        processed_description = f"{history_instruction}{processed_description}"
                    else:
                        # No history, use input as-is
                        if "{{input}}" in processed_description:
                            processed_description = processed_description.replace("{{input}}", input)
                        else:
                            processed_description = f"{processed_description}\n{input}"
                else:
                    # Fallback if no current input
                    if history_instruction:
                        processed_description = f"{history_instruction}{processed_description}"

                if file_path and file_type:
                    file_content = self.process_file_content(file_path, file_type)
                    processed_description += file_content

                if self.knowledge_sources:
                    knowledge_instruction = (
                        "The agent has access to the following knowledge sources to inform its response:\n"
                        f"{self._summarize_knowledge_sources()}\n"
                        "Use these sources to provide accurate and relevant information in the response. "
                        "If the task requires specific details from these sources, reference them explicitly in the response."
                    )
                    processed_description = f"{knowledge_instruction}\n\n{processed_description}"

                direct_answer_mode = self.raw_output
                task = Task(
                    description=(
                        f"{processed_description}\n\n"
                        "The output should not include any thoughts, reasoning, or intermediate steps in the response."
                        "Return only the final output as specified in `expected_output`. "
                        "Do not include any thoughts, reasoning, intermediate steps, or additional explanations. "
                        "If a tool returns no results or an empty response (e.g., empty URLs or no data), generate a professional, concise, and context-appropriate response acknowledging the lack of information and offering assistance with related topics. "
                        "Do not include any tool response text or phrases like 'my circuits are buzzing' in the output."
                    ),
                    expected_output=processed_expected_output,
                    agent=self.agent,
                    **({"output_json": AgentOutput} if direct_answer_mode else {})
                )

                # CRITICAL FIX: Final safeguard - ensure clean environment before Crew creation
                final_storage_path = os.environ.get("CREWAI_STORAGE_DIR", "temp/crew_default")
                validated_storage_path = validate_path_length(final_storage_path, max_length=150)
                os.environ["CREWAI_STORAGE_DIR"] = validated_storage_path
                self.logger.debug(f"[{self.execution_id}] Final storage path: {validated_storage_path}")
                
                # Resolve embedder config fresh per execution. In production this
                # mints a short-lived ADC access token for the Vertex embedder,
                # so it must not be reused from a module-level cache.
                current_embedder_config = get_crewai_embedder()

                # ENHANCED FIX: Create completely fresh Crew instance with request-specific isolation
                crew = Crew(
                    agents=[self.agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=self.verbose,
                    planning=self.enable_planning,
                    cache=False,
                    memory=False,
                    # Request-specific configuration to prevent shared state
                    max_rpm=None,  # Disable any shared rate limiting
                    share_crew=False,  # Ensure no crew sharing
                    step_callback=None,  # No shared callbacks
                    task_callback=None,  # No shared task callbacks
                    embedder=current_embedder_config,  # Use Vertex/Gemini embeddings for knowledge
                    knowledge=None
                )

                self.logger.info(f"Initiating CrewAI execution (Attempt: {attempt})")
                if status_queue:
                    await status_queue.put("Analyzing the task...")

                    await status_queue.put(f"Preparing for the task execution..")
                    
                    # if self.tools:
                    #     await status_queue.put(f"Preparing {len(self.tools)} tools for your request")
                    
                    if hasattr(self.llm_client, 'model'):
                        # model_name = self.llm_client.model.split('/')[-1] if '/' in self.llm_client.model else self.llm_client.model
                        # await status_queue.put(f"Using {model_name} to process your request")
                        await status_queue.put(f"Executing the task..")

                    if self.knowledge_sources:
                        await status_queue.put(f"Loading {len(self.knowledge_sources)} knowledge source(s)")
                        try:
                            await status_queue.put("Knowledge sources loaded")
                        except Exception:
                            pass

                    # Execute the actual model
                    result = await crew.kickoff_async()
                    
                    await status_queue.put("")

                else:
                    # If no status queue, just run the model
                    result = await crew.kickoff_async()

                if not direct_answer_mode:
                    raw_output = ""
                    if hasattr(result, 'tasks_output') and result.tasks_output:
                        raw_output = result.tasks_output[0].raw.strip() if hasattr(result.tasks_output[0], 'raw') else str(result.tasks_output[0])
                    elif hasattr(result, 'raw'):
                        raw_output = result.raw.strip()
                        self.logger.info(f"result.raw: {sanitize_for_logging(raw_output)}")
                    elif isinstance(result, str):
                        raw_output = result.strip()
                    else:
                        raw_output = str(result)


                    if status_queue:
                        await status_queue.put("Finalizing results..")
                    
                    final_output = await self.clean_agent_output_with_llm(
                        expected_output=processed_expected_output,
                        user_input=input, 
                        final_output=raw_output
                    )
                else:
                    final_output = ""
                    try:
                        if hasattr(result, 'tasks_output') and result.tasks_output:
                            task_output = result.tasks_output[0]
                            
                            # Try pydantic first (most reliable)
                            if hasattr(task_output, 'pydantic') and task_output.pydantic:
                                final_output = task_output.pydantic.answer
                            # Try JSON parsing
                            elif hasattr(task_output, 'json') and task_output.json:
                                if isinstance(task_output.json, str):
                                    json_data = json.loads(task_output.json)
                                    final_output = json_data.get('answer', '')
                                elif isinstance(task_output.json, dict):
                                    final_output = task_output.json.get('answer', '')
                                else:
                                    final_output = str(task_output.json)
                            # Try direct output access
                            elif hasattr(task_output, 'output'):
                                final_output = task_output.output
                            else:
                                final_output = str(task_output)
                        else:
                            final_output = str(result)
                            
                    except (json.JSONDecodeError, AttributeError, Exception) as e:
                        self.logger.error(f"Error extracting structured output: {e}")
                        # Safe fallback
                        final_output = str(result)

                final_output = final_output.strip()

                self.logger.info(f"Task '{sanitize_for_logging(task_name or 'Unnamed Task')}' completed (Attempt: {attempt})")
                self.logger.debug(f"Raw LLM output: {sanitize_for_logging(final_output)}")

                # ENHANCED FIX: Cleanup request-specific resources with memory optimization
                self._cleanup_request_resources()
                
                # DIRECT CLEANUP: Also clean up temp directories directly
                if temp_memory_dir and os.path.exists(temp_memory_dir):
                    import shutil
                    try:
                        shutil.rmtree(temp_memory_dir, ignore_errors=True)
                        self.logger.debug(f"[{self.execution_id}] Direct cleanup - removed memory dir: {temp_memory_dir}")
                    except Exception as e:
                        self.logger.warning(f"[{self.execution_id}] Failed to directly cleanup memory dir: {e}")
                
                if temp_cache_dir and os.path.exists(temp_cache_dir):
                    import shutil
                    try:
                        shutil.rmtree(temp_cache_dir, ignore_errors=True)
                        self.logger.debug(f"[{self.execution_id}] Direct cleanup - removed cache dir: {temp_cache_dir}")
                    except Exception as e:
                        self.logger.warning(f"[{self.execution_id}] Failed to directly cleanup cache dir: {e}")
                
                # MEMORY OPTIMIZATION: Force garbage collection after successful execution
                collected = gc.collect()
                if collected > 0:
                    self.logger.debug(f"[{self.execution_id}] Post-execution GC freed {collected} objects")

                return final_output

            except Exception as e:
                self.logger.error(f"Error during task execution: {sanitize_for_logging(e)} (Attempt: {attempt})", exc_info=True)
                if attempt < max_retries:
                    self.logger.info(f"Retrying task execution (Attempt {attempt + 1}/{max_retries}) after {retry_delay} seconds")
                    await asyncio.sleep(retry_delay)
                    attempt += 1
                    continue
                else:
                    self.logger.error(f"Max retries ({max_retries}) reached for error: {sanitize_for_logging(e)}")
                    # Cleanup even on failure
                    self._cleanup_request_resources()
                    raise Exception(f"Error: Failed after {max_retries} attempts: {str(e)}")
        
        # ENHANCED CLEANUP: Always clean up temp directories after execution
        if temp_memory_dir and os.path.exists(temp_memory_dir):
            import shutil
            try:
                shutil.rmtree(temp_memory_dir, ignore_errors=True)
                self.logger.debug(f"[{self.execution_id}] Post-execution cleanup - removed memory dir: {temp_memory_dir}")
            except Exception as e:
                self.logger.warning(f"[{self.execution_id}] Failed to cleanup memory dir: {e}")
        
        if temp_cache_dir and os.path.exists(temp_cache_dir):
            import shutil
            try:
                shutil.rmtree(temp_cache_dir, ignore_errors=True)
                self.logger.debug(f"[{self.execution_id}] Post-execution cleanup - removed cache dir: {temp_cache_dir}")
            except Exception as e:
                self.logger.warning(f"[{self.execution_id}] Failed to cleanup cache dir: {e}")

    def _cleanup_request_resources(self):
        """Clean up request-specific resources to prevent shared state"""
        try:
            context = get_execution_context()
            
            # Clean up temporary directories
            if 'temp_memory_dir' in context:
                import shutil
                if os.path.exists(context['temp_memory_dir']):
                    shutil.rmtree(context['temp_memory_dir'], ignore_errors=True)
                    self.logger.debug(f"[{self.execution_id}] Cleaned up memory dir: {context['temp_memory_dir']}")
            
            if 'temp_cache_dir' in context:
                import shutil
                if os.path.exists(context['temp_cache_dir']):
                    shutil.rmtree(context['temp_cache_dir'], ignore_errors=True)
                    self.logger.debug(f"[{self.execution_id}] Cleaned up cache dir: {context['temp_cache_dir']}")
            
            # Clean up environment variables for this request using shortened IDs
            short_id = context.get('execution_id', '')[:6]  # Get ultra-shortened ID
            if short_id:
                env_vars_to_remove = [k for k in os.environ.keys() if short_id in k]
                for env_var in env_vars_to_remove:
                    del os.environ[env_var]
                    self.logger.debug(f"[{self.execution_id}] Cleaned up env var: {env_var}")
            
            # OPTIMIZED CLEANUP: Only clean up temporary execution-specific directories, not knowledge base directories
            # Don't clean up knowledge base storage directories as they should be reused
            temp_storage_vars = [k for k in os.environ.keys() if k.startswith('CREWAI_STORAGE') and 'temp' in os.environ.get(k, '')]
            for var in temp_storage_vars:
                storage_path = os.environ.get(var, '')
                if storage_path and os.path.exists(storage_path) and 'temp' in storage_path:
                    import shutil
                    try:
                        shutil.rmtree(storage_path, ignore_errors=True)
                        self.logger.debug(f"[{self.execution_id}] Cleaned up temporary storage dir: {storage_path}")
                    except Exception as e:
                        self.logger.warning(f"[{self.execution_id}] Failed to cleanup temp storage dir {storage_path}: {e}")
            
            # Reset execution-specific environment variables only
            if short_id:
                execution_env_vars = [k for k in os.environ.keys() if f"_{short_id}" in k]
                for var in execution_env_vars:
                    del os.environ[var]
                    self.logger.debug(f"[{self.execution_id}] Cleaned up execution-specific env var: {var}")
            
            # MEMORY OPTIMIZATION: Clear large objects and references
            if hasattr(self, 'tools') and self.tools:
                # Clear tools list to free memory safely
                for tool in self.tools:
                    if hasattr(tool, '__dict__'):
                        try:
                            tool.__dict__.clear()
                        except (ValueError, TypeError, AttributeError) as e:
                            # Some tools might have frozen or protected attributes
                            self.logger.debug(f"[{self.execution_id}] Could not clear tool dict: {e}")
                            continue
                self.tools.clear()
                self.logger.debug(f"[{self.execution_id}] Cleared tools references")
            
            if hasattr(self, 'knowledge_sources') and self.knowledge_sources:
                # Clear knowledge sources to free memory
                self.knowledge_sources.clear()
                self.logger.debug(f"[{self.execution_id}] Cleared knowledge sources")
            
            # Force garbage collection of agent and crew instances
            if hasattr(self, 'agent') and self.agent:
                self.logger.debug(f"[{self.execution_id}] Agent instance {id(self.agent)} will be garbage collected")
                # Clear agent's internal state if possible, but handle frozen fields safely
                if hasattr(self.agent, '__dict__'):
                    agent_dict = self.agent.__dict__.copy()
                    for key in agent_dict:
                        if key not in ['role', 'goal', 'backstory', 'id']:  # Skip essential and frozen attributes
                            try:
                                setattr(self.agent, key, None)
                            except (ValueError, TypeError) as e:
                                # Skip frozen fields or other immutable attributes
                                self.logger.debug(f"[{self.execution_id}] Skipped frozen/immutable agent field '{key}': {e}")
                                continue
                self.agent = None
                
            if hasattr(self, 'crew') and self.crew:
                # Clear crew's internal state safely
                if hasattr(self.crew, '__dict__'):
                    crew_dict = self.crew.__dict__.copy()
                    for key in crew_dict:
                        try:
                            setattr(self.crew, key, None)
                        except (ValueError, TypeError) as e:
                            # Skip frozen fields or other immutable attributes
                            self.logger.debug(f"[{self.execution_id}] Skipped frozen/immutable crew field '{key}': {e}")
                            continue
                self.crew = None
            
            # Clear LLM client references
            if hasattr(self, 'llm_client'):
                self.llm_client = None
            
            # Force immediate garbage collection for this cleanup
            collected = gc.collect()
            if collected > 0:
                self.logger.debug(f"[{self.execution_id}] Cleanup GC freed {collected} objects")
            
            _EXECUTION_CONTEXT.set(None)
            self.logger.debug(f"[{self.execution_id}] Request resource cleanup completed")
            
        except Exception as e:
            # Log the error but don't let cleanup failures affect the main execution
            self.logger.warning(f"[{self.execution_id}] Error during resource cleanup: {e}")
            # Try to at least clear the main references even if detailed cleanup fails
            try:
                if hasattr(self, 'agent'):
                    self.agent = None
                if hasattr(self, 'crew'):
                    self.crew = None
                if hasattr(self, 'tools'):
                    self.tools = []
                if hasattr(self, 'knowledge_sources'):
                    self.knowledge_sources = []
            except Exception as fallback_error:
                self.logger.warning(f"[{self.execution_id}] Even fallback cleanup failed: {fallback_error}")

def cleanup_old_tmp_directories(max_age_hours: int = 24, dry_run: bool = False) -> Dict[str, int]:
    """
    Standalone utility function to clean up old accumulated temp/ directories.
    This can be called periodically to prevent disk space issues.
    
    Args:
        max_age_hours: Delete directories older than this many hours (default: 24)
        dry_run: If True, only report what would be deleted without actually deleting
        
    Returns:
        Dictionary with cleanup statistics
    """
    import glob
    import shutil
    import fnmatch
    
    logger = logging.getLogger("tmp_cleanup")
    current_time = time.time()
    age_threshold = max_age_hours * 3600  # Convert to seconds
    
    # Patterns for directories that can be safely deleted (accumulated over time)
    cleanup_patterns = [
        'temp/cm_*',           # Memory directories (temp/cm_{short_id})
        'temp/cc_*',           # Cache directories (temp/cc_{short_id})
        'temp/cs_*',           # Storage directories (temp/cs_{short_id}) - YOUR ISSUE!
        'temp/crew_safe_*',    # Safe path directories
        'temp/crew_default*',  # Default crew storage
        'temp/crewai_base*',   # Base CrewAI storage
        'temp/crewai_memory_*', # Memory paths with full execution IDs
        'temp/crewai_cache_*',  # Cache paths with full execution IDs
        'temp/*crewai*tmp*',   # Any crewai temp files
        'temp/*crew*temp*',    # Any crew temp files
    ]
    
    # Directories to always preserve (contain valuable reusable data)
    preserve_patterns = [
        'temp/embed_cache*',        # Embedding cache - preserve for reuse
        'temp/vector_store*',       # Vector store - preserve for reuse  
        'temp/crewai_db*',          # Database storage - CRITICAL: preserve SQLite databases
        'temp/crewai_cache*',       # Cache storage - preserve for reuse
        'temp/crewai_storage_kb_*', # Knowledge base storage - preserve for reuse
    ]
    
    stats = {
        'scanned': 0,
        'deleted': 0,
        'preserved': 0,
        'errors': 0,
        'total_size_mb': 0
    }
    
    try:
        logger.info(f"Starting cleanup of temp/ directories older than {max_age_hours} hours {'(DRY RUN)' if dry_run else ''}")
        
        all_candidates = set()
        for pattern in cleanup_patterns:
            matching_items = glob.glob(pattern)
            all_candidates.update(matching_items)
        
        for item_path in all_candidates:
            stats['scanned'] += 1
            
            try:
                # Check if item exists and get its age
                if not os.path.exists(item_path):
                    continue
                    
                # Get modification time
                mtime = os.path.getmtime(item_path)
                age_seconds = current_time - mtime
                
                # Skip if not old enough
                if age_seconds < age_threshold:
                    logger.debug(f"Skipping recent item: {item_path} (age: {age_seconds/3600:.1f}h)")
                    continue
                
                # Check if this should be preserved
                should_preserve = False
                for preserve_pattern in preserve_patterns:
                    if fnmatch.fnmatch(item_path, preserve_pattern):
                        should_preserve = True
                        stats['preserved'] += 1
                        # logger.debug(f"Preserved important directory: {item_path}")
                        break
                
                if should_preserve:
                    continue
                
                # Calculate size before deletion
                size_mb = 0
                if os.path.isdir(item_path):
                    try:
                        # Calculate directory size
                        for dirpath, dirnames, filenames in os.walk(item_path):
                            for filename in filenames:
                                filepath = os.path.join(dirpath, filename)
                                try:
                                    size_mb += os.path.getsize(filepath) / (1024 * 1024)
                                except (OSError, IOError):
                                    pass
                    except Exception:
                        pass
                elif os.path.isfile(item_path):
                    try:
                        size_mb = os.path.getsize(item_path) / (1024 * 1024)
                    except (OSError, IOError):
                        pass
                
                stats['total_size_mb'] += size_mb
                
                # Delete the item (unless dry run)
                if not dry_run:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                    elif os.path.isfile(item_path):
                        os.remove(item_path)
                
                stats['deleted'] += 1
                age_hours = age_seconds / 3600
                logger.info(f"{'Would delete' if dry_run else 'Deleted'} old temp item: {item_path} (age: {age_hours:.1f}h, size: {size_mb:.1f}MB)")
                
            except Exception as e:
                stats['errors'] += 1
                logger.warning(f"Error processing {item_path}: {e}")
        
        logger.info(f"Cleanup completed: scanned={stats['scanned']}, deleted={stats['deleted']}, "
                   f"preserved={stats['preserved']}, errors={stats['errors']}, "
                   f"freed_space={stats['total_size_mb']:.1f}MB")
        
    except Exception as e:
        logger.error(f"Error during temp/ cleanup: {e}")
        stats['errors'] += 1
    
    return stats

def get_api_key():
    keys = os.getenv("RANDOM_GEMINI_API_KEYS", "")
    key_list = [key.strip() for key in keys.split(",") if key.strip()]
    if not key_list:
        raise ValueError("No API keys found in GEMINI_API_KEYS environment variable.")
    return random.choice(key_list)