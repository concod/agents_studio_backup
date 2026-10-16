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

class LangGraphExecutorStream:
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
        self.suggestive_questions = []  # Side-channel: captured from tool API responses
        self.display_metadata = None  # Side-channel: captured from image generation tool responses
        self.mcp_servers_config = mcp_servers or []
        self._mcp_manager = None  # Started in create()

        self.logger = logging.getLogger(f"langraph_executor_stream_{self.execution_id}")
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
        
        # Internal LLM for payload generation
        self.internal_llm = get_llm(
            temperature=0.3,
            max_output_tokens=4096,
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
    
    def _create_tools(self, tools_config: List[Dict]) ->None:
        """Create LangChain-compatible tools from tools_config"""
        from langchain_core.tools import StructuredTool
        from pydantic import BaseModel, Field
        
        for tool_config in tools_config:
            tool_schema = tool_config["schema"]
            tool_headers = tool_config.get("auth", {}).get("headers", {})
            tool_params = tool_config.get("auth", {}).get("params", {})
            tool_data_connector = tool_config.get("data_connector", None)
            
            tool_name = tool_schema["info"]["title"].lower().replace(" ", "_")
            tool_description = f"Calls API: {tool_schema['info']['title']}. {tool_schema['info'].get('description', '')}"
            
            # Create a tool function
            async def tool_func(input_data: str, _tool_schema=tool_schema, _tool_headers=tool_headers, 
                              _tool_params=tool_params, _tool_data_connector=tool_data_connector):
                """Execute the API tool"""
                try:
                    # Generate payload using the same logic as TaskExecutor
                    result = await self._generate_and_call_api(
                        input_data, _tool_schema, _tool_headers, _tool_params, _tool_data_connector
                    )
                    
                    # Capture suggestive_questions from tool response (side-channel)
                    # and remove from result so the LLM doesn't duplicate them in its text output
                    if isinstance(result, dict) and "suggestive_questions" in result:
                        sq = result.pop("suggestive_questions", None)
                        if isinstance(sq, list) and sq:
                            self.suggestive_questions = sq

                    # Capture display_metadata from image generation tool response (side-channel)
                    if isinstance(result, dict) and "display_metadata" in result:
                        dm = result.pop("display_metadata", None)
                        if isinstance(dm, dict) and dm:
                            self.display_metadata = dm

                    return json.dumps(result)
                except Exception as e:
                    self.logger.error(f"Tool execution error: {str(e)}")
                    return json.dumps({"error": str(e)})
            
            # Create LangChain structured tool
            lang_tool = StructuredTool.from_function(
                func=tool_func,
                name=tool_name,
                description=tool_description,
                coroutine=tool_func  # async version
            )
            
            self.tools.append(lang_tool)
            self.logger.debug(f"Created LangChain tool: {tool_name}")
    
    def _extract_final_json_object(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Extract the final valid JSON object from LLM response"""
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
        
        for json_obj in reversed(valid_jsons):
            if ('payload' in json_obj and 'endpoint_url' in json_obj) or 'error' in json_obj:
                return json_obj
        
        return valid_jsons[-1] if valid_jsons else None

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
            # If omitted, downstream BigQuery client uses Application Default Credentials.
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
                               tool_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate API payload using LLM"""
        self.logger.debug(f"[Payload Gen] Starting payload generation for: {tool_name or 'API Tool'}")
        
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

Perform the following steps:
1. Analyze the OpenAPI schema, focusing on required fields, types, constraints, parameters, and response structure.
2. Validate that the user input contains all necessary information to generate a valid payload based on the schema's required fields.
3. If validation passes, generate a valid JSON payload and the endpoint URL for the API call.
4. Construct the endpoint URL using the server URL from the schema's "servers" field and the correct path from the "paths" field.
5. If schema has a db_config field, set it to null in the payload — it will be filled in separately.

Return ONLY a single JSON object:
- If successful: {{"payload": <payload_data>, "endpoint_url": "<url>"}}
- If error: {{"error": "<error_message>"}}

Do NOT include explanatory text. Return ONLY the final JSON object."""

        for attempt in range(2):
            try:
                self.logger.debug(f"[Payload Gen] LLM attempt {attempt + 1}/2")
                messages = [HumanMessage(content=payload_prompt)]
                response = await self.internal_llm.ainvoke(messages)
                result_str = response.content
                
                self.logger.debug(f"[Payload Gen] LLM response length: {len(str(result_str))} chars")
                # Log raw LLM response for debugging
                self.logger.debug(f"[Payload Gen] Raw LLM response: {sanitize_for_logging(result_str[:1000])}")

                if "```" in result_str:
                    result_str = re.sub(r"```(json)?", "", result_str).strip()
                    self.logger.debug(f"[Payload Gen] After code block removal: {sanitize_for_logging(result_str[:1000])}")

                result_json = self._extract_final_json_object(result_str)
                if not result_json:
                    self.logger.warning(f"[Payload Gen] No valid JSON found in LLM response (attempt {attempt + 1}). Response was: {sanitize_for_logging(result_str[:500])}")
                    if attempt == 0:
                        continue
                    self.logger.error(f"[Payload Gen] Failed to extract JSON after 2 attempts")
                    return {"error": "No valid JSON in LLM response"}

                if "error" in result_json:
                    self.logger.warning(f"[Payload Gen] LLM returned error: {result_json['error']}")
                    return result_json

                payload = result_json.get("payload")
                endpoint_url = result_json.get("endpoint_url")
                
                if not endpoint_url:
                    self.logger.error(f"[Payload Gen] Missing endpoint_url in LLM response")
                    return {"error": "Missing endpoint_url"}

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
            # Generate payload using LLM
            result = await self._generate_payload(input_data, tool_schema, tool_data_connector, tool_name)
            
            if not result or "error" in result:
                self.logger.error(f"[API Call] Payload generation failed: {result.get('error', 'Unknown error')}")
                return result

            payload = result.get("payload")
            endpoint_url = result.get("endpoint_url")
            if not endpoint_url:
                self.logger.error(f"[API Call] Missing endpoint URL in payload result")
                return {"error": "Missing endpoint URL"}

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

            # Make the API request
            self.logger.info(f"[API Call] Sending {method.upper()} request to {endpoint_url}")
            call_start = time.time()
            
            timeout = aiohttp.ClientTimeout(total=1200, connect=30, sock_read=900)
            async with aiohttp.ClientSession(timeout=timeout) as session:
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
                                return result
                            except:
                                result = {"result": await response.text()}
                                self.logger.info(f"[API Call] Success - Text response")
                                self.logger.debug(f"[API Call] Response text: {sanitize_for_logging(str(result.get('result', ''))[:500])}")
                                return result
                        else:
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
                                return result
                            except:
                                result = {"result": await response.text()}
                                self.logger.info(f"[API Call] Success - Text response")
                                self.logger.debug(f"[API Call] Response text: {sanitize_for_logging(str(result.get('result', ''))[:500])}")
                                return result
                        else:
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
        temp_logger = logging.getLogger(f"langraph_executor_stream_loader_{execution_id}")
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
    async def _load_knowledge_sources(knowledge_base_id: Optional[str] = None, executor_logger: Optional[logging.Logger] = None) -> List[str]:
        """
        Load knowledge sources from knowledge_base.json based on knowledge_base_id.
        Returns a list of knowledge content strings.
        """
        logger = executor_logger or logging.getLogger(f"langraph_executor_loader")
        if not knowledge_base_id:
            logger.debug("No knowledge_base_id provided in agent config")
            return []

        knowledge_contents = []
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
                        knowledge_contents.append(source_content)
                        logger.debug(f"Loaded text knowledge source: {source_id}")

                    elif source_type == "file":
                        file_path = f"{source.get('filename')}"
                        actual_file_path = f"knowledge/{file_path}"
                        if not actual_file_path or not os.path.exists(actual_file_path):
                            logger.error(f"File path missing or invalid for source: {source_id} ({source_content})")
                            continue

                        file_extension = os.path.splitext(file_path)[1].lower()
                        if file_extension == ".txt":
                            # Read text file content
                            async with aiofiles.open(actual_file_path, "r", encoding="utf-8") as f:
                                file_content = await f.read()
                                knowledge_contents.append(f"=== {source.get('filename')} ===\n{file_content}")
                                logger.debug(f"Loaded text file knowledge source: {source_id} ({file_path})")
                        elif file_extension == ".pdf":
                            # Read PDF content (note: PyPDF2 doesn't have async support, using sync in thread)
                            try:
                                import PyPDF2
                                # Read file content first
                                async with aiofiles.open(actual_file_path, "rb") as pdf_file:
                                    pdf_bytes = await pdf_file.read()
                                # Parse PDF
                                from io import BytesIO
                                pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
                                pdf_text = ""
                                for page in pdf_reader.pages:
                                    pdf_text += page.extract_text()
                                knowledge_contents.append(f"=== {source.get('filename')} ===\n{pdf_text}")
                                logger.debug(f"Loaded PDF knowledge source: {source_id} ({file_path})")
                            except Exception as e:
                                logger.error(f"Error reading PDF {file_path}: {str(e)}")
                        elif file_extension == ".csv":
                            # Read CSV content
                            async with aiofiles.open(actual_file_path, "r", encoding="utf-8") as f:
                                csv_content = await f.read()
                                knowledge_contents.append(f"=== {source.get('filename')} ===\n{csv_content}")
                                logger.debug(f"Loaded CSV knowledge source: {source_id} ({file_path})")
                        else:
                            logger.warning(f"Unsupported file extension: {file_extension} for source: {source_id}")
                            continue

                    else:
                        logger.warning(f"Unsupported knowledge source type: {source_type} for source: {source_id}")
                except Exception as e:
                    logger.error(f"Error loading knowledge source {source_id}: {sanitize_for_logging(str(e))}")
                    continue

            logger.info(f"Loaded {len(knowledge_contents)} knowledge sources for knowledge base ID: {knowledge_base_id}")
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
            system_prompt += f"""

Knowledge Base:
{self.knowledge_content}

Use the information from the knowledge base above to answer the user's questions accurately."""

        system_prompt += """

Respond to the user's request thoughtfully.

CRITICAL INSTRUCTIONS FOR YOUR RESPONSE:
1. DO NOT include any internal reasoning, thoughts, or planning in your response.
2. DO NOT start your response with phrases like "The user is asking...", "This is a simple question...", "I should...", "Let me...", "I will...", "First, I need to...", etc.
3. DO NOT use prefixes like 'THOUGHT:', 'REASONING:', 'PLAN:', or 'INTERNAL:' in your response.
4. JUST provide the direct, helpful answer to the user's question.
5. If you used tools, use the data returned to provide a comprehensive answer.
6. Your response should read naturally as if you're directly addressing the user - no meta-commentary about your own process."""

        
        # Bind tools to LLM if available
        if self.tools:
            llm_with_tools = self.llm.bind_tools(self.tools)
            self.logger.debug(f"[Agent Node] Bound {len(self.tools)} tools to LLM")
        else:
            llm_with_tools = self.llm
            self.logger.debug(f"[Agent Node] No tools available, using LLM directly")
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages")
        ])
        
        # Invoke the LLM with streaming for "Real Thoughts"
        self.logger.debug(f"[Agent Node] Invoking LLM with streaming")
        chain = prompt | llm_with_tools
        
        full_response = None
        
        # Stream the chunks to capture "Thoughts" in real-time
        async for chunk in chain.astream({"messages": messages}):
            if full_response is None:
                full_response = chunk
            else:
                full_response += chunk
            
            # If we have content (text), stream it as a "Thought"
            if chunk.content and self.status_queue:
                # We prefix with THOUGHT: so the UI knows how to handle it
                await self.status_queue.put(f"THOUGHT: {chunk.content}")
        
        # Log response details
        self.logger.info(f"[Agent Node] LLM responded with message type: {type(full_response).__name__}")
        if hasattr(full_response, "tool_calls") and full_response.tool_calls:
            self.logger.info(f"[Agent Node] LLM requested {len(full_response.tool_calls)} tool call(s)")
        
        # Update state
        new_messages = list(messages)
        new_messages.append(full_response)
        
        return {
            **state,
            "messages": new_messages
        }

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
                
                self.logger.info(f"[Action Node] [{idx}/{len(last_message.tool_calls)}] Executing tool: {tool_name}")
                if self.status_queue:
                    await self.status_queue.put(f"Executing tool: {tool_name} with arguments: {json.dumps(tool_args)}")
                
                # Find the tool in our tools list
                tool_result = None
                tool_found = False
                for tool in self.tools:
                    if tool.name == tool_name:
                        tool_found = True
                        try:
                            start_time = time.time()
                            
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
            
            os.makedirs(temp_memory_dir, exist_ok=True)
            os.makedirs(temp_cache_dir, exist_ok=True)
            
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
                
                # Execute the graph using astream for real-time thoughts
                self.logger.info(">>> Executing LangGraph workflow (Streaming mode) <<<")
                graph_start = time.time()
                
                final_state = None
                async for event in self.graph.astream(initial_state, stream_mode="updates"):
                    # event is a dict mapping node_name -> update
                    for node_name, update in event.items():
                        self.logger.debug(f"[Stream] Node '{node_name}' completed")
                        
                        # Merge update into our tracked state to keep it up to date
                        # In 'updates' mode, only the delta is returned
                        for key, value in update.items():
                            if initial_state.get(key) is None:
                                initial_state[key] = value
                            elif isinstance(initial_state[key], list) and isinstance(value, list):
                                initial_state[key].extend(value)
                            else:
                                initial_state[key] = value
                        
                        # Send granular updates based on the node
                        if self.status_queue:
                            if node_name == "action":
                                await self.status_queue.put("STATUS: Tool execution completed. Refining response..")
                
                # The initial_state now contains the accumulated results from the stream
                final_state = initial_state
                
                graph_duration = time.time() - graph_start
                self.logger.info(f">>> Graph execution completed in {graph_duration:.2f}s <<<")
                
                # Extract final response
                final_messages = final_state.get("messages", [])
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
                
                # Clean up internal reasoning/planning from the final result
                if isinstance(result, str):
                    original_result = result
                    
                    # Remove THOUGHT:, REASONING:, PLAN: prefixes and their content lines
                    result = re.sub(r'^\s*THOUGHT:?\s*[^\n]*\n?', '', result, flags=re.IGNORECASE | re.MULTILINE)
                    result = re.sub(r'^\s*REASONING:?\s*[^\n]*\n?', '', result, flags=re.IGNORECASE | re.MULTILINE)
                    result = re.sub(r'^\s*PLAN:?\s*[^\n]*\n?', '', result, flags=re.IGNORECASE | re.MULTILINE)
                    result = re.sub(r'^\s*INTERNAL:?\s*[^\n]*\n?', '', result, flags=re.IGNORECASE | re.MULTILINE)
                    
                    # Remove common reasoning patterns that indicate internal monologue
                    reasoning_patterns = [
                        r'^The user is asking[^\n]*\n?',
                        r'^This is a (simple|general|basic)[^\n]*\n?',
                        r'^I (should|will|need to|must)[^\n]*\n?',
                        r'^Let me[^\n]*\n?',
                        r'^First,? I (should|will|need)[^\n]*\n?',
                        r'^Since[^\n]*(does not require|doesn\'t require)[^\n]*\n?',
                        r'^This (does not require|doesn\'t require)[^\n]*\n?',
                        r'^(Based on|Given) (this|the)[^\n]*analysis[^\n]*\n?',
                        r'^I (do not|don\'t) have access to[^\n]*\n?',
                        r'^I cannot provide[^\n]*real-time[^\n]*\n?',
                    ]
                    
                    for pattern in reasoning_patterns:
                        result = re.sub(pattern, '', result, flags=re.IGNORECASE | re.MULTILINE)
                    
                    # Remove any remaining inline THOUGHT patterns
                    result = re.sub(r'THOUGHT:?\s*[^\n]*\n?', '', result, flags=re.IGNORECASE)
                    
                    # Clean up excessive newlines
                    result = re.sub(r'\n{3,}', '\n\n', result)
                    result = result.strip()
                    
                    if original_result != result:
                        self.logger.debug(f"Cleaned result: removed internal reasoning/planning content")
                    
                    # If result became empty after cleanup, it means it was only internal monologue
                    if not result:
                        result = "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
                        self.logger.warning("Result was empty after removing internal monologue")
                
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
        """Cleanup resources"""
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
        
        # Cleanup temporary directories
        context = get_execution_context()
        if context:
            temp_memory_dir = context.get('temp_memory_dir')
            temp_cache_dir = context.get('temp_cache_dir')
            
            for temp_dir in [temp_memory_dir, temp_cache_dir]:
                if temp_dir and os.path.exists(temp_dir):
                    try:
                        import shutil
                        shutil.rmtree(temp_dir)
                        self.logger.debug(f"Removed temporary directory: {temp_dir}")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove temporary directory {temp_dir}: {e}")
        
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

