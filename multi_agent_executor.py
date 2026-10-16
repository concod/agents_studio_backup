import os
import requests
import random
import json
import re
import uuid
import logging
import aiohttp
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from crewai import Crew, Process, Task, Agent as CrewAgent, LLM
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import jwt
import PyPDF2
from PIL import Image
import base64
import csv

load_dotenv()

# Disable CrewAI telemetry
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"

# File type mappings
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

class GeminiFailoverLLM:
    def __init__(self, model: str, api_keys: List[str], temperature: float = 1, logger: logging.Logger = None):
        self.model = model
        self.api_keys = api_keys
        self.temperature = temperature
        self.logger = logger

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

class MultiAgentExecutor:
    """
    Orchestrates multiple agents with a manager agent that delegates tasks dynamically.
    Handles user input, preserves context, and ensures robust error handling with logging.
    """
    def __init__(
        self,
        multi_agent_config: Dict[str, Any],
        worker_agent_configs: List[Dict[str, Any]],
        execution_id: Optional[str] = None,
        log_file: Optional[str] = None,
        log_url: Optional[str] = None,
        user_email: Optional[str] = None
    ):
        self.execution_id = execution_id or str(uuid.uuid4())
        self.logger = logging.getLogger(f"multi_agent_executor_{self.execution_id}")
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        if log_file:
            try:
                file_handler = logging.FileHandler(log_file, encoding="utf-8")
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.error(f"Failed to create log file handler for {log_file}: {self._sanitize_for_logging(e)}")

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.multi_agent_config = multi_agent_config
        self.worker_agent_configs = worker_agent_configs
        self.log_url = log_url
        self.user_email = user_email
        self._validate_configs()

        # Generate JWT token if user_email is provided
        if user_email:
            self.token = self._generate_token(user_email)
            self.logger.debug(f"Generated token for user: {user_email}")
        else:
            self.token = None
            self.logger.debug("No user email provided, token not generated")

        # Initialize LLM clients
        api_keys_string = os.getenv("RANDOM_GEMINI_API_KEYS", "")
        gemini_keys = [key.strip() for key in api_keys_string.split(",") if key.strip()]
        self.llm_client = GeminiFailoverLLM(
            model="gemini/gemini-2.5-flash",
            api_keys=gemini_keys,
            temperature=0.9,
            logger=self.logger
        )
        self.internal_llm_client = LLM(
            model="gemini/gemini-2.5-flash",
            api_key=os.getenv("INTERNAL_GEMINI_API_KEY")
        )

        # Initialize Schema and Payload Agents
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
            goal="Generate accurate payloads and endpoint URLs for API tools",
            backstory="Expert at creating valid API payloads and determining endpoint URLs.",
            verbose=False,
            allow_delegation=False,
            llm=self.internal_llm_client
        )

        # Initialize Manager Agent
        self.manager_agent = CrewAgent(
            role=self.multi_agent_config.get("role", "Coordinator"),
            goal=self.multi_agent_config.get("goal", "Delegate tasks efficiently."),
            backstory=self.multi_agent_config.get("backstory", "Orchestrates worker agents."),
            llm=self.llm_client,
            verbose=True,
            allow_delegation=True
        )
        self.logger.info(f"Initialized Manager Agent: {self.manager_agent.role}")

        # Initialize Worker Agents
        self.worker_agents = []
        self.worker_map = {}
        for config in self.worker_agent_configs:
            tools = self._load_agent_tools(config)
            agent = CrewAgent(
                role=config["role"],
                goal=config["goal"],
                backstory=config["backstory"],
                llm=self.llm_client,
                tools=tools,
                verbose=True,
                allow_delegation=False
            )
            self.worker_agents.append(agent)
            self.worker_map[config["id"]] = {
                "agent": agent,
                "config": config
            }
            self.logger.info(f"Initialized Worker Agent: {agent.role} (ID: {config['id']}, Name: {config['name']})")

        if not self.worker_agents:
            self.logger.warning("No worker agents initialized.")

    def _sanitize_for_logging(self, text: Any) -> str:
        """Sanitizes strings for logging, preserving emojis and handling non-UTF-8 bytes."""
        try:
            text_str = str(text)
            return text_str.encode("utf-8", errors="replace").decode("utf-8")
        except Exception:
            return str(text).encode("utf-8", errors="replace").decode("utf-8")

    def _generate_token(self, email: str, expires_delta: timedelta = timedelta(minutes=20)) -> str:
        """Generate a JWT token for the given email."""
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

    def _validate_configs(self):
        """Validates configurations."""
        required_manager_fields = ["role", "goal", "backstory", "description", "expected_output", "agent_ids"]
        required_worker_fields = ["id", "role", "goal", "backstory", "instructions", "expectedOutput"]

        for field in required_manager_fields:
            if field not in self.multi_agent_config:
                self.logger.warning(f"Missing manager config field: {field}. Using default.")
                self.multi_agent_config[field] = (
                    f"Default {field}" if field != "expected_output"
                    else "Formatted output containing each agent's result."
                    if field != "agent_ids" else []
                )

        for config in self.worker_agent_configs:
            for field in required_worker_fields:
                if field not in config:
                    self.logger.warning(f"Missing worker config field: {field} for agent {config.get('id', 'unknown')}. Using default.")
                    config[field] = f"Default {field}"
            if "name" not in config:
                config["name"] = config["role"]
                self.logger.warning(f"Missing name for agent {config['id']}, using role: {config['name']}.")

    def encode_image_to_base64(self, image_path: str) -> str:
        self.logger.debug(f"Encoding image: {self._sanitize_for_logging(image_path)}")
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            self.logger.error(f"Error encoding image: {self._sanitize_for_logging(e)}")
            return f"Error encoding image: {self._sanitize_for_logging(e)}"

    def read_csv_as_text(self, csv_path: str) -> str:
        self.logger.debug(f"Reading CSV: {self._sanitize_for_logging(csv_path)}")
        try:
            with open(csv_path, "r", encoding="utf-8", errors="replace") as csv_file:
                content = csv_file.read()
                return content
        except Exception as e:
            self.logger.error(f"Error reading CSV: {self._sanitize_for_logging(e)}")
            return f"Error reading CSV: {self._sanitize_for_logging(e)}"

    def read_json_as_text(self, json_path: str) -> str:
        self.logger.debug(f"Reading JSON: {self._sanitize_for_logging(json_path)}")
        try:
            with open(json_path, "r", encoding="utf-8", errors="replace") as json_file:
                data = json.load(json_file)
                return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error reading JSON: {self._sanitize_for_logging(e)}")
            return f"Error reading JSON: {self._sanitize_for_logging(e)}"

    def read_txt_as_text(self, txt_path: str) -> str:
        self.logger.debug(f"Reading TXT: {self._sanitize_for_logging(txt_path)}")
        try:
            with open(txt_path, "r", encoding="utf-8", errors="replace") as txt_file:
                return txt_file.read()
        except Exception as e:
            self.logger.error(f"Error reading TXT: {self._sanitize_for_logging(e)}")
            return f"Error reading TXT: {self._sanitize_for_logging(e)}"

    def read_pdf_as_text(self, pdf_path: str) -> str:
        self.logger.debug(f"Reading PDF: {self._sanitize_for_logging(pdf_path)}")
        try:
            with open(pdf_path, "rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = "".join(page.extract_text() or "" for page in reader.pages)
                return text.strip() or "No readable text in PDF"
        except Exception as e:
            self.logger.error(f"Error reading PDF: {self._sanitize_for_logging(e)}")
            return f"Error reading PDF: {self._sanitize_for_logging(e)}"

    def process_file_content(self, file_path: str, file_type: str) -> str:
        self.logger.debug(f"Processing file: {self._sanitize_for_logging(file_path)}, type: {file_type}")
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
        sanitized_result = self._sanitize_for_logging(result)
        return str(result)

    async def generate_payload(self, user_input: str, schema: dict, tool_data_connector: Optional[dict] = None) -> Dict[str, Any]:
        self.logger.debug(f"Generating payload and endpoint URL for input: {self._sanitize_for_logging(user_input)}")
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

        connector_info = f"db_informations:\n{json.dumps(tool_data_connector, indent=2, ensure_ascii=False)}\n" if tool_data_connector else ""
        payload_task = Task(
            description=f"""
            User Input: '{self._sanitize_for_logging(user_input)}'
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
            
            Return the response as a JSON object with two keys (unless an error occurs):
            - "payload": The JSON payload or query parameters (or null if not applicable).
            - "endpoint_url": The full URL for the API endpoint, corresponding to the {method.upper()} endpoint in the schema.
            If an error occurs due to missing required fields, return a JSON object with a single key:
            - "error": A message describing the missing required fields.
            Ensure the endpoint_url is valid and matches the schema. Do NOT use example values from the schema unless explicitly provided in the user input.
            """,
            expected_output="JSON object with 'payload' and 'endpoint_url', or 'error' if required fields are missing",
            agent=self.payload_agent
        )
        crew = Crew(agents=[self.payload_agent], tasks=[payload_task], process=Process.sequential)
        result = await crew.kickoff_async()

        try:
            if hasattr(result, 'raw'):
                result_str = str(result.raw)
            else:
                result_str = str(result)

            self.logger.debug(f"Raw result from Crew: {self._sanitize_for_logging(result_str)}")

            start_marker = "```json\n"
            end_marker = "\n```"
            start_idx = result_str.find(start_marker)
            end_idx = result_str.rfind(end_marker)

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                result_str = result_str[start_idx + len(start_marker):end_idx].strip()
            else:
                self.logger.error("No valid JSON block found in agent response")
                return {"error": "No valid JSON block found in agent response"}

            if not result_str:
                self.logger.error("Empty JSON content after extracting block")
                return {"error": "Empty JSON content from agent"}

            try:
                result_json = json.loads(result_str)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON: {self._sanitize_for_logging(result_str)}, error: {self._sanitize_for_logging(e)}")
                return {"error": f"Invalid JSON response from agent: {self._sanitize_for_logging(e)}"}

            if "error" in result_json:
                self.logger.error(f"Payload generation failed: {self._sanitize_for_logging(result_json['error'])}")
                return {"error": result_json["error"]}
            
            payload = result_json.get("payload")
            endpoint_url = result_json.get("endpoint_url")
            
            if not endpoint_url:
                self.logger.error("Missing endpoint_url in agent response")
                return {"error": "Missing endpoint_url in agent response"}
            
            self.logger.debug(f"Generated payload: {json.dumps(payload, ensure_ascii=False) if payload else 'null'}")
            self.logger.debug(f"Generated endpoint URL: {endpoint_url}")
            return {"payload": payload, "endpoint_url": endpoint_url}

        except Exception as e:
            self.logger.error(f"Unexpected error during payload generation: {self._sanitize_for_logging(e)}")
            return {"error": f"Unexpected error during payload generation: {self._sanitize_for_logging(e)}"}

    def _load_agent_tools(self, agent_config: Dict[str, Any]) -> List:
        """Loads tools for an agent."""
        tools = []
        tools_config = agent_config.get("tools", [])
        agent_id = agent_config.get("id", "unknown")

        for tool_config in tools_config:
            if not isinstance(tool_config, dict) or "schema" not in tool_config:
                self.logger.warning(f"Skipping invalid tool config for agent {agent_id}: {tool_config}")
                continue

            tool_schema = tool_config.get("schema")
            if not isinstance(tool_schema, dict) or "info" not in tool_schema or "paths" not in tool_schema:
                self.logger.warning(f"Skipping invalid schema for tool {tool_config.get('id')} in agent {agent_id}")
                continue

            tool_headers = tool_config.get("auth", {}).get("headers", {}) or {}
            tool_params = tool_config.get("auth", {}).get("params", {}) or {}
            tool_data_connector = tool_config.get("data_connector", None)

            tool_name = tool_schema.get("info", {}).get("title", f"tool_{tool_config.get('id')}")
            tool_name = tool_name.lower().replace(" ", "_")
            tool_title = tool_schema.get("info", {}).get("title", "Unknown API")
            tool_description = tool_schema.get("info", {}).get("description", "")
            tool_description = f"Calls API: {tool_title}. {tool_description}"

            class ApiCallerToolArgsSchema(BaseModel):
                input_text: str = Field(..., description="The input text to process with the API tool")

            class ApiCallerTool(BaseTool):
                name: str = tool_name
                description: str = tool_description
                args_schema: type[BaseModel] = ApiCallerToolArgsSchema
                
                def __init__(self, _tool_schema=tool_schema, _tool_headers=tool_headers, 
                             _tool_params=tool_params, _tool_data_connector=tool_data_connector, 
                             _agent_id=agent_id, _executor=self):
                    super().__init__()
                    self._tool_schema = _tool_schema
                    self._tool_headers = _tool_headers
                    self._tool_params = _tool_params
                    self._tool_data_connector = _tool_data_connector
                    self._agent_id = _agent_id
                    self._executor = _executor
                
                async def _make_request(self, method: str, url: str, headers: dict, params: dict = None, json: dict = None) -> dict:
                    async with aiohttp.ClientSession() as session:
                        max_retries = 2
                        retry_count = 0
                        while True:
                            try:
                                if method.lower() == "get":
                                    async with session.get(url, headers=headers, params=params) as response:
                                        if response.status == 200:
                                            try:
                                                return await response.json()
                                            except ValueError:
                                                return {"result": await response.text()}
                                        elif response.status == 400 and retry_count < max_retries:
                                            retry_count += 1
                                            self._executor.logger.debug(f"Retrying request after 400 error (attempt {retry_count}/{max_retries})")
                                            await asyncio.sleep(1)
                                            continue
                                        else:
                                            try:
                                                error_data = await response.json()
                                                return {"error": error_data.get('detail', str(response.status))}
                                            except ValueError:
                                                return {"error": await response.text() or str(response.status)}
                                else:  # POST
                                    async with session.post(url, headers=headers, params=params, json=json) as response:
                                        if response.status == 200:
                                            try:
                                                return await response.json()
                                            except ValueError:
                                                return {"result": await response.text()}
                                        elif response.status == 400 and retry_count < max_retries:
                                            retry_count += 1
                                            self._executor.logger.debug(f"Retrying request after 400 error (attempt {retry_count}/{max_retries})")
                                            await asyncio.sleep(1)
                                            continue
                                        else:
                                            try:
                                                error_data = await response.json()
                                                return {"error": error_data.get('detail', str(response.status))}
                                            except ValueError:
                                                return {"error": await response.text() or str(response.status)}
                            except aiohttp.ClientError as e:
                                return {"error": f"HTTP request failed: {str(e)}"}
                            except Exception as e:
                                return {"error": f"Unexpected error: {str(e)}"}
                            break
                        return {"error": "Max retries exceeded"}

                def _run(self, input_text: str) -> str:
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            result = loop.run_until_complete(self._async_run(input_text))
                        finally:
                            loop.close()
                        return json.dumps(result)
                    except Exception as e:
                        self._executor.logger.error(f"Tool returned error: {self._executor._sanitize_for_logging(e)}")
                        return json.dumps({"error": self._executor._sanitize_for_logging(str(e))})

                async def _async_run(self, input_text: str) -> dict:
                    try:
                        result = await self._executor.generate_payload(input_text, self._tool_schema, self._tool_data_connector)
                        
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

                        if method == "get" and payload:
                            params.update(payload)

                        self._executor.logger.debug(f"Calling {method.upper()} {endpoint_url}")
                        return await self._make_request(
                            method=method,
                            url=endpoint_url,
                            headers=headers,
                            params=params if params else None,
                            json=payload if method == "post" and payload else None
                        )

                    except Exception as e:
                        self._executor.logger.error(f"Tool returned error: {self._executor._sanitize_for_logging(e)}")
                        return {"error": self._executor._sanitize_for_logging(str(e))}

            api_tool = ApiCallerTool()
            tools.append(api_tool)
            self.logger.info(f"Loaded tool {tool_name} for agent {agent_id}")

        return tools

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

    class AgentOutput(BaseModel):
        answer: str
        class Config:
            extra = "forbid"

    async def execute_task(
        self,
        user_input: str,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        max_retries = 3
        retry_delay = 5
        attempt = 1

        while attempt <= max_retries:
            try:
                if len(self.worker_agents) < 2:
                    self.logger.error(f"At least two worker agents are required.")
                    return "Error: At least two worker agents are required."

                self.logger.info(f"Starting task execution with user_input: '{self._sanitize_for_logging(user_input)}' (Attempt: {attempt})")
                if file_path:
                    self.logger.info(f"Processing file: {self._sanitize_for_logging(file_path)} (Attempt: {attempt})")
                if history:
                    self.logger.info(f"Received chat history with {len(history)} entries (Attempt: {attempt})")
                else:
                    self.logger.info(f"No chat history provided (Attempt: {attempt})")

                # Prepare file content
                file_content = ""
                if file_path and file_type:
                    file_content = self.process_file_content(file_path, file_type)
                    self.logger.debug(f"Processed file content: {self._sanitize_for_logging(file_content[:100])}{'...' if len(file_content) > 100 else ''}")

                # Prepare context from history
                if history:
                    context = self._build_context_from_history(history)
                    self.logger.debug(f"Generated context from history: {self._sanitize_for_logging(context)}")
                    history_instruction = (
                        "CHAT HISTORY FOR CONTEXT:\n"
                        f"{context}\n\n"
                        "IMPORTANT: Use the above chat history to understand context and enhance the current user input below. "
                        "Before processing the user's request, analyze if the input contains pronouns or references like "
                        "'this', 'it', 'he', 'she', 'his', 'her', 'that', 'those', 'them', etc. that need clarification. "
                        "If so, modify the user input to be more specific and contextual based on the chat history. "
                        "For example:\n"
                        "- 'What about it?' → 'What about [specific topic from history]?'\n"
                        "- 'How does this work?' → 'How does [specific item from history] work?'\n"
                        "- 'Tell me more about her' → 'Tell me more about [specific person from history]'\n"
                        "After enhancing the input, proceed with coordinating the agents using the enhanced version.\n\n"
                    )
                else:
                    context = ""
                    history_instruction = ""

                # Prepare worker metadata
                worker_metadata = [
                    {
                        "id": config["id"],
                        "name": config["name"],
                        "role": config["role"],
                        "instructions": config["instructions"],
                        "expectedOutput": config["expectedOutput"]
                    }
                    for config in self.worker_agent_configs
                ]
                self.logger.info(f"Worker metadata prepared: {self._sanitize_for_logging(json.dumps([{'id': w['id'], 'name': w['name']} for w in worker_metadata], indent=2))}")

                # Create task to determine agent order
                manager_description = self.multi_agent_config.get("description", "Coordinate the processing of the user request by delegating to worker agents in sequence.")
                agent_order_task_description = (
                    f"Manager Description: {manager_description}\n"
                    f"User Input: '{self._sanitize_for_logging(user_input)}'\n"
                    f"File Content: {file_content if file_content else 'None'}\n"
                    f"Available Agents: {json.dumps([{'id': w['id'], 'name': w['name'], 'role': w['role'], 'instructions': w['instructions']} for w in worker_metadata], indent=2)}\n"
                    f"Task: Based on the manager's description, determine the sequence of agents to process the user request. The sequence should follow the steps outlined in the manager description, ensuring each agent processes the output of the previous agent (except the first, which processes the user input). The same agent may be used multiple times if required by the description.\n"
                    f"Return ONLY a JSON array of agent IDs in the order they should be executed, with no additional text, explanations, thoughts, or formatting. For example: ['agent_1', 'agent_2', 'agent_2']."
                )

                agent_order_task = Task(
                    description=agent_order_task_description,
                    expected_output="A JSON array of agent IDs in execution order",
                    agent=self.manager_agent,
                    output_json=self.AgentOutput
                )

                order_crew = Crew(
                    agents=[self.manager_agent],
                    tasks=[agent_order_task],
                    process=Process.sequential,
                    verbose=True
                )

                self.logger.info(f"Starting crew to determine agent order (Attempt: {attempt})")
                order_result = await order_crew.kickoff_async()
                self.logger.info(f"Agent order crew execution completed (Attempt: {attempt})")

                try:
                    result_str = str(order_result.raw if hasattr(order_result, 'raw') else order_result).strip()
                    json_match = re.search(r'\[\s*"[0-9a-f-]+"(?:\s*,\s*"[0-9a-f-]+")*\s*\]', result_str)
                    if json_match:
                        json_str = json_match.group(0)
                        agent_ids = json.loads(json_str)
                        self.logger.info(f"LLM-determined agent order: {self._sanitize_for_logging(json.dumps(agent_ids))}")
                    else:
                        raise json.JSONDecodeError("No valid JSON array found in the response", result_str, 0)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse agent order from LLM response: {self._sanitize_for_logging(str(e))}. Falling back to multi_agent_config agent_ids.")
                    agent_ids = self.multi_agent_config.get("agent_ids", [])

                available_agent_ids = [w["id"] for w in worker_metadata]
                missing_agents = [aid for aid in agent_ids if aid not in available_agent_ids]
                if missing_agents:
                    self.logger.error(f"Missing required agents with IDs: {missing_agents}")
                    return f"Error: Missing required agents with IDs: {', '.join(missing_agents)}"

                ordered_worker_agents = []
                seen_ids = set()
                for aid in agent_ids:
                    for config in self.worker_agent_configs:
                        if config["id"] == aid and aid not in seen_ids:
                            for agent in self.worker_agents:
                                if agent.role == config["role"]:
                                    ordered_worker_agents.append(agent)
                                    seen_ids.add(aid)
                                    break
                            break

                if len(ordered_worker_agents) < 2:
                    self.logger.error(f"Insufficient valid agents in sequence: {len(ordered_worker_agents)}")
                    return "Error: At least two valid agents are required."

                instructions = []
                
                # Present user input for contextual enhancement
                instructions.append("CURRENT USER INPUT FOR CONTEXTUAL ENHANCEMENT:")
                instructions.append(f"'{self._sanitize_for_logging(user_input)}'")
                instructions.append("")  # Empty line for separation
                
                if file_content:
                    instructions.append(f"File Content: {file_content}")
                    instructions.append("")  # Empty line for separation
                
                # Add history context for enhancement
                if history_instruction:
                    instructions.append(history_instruction)

                instructions.append("\nFollow these steps exactly:")
                for i, agent in enumerate(ordered_worker_agents):
                    agent_config = next((config for config in self.worker_agent_configs if config["role"] == agent.role), None)
                    if agent_config:
                        if i == 0:
                            instructions.append(
                                f"{i+1}. Delegate to '{agent_config['name']}' to process the user input.\n"
                                f"   - Role: {agent_config['role']}\n"
                                f"   - Instructions: {agent_config['instructions']}\n"
                                f"   - Expected output: {agent_config['expectedOutput']}"
                            )
                        else:
                            prev_agent_config = next((config for config in self.worker_agent_configs if config["role"] == ordered_worker_agents[i-1].role), None)
                            instructions.append(
                                f"{i+1}. Pass the output from '{prev_agent_config['name']}' to '{agent_config['name']}'.\n"
                                f"   - Role: {agent_config['role']}\n"
                                f"   - Instructions: {agent_config['instructions']}\n"
                                f"   - Expected output: {agent_config['expectedOutput']}"
                            )

                instructions.append(
                    f"{len(ordered_worker_agents)+1}. Collect the output from each agent and format the final result as follows:\n"
                    f"{'-' * 40}\n" +
                    "\n".join([f"{next(config['name'] for config in self.worker_agent_configs if config['role'] == agent.role)} Output: <output from {next(config['name'] for config in self.worker_agent_configs if config['role'] == agent.role)}>" for agent in ordered_worker_agents]) + "\n" +
                    f"{'-' * 40}\n"
                    "Do not modify or rephrase any of the agent outputs. Display each exactly as received."
                )

                manager_task_description = f"{self.multi_agent_config.get('description', 'Coordinate the processing of the user request.')}\n\n" + "\n".join(instructions)

                manager_task = Task(
                    description=(
                        f"{manager_task_description}\n\n"
                        "The output should not include any thoughts, reasoning, or intermediate steps in the response."
                        "Return only the final output as specified in `expected_output`. "
                        "Do not include any thoughts, reasoning, intermediate steps, or additional explanations. "
                        "If a tool returns no results or an empty response (e.g., empty URLs or no data), generate a professional, concise, and context-appropriate response acknowledging the lack of information and offering assistance with related topics."
                    ),
                    expected_output=self.multi_agent_config.get("expected_output"),
                    agent=self.manager_agent,
                    output_json=self.AgentOutput
                )

                crew = Crew(
                    agents=[self.manager_agent] + ordered_worker_agents,
                    tasks=[manager_task],
                    process=Process.sequential,
                    manager=self.manager_agent,
                    verbose=True
                )

                self.logger.info(f"Starting crew execution (Attempt: {attempt})")
                result = await crew.kickoff_async()
                self.logger.info(f"Crew execution completed (Attempt: {attempt})")

                final_output = ""
                try:
                    if hasattr(result, 'tasks_output') and result.tasks_output:
                        task_output = result.tasks_output[0]
                        if hasattr(task_output, 'pydantic') and task_output.pydantic:
                            final_output = task_output.pydantic.answer
                        elif hasattr(task_output, 'json') and task_output.json:
                            if isinstance(task_output.json, str):
                                json_data = json.loads(task_output.json)
                                final_output = json_data.get('answer', '')
                            elif isinstance(task_output.json, dict):
                                final_output = task_output.json.get('answer', '')
                            else:
                                final_output = str(task_output.json)
                        elif hasattr(task_output, 'output'):
                            final_output = task_output.output
                        else:
                            final_output = str(task_output)
                    else:
                        final_output = str(result)
                except Exception as e:
                    self.logger.error(f"Error extracting structured output: {self._sanitize_for_logging(e)}")
                    final_output = str(result)

                final_output = final_output.strip()
                if not final_output:
                    self.logger.warning(f"Empty result after processing")
                    return "No output generated."

                self.logger.info(f"Final output: '{self._sanitize_for_logging(final_output[:100])}{'...' if len(final_output) > 100 else ''}'")
                return final_output

            except Exception as e:
                self.logger.error(f"Error during execution: {self._sanitize_for_logging(e)} (Attempt: {attempt})", exc_info=True)
                if attempt < max_retries:
                    self.logger.info(f"Retrying task execution (Attempt {attempt + 1}/{max_retries}) after {retry_delay} seconds")
                    await asyncio.sleep(retry_delay)
                    attempt += 1
                    continue
                else:
                    self.logger.error(f"Max retries ({max_retries}) reached for error: {self._sanitize_for_logging(e)}")
                    return f"Error: Failed after {max_retries} attempts: {str(e)}"