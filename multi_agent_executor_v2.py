import os
import requests
import random
import json
import re
import uuid
import logging
import aiohttp
import asyncio
import time
from typing import Optional, Dict, Any, List, Union
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
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict

# Optional psutil import for system metrics (install with: pip install psutil)
try:
    import psutil  # type: ignore  # Optional dependency
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import common functionality from task_executor.py
from task_executor import (
    # Constants
    ALLOWED_FILE_TYPES,
    EXTENSION_TO_MIME,
    
    # Utility functions
    sanitize_for_logging,
    generate_token,
    
    # LLM class
    GeminiFailoverLLM,
    
    # Context management
    get_execution_context,
    set_execution_context,
    
    # Task executor for reference
    TaskExecutor
)

load_dotenv()

# Disable CrewAI telemetry
os.environ["CREWAI_TELEMETRY_ENABLED"] = "false"

# ===== V3 HYBRID APPROACH COMPONENTS =====

class ExecutionStrategy(Enum):
    """Execution strategies for different phases and scenarios"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    DYNAMIC = "dynamic"

@dataclass
class ExecutionPhase:
    """Represents a phase in the execution plan"""
    name: str
    strategy: ExecutionStrategy
    agents: List[str]
    conditions: Optional[Dict[str, Any]] = None
    dependencies: Optional[List[str]] = None
    domain: Optional[str] = None
    parallel_groups: Optional[List[List[str]]] = None  # For parallel execution grouping

@dataclass
class ExecutionContext:
    """Maintains execution state and history"""
    execution_id: str
    current_phase: Optional[str] = None
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    skipped_agents: List[Dict[str, str]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    agent_logs: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    execution_metrics: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def log_skipped_agent(self, agent_id: str, reason: str):
        """Log when an agent is skipped"""
        skip_info = {
            "agent_id": agent_id, 
            "reason": reason, 
            "phase": self.current_phase,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.skipped_agents.append(skip_info)
    
    def log_error(self, agent_id: str, error: str, phase: str):
        """Log execution errors"""
        error_info = {
            "agent_id": agent_id, 
            "error": error, 
            "phase": phase,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        self.errors.append(error_info)
    
    def log_agent_execution(self, agent_id: str, agent_name: str, action: str, details: Dict[str, Any] = None):
        """Log individual agent actions"""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "agent_name": agent_name,
            "phase": self.current_phase,
            "details": details or {}
        }
        self.agent_logs[agent_id].append(log_entry)
    
    def start_execution(self):
        """Mark the start of execution"""
        self.start_time = datetime.now(timezone.utc)
        self.execution_metrics["start_time"] = self.start_time.isoformat()
        self.execution_metrics["process_id"] = os.getpid()
        
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                self.execution_metrics["initial_memory_mb"] = process.memory_info().rss / 1024 / 1024
                self.execution_metrics["cpu_count"] = psutil.cpu_count()
            except Exception:
                # Silently handle psutil errors
                pass
    
    def end_execution(self):
        """Mark the end of execution and calculate metrics"""
        self.end_time = datetime.now(timezone.utc)
        self.execution_metrics["end_time"] = self.end_time.isoformat()
        
        if self.start_time:
            duration = (self.end_time - self.start_time).total_seconds()
            self.execution_metrics["total_duration_seconds"] = duration
            self.execution_metrics["total_duration_readable"] = f"{duration:.2f}s"
        
        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process()
                self.execution_metrics["final_memory_mb"] = process.memory_info().rss / 1024 / 1024
                self.execution_metrics["peak_memory_mb"] = process.memory_info().peak_wss / 1024 / 1024 if hasattr(process.memory_info(), 'peak_wss') else "N/A"
            except Exception:
                # Silently handle psutil errors
                pass
        
        # Calculate summary statistics
        self.execution_metrics["total_agents_executed"] = len([logs for logs in self.agent_logs.values() if logs])
        self.execution_metrics["total_errors"] = len(self.errors)
        self.execution_metrics["total_skipped"] = len(self.skipped_agents)

class BaseDomainManager:
    """Base class for domain-specific managers"""
    
    def __init__(self, domain_name: str, logger: logging.Logger):
        self.domain_name = domain_name
        self.logger = logger
    
    async def should_execute_phase(self, phase: ExecutionPhase, context: ExecutionContext) -> bool:
        """Decide if a phase should be executed"""
        return True  # Default: execute all phases
    
    async def should_execute_agent(self, agent_id: str, input_data: Any, context: ExecutionContext) -> bool:
        """Decide if a specific agent should be executed"""
        return True  # Default: execute all agents
    
    async def choose_next_agent(self, available_agents: List[str], current_input: Any, 
                              context: ExecutionContext, execution_history: List[Dict]) -> Optional[str]:
        """Choose the next agent for dynamic execution"""
        return available_agents[0] if available_agents else None
    
    async def evaluate_conditions(self, conditions: Dict[str, Any], context: ExecutionContext) -> Dict[str, bool]:
        """Evaluate conditions for conditional execution"""
        # Default implementation - always return True for all conditions
        return {key: True for key in conditions.keys()} if conditions else {}
    
    async def prepare_agent_input(self, agent_id: str, phase_input: Any, context: ExecutionContext) -> Any:
        """Prepare input for a specific agent"""
        return phase_input  # Default: pass through

class DataDomainManager(BaseDomainManager):
    """Manages data-related agents and decisions"""
    
    def __init__(self, logger: logging.Logger):
        super().__init__("data", logger)
    
    async def should_execute_agent(self, agent_id: str, input_data: Any, context: ExecutionContext) -> bool:
        """Data-specific logic for agent execution"""
        # Example: Skip data cleaning if data is already clean
        if agent_id.endswith("_cleaner") and self._is_data_clean(input_data):
            self.logger.info(f"Skipping {agent_id} - data appears to be clean")
            return False
        return True
    
    def _is_data_clean(self, data: Any) -> bool:
        """Check if data appears to be clean (simple heuristic)"""
        # This would contain domain-specific logic
        return False  # Default: assume data needs cleaning

class AnalysisDomainManager(BaseDomainManager):
    """Manages analysis-related agents and decisions"""
    
    def __init__(self, logger: logging.Logger):
        super().__init__("analysis", logger)
    
    async def choose_next_agent(self, available_agents: List[str], current_input: Any, 
                              context: ExecutionContext, execution_history: List[Dict]) -> Optional[str]:
        """Analysis-specific dynamic agent selection"""
        # Example: Choose different analyzers based on data type or size
        if self._is_large_dataset(current_input):
            preferred_agents = [a for a in available_agents if "heavy" in a or "parallel" in a]
            if preferred_agents:
                return preferred_agents[0]
        
        return available_agents[0] if available_agents else None
    
    def _is_large_dataset(self, data: Any) -> bool:
        """Check if dataset is large"""
        # This would contain domain-specific logic
        return False

class SynthesisDomainManager(BaseDomainManager):
    """Manages synthesis and integration agents"""
    
    def __init__(self, logger: logging.Logger):
        super().__init__("synthesis", logger)

class OutputDomainManager(BaseDomainManager):
    """Manages output and reporting agents"""
    
    def __init__(self, logger: logging.Logger):
        super().__init__("output", logger)

# ===== END V3 COMPONENTS =====

class MultiAgentExecutorV2:
    """
    Enhanced MultiAgentExecutor V3 - Hybrid Hierarchical + Dynamic Approach
    
    Features:
    - Leverages proven functionality from task_executor.py
    - Hierarchical domain-based management for scalability
    - Dynamic decision-making for flexibility
    - Multiple execution strategies (sequential, parallel, conditional, dynamic)
    - Backward compatible with V2 interface
    - Advanced error handling and context management
    
    Execution Strategies:
    - SEQUENTIAL: Agents execute one after another (V2 behavior)
    - PARALLEL: Agents execute simultaneously within phases
    - CONDITIONAL: Agents execute based on runtime conditions
    - DYNAMIC: Real-time agent selection based on intermediate results
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
        self.log_file = log_file  # Store log file for TaskExecutor instances
        self.logger = logging.getLogger(f"multi_agent_executor_v2_{self.execution_id}")
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

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

        self.multi_agent_config = multi_agent_config
        self.worker_agent_configs = worker_agent_configs
        self.log_url = log_url
        self.user_email = user_email
        self._validate_configs()

        # Generate JWT token if user_email is provided (using function from task_executor.py)
        if user_email:
            self.token = generate_token(user_email)
            self.logger.debug(f"Generated token for user: {user_email}")
        else:
            self.token = None
            self.logger.debug("No user email provided, token not generated")

        # Initialize LLM clients using the same approach as task_executor.py
        api_keys_string = os.getenv("RANDOM_GEMINI_API_KEYS", "")
        gemini_keys = [key.strip() for key in api_keys_string.split(",") if key.strip()]
        self.llm_client = GeminiFailoverLLM(
            model="gemini/gemini-2.5-flash",
            api_keys=gemini_keys,
            temperature=0.9,
            executor_logger=self.logger
        )
        # Note: internal_llm_client removed - was only used by schema/payload agents which are handled by TaskExecutor instances
        # Note: Schema and Payload agents removed - functionality handled by individual TaskExecutor instances

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

        # Initialize Worker Agents using TaskExecutor approach
        self.worker_agents = []
        self.worker_map = {}
        self.task_executors = {}  # Store TaskExecutor instances for each worker
        self.agent_performance_metrics = {}  # Track individual agent performance
        
        for config in self.worker_agent_configs:
            # Create a TaskExecutor instance for each worker agent
            # This allows us to leverage all the proven functionality from task_executor.py
            agent_start_time = time.time()
            try:
                self.logger.debug(f"🔧 Initializing Worker Agent: {config.get('name', 'Unknown')} (ID: {config['id']})")
                
                task_executor = TaskExecutor(
                    agent_config=config,
                    tools_config=config.get("tools", []),
                    user_email=user_email,
                    execution_id=f"{self.execution_id}_{config['id']}",
                    log_file=self.log_file,  # Use same log file as multi-agent executor
                    verbose=True,
                    raw_output=False
                )
                
                self.task_executors[config["id"]] = task_executor
                self.worker_agents.append(task_executor.agent)
                self.worker_map[config["id"]] = {
                    "agent": task_executor.agent,
                    "config": config,
                    "executor": task_executor
                }
                
                # Initialize performance tracking for this agent
                self.agent_performance_metrics[config["id"]] = {
                    "initialization_time": time.time() - agent_start_time,
                    "executions": 0,
                    "total_execution_time": 0.0,
                    "success_count": 0,
                    "error_count": 0,
                    "last_execution": None
                }
                
                tools_count = len(config.get("tools", []))
                
                # Configure TaskExecutor for detailed logging
                self._configure_task_executor_logging(task_executor, config["id"])
                
                self.logger.info(f"✅ Initialized Worker Agent: {task_executor.agent.role}")
                self.logger.info(f"   📋 ID: {config['id']}, Name: {config['name']}")
                self.logger.info(f"   🛠️  Tools: {tools_count}, Init Time: {self.agent_performance_metrics[config['id']]['initialization_time']:.3f}s")
                
                if tools_count > 0:
                    tool_names = [tool.get('name', 'Unknown') for tool in config.get('tools', [])][:3]
                    tools_preview = ', '.join(tool_names) + ('...' if tools_count > 3 else '')
                    self.logger.debug(f"   🔧 Available Tools: {tools_preview}")
                    
            except Exception as e:
                self.logger.error(f"❌ Failed to initialize worker agent {config['id']}: {sanitize_for_logging(e)}")
                self.logger.debug(f"   Agent Config: {sanitize_for_logging(config)}")
                continue

        if not self.worker_agents:
            self.logger.warning("⚠️  No worker agents initialized.")
        else:
            total_init_time = sum(metrics['initialization_time'] for metrics in self.agent_performance_metrics.values())
            self.logger.info(f"🎯 Successfully initialized {len(self.worker_agents)} worker agents (Total init time: {total_init_time:.3f}s)")

        # ===== V3 HYBRID APPROACH INITIALIZATION =====
        
        # Initialize domain managers for hierarchical organization
        self.domain_managers = {
            "data": DataDomainManager(self.logger),
            "analysis": AnalysisDomainManager(self.logger),
            "synthesis": SynthesisDomainManager(self.logger),
            "output": OutputDomainManager(self.logger)
        }
        self.logger.info(f"🏗️  Initialized {len(self.domain_managers)} domain managers: {list(self.domain_managers.keys())}")
        
        # Initialize V3 execution context
        self.v3_execution_context = ExecutionContext(execution_id=self.execution_id)
        
        # Agent domain mapping (can be configured based on agent roles)
        self.agent_domain_mapping = self._create_agent_domain_mapping()
        
        # Log domain distribution of agents
        domain_distribution = defaultdict(list)
        for agent_id, domain in self.agent_domain_mapping.items():
            agent_name = self.worker_map.get(agent_id, {}).get('config', {}).get('name', agent_id)
            domain_distribution[domain].append(agent_name)
        
        for domain, agents in domain_distribution.items():
            self.logger.debug(f"   🎭 {domain.capitalize()} Domain: {', '.join(agents)}")
        
        # Enable V3 features flag (can be toggled for backward compatibility)
        self.enable_v3_features = True
        
        # Log initialization summary
        self.logger.info("🚀 V3 Hybrid features initialized successfully")
        self.logger.info(f"📊 Execution Summary:")
        self.logger.info(f"   🆔 Execution ID: {self.execution_id}")
        self.logger.info(f"   👑 Manager: {self.manager_agent.role}")
        self.logger.info(f"   👥 Worker Agents: {len(self.worker_agents)}")
        self.logger.info(f"   🏗️  Domain Managers: {len(self.domain_managers)}")
        self.logger.info(f"   ⚡ V3 Features: {'Enabled' if self.enable_v3_features else 'Disabled'}")
        
        if user_email:
            self.logger.info(f"   👤 User: {user_email}")
        if log_file:
            self.logger.info(f"   📝 Log File: {log_file}")

    def _validate_configs(self):
        """Validates configurations (reusing approach from original)."""
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

    def _create_agent_domain_mapping(self) -> Dict[str, str]:
        """Create mapping between agent IDs and their domains based on roles/names"""
        mapping = {}
        
        for config in self.worker_agent_configs:
            agent_id = config["id"]
            role = config.get("role", "").lower()
            name = config.get("name", "").lower()
            
            # Simple heuristic-based domain mapping
            # Users can override this by providing domain info in config
            if config.get("domain"):
                mapping[agent_id] = config["domain"]
            elif any(keyword in role or keyword in name for keyword in ["data", "collect", "clean", "process", "extract"]):
                mapping[agent_id] = "data"
            elif any(keyword in role or keyword in name for keyword in ["analyz", "calculate", "compute", "model", "predict"]):
                mapping[agent_id] = "analysis"
            elif any(keyword in role or keyword in name for keyword in ["synth", "combin", "merge", "integrat", "consolidat"]):
                mapping[agent_id] = "synthesis"
            elif any(keyword in role or keyword in name for keyword in ["report", "format", "present", "output", "visual"]):
                mapping[agent_id] = "output"
            else:
                # Default domain for unclear roles
                mapping[agent_id] = "analysis"
        
        self.logger.debug(f"🎭 Agent domain mapping created:")
        for agent_id, domain in mapping.items():
            agent_name = next((c['name'] for c in self.worker_agent_configs if c['id'] == agent_id), agent_id)
            self.logger.debug(f"   {agent_name} ({agent_id}) → {domain} domain")
        return mapping

    def encode_image_to_base64(self, image_path: str) -> str:
        """Use TaskExecutor's image encoding function"""
        # We can use any TaskExecutor instance since this is a utility function
        if self.task_executors:
            executor = next(iter(self.task_executors.values()))
            return executor.encode_image_to_base64(image_path)
        else:
            # Fallback implementation
            self.logger.debug(f"Encoding image: {sanitize_for_logging(image_path)}")
            try:
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode("utf-8")
            except Exception as e:
                self.logger.error(f"Error encoding image: {sanitize_for_logging(e)}")
                return f"Error encoding image: {sanitize_for_logging(e)}"

    def read_csv_as_text(self, csv_path: str) -> str:
        """Use TaskExecutor's CSV reading function"""
        if self.task_executors:
            executor = next(iter(self.task_executors.values()))
            return executor.read_csv_as_text(csv_path)
        else:
            # Fallback implementation
            self.logger.debug(f"Reading CSV: {sanitize_for_logging(csv_path)}")
            try:
                with open(csv_path, "r", encoding="utf-8", errors="replace") as csv_file:
                    content = csv_file.read()
                    return content
            except Exception as e:
                self.logger.error(f"Error reading CSV: {sanitize_for_logging(e)}")
                return f"Error reading CSV: {sanitize_for_logging(e)}"

    def read_json_as_text(self, json_path: str) -> str:
        """Use TaskExecutor's JSON reading function"""
        if self.task_executors:
            executor = next(iter(self.task_executors.values()))
            return executor.read_json_as_text(json_path)
        else:
            # Fallback implementation
            self.logger.debug(f"Reading JSON: {sanitize_for_logging(json_path)}")
            try:
                with open(json_path, "r", encoding="utf-8", errors="replace") as json_file:
                    data = json.load(json_file)
                    return json.dumps(data, indent=2, ensure_ascii=False)
            except Exception as e:
                self.logger.error(f"Error reading JSON: {sanitize_for_logging(e)}")
                return f"Error reading JSON: {sanitize_for_logging(e)}"

    def read_txt_as_text(self, txt_path: str) -> str:
        """Use TaskExecutor's text reading function"""
        if self.task_executors:
            executor = next(iter(self.task_executors.values()))
            return executor.read_txt_as_text(txt_path)
        else:
            # Fallback implementation
            self.logger.debug(f"Reading TXT: {sanitize_for_logging(txt_path)}")
            try:
                with open(txt_path, "r", encoding="utf-8", errors="replace") as txt_file:
                    return txt_file.read()
            except Exception as e:
                self.logger.error(f"Error reading TXT: {sanitize_for_logging(e)}")
                return f"Error reading TXT: {sanitize_for_logging(e)}"

    def read_pdf_as_text(self, pdf_path: str) -> str:
        """Use TaskExecutor's PDF reading function"""
        if self.task_executors:
            executor = next(iter(self.task_executors.values()))
            return executor.read_pdf_as_text(pdf_path)
        else:
            # Fallback implementation
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
        """Use TaskExecutor's file processing function"""
        if self.task_executors:
            executor = next(iter(self.task_executors.values()))
            return executor.process_file_content(file_path, file_type)
        else:
            # Fallback implementation
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

    # Note: analyze_schema() removed - functionality handled by individual TaskExecutor instances

    # Note: generate_payload() removed - functionality handled by individual TaskExecutor instances

    def _build_context_from_history(self, history: List[Dict[str, str]]) -> str:
        """Use TaskExecutor's context building function"""
        if self.task_executors:
            executor = next(iter(self.task_executors.values()))
            return executor._build_context_from_history(history)
        else:
            # Fallback implementation
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
        """Use TaskExecutor's sentence extraction function"""
        if self.task_executors:
            executor = next(iter(self.task_executors.values()))
            return executor._extract_first_sentences(text, max_sentences)
        else:
            # Fallback implementation
            if not text:
                return ""
            
            text = text.strip()
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            if len(sentences) <= max_sentences:
                return text
            
            result = ' '.join(sentences[:max_sentences])
            
            if not result.endswith(('.', '!', '?')):
                result += '...'
            
            return result

    class AgentOutput(BaseModel):
        answer: str
        class Config:
            extra = "forbid"

    # ===== AGENT TRACKING AND LOGGING METHODS =====
    
    def _log_agent_start(self, agent_id: str, agent_name: str, phase: str, input_preview: str = "") -> float:
        """Log agent execution start and return start time"""
        start_time = time.time()
        
        # Update performance metrics
        if agent_id in self.agent_performance_metrics:
            self.agent_performance_metrics[agent_id]["executions"] += 1
            self.agent_performance_metrics[agent_id]["last_execution"] = datetime.now(timezone.utc).isoformat()
        
        # Log to execution context
        self.v3_execution_context.log_agent_execution(
            agent_id, agent_name, "execution_start", {
                "phase": phase,
                "input_preview": input_preview[:100] if input_preview else "No input",
                "start_time": start_time
            }
        )
        
        self.logger.info(f"🚀 Starting Agent: {agent_name} (ID: {agent_id}) in {phase} phase")
        if input_preview:
            self.logger.debug(f"   📝 Input Preview: {input_preview[:150]}{'...' if len(input_preview) > 150 else ''}")
        
        return start_time
    
    def _log_agent_success(self, agent_id: str, agent_name: str, start_time: float, output_preview: str = ""):
        """Log successful agent execution"""
        execution_time = time.time() - start_time
        
        # Update performance metrics
        if agent_id in self.agent_performance_metrics:
            self.agent_performance_metrics[agent_id]["success_count"] += 1
            self.agent_performance_metrics[agent_id]["total_execution_time"] += execution_time
        
        # Log to execution context
        self.v3_execution_context.log_agent_execution(
            agent_id, agent_name, "execution_success", {
                "execution_time": execution_time,
                "output_preview": output_preview[:100] if output_preview else "No output"
            }
        )
        
        self.logger.info(f"✅ Agent Completed: {agent_name} (Duration: {execution_time:.3f}s)")
        if output_preview:
            self.logger.debug(f"   📝 Output Preview: {output_preview[:150]}{'...' if len(output_preview) > 150 else ''}")
    
    def _log_agent_error(self, agent_id: str, agent_name: str, start_time: float, error: str):
        """Log agent execution error"""
        execution_time = time.time() - start_time
        
        # Update performance metrics
        if agent_id in self.agent_performance_metrics:
            self.agent_performance_metrics[agent_id]["error_count"] += 1
            self.agent_performance_metrics[agent_id]["total_execution_time"] += execution_time
        
        # Log to execution context
        self.v3_execution_context.log_agent_execution(
            agent_id, agent_name, "execution_error", {
                "execution_time": execution_time,
                "error": error
            }
        )
        
        self.logger.error(f"❌ Agent Failed: {agent_name} (Duration: {execution_time:.3f}s)")
        self.logger.error(f"   ⚠️  Error: {sanitize_for_logging(error)}")
    
    def _log_agent_skipped(self, agent_id: str, agent_name: str, reason: str):
        """Log agent skip"""
        self.logger.warning(f"⏭️  Skipping Agent: {agent_name} - {reason}")
    
    def _configure_task_executor_logging(self, task_executor, agent_id: str):
        """Configure TaskExecutor for detailed logging to capture tool execution details"""
        try:
            if hasattr(task_executor, 'logger'):
                # Set TaskExecutor logger to DEBUG level for detailed tool logging
                task_executor.logger.setLevel(logging.DEBUG)
                
                # Ensure the TaskExecutor logger is properly configured
                agent_name = self.worker_map.get(agent_id, {}).get('config', {}).get('name', agent_id)
                self.logger.debug(f"   🔧 Configured {agent_name} TaskExecutor for detailed logging (DEBUG level)")
                
                # If the TaskExecutor has tools, log that tool execution details will be captured
                if hasattr(task_executor, 'tools') and task_executor.tools:
                    self.logger.debug(f"   🛠️  {agent_name} will capture detailed tool execution logs for {len(task_executor.tools)} tools")
                    
                # Store reference for logging aggregation
                if not hasattr(self, 'configured_executors'):
                    self.configured_executors = {}
                self.configured_executors[agent_id] = task_executor
                
        except Exception as e:
            self.logger.warning(f"⚠️  Could not configure detailed logging for agent {agent_id}: {e}")
    
    async def _parse_agent_order_response(self, order_result, worker_metadata: List[Dict]) -> List[str]:
        """Enhanced agent order parsing with multiple strategies and detailed logging"""
        available_agent_ids = [w["id"] for w in worker_metadata]
        agent_names_to_ids = {w["name"].lower(): w["id"] for w in worker_metadata}
        
        try:
            # Extract the raw response
            if hasattr(order_result, 'tasks_output') and order_result.tasks_output:
                task_output = order_result.tasks_output[0]
                if hasattr(task_output, 'pydantic') and task_output.pydantic:
                    result_str = task_output.pydantic.answer
                elif hasattr(task_output, 'json') and task_output.json:
                    if isinstance(task_output.json, str):
                        json_data = json.loads(task_output.json)
                        result_str = json_data.get('answer', task_output.json)
                    elif isinstance(task_output.json, dict):
                        result_str = task_output.json.get('answer', str(task_output.json))
                    else:
                        result_str = str(task_output.json)
                elif hasattr(task_output, 'raw'):
                    result_str = str(task_output.raw)
                else:
                    result_str = str(task_output)
            else:
                result_str = str(order_result.raw if hasattr(order_result, 'raw') else order_result)
            
            result_str = result_str.strip()
            
            # TEMPORARY: Enhanced debug logging to diagnose the issue
            self.logger.info(f"🔍 DEBUGGING: Raw LLM response for agent order:")
            self.logger.info(f"   Length: {len(result_str)} characters")
            self.logger.info(f"   Type: {type(result_str)}")
            self.logger.info(f"   Contains '[': {('[' in result_str)}")
            self.logger.info(f"   Contains ']': {(']' in result_str)}")
            self.logger.info(f"   First 300 chars: {repr(result_str[:300])}")
            if len(result_str) > 300:
                self.logger.info(f"   Last 100 chars: {repr(result_str[-100:])}")
            
            if not result_str:
                self.logger.warning("⚠️  Empty response from LLM for agent order")
                return []
            
            # Strategy 1: Look for JSON array patterns (enhanced flexible regex)
            json_patterns = [
                # Most flexible - any JSON array with quoted strings
                r'\[\s*["\'][^"\']+["\'](?:\s*,\s*["\'][^"\']+["\'])*\s*\]',
                # UUID patterns - more flexible
                r'\[\s*["\'][0-9a-f-]{20,}["\'](?:\s*,\s*["\'][0-9a-f-]{20,}["\'])*\s*\]',
                # Any alphanumeric IDs
                r'\[\s*["\'][a-zA-Z0-9_-]{5,}["\'](?:\s*,\s*["\'][a-zA-Z0-9_-]{5,}["\'])*\s*\]',
                # Even more flexible - any array-like structure
                r'\[.*?\]',
                # Handle code blocks
                r'```(?:json)?\s*(\[.*?\])\s*```',
            ]
            
            for i, pattern in enumerate(json_patterns):
                json_match = re.search(pattern, result_str, re.DOTALL)
                if json_match:
                    try:
                        # Handle code block pattern with capture group
                        if json_match.groups():
                            json_str = json_match.group(1)
                        else:
                            json_str = json_match.group(0)
                        
                        self.logger.debug(f"🔍 Strategy 1.{i+1} found potential JSON: {json_str[:100]}{'...' if len(json_str) > 100 else ''}")
                        
                        parsed_ids = json.loads(json_str)
                        if isinstance(parsed_ids, list):
                            self.logger.debug(f"✅ Strategy 1.{i+1} successful: Found JSON array with {len(parsed_ids)} agents")
                            
                            # Clean up the IDs (remove quotes, etc.)
                            cleaned_ids = []
                            for item in parsed_ids:
                                item_str = str(item).strip('\"').strip("'")
                                cleaned_ids.append(item_str)
                            
                            # Validate that all IDs exist
                            valid_ids = [aid for aid in cleaned_ids if aid in available_agent_ids]
                            if len(valid_ids) == len(cleaned_ids):
                                self.logger.info(f"🎯 LLM-determined agent order: {sanitize_for_logging(json.dumps(valid_ids))}")
                                return valid_ids
                            elif len(valid_ids) > 0:
                                invalid_ids = [aid for aid in cleaned_ids if aid not in available_agent_ids]
                                self.logger.warning(f"⚠️  Some agent IDs not found: {invalid_ids}. Using valid ones: {len(valid_ids)}/{len(cleaned_ids)}")
                                return valid_ids
                    except json.JSONDecodeError as e:
                        self.logger.debug(f"❌ Strategy 1.{i+1} JSON decode failed: {e}")
                        continue
            
            # Strategy 2: Look for agent names and map to IDs
            self.logger.debug("🔍 Strategy 2: Searching for agent names in response")
            found_names = []
            result_lower = result_str.lower()
            for name, agent_id in agent_names_to_ids.items():
                if name in result_lower:
                    found_names.append((name, agent_id))
            
            if found_names:
                # Sort by order of appearance in text
                name_positions = [(name, agent_id, result_lower.find(name)) for name, agent_id in found_names]
                name_positions.sort(key=lambda x: x[2])  # Sort by position
                mapped_ids = [agent_id for name, agent_id, pos in name_positions]
                self.logger.info(f"🎯 Strategy 2 successful: Mapped agent names to IDs: {len(mapped_ids)} agents")
                self.logger.debug(f"   Name mapping: {[(name, agent_id) for name, agent_id, pos in name_positions]}")
                return mapped_ids
            
            # Strategy 3: Look for any valid agent IDs anywhere in the text
            self.logger.debug("🔍 Strategy 3: Searching for any valid agent IDs in text")
            found_agent_ids = []
            for agent_id in available_agent_ids:
                if agent_id in result_str:
                    found_agent_ids.append(agent_id)
                    self.logger.debug(f"   Found agent ID '{agent_id}' in response")
            
            if found_agent_ids:
                self.logger.info(f"🎯 Strategy 3 successful: Found {len(found_agent_ids)} valid agent IDs in text")
                return found_agent_ids
            
            # Strategy 4: Try to parse any JSON structure
            self.logger.debug("🔍 Strategy 4: Attempting to parse any JSON structure")
            try:
                # Look for any JSON structure
                json_start = result_str.find('{')
                json_array_start = result_str.find('[')
                
                if json_array_start != -1 and (json_start == -1 or json_array_start < json_start):
                    # Array comes first, try to parse it
                    potential_json = result_str[json_array_start:]
                    # Find the closing bracket
                    bracket_count = 0
                    end_pos = -1
                    for i, char in enumerate(potential_json):
                        if char == '[':
                            bracket_count += 1
                        elif char == ']':
                            bracket_count -= 1
                            if bracket_count == 0:
                                end_pos = i + 1
                                break
                    
                    if end_pos != -1:
                        json_candidate = potential_json[:end_pos]
                        parsed_data = json.loads(json_candidate)
                        if isinstance(parsed_data, list):
                            # Check if items are valid agent IDs or names
                            mapped_ids = []
                            for item in parsed_data:
                                item_str = str(item).strip('\"').strip("'")
                                if item_str in available_agent_ids:
                                    mapped_ids.append(item_str)
                                elif item_str.lower() in agent_names_to_ids:
                                    mapped_ids.append(agent_names_to_ids[item_str.lower()])
                            
                            if mapped_ids:
                                self.logger.info(f"🎯 Strategy 4 successful: Parsed JSON structure with {len(mapped_ids)} valid agents")
                                return mapped_ids
            
            except (json.JSONDecodeError, IndexError) as e:
                self.logger.debug(f"❌ Strategy 4 failed: {e}")
            
            # Log failure details
            self.logger.warning(f"❌ All parsing strategies failed")
            self.logger.debug(f"   Available agent IDs: {available_agent_ids}")
            self.logger.debug(f"   Available agent names: {list(agent_names_to_ids.keys())}")
            self.logger.debug(f"   Response length: {len(result_str)} characters")
            
            return []
            
        except Exception as e:
            self.logger.error(f"❌ Unexpected error in agent order parsing: {sanitize_for_logging(e)}")
            return []
    
    def _log_detailed_tool_execution(self, agent_id: str, agent_name: str, tool_name: str, input_data: str, output_data: str, execution_time: float):
        """Log detailed tool execution information"""
        self.logger.info(f"🔧 Tool Execution: {agent_name} -> {tool_name}")
        self.logger.info(f"   ⏱️  Duration: {execution_time:.3f}s")
        self.logger.debug(f"   📤 Input Data: {input_data[:200]}{'...' if len(input_data) > 200 else ''}")
        self.logger.debug(f"   📥 Output Data: {output_data[:200]}{'...' if len(output_data) > 200 else ''}")
        
        # Log to execution context for tracking
        self.v3_execution_context.log_agent_execution(
            agent_id, agent_name, "tool_execution", {
                "tool_name": tool_name,
                "execution_time": execution_time,
                "input_preview": input_data[:100],
                "output_preview": output_data[:100]
            }
        )
    
    def _aggregate_individual_agent_logs(self):
        """Aggregate logs from individual TaskExecutor instances"""
        self.logger.info("📋 Aggregating individual agent logs from TaskExecutor instances...")
        
        for agent_id, executor in self.task_executors.items():
            agent_name = self.worker_map.get(agent_id, {}).get('config', {}).get('name', agent_id)
            
            # Get the agent's individual logger info if available
            agent_logger = executor.logger if hasattr(executor, 'logger') else None
            if agent_logger:
                # Log agent-specific information
                self.logger.debug(f"   👤 {agent_name} Logger: {len(agent_logger.handlers)} handlers configured")
                self.logger.debug(f"   📝 {agent_name} Logger Level: {agent_logger.level}")
                
                # Log if the agent logger is writing to the same file as multi-agent logger
                if hasattr(executor, 'log_file'):
                    self.logger.debug(f"   📁 {agent_name} Log File: {executor.log_file}")
                
                # Try to get agent execution stats from the executor if available
                if hasattr(executor, 'execution_stats'):
                    stats = executor.execution_stats
                    self.logger.info(f"   📊 {agent_name} Stats: {stats}")
            
            # Log agent tools information with more detail
            if hasattr(executor, 'tools') and executor.tools:
                tools_info = [getattr(tool, 'name', 'Unknown') for tool in executor.tools]
                self.logger.info(f"   🛠️  {agent_name} Available Tools ({len(executor.tools)}):")
                for i, tool in enumerate(executor.tools[:5]):  # Show first 5 tools
                    tool_name = getattr(tool, 'name', f'Tool_{i}')
                    tool_desc = getattr(tool, 'description', 'No description')
                    self.logger.debug(f"      {i+1}. {tool_name}: {tool_desc[:100]}{'...' if len(tool_desc) > 100 else ''}")
                if len(executor.tools) > 5:
                    self.logger.debug(f"      ... and {len(executor.tools) - 5} more tools")
            else:
                self.logger.debug(f"   🛠️  {agent_name}: No tools configured")
    
    def _log_execution_summary(self):
        """Log comprehensive execution summary"""
        self.logger.info("📊 " + "="*60)
        self.logger.info("📊 MULTI-AGENT EXECUTION SUMMARY")
        self.logger.info("📊 " + "="*60)
        
        # Overall execution metrics
        if hasattr(self.v3_execution_context, 'execution_metrics'):
            metrics = self.v3_execution_context.execution_metrics
            if 'total_duration_seconds' in metrics:
                self.logger.info(f"⏱️  Total Execution Time: {metrics['total_duration_seconds']:.3f}s")
            if 'total_agents_executed' in metrics:
                self.logger.info(f"👥 Agents Executed: {metrics['total_agents_executed']}")
            if 'total_errors' in metrics:
                self.logger.info(f"❌ Total Errors: {metrics['total_errors']}")
            if 'total_skipped' in metrics:
                self.logger.info(f"⏭️  Total Skipped: {metrics['total_skipped']}")
        
        # Individual agent performance
        self.logger.info("🛠️  Agent Performance Summary:")
        for agent_id, metrics in self.agent_performance_metrics.items():
            agent_name = self.worker_map.get(agent_id, {}).get('config', {}).get('name', agent_id)
            if metrics['executions'] > 0:
                avg_time = metrics['total_execution_time'] / metrics['executions']
                success_rate = (metrics['success_count'] / metrics['executions']) * 100
                self.logger.info(f"   {agent_name}: {metrics['executions']} runs, avg {avg_time:.3f}s, {success_rate:.1f}% success")
            else:
                self.logger.info(f"   {agent_name}: Not executed")
        
        self.logger.info("📊 " + "="*60)
    
    # ===== V3 HYBRID EXECUTION METHODS =====
    
    async def execute_task_v3(
        self,
        user_input: str,
        execution_strategy: ExecutionStrategy = ExecutionStrategy.DYNAMIC,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        custom_phases: Optional[List[ExecutionPhase]] = None
    ) -> str:
        """
        V3 Hybrid execution with multiple strategies
        
        Args:
            user_input: The user's request
            execution_strategy: How to execute the agents (SEQUENTIAL, PARALLEL, CONDITIONAL, DYNAMIC)
            file_path: Optional file to process
            file_type: Type of the file
            history: Chat history for context
            custom_phases: Custom execution phases (if None, will be generated by manager)
        
        Returns:
            Formatted results from all executed agents
        """
        # Start execution logging and metrics
        execution_start_time = time.time()
        self.logger.info("🚀 " + "="*60)
        self.logger.info(f"🚀 STARTING V3 MULTI-AGENT EXECUTION")
        self.logger.info("🚀 " + "="*60)
        self.logger.info(f"📝 User Input: {user_input[:200]}{'...' if len(user_input) > 200 else ''}")
        self.logger.info(f"🛠️  Strategy: {execution_strategy.value.upper()}")
        self.logger.info(f"📁 File: {'Yes (' + file_path + ')' if file_path else 'No'}")
        self.logger.info(f"📜 History: {'Yes (' + str(len(history)) + ' entries)' if history else 'No'}")
        self.logger.info(f"🏗️  Custom Phases: {'Yes' if custom_phases else 'No'}")
        
        if not self.enable_v3_features:
            self.logger.warning("⚠️  Hybrid V3 features disabled - falling back to the older version of execution")
            return await self.execute_task(user_input, file_path, file_type, history)
        
        try:
            # Reset execution context and start tracking
            self.v3_execution_context = ExecutionContext(execution_id=self.execution_id)
            self.v3_execution_context.start_execution()
            self.v3_execution_context.results["original_input"] = user_input
            
            self.logger.info("🔄 Resetting execution context for new run")
            
            # Phase 1: Create execution plan
            self.logger.info("🧠 Phase 1: Creating intelligent execution plan...")
            if custom_phases:
                execution_phases = custom_phases
                self.logger.info(f"📋 Using {len(custom_phases)} custom execution phases")
                for i, phase in enumerate(custom_phases, 1):
                    self.logger.debug(f"   Phase {i}: {phase.name} ({phase.strategy.value}, {len(phase.agents)} agents)")
            else:
                planning_start = time.time()
                execution_phases = await self._create_execution_plan_v3(
                    user_input, file_path, file_type, history, execution_strategy
                )
                planning_time = time.time() - planning_start
                self.logger.info(f"✅ Execution plan created in {planning_time:.3f}s with {len(execution_phases)} phases")
            
            # Phase 2: Execute plan with hybrid approach
            self.logger.info(f"⚡ Phase 2: Executing {len(execution_phases)} phases with hybrid approach...")
            execution_phase_start = time.time()
            
            results = await self._execute_phases_v3(execution_phases, user_input, file_path, file_type, history)
            
            execution_phase_time = time.time() - execution_phase_start
            self.logger.info(f"✅ Phase execution completed in {execution_phase_time:.3f}s")
            
            # Phase 3: Format final output
            self.logger.info("📋 Phase 3: Formatting final output...")
            formatting_start = time.time()
            
            final_result = await self._format_v3_results(results)
            
            formatting_time = time.time() - formatting_start
            self.logger.info(f"✅ Output formatting completed in {formatting_time:.3f}s")
            
            # End execution tracking and log summary
            self.v3_execution_context.end_execution()
            total_execution_time = time.time() - execution_start_time
            
            self.logger.info("🎆 V3 EXECUTION COMPLETED SUCCESSFULLY")
            self.logger.info(f"⏱️  Total Time: {total_execution_time:.3f}s (Planning: {planning_time:.3f}s, Execution: {execution_phase_time:.3f}s, Formatting: {formatting_time:.3f}s)")
            
            # Aggregate individual agent logs
            self._aggregate_individual_agent_logs()
            
            # Log detailed execution summary
            self._log_execution_summary()
            
            return final_result
            
        except Exception as e:
            # End execution tracking even on failure
            self.v3_execution_context.end_execution()
            execution_time = time.time() - execution_start_time
            
            self.logger.error(f"❌ V3 execution failed after {execution_time:.3f}s: {sanitize_for_logging(e)}")
            self.logger.error(f"   Stack trace available in debug logs")
            self.logger.debug(f"V3 Execution Error Details:", exc_info=True)
            
            # Log what we managed to accomplish before failure
            if hasattr(self, 'v3_execution_context') and self.v3_execution_context.agent_logs:
                self.logger.info(f"📄 Partial execution summary before failure:")
                for agent_id, logs in self.v3_execution_context.agent_logs.items():
                    agent_name = self.worker_map.get(agent_id, {}).get('config', {}).get('name', agent_id)
                    self.logger.info(f"   {agent_name}: {len(logs)} actions logged")
            
            # Graceful fallback to V2
            self.logger.info("🔄 Gracefully falling back to older V2 execution...")
            fallback_start = time.time()
            
            try:
                result = await self.execute_task(user_input, file_path, file_type, history)
                fallback_time = time.time() - fallback_start
                self.logger.info(f"✅ V2 fallback completed successfully in {fallback_time:.3f}s")
                return result
            except Exception as fallback_error:
                fallback_time = time.time() - fallback_start
                self.logger.error(f"❌ V2 fallback also failed after {fallback_time:.3f}s: {sanitize_for_logging(fallback_error)}")
                raise Exception(f"Both V3 and V2 execution failed. V3 error: {str(e)}, V2 error: {str(fallback_error)}")
    
    async def _create_execution_plan_v3(
        self,
        user_input: str,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        strategy: ExecutionStrategy = ExecutionStrategy.DYNAMIC
    ) -> List[ExecutionPhase]:
        """Create an intelligent execution plan based on the request and strategy"""
        
        # Enhanced manager prompt for V3 planning
        manager_description = self.multi_agent_config.get("description", "Coordinate the processing of the user request.")
        
        planning_prompt = f"""
        EXECUTION PLANNING FOR MULTI-AGENT SYSTEM
        
        User Request: "{user_input}"
        Available Agents: {json.dumps([{
            'id': config['id'], 
            'name': config['name'], 
            'role': config['role'],
            'domain': self.agent_domain_mapping.get(config['id'], 'unknown')
        } for config in self.worker_agent_configs], indent=2)}
        
        Manager Description: {manager_description}
        Requested Strategy: {strategy.value}
        File Provided: {"Yes" if file_path else "No"}
        History Provided: {"Yes" if history else "No"}
        
        Create an optimal execution plan with the following phases:
        
        For SEQUENTIAL strategy:
        - Create a linear sequence of agents
        - Each agent processes the output of the previous
        
        For PARALLEL strategy:
        - Group agents that can work simultaneously
        - Identify dependencies between groups
        
        For CONDITIONAL strategy:
        - Define conditions for each agent's execution
        - Specify fallback agents
        
        For DYNAMIC strategy:
        - Start with initial agents
        - Allow for runtime decision-making
        
        Return a JSON array of execution phases:
        [{{
            "name": "phase_name",
            "strategy": "sequential|parallel|conditional|dynamic", 
            "agents": ["agent_id1", "agent_id2"],
            "domain": "data|analysis|synthesis|output",
            "conditions": {{"optional": "conditions"}},
            "dependencies": ["previous_phase_names"]
        }}]
        """
        
        planning_task = Task(
            description=planning_prompt,
            expected_output="JSON array of execution phases with strategy, agents, and dependencies",
            agent=self.manager_agent,
            output_json=self.AgentOutput
        )
        
        planning_crew = Crew(
            agents=[self.manager_agent],
            tasks=[planning_task],
            process=Process.sequential,
            verbose=True
        )
        
        try:
            planning_result = await planning_crew.kickoff_async()
            phases = self._parse_execution_plan(planning_result, strategy)
            self.logger.info(f"Created execution plan with {len(phases)} phases")
            return phases
            
        except Exception as e:
            self.logger.warning(f"Failed to create V3 execution plan: {e}. Using fallback plan.")
            return self._create_fallback_execution_plan(strategy)
    
    def _parse_execution_plan(self, planning_result: Any, default_strategy: ExecutionStrategy) -> List[ExecutionPhase]:
        """Parse the manager's execution plan response into ExecutionPhase objects"""
        try:
            result_str = str(planning_result.raw if hasattr(planning_result, 'raw') else planning_result)
            
            # Extract JSON from the response
            json_match = re.search(r'\[.*\]', result_str, re.DOTALL)
            if json_match:
                phases_data = json.loads(json_match.group(0))
                
                phases = []
                for phase_data in phases_data:
                    try:
                        strategy = ExecutionStrategy(phase_data.get("strategy", default_strategy.value))
                    except ValueError:
                        strategy = default_strategy
                    
                    phase = ExecutionPhase(
                        name=phase_data.get("name", f"phase_{len(phases)}"),
                        strategy=strategy,
                        agents=phase_data.get("agents", []),
                        conditions=phase_data.get("conditions"),
                        dependencies=phase_data.get("dependencies"),
                        domain=phase_data.get("domain")
                    )
                    phases.append(phase)
                
                return phases
            else:
                raise ValueError("No valid JSON found in planning response")
                
        except Exception as e:
            self.logger.error(f"Failed to parse execution plan: {e}")
            return self._create_fallback_execution_plan(default_strategy)
    
    def _create_fallback_execution_plan(self, strategy: ExecutionStrategy) -> List[ExecutionPhase]:
        """Create a simple fallback execution plan when intelligent planning fails"""
        agent_ids = [config["id"] for config in self.worker_agent_configs]
        
        if strategy == ExecutionStrategy.PARALLEL:
            # Group agents by domain for parallel execution
            domain_groups = {}
            for agent_id in agent_ids:
                domain = self.agent_domain_mapping.get(agent_id, "analysis")
                if domain not in domain_groups:
                    domain_groups[domain] = []
                domain_groups[domain].append(agent_id)
            
            phases = []
            for domain, agents in domain_groups.items():
                phases.append(ExecutionPhase(
                    name=f"{domain}_phase",
                    strategy=ExecutionStrategy.PARALLEL,
                    agents=agents,
                    domain=domain
                ))
            return phases
        else:
            # Default sequential plan
            return [ExecutionPhase(
                name="main_phase",
                strategy=ExecutionStrategy.SEQUENTIAL,
                agents=agent_ids
            )]
    
    async def _execute_phases_v3(
        self,
        phases: List[ExecutionPhase],
        user_input: str,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Execute phases using the hybrid approach"""
        
        results = {"original_input": user_input}
        current_input = user_input
        
        # Prepare file content if provided
        file_content = ""
        if file_path and file_type:
            file_content = self.process_file_content(file_path, file_type)
            current_input += f"\n\nFile Content:\n{file_content}"
        
        # Prepare history context
        history_context = ""
        if history:
            history_context = self._build_context_from_history(history)
        
        for phase in phases:
            self.v3_execution_context.current_phase = phase.name
            self.logger.info(f"Executing phase: {phase.name} with strategy: {phase.strategy.value}")
            
            # Check if phase should be executed
            if phase.domain and phase.domain in self.domain_managers:
                domain_manager = self.domain_managers[phase.domain]
                should_execute = await domain_manager.should_execute_phase(phase, self.v3_execution_context)
                if not should_execute:
                    self.logger.info(f"Skipping phase {phase.name} - domain manager decision")
                    continue
            
            # Execute based on strategy
            try:
                if phase.strategy == ExecutionStrategy.SEQUENTIAL:
                    phase_result = await self._execute_sequential_v3(phase, current_input, history_context)
                elif phase.strategy == ExecutionStrategy.PARALLEL:
                    phase_result = await self._execute_parallel_v3(phase, current_input, history_context)
                elif phase.strategy == ExecutionStrategy.CONDITIONAL:
                    phase_result = await self._execute_conditional_v3(phase, current_input, history_context)
                elif phase.strategy == ExecutionStrategy.DYNAMIC:
                    phase_result = await self._execute_dynamic_v3(phase, current_input, history_context)
                else:
                    # Fallback to sequential
                    phase_result = await self._execute_sequential_v3(phase, current_input, history_context)
                
                results[phase.name] = phase_result
                
                # Update input for next phase (use last result as input)
                if isinstance(phase_result, dict) and phase_result:
                    # For parallel results, combine them
                    if phase.strategy == ExecutionStrategy.PARALLEL:
                        combined_output = "\n".join([f"{k}: {v}" for k, v in phase_result.items() if v])
                        current_input = combined_output if combined_output else current_input
                    else:
                        current_input = str(phase_result)
                elif isinstance(phase_result, str):
                    current_input = phase_result
                
                self.logger.info(f"Phase {phase.name} completed successfully")
                
            except Exception as e:
                self.logger.error(f"Phase {phase.name} failed: {sanitize_for_logging(e)}")
                self.v3_execution_context.log_error("phase", str(e), phase.name)
                # Continue with error as input for debugging
                current_input = f"Error in {phase.name}: {str(e)}"
                results[phase.name] = {"error": str(e)}
        
        return results
    
    async def _execute_sequential_v3(self, phase: ExecutionPhase, phase_input: str, history_context: str) -> str:
        """Execute agents sequentially within a phase"""
        current_input = phase_input
        
        self.logger.info(f"🔄 Executing SEQUENTIAL phase with {len(phase.agents)} agents")
        
        for i, agent_id in enumerate(phase.agents, 1):
            if agent_id not in self.task_executors:
                self.logger.warning(f"⚠️  Agent {agent_id} not found in task executors - skipping")
                continue
            
            config = self.worker_map[agent_id]["config"]
            agent_name = config["name"]
            
            self.logger.info(f"🔢 Sequential Step {i}/{len(phase.agents)}: {agent_name}")
            
            # Check if agent should be executed
            domain = self.agent_domain_mapping.get(agent_id)
            if domain and domain in self.domain_managers:
                domain_manager = self.domain_managers[domain]
                should_execute = await domain_manager.should_execute_agent(
                    agent_id, current_input, self.v3_execution_context
                )
                if not should_execute:
                    self._log_agent_skipped(agent_id, agent_name, "Domain manager decision")
                    self.v3_execution_context.log_skipped_agent(agent_id, "Domain manager decision")
                    continue
            
            executor = self.task_executors[agent_id]
            
            # Start agent execution logging
            start_time = self._log_agent_start(agent_id, agent_name, phase.name, current_input)
            
            try:
                # Prepare agent input
                agent_input = current_input
                if history_context and current_input == phase_input:  # First agent gets history
                    agent_input = f"{history_context}\n\nCurrent Request: {current_input}"
                    self.logger.debug(f"   📜 First agent in sequence - adding history context")
                
                # Log detailed agent execution start
                # self.logger.debug(f"   📋 Task Description: {config['instructions'][:150]}{'...' if len(config['instructions']) > 150 else ''}")
                # self.logger.debug(f"   🎯 Expected Output: {config['expectedOutput'][:100]}{'...' if len(config['expectedOutput']) > 100 else ''}")
                
                # Execute the agent task
                result = await executor.execute_task(
                    description=config["instructions"],
                    expected_output=config["expectedOutput"],
                    task_name=config["name"],
                    input=agent_input,
                    user_email=self.user_email
                )
                
                # Log successful execution with detailed result
                self._log_agent_success(agent_id, agent_name, start_time, result)
                
                # Log result details
                self.logger.debug(f"   ✅ Result Length: {len(result)} characters")
                if hasattr(executor, 'tools') and executor.tools:
                    self.logger.debug(f"   🛠️  Agent used {len(executor.tools)} available tools during execution")
                
                current_input = result
                self.logger.debug(f"   🔄 Output becomes input for next agent (length: {len(result)} chars)")
                
            except Exception as e:
                # Log agent error
                self._log_agent_error(agent_id, agent_name, start_time, str(e))
                self.v3_execution_context.log_error(agent_id, str(e), phase.name)
                
                # Continue with error as input for debugging
                error_message = f"Error from {config['name']}: {str(e)}"
                current_input = error_message
                self.logger.debug(f"   🔄 Error becomes input for next agent: {error_message[:100]}...")
        
        return current_input
    
    async def _execute_parallel_v3(self, phase: ExecutionPhase, phase_input: str, history_context: str) -> Dict[str, str]:
        """Execute agents in parallel within a phase"""
        tasks = []
        agent_configs = {}
        
        for agent_id in phase.agents:
            if agent_id not in self.task_executors:
                continue
            
            # Check if agent should be executed
            domain = self.agent_domain_mapping.get(agent_id)
            if domain and domain in self.domain_managers:
                domain_manager = self.domain_managers[domain]
                should_execute = await domain_manager.should_execute_agent(
                    agent_id, phase_input, self.v3_execution_context
                )
                if not should_execute:
                    self.v3_execution_context.log_skipped_agent(agent_id, "Domain manager decision")
                    continue
            
            executor = self.task_executors[agent_id]
            config = self.worker_map[agent_id]["config"]
            agent_configs[agent_id] = config
            
            # Prepare agent input
            agent_input = phase_input
            if history_context:
                agent_input = f"{history_context}\n\nCurrent Request: {phase_input}"
            
            task = executor.execute_task(
                description=config["instructions"],
                expected_output=config["expectedOutput"],
                task_name=config["name"],
                input=agent_input,
                user_email=self.user_email
            )
            tasks.append((agent_id, task))
        
        # Execute all agents in parallel
        results = {}
        if tasks:
            completed_tasks = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            for (agent_id, _), result in zip(tasks, completed_tasks):
                config = agent_configs[agent_id]
                if isinstance(result, Exception):
                    self.logger.error(f"Parallel agent {config['name']} failed: {result}")
                    self.v3_execution_context.log_error(agent_id, str(result), phase.name)
                    results[agent_id] = f"Error: {str(result)}"
                else:
                    results[agent_id] = result
                    self.logger.info(f"Parallel agent {config['name']} completed successfully")
        
        return results
    
    async def _execute_conditional_v3(self, phase: ExecutionPhase, phase_input: str, history_context: str) -> Dict[str, str]:
        """Execute agents based on runtime conditions"""
        if not phase.conditions:
            # If no conditions specified, execute all agents
            return await self._execute_parallel_v3(phase, phase_input, history_context)
        
        results = {}
        domain = phase.domain or "analysis"
        
        if domain in self.domain_managers:
            domain_manager = self.domain_managers[domain]
            conditions_met = await domain_manager.evaluate_conditions(phase.conditions, self.v3_execution_context)
        else:
            # Default: all conditions are met
            conditions_met = {agent_id: True for agent_id in phase.agents}
        
        for agent_id in phase.agents:
            if agent_id not in self.task_executors:
                continue
                
            if not conditions_met.get(agent_id, False):
                self.v3_execution_context.log_skipped_agent(agent_id, "Condition not met")
                continue
            
            executor = self.task_executors[agent_id]
            config = self.worker_map[agent_id]["config"]
            
            try:
                agent_input = phase_input
                if history_context:
                    agent_input = f"{history_context}\n\nCurrent Request: {phase_input}"
                
                result = await executor.execute_task(
                    description=config["instructions"],
                    expected_output=config["expectedOutput"],
                    task_name=config["name"],
                    input=agent_input,
                    user_email=self.user_email
                )
                
                results[agent_id] = result
                self.logger.info(f"Conditional agent {config['name']} executed successfully")
                
            except Exception as e:
                self.logger.error(f"Conditional agent {agent_id} failed: {sanitize_for_logging(e)}")
                self.v3_execution_context.log_error(agent_id, str(e), phase.name)
                results[agent_id] = f"Error: {str(e)}"
        
        return results
    
    async def _execute_dynamic_v3(self, phase: ExecutionPhase, phase_input: str, history_context: str) -> List[Dict[str, Any]]:
        """Execute agents with dynamic selection based on runtime conditions"""
        results = []
        current_input = phase_input
        if history_context:
            current_input = f"{history_context}\n\nCurrent Request: {phase_input}"
        
        domain = phase.domain or "analysis"
        domain_manager = self.domain_managers.get(domain)
        
        # Safety counter to prevent infinite loops
        max_iterations = len(phase.agents) + 2
        iteration = 0
        
        while iteration < max_iterations:
            if not domain_manager:
                # Fallback: execute agents sequentially
                remaining_agents = [a for a in phase.agents if not any(r.get("agent_id") == a for r in results)]
                if not remaining_agents:
                    break
                next_agent_id = remaining_agents[0]
            else:
                # Let domain manager choose next agent
                next_agent_id = await domain_manager.choose_next_agent(
                    available_agents=phase.agents,
                    current_input=current_input,
                    context=self.v3_execution_context,
                    execution_history=results
                )
            
            if not next_agent_id or next_agent_id not in self.task_executors:
                break
            
            executor = self.task_executors[next_agent_id]
            config = self.worker_map[next_agent_id]["config"]
            
            try:
                result = await executor.execute_task(
                    description=config["instructions"],
                    expected_output=config["expectedOutput"],
                    task_name=config["name"],
                    input=current_input,
                    user_email=self.user_email
                )
                
                results.append({
                    "agent_id": next_agent_id,
                    "agent_name": config["name"],
                    "result": result,
                    "iteration": iteration
                })
                
                current_input = result  # Output becomes input for next iteration
                self.logger.info(f"Dynamic agent {config['name']} executed successfully (iteration {iteration})")
                
            except Exception as e:
                self.logger.error(f"Dynamic agent {next_agent_id} failed: {sanitize_for_logging(e)}")
                self.v3_execution_context.log_error(next_agent_id, str(e), phase.name)
                results.append({
                    "agent_id": next_agent_id,
                    "agent_name": config["name"],
                    "result": f"Error: {str(e)}",
                    "iteration": iteration
                })
                current_input = f"Error from {config['name']}: {str(e)}"
            
            iteration += 1
        
        return results
    
    async def _format_v3_results(self, results: Dict[str, Any]) -> str:
        """Format V3 execution results according to manager's expected_output format"""
        
        # Get manager's expected output format
        expected_output = self.multi_agent_config.get("expected_output", "")
        
        if not expected_output:
            # Fallback to default V3 format if no expected_output specified
            return await self._format_v3_results_default(results)
        
        # Prepare agent results for formatting
        agent_results = []
        for phase_name, phase_result in results.items():
            if phase_name == "original_input":
                continue
                
            if isinstance(phase_result, dict):
                for key, value in phase_result.items():
                    if key != "error":
                        agent_results.append(f"{key}: {value}")
                    else:
                        agent_results.append(f"Error in {key}: {value}")
            elif isinstance(phase_result, list):
                for item in phase_result:
                    if isinstance(item, dict):
                        agent_name = item.get("agent_name", "Agent")
                        result = item.get("result", "No result")
                        agent_results.append(f"{agent_name}: {result}")
            else:
                agent_results.append(f"{phase_name}: {phase_result}")
        
        combined_results = "\n".join(agent_results)
        
        # Use manager agent to format according to expected_output
        formatting_task = Task(
            description=f"""
            You have received the following results from worker agents:
            
            {combined_results}
            
            Format these results according to the following expected output format:
            {expected_output}
            
            Present the results in the exact format specified above. Do not include any reasoning, thoughts, or additional explanations.
            Return only the formatted output as specified in the expected output format.
            """,
            expected_output=expected_output,
            agent=self.manager_agent,
            output_json=self.AgentOutput
        )
        
        formatting_crew = Crew(
            agents=[self.manager_agent],
            tasks=[formatting_task], 
            process=Process.sequential,
            verbose=False
        )
        
        try:
            formatting_result = await formatting_crew.kickoff_async()
            
            # Extract the formatted result
            if hasattr(formatting_result, 'tasks_output') and formatting_result.tasks_output:
                task_output = formatting_result.tasks_output[0]
                if hasattr(task_output, 'pydantic') and task_output.pydantic:
                    formatted_output = task_output.pydantic.answer
                elif hasattr(task_output, 'json') and task_output.json:
                    if isinstance(task_output.json, str):
                        json_data = json.loads(task_output.json)
                        formatted_output = json_data.get('answer', '')
                    elif isinstance(task_output.json, dict):
                        formatted_output = task_output.json.get('answer', '')
                    else:
                        formatted_output = str(task_output.json)
                elif hasattr(task_output, 'output'):
                    formatted_output = task_output.output
                else:
                    formatted_output = str(task_output)
            else:
                formatted_output = str(formatting_result)
            
            # Clean up the output
            formatted_output = formatted_output.strip()
            
            if formatted_output:
                return formatted_output
            else:
                self.logger.warning("Manager formatting returned empty result, using default format")
                return await self._format_v3_results_default(results)
                
        except Exception as e:
            self.logger.error(f"Error formatting results with manager: {sanitize_for_logging(e)}")
            self.logger.warning("Falling back to default V3 format")
            return await self._format_v3_results_default(results)
    
    async def _format_v3_results_default(self, results: Dict[str, Any]) -> str:
        """Default V3 formatting when no expected_output is specified or formatting fails"""
        output_lines = []
        output_lines.append("=" * 60)
        output_lines.append("MULTI-AGENT EXECUTION RESULTS (V3 Hybrid)")
        output_lines.append("=" * 60)
        
        for phase_name, phase_result in results.items():
            if phase_name == "original_input":
                continue
                
            output_lines.append(f"\n📋 PHASE: {phase_name.upper()}")
            output_lines.append("-" * 40)
            
            if isinstance(phase_result, dict):
                for key, value in phase_result.items():
                    if key != "error":
                        output_lines.append(f"🤖 {key}: {value}")
                    else:
                        output_lines.append(f"❌ Error: {value}")
            elif isinstance(phase_result, list):
                for i, item in enumerate(phase_result):
                    if isinstance(item, dict):
                        agent_name = item.get("agent_name", f"Agent {i}")
                        result = item.get("result", "No result")
                        output_lines.append(f"🤖 {agent_name}: {result}")
            else:
                output_lines.append(f"📄 Result: {phase_result}")
        
        # Add execution summary
        if self.v3_execution_context.skipped_agents:
            output_lines.append(f"\n⏭️  Skipped Agents: {len(self.v3_execution_context.skipped_agents)}")
        
        if self.v3_execution_context.errors:
            output_lines.append(f"❌ Errors: {len(self.v3_execution_context.errors)}")
        
        output_lines.append("=" * 60)
        
        return "\n".join(output_lines)

    async def _format_v2_results(self, agent_outputs: Dict[str, Dict[str, str]], agent_ids: List[str]) -> str:
        """Format V2 execution results according to manager's expected_output format"""
        
        # Get manager's expected output format
        expected_output = self.multi_agent_config.get("expected_output", "")
        
        if not expected_output:
            # Fallback to default V2 format if no expected_output specified
            return self._format_v2_results_default(agent_outputs, agent_ids)
        
        # Prepare agent results for formatting
        agent_results = []
        for agent_id in agent_ids:
            if agent_id in agent_outputs:
                agent_info = agent_outputs[agent_id]
                agent_results.append(f"{agent_info['name']}: {agent_info['output']}")
        
        combined_results = "\n".join(agent_results)
        
        # Use manager agent to format according to expected_output
        formatting_task = Task(
            description=f"""
            You have received the following results from worker agents:
            
            {combined_results}
            
            Format these results according to the following expected output format:
            {expected_output}
            
            Present the results in the exact format specified above. Do not include any reasoning, thoughts, or additional explanations.
            Return only the formatted output as specified in the expected output format.
            """,
            expected_output=expected_output,
            agent=self.manager_agent,
            output_json=self.AgentOutput
        )
        
        formatting_crew = Crew(
            agents=[self.manager_agent],
            tasks=[formatting_task], 
            process=Process.sequential,
            verbose=False
        )
        
        try:
            formatting_result = await formatting_crew.kickoff_async()
            
            # Extract the formatted result
            if hasattr(formatting_result, 'tasks_output') and formatting_result.tasks_output:
                task_output = formatting_result.tasks_output[0]
                if hasattr(task_output, 'pydantic') and task_output.pydantic:
                    formatted_output = task_output.pydantic.answer
                elif hasattr(task_output, 'json') and task_output.json:
                    if isinstance(task_output.json, str):
                        json_data = json.loads(task_output.json)
                        formatted_output = json_data.get('answer', '')
                    elif isinstance(task_output.json, dict):
                        formatted_output = task_output.json.get('answer', '')
                    else:
                        formatted_output = str(task_output.json)
                elif hasattr(task_output, 'output'):
                    formatted_output = task_output.output
                else:
                    formatted_output = str(task_output)
            else:
                formatted_output = str(formatting_result)
            
            # Clean up the output
            formatted_output = formatted_output.strip()
            
            if formatted_output:
                return formatted_output
            else:
                self.logger.warning("Manager formatting returned empty result, using default V2 format")
                return self._format_v2_results_default(agent_outputs, agent_ids)
                
        except Exception as e:
            self.logger.error(f"Error formatting V2 results with manager: {sanitize_for_logging(e)}")
            self.logger.warning("Falling back to default V2 format")
            return self._format_v2_results_default(agent_outputs, agent_ids)
    
    def _format_v2_results_default(self, agent_outputs: Dict[str, Dict[str, str]], agent_ids: List[str]) -> str:
        """Default V2 formatting when no expected_output is specified or formatting fails"""
        final_output_lines = ["-" * 40]
        for agent_id in agent_ids:
            if agent_id in agent_outputs:
                agent_info = agent_outputs[agent_id]
                final_output_lines.append(f"{agent_info['name']} Output: {agent_info['output']}")
        final_output_lines.append("-" * 40)
        return "\n".join(final_output_lines)

    # ===== V3 CONVENIENCE METHODS =====
    
    async def execute_sequential(self, user_input: str, **kwargs) -> str:
        """Convenience method for sequential execution (V2 behavior)"""
        return await self.execute_task_v3(
            user_input=user_input,
            execution_strategy=ExecutionStrategy.SEQUENTIAL,
            **kwargs
        )
    
    async def execute_parallel(self, user_input: str, **kwargs) -> str:
        """Convenience method for parallel execution"""
        return await self.execute_task_v3(
            user_input=user_input,
            execution_strategy=ExecutionStrategy.PARALLEL,
            **kwargs
        )
    
    async def execute_dynamic(self, user_input: str, **kwargs) -> str:
        """Convenience method for dynamic execution (most flexible)"""
        return await self.execute_task_v3(
            user_input=user_input,
            execution_strategy=ExecutionStrategy.DYNAMIC,
            **kwargs
        )
    
    async def execute_conditional(self, user_input: str, **kwargs) -> str:
        """Convenience method for conditional execution"""
        return await self.execute_task_v3(
            user_input=user_input,
            execution_strategy=ExecutionStrategy.CONDITIONAL,
            **kwargs
        )
    
    def enable_v3(self, enable: bool = True):
        """Enable or disable V3 features"""
        self.enable_v3_features = enable
        self.logger.info(f"V3 features {'enabled' if enable else 'disabled'}")
    
    def create_custom_phase(
        self,
        name: str,
        agents: List[str],
        strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
        conditions: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        domain: Optional[str] = None
    ) -> ExecutionPhase:
        """Helper method to create custom execution phases"""
        return ExecutionPhase(
            name=name,
            strategy=strategy,
            agents=agents,
            conditions=conditions,
            dependencies=dependencies,
            domain=domain
        )
    
    def get_agent_domain(self, agent_id: str) -> str:
        """Get the domain for a specific agent"""
        return self.agent_domain_mapping.get(agent_id, "unknown")
    
    def get_agents_by_domain(self, domain: str) -> List[str]:
        """Get all agent IDs for a specific domain"""
        return [agent_id for agent_id, agent_domain in self.agent_domain_mapping.items() if agent_domain == domain]
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of the last execution"""
        if not hasattr(self, 'v3_execution_context'):
            return {"error": "No execution context available"}
        
        return {
            "execution_id": self.v3_execution_context.execution_id,
            "current_phase": self.v3_execution_context.current_phase,
            "total_phases": len(self.v3_execution_context.execution_history),
            "skipped_agents": len(self.v3_execution_context.skipped_agents),
            "errors": len(self.v3_execution_context.errors),
            "skipped_details": self.v3_execution_context.skipped_agents,
            "error_details": self.v3_execution_context.errors
        }

    # ===== END V3 METHODS =====

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

         # Prepare context from history using TaskExecutor's proven method
        if history:
            self.logger.info(f"Received chat history with {len(history)} entries (Attempt: {attempt})")
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
                "- 'What about it?' → 'What about [specific topic from history]?'\n"
                "- 'How does this work?' → 'How does [specific item from history] work?'\n"
                "- 'Tell me more about her' → 'Tell me more about [specific person from history]'\n"
                "After enhancing the input, proceed with coordinating the agents using the enhanced version.\n\n"
            )
        else:
            self.logger.info(f"No chat history provided (Attempt: {attempt})")
            context = ""
            history_instruction = ""

            
        while attempt <= max_retries:
            try:
                if len(self.worker_agents) < 2:
                    self.logger.error(f"At least two worker agents are required.")
                    return "Error: At least two worker agents are required."

                self.logger.info(f"Starting task execution with user_input: '{sanitize_for_logging(user_input)}' (Attempt: {attempt})")
                if file_path:
                    self.logger.info(f"Processing file: {sanitize_for_logging(file_path)} (Attempt: {attempt})")
                

                # Set execution context for all task executors
                for executor in self.task_executors.values():
                    executor.expected_user_input = user_input

                # Prepare file content using the enhanced file processing
                file_content = ""
                if file_path and file_type:
                    file_content = self.process_file_content(file_path, file_type)
                    self.logger.debug(f"Processed file content: {sanitize_for_logging(file_content[:100])}{'...' if len(file_content) > 100 else ''}")

               

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
                self.logger.info(f"Worker metadata prepared: {sanitize_for_logging(json.dumps([{'id': w['id'], 'name': w['name']} for w in worker_metadata], indent=2))}")

                # Create task to determine agent order (using the same logic as original)
                manager_description = self.multi_agent_config.get("description", "Coordinate the processing of the user request by delegating to worker agents in sequence.")
                agent_order_task_description = (
                    f"Manager Description: {manager_description}\n"
                    f"User Input: '{sanitize_for_logging(user_input)}'\n"
                    f"File Content: {file_content if file_content else 'None'}\n"
                    f"Available Agents: {json.dumps([{'id': w['id'], 'name': w['name'], 'role': w['role'], 'instructions': w['instructions']} for w in worker_metadata], indent=2)}\n"
                    f"Task: Based on the manager's description, determine the sequence of agents to process the user request. The sequence should follow the steps outlined in the manager description, ensuring each agent processes the output of the previous agent (except the first, which processes the user input). The same agent may be used multiple times if required by the description.\n"
                    f"CRITICAL: Return ONLY a valid JSON array of agent IDs. Do not include any explanations, thoughts, or additional text.\n"
                    f"Format: [\"agent_id_1\", \"agent_id_2\", \"agent_id_3\"]\n"
                    f"Example with actual IDs: [\"1631a8e0-e4d0-457b-b5a9-8f796225f301\", \"2742b9f1-f5e1-568c-c6ba-9g807336g412\"]"
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

                # Enhanced agent order parsing with better error handling
                self.logger.debug(f"🎨 Attempting to parse LLM response for agent order...")
                agent_ids = await self._parse_agent_order_response(order_result, worker_metadata)
                
                if not agent_ids:
                    self.logger.warning("⚠️  Could not parse agent order from LLM. Using fallback strategies...")
                    
                    # First fallback: multi_agent_config agent_ids
                    agent_ids = self.multi_agent_config.get("agent_ids", [])
                    if agent_ids:
                        self.logger.info(f"✅ Using agent_ids from multi_agent_config: {len(agent_ids)} agents")
                    else:
                        # Ultimate fallback: use all available agents in order
                        agent_ids = [w["id"] for w in worker_metadata]
                        self.logger.info(f"✅ Using all available agents as ultimate fallback: {len(agent_ids)} agents")
                        self.logger.debug(f"   Fallback agent order: {[w['name'] for w in worker_metadata]}")
                else:
                    self.logger.info(f"✅ Successfully parsed agent order from LLM: {len(agent_ids)} agents")

                available_agent_ids = [w["id"] for w in worker_metadata]
                missing_agents = [aid for aid in agent_ids if aid not in available_agent_ids]
                if missing_agents:
                    self.logger.error(f"Missing required agents with IDs: {missing_agents}")
                    return f"Error: Missing required agents with IDs: {', '.join(missing_agents)}"

                # Execute agents in sequence using TaskExecutor instances
                agent_outputs = {}
                current_input = user_input
                
                for i, agent_id in enumerate(agent_ids):
                    if agent_id not in self.task_executors:
                        self.logger.error(f"TaskExecutor not found for agent ID: {agent_id}")
                        continue
                    
                    executor = self.task_executors[agent_id]
                    config = self.worker_map[agent_id]["config"]
                    
                    # Prepare task description and expected output
                    task_description = config["instructions"]
                    expected_output = config["expectedOutput"]
                    
                    # Add history context if available
                    full_task_description = task_description
                    if i == 0 and history_instruction:
                        full_task_description = f"{history_instruction}{task_description}"
                    
                    # Enhanced V2 execution logging
                    self.logger.info(f"🚀 Executing agent {config['name']} (ID: {agent_id}) - Step {i+1}/{len(agent_ids)}")
                    # self.logger.debug(f"   📋 V2 Task Description: {full_task_description[:200]}{'...' if len(full_task_description) > 200 else ''}")
                    # self.logger.debug(f"   🎯 V2 Expected Output: {expected_output[:100]}{'...' if len(expected_output) > 100 else ''}")
                    self.logger.debug(f"   📤 Input: {sanitize_for_logging(current_input[:2150])}{'...' if len(current_input) > 2150 else ''}")
                    
                    # Log tool information for this agent
                    if hasattr(executor, 'tools') and executor.tools:
                        self.logger.debug(f"   🛠️  Available Tools: {len(executor.tools)} tools ready")
                        for j, tool in enumerate(executor.tools[:3]):  # Show first 3 tools
                            tool_name = getattr(tool, 'name', f'Tool_{j}')
                            self.logger.debug(f"      {j+1}. {tool_name}")
                        if len(executor.tools) > 3:
                            self.logger.debug(f"      ... and {len(executor.tools) - 3} more tools")
                    
                    agent_start_time = time.time()
                    
                    try:
                        # Use TaskExecutor's proven execute_task method
                        result = await executor.execute_task(
                            description=full_task_description,
                            expected_output=expected_output,
                            task_name=config["name"],
                            file_path=file_path if i == 0 else None,  # Only first agent gets file
                            file_type=file_type if i == 0 else None,
                            input=current_input,
                            user_email=self.user_email,
                            history=history if i == 0 else None  # Only first agent gets history
                        )
                        
                        agent_duration = time.time() - agent_start_time
                        self.logger.info(f"✅  Agent {config['name']} completed successfully in {agent_duration:.3f}s")
                        self.logger.debug(f"   📥  Result: {result[:3200]}{'...' if len(result) > 3200 else ''}")
                        self.logger.debug(f"   📈  Result Length: {len(result)} characters")
                        
                        agent_outputs[agent_id] = {
                            "name": config["name"],
                            "output": result
                        }
                        
                        # The output becomes input for the next agent
                        current_input = result
                        self.logger.debug(f"   🔄 V2 Output becomes input for next agent ({len(result)} chars)")
                        
                    except Exception as e:
                        agent_duration = time.time() - agent_start_time
                        self.logger.error(f"❌ Agent {config['name']} failed after {agent_duration:.3f}s: {sanitize_for_logging(e)}")
                        self.logger.debug(f"   ⚠️  Error Details: {sanitize_for_logging(str(e))}")
                        
                        agent_outputs[agent_id] = {
                            "name": config["name"],
                            "output": f"Error: {sanitize_for_logging(str(e))}"
                        }
                        # Continue with the error as input for debugging
                        error_message = f"Error from {config['name']}: {sanitize_for_logging(str(e))}"
                        current_input = error_message
                        self.logger.debug(f"   🔄 Error becomes input for next agent: {error_message[:100]}...")

                # Format final output according to manager's expected_output format
                final_output = await self._format_v2_results(agent_outputs, agent_ids)
                
                self.logger.info(f"Multi-agent execution completed (Attempt: {attempt})")
                return final_output

            except Exception as e:
                self.logger.error(f"Error during execution: {sanitize_for_logging(e)} (Attempt: {attempt})", exc_info=True)
                if attempt < max_retries:
                    self.logger.info(f"Retrying task execution (Attempt {attempt + 1}/{max_retries}) after {retry_delay} seconds")
                    await asyncio.sleep(retry_delay)
                    attempt += 1
                    continue
                else:
                    self.logger.error(f"Max retries ({max_retries}) reached for error: {sanitize_for_logging(e)}")
                    return f"Error: Failed after {max_retries} attempts: {str(e)}"