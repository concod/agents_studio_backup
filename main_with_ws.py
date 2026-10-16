from fastapi import APIRouter, FastAPI, HTTPException, UploadFile, File, Form, Request, status, Depends, Response, Query, Cookie, Body, WebSocket, WebSocketDisconnect
import httpx
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Optional, List, Union, Any, Set
import os
import pathlib


from ddtrace import patch_all
from ddtrace.runtime import RuntimeMetrics
from ddtrace.profiling import Profiler
patch_all()
RuntimeMetrics.enable()
profiler = Profiler()
profiler.start()

import re
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
import logging
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import time
import asyncio
import concurrent.futures
import psycopg2
from psycopg2 import pool, sql, errors
import jwt
from google.oauth2 import id_token
from google.auth.transport import requests as grequests
from google.oauth2 import service_account
import pytz
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from DB.async_session_logs import AsyncSessionLogsDB
from database import DB_CONFIG
import aiofiles
import aiofiles.os
import gc
import psutil
import weakref
from functools import wraps

from api.auth import router as auth_router
from api.session_logs import router as session_logs_router
from api.users import router as users_router
from api.tenants import router as tenants_router, get_tenant_auth_user
from contextlib import asynccontextmanager
from config import settings

import pandas as pd
import requests
from io import BytesIO, StringIO
from DB.runtime import get_db, engine
from sqlalchemy.ext.asyncio import AsyncSession
from repositories.user_repository import UsersRepository

session_logs_db = AsyncSessionLogsDB()
app_router = APIRouter()
load_dotenv()
router = APIRouter()

ENABLE_AGENT_RUN = True
MIN_AGENTSFOR_MULTI = 1

# Memory Management Configuration
MEMORY_THRESHOLD_MB = int(os.getenv("MEMORY_THRESHOLD_MB", "1024"))  # 1GB default
GC_FREQUENCY = int(os.getenv("GC_FREQUENCY", "5"))  # Every 5 requests
FORCE_GC_THRESHOLD_MB = int(os.getenv("FORCE_GC_THRESHOLD_MB", "2048"))  # 2GB force GC
TOOLS_UNIVERSE_API_KEY = os.getenv("TEXT2SQL_API_TOKEN", "")

# Global counters for memory management
_request_counter = 0
_active_executors = weakref.WeakSet()

def get_memory_usage():
    """Get current memory usage in MB"""
    try:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0

def memory_monitor(func):
    """Decorator to monitor and manage memory usage"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        global _request_counter
        _request_counter += 1
        
        # Get initial memory
        initial_memory = get_memory_usage()
        
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            # Get final memory
            final_memory = get_memory_usage()
            memory_diff = final_memory - initial_memory
            
            # Log memory usage if significant
            if memory_diff > 50:  # Log if memory increased by more than 50MB
                logging.info(f"Memory usage increased by {memory_diff:.1f}MB during request")
            
            # Periodic garbage collection
            if _request_counter % GC_FREQUENCY == 0:
                collected = gc.collect()
                if collected > 0:
                    logging.info(f"Garbage collector freed {collected} objects")
            
            # Force garbage collection if memory usage is high
            if final_memory > FORCE_GC_THRESHOLD_MB:
                logging.warning(f"High memory usage detected: {final_memory:.1f}MB, forcing garbage collection")
                collected = gc.collect()
                new_memory = get_memory_usage()
                logging.info(f"Forced GC freed {collected} objects, memory reduced from {final_memory:.1f}MB to {new_memory:.1f}MB")
    
    return wrapper

def cleanup_executor_resources(executor):
    """Enhanced cleanup for executor resources"""
    if executor:
        try:
            # Call existing cleanup method
            if hasattr(executor, '_cleanup_request_resources'):
                executor._cleanup_request_resources()
            
            # Clear any large attributes
            if hasattr(executor, 'tools'):
                executor.tools = []
            if hasattr(executor, 'knowledge_sources'):
                executor.knowledge_sources = []
            if hasattr(executor, 'agent'):
                executor.agent = None
            if hasattr(executor, 'crew'):
                executor.crew = None
                
            # Remove from active executors tracking
            _active_executors.discard(executor)
            
        except Exception as e:
            logging.warning(f"Error during executor cleanup: {e}")
        finally:
            executor = None
AGENT_API_KEY = os.getenv("AGENT_API_KEY", "")
PUBLIC_AGENT_API_KEY = os.getenv("PUBLIC_AGENT_API_KEY", "")

IA_CHATBOT_AGENT_ID_PROD = os.getenv("IA_CHATBOT_AGENT_ID_PROD", "")


if ENABLE_AGENT_RUN:
    from task_executor import TaskExecutor
    from langraph_executor import LangGraphExecutor
    from langraph_executor_stream import LangGraphExecutorStream
    # from multi_agent_executor import MultiAgentExecutor
    from multi_agent_executor_v2 import MultiAgentExecutorV2 as MultiAgentExecutor
    import psycopg2
    from psycopg2 import Error as PostgresError
    from google.oauth2 import service_account
    from google.cloud import bigquery

import logging
# Import centralized logger manager
# from app.middleware.logger_manager import get_logger_manager, get_logger, configure_file_logger
logger = logging.getLogger(__name__)

# logger = get_logger(__name__)


# BigQuery Service Account Credentials
def get_bigquery_credentials() -> Dict[str, Any]:
    """
    Load BigQuery service account credentials from environment variable.
    Returns empty dict if not set or if parsing fails.
    """
    credentials_json_str = os.getenv("BIGQUERY_SA_JSON")
    if not credentials_json_str:
        return {}
    try:
        return json.loads(credentials_json_str)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse BIGQUERY_SA_JSON: {str(e)}")
        return {}


# ✅ Application lifecycle context manager
# Global Whisper model instance (loaded once at startup)
whisper_model = None
whisper_executor = None
whisper_device = None  # "cuda", "cpu", or "auto"
whisper_compute_type = None  # "int8", "int8_float16", "float16", "float32"
is_faster_whisper_loaded = False  # Track which library is actually loaded

@asynccontextmanager
async def lifespan(app):
    """
    Initialize the async database engine at startup and dispose it at shutdown.
    Also pre-load Whisper model for transcription API.
    """
    global whisper_model, whisper_executor, whisper_device, whisper_compute_type, is_faster_whisper_loaded
    
    try:
        # test connection
        async with engine.begin() as conn:
            await conn.run_sync(lambda _: None)
        print("✅ Database connection pool established")
        
        # Pre-load Whisper model at startup with optimizations
        try:
            import concurrent.futures
            
            # Configuration
            whisper_model_name = os.getenv("WHISPER_MODEL", "base")
            use_faster_whisper = os.getenv("USE_FASTER_WHISPER", "true").lower() == "true"
            device_config = os.getenv("WHISPER_DEVICE", "auto")  # "cuda", "cpu", or "auto"
            compute_type_config = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # "int8", "int8_float16", "float16", "float32"
            max_workers = int(os.getenv("WHISPER_MAX_WORKERS", "3"))  # Increase for better parallelism
            
            print(f"🔄 Loading Whisper model '{whisper_model_name}' (this may take a moment)...")
            print(f"   Using faster-whisper: {use_faster_whisper}")
            print(f"   Device config: {device_config}, Compute Type: {compute_type_config}")
            
            # Try faster-whisper first (much faster, uses CTranslate2)
            if use_faster_whisper:
                try:
                    from faster_whisper import WhisperModel
                    
                    # Auto-detect device if set to "auto"
                    if device_config == "auto":
                        try:
                            import torch
                            detected_device = "cuda" if torch.cuda.is_available() else "cpu"
                        except:
                            detected_device = "cpu"
                    else:
                        detected_device = device_config
                    
                    whisper_device = detected_device
                    whisper_compute_type = compute_type_config
                    
                    print(f"   Detected device: {whisper_device}")
                    
                    # Load model in a thread to avoid blocking startup
                    def load_faster_whisper():
                        return WhisperModel(
                            whisper_model_name,
                            device=whisper_device,
                            compute_type=whisper_compute_type,
                            num_workers=1,  # Per-model workers
                            download_root=None  # Use cache
                        )
                    
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(load_faster_whisper)
                        whisper_model = future.result()
                    
                    is_faster_whisper_loaded = True
                    print("✅ Faster-Whisper model loaded successfully")
                except ImportError:
                    print("⚠️  faster-whisper not installed, falling back to openai-whisper")
                    print("   Install with: pip install faster-whisper")
                    use_faster_whisper = False
                except Exception as e:
                    print(f"⚠️  Failed to load faster-whisper: {e}")
                    print("   Falling back to openai-whisper")
                    use_faster_whisper = False
            
            # Fallback to openai-whisper if faster-whisper failed
            if not is_faster_whisper_loaded:
                import whisper
                
                # Load model in a thread to avoid blocking startup
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(whisper.load_model, whisper_model_name)
                    whisper_model = future.result()
                
                whisper_device = "cpu"  # openai-whisper doesn't expose device easily
                whisper_compute_type = "float32"
                is_faster_whisper_loaded = False  # Explicitly set to False for openai-whisper
                print("✅ OpenAI-Whisper model loaded successfully")
            
            # Create thread pool executor for transcription tasks
            # Increased workers for better parallelism with optimized model
            whisper_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix="whisper_transcribe"
            )
            print(f"✅ Whisper executor created with {max_workers} workers")
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to load Whisper model: {e}")
            import traceback
            print(traceback.format_exc())
            print("   Transcription API will not be available")
            whisper_model = None
            whisper_executor = None
            whisper_device = None
            whisper_compute_type = None
            is_faster_whisper_loaded = False
        
        yield
    finally:
        await engine.dispose()
        print("🧹 Database connection pool disposed")
        
        # Cleanup Whisper executor
        if whisper_executor:
            whisper_executor.shutdown(wait=True)
            print("🧹 Whisper executor shutdown")


app = FastAPI(lifespan=lifespan)
app.include_router(app_router)

app.include_router(session_logs_router, prefix="/api", tags=["session_logs"])
app.include_router(auth_router, prefix="/api", tags=["auth"])
app.include_router(users_router, prefix="/api", tags=["users"])
app.include_router(tenants_router, prefix="/api", tags=["tenants"])


# # Database connection configuration
# DB_CONFIG = {
#     "host": os.getenv("DB_HOST", "10.103.0.3"),
#     "port": os.getenv("DB_PORT", "5432"),
#     "database": os.getenv("DB_NAME", "postgres"),
#     "user": os.getenv("DB_USER", "postgres"),
#     "password": os.getenv("DB_PASSWORD", "impact@123")
# }

# Connection pool for database connections
db_pool, db_connection = None, None

# Add general exception handlers
@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: HTTPException):
    return page_templates.TemplateResponse("404.html", {"request": request}, status_code=404)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if exc.status_code == 404:
        return page_templates.TemplateResponse("404.html", {"request": request}, status_code=404)
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# Mount logs directory for static file serving (optional)
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
app.mount("/logs", StaticFiles(directory=LOG_DIR), name="logs")

# Mount uploads directory
UPLOAD_DIR = Path("uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


# Set up Jinja2 templates
templates = Jinja2Templates(directory="static/pages")
page_templates = Jinja2Templates(directory="static/pages")

# Load base URL from environment variable
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")



# Add CORS middleware to handle cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        BASE_URL,
        "http://localhost:8004",
        "http://127.0.0.1:8004",
        "http://localhost:8123",
        "http://127.0.0.1:8123",
        "http://ajunsmachine.theworkpc.com:8123",
        "http://ajunsmachine.theworkpc.com:5173",
        "http://ajunsmachine.theworkpc.com:3000",
        "http://localhost:3000",
        # Impact Agents domains
        "https://impact-agents.ai",
        "https://www.impact-agents.ai",
        "https://www.app.impact-agents.ai",
        "https://app.impact-agents.ai",
        "https://smart-ai-image-editor.impact-agents.ai",
        # Impact Analytics domains
        "https://impactanalytics.co",
        "https://www.impactanalytics.co",
        "https://sandbox.impactanalytics.co",
        "http://impactanalytics.ai",
        "https://impactanalytics.ai",
        "https://ia-stg.webflow.io",
        "https://sample-site-a9e933.design.webflow.com"
    ],  # Adjust for production
    # Allow any subdomain depth under impact-agents.ai (e.g., app, tools, www.app, etc.)
    allow_origin_regex=r"^https://([a-zA-Z0-9-]+\.)*impact-agents\.ai$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define paths that don't require authentication
EXCLUDED_PATHS = [
    "/glogin",
    "/login",
    "/static",
    "/pages",
    "/favicon.ico",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/api/logs",
    "/api/multi_agent/logs",
    "/api/sessions-logs",
    "/api/sessions-login",  # Add login endpoint to excluded paths
    "/api/register",  # Add register endpoint to excluded paths
    "/api/fetch_session_logs",
    "/api/filtered-logs",
    "/api/user-sessions",
    "/chat",  # This will match /chat and all paths that start with /chat/
    "/ia_chatbot",  # This will match /ia_chatbot and all paths that start with /ia_chatbot/
    "/reset-session",
    "/api/agent/execute",
    "/api/agent/execute/ws",
    "/api/agent/langgraph/execute",
    "/api/agent/image-extraction/execute",
    "/api/generate_captcha_token",
    "/ia_chatbot_embed",
    "/api/ia_chatbot",
    "/api/tenants",  # Exclude tenant APIs from global auth - they have their own auth
    "/api/agents/custom/store-hub",  # Exclude store-hub APIs from global auth - they use tenant auth (log_users)
    "/api/session/execution-feedback",
    "/sessions-login",
    "/sessions",
    "/ib_agent",
    "/ia_digital_agent",
    "/leslies-pool-agent",
    "/api/multi-agent/execute",
    "/psp-agent",
    "/psp-agent-v2",
    "/psp-agent-v3",
    "/usc-agent",
    "/psp_sop",
    "/ariat-agent",
    "/api/datadog/logs",
    "/api/geo/location",
    "/api/geo/location/",
    "/tsc-banner-agent",
    "/ia-marketing-agent",
    "/api/proxy/audit-image",
    "/api/transcribe",  # Whisper transcription endpoint - no auth required
    "/api/whisper/transcribe",  # Alternative Whisper transcription endpoint - no auth required
    "/whisper-transcribe",  # Whisper transcription HTML page - no auth required
    "/run_agent",  # LangGraph demo page - no auth required
    "/ib_bi_agent",
]



# Initialize Limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Import and setup Store Hub application (DDD architecture)
from applications.store_hub.api.routes import router as store_hub_base_router, create_routes
store_hub_router = create_routes(limiter, get_db, get_tenant_auth_user)
app.include_router(store_hub_router, tags=["Store Hub"])

# Error handlers
@app.exception_handler(404)
async def not_found_exception_handler(_: Request, __: HTTPException) -> JSONResponse:
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"detail": "Not Found"},
    )


@app.exception_handler(500)
async def server_error_exception_handler(_: Request, __: HTTPException) -> JSONResponse:
    """Handle 500 errors."""
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
    )


# Custom rate limit handler
@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logger.warning(f"Rate limit exceeded for IP: {request.client.host}")
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded: 10 per 1 minute. Please try again later."}
    )

# Direct chat route - add this early before middleware
@app.get("/chat", response_class=HTMLResponse)
@limiter.limit("60/minute")  # Page serving endpoint
async def serve_chat_page(request: Request):
    return FileResponse("static/chat_common.html")

@app.get("/ia_digital_agent", response_class=HTMLResponse)
@limiter.limit("60/minute")  # Page serving endpoint
async def serve_ia_digital_agent_page(request: Request):
    return FileResponse("static/chat.html")

# @app.get("/leslies-pool-agent", response_class=HTMLResponse)
# @limiter.limit("60/minute")  # Page serving endpoint
# async def serve_leslies_agent_page(request: Request):
#     return FileResponse("static/leslies-pool-agent.html")
@app.get("/psp-agent", response_class=HTMLResponse)
@limiter.limit("60/minute")  # Page serving endpoint
async def serve_psp_agent_page(request: Request):
    return FileResponse("static/pet-supplies-plus-agent.html")

@app.get("/psp-agent-v2", response_class=HTMLResponse)
@limiter.limit("60/minute")  # Page serving endpoint
async def serve_psp_agent_page(request: Request):
    return FileResponse("static/pet-supplies-plus-agent-v2.html")


@app.get("/psp-agent-v3", response_class=HTMLResponse)
@limiter.limit("60/minute")  # Page serving endpoint
async def serve_psp_agent_v3_page(request: Request):
    return FileResponse("static/pet-supplies-plus-agent-v3.html")


@app.get("/tsc-banner-agent", response_class=HTMLResponse)
@limiter.limit("60/minute")  # Page serving endpoint
async def serve_psp_agent_page(request: Request):
    return FileResponse("static/tsc-banner-agent.html")

@app.get("/ia_chatbot", response_class=HTMLResponse)
@limiter.limit("60/minute")  # Page serving endpoint
async def serve_ia_chat_page(request: Request):
    return FileResponse("static/ia_chatbot.html")

@app.get("/ib_agent", response_class=HTMLResponse)
@limiter.limit("60/minute")  # Page serving endpoint
async def serve_ib_agent_page(request: Request):
    return FileResponse("static/ib_agent.html")

@app.get("/ib_bi_agent", response_class=HTMLResponse)
@limiter.limit("60/minute")  # Page serving endpoint
async def serve_ib_bi_agent_page(request: Request):
    return FileResponse("static/ib_bi_agent.html")

@app.get("/ia_chatbot_embed", response_class=HTMLResponse)
@limiter.limit("60/minute")  # Page serving endpoint
async def ia_chatbot_embed(request: Request):
    return FileResponse("static/ia_chatbot_embed.html")

@app.get("/ia-marketing-agent", response_class=HTMLResponse)
@limiter.limit("60/minute")
async def serve_ia_marketing_agent_page(request: Request):
    return FileResponse("static/ia-marketing-agent.html")

# @app.get("/ariat-agent", response_class=HTMLResponse)
# @limiter.limit("60/minute")  # Page serving endpoint
# async def serve_ariat_agent_page(request: Request):
#     return FileResponse("static/ariat_agent.html")

@app.get("/psp_sop", response_class=HTMLResponse)
@limiter.limit("60/minute")  # Page serving endpoint
async def serve_psp_sop_page(request: Request):
    return FileResponse("static/pop_sop.html")

@app.get("/pet_pantry_sop", response_class=HTMLResponse)
@limiter.limit("60/minute")  # Page serving endpoint
async def serve_psp_sop_duplicate_page(request: Request):
    return FileResponse("static/pop_sop_duplicate.html")

@app.get("/complete_pet_banner", response_class=HTMLResponse)
@limiter.limit("60/minute")  # Page serving endpoint
async def serve_psp_agent_v2_duplicate_page(request: Request):
    return FileResponse("static/pet-supplies-plus-agent-v2-duplicate.html")

@app.get("/assortment_recommendation", response_class=HTMLResponse)
@limiter.limit("60/minute")  # Page serving endpoint
async def serve_ib_agent_duplicate_page(request: Request):
    return FileResponse("static/ib_agent_duplicate.html")

@app.get("/usc-agent", response_class=HTMLResponse)
@limiter.limit("60/minute")  # Page serving endpoint
async def serve_usc_agent_page(request: Request):
    return FileResponse("static/usc_agent-v2.html")

@app.get("/label-compliance-agent", response_class=HTMLResponse)
@limiter.limit("60/minute")  # Page serving endpoint
async def serve_label_compliance_agent_page(request: Request):
    return FileResponse("static/label-compliance-agent.html")

@app.get("/whisper-transcribe", response_class=HTMLResponse)
@limiter.limit("60/minute")  # Page serving endpoint
async def serve_whisper_transcribe_page(request: Request):
    return FileResponse("static/whisper-transcribe.html")

@app.get("/run_agent", response_class=HTMLResponse)
@limiter.limit("60/minute")
async def serve_langgraph_demo_page(request: Request):
    return FileResponse("static/langgraph_ws_demo.html")

# Hardcoded credentials for the cloned chatbot page
CLONE_CHATBOT_USERS = {
    "admin": "monday@gent9544",
    "nirav.parekh@impactanalytics.co": "niravs@gent4322",
    "ajun.ravi@impactanalytics.co": "monday@gent777",
    "guest": "guest123"
}

# Set to store active sessions
ACTIVE_SESSIONS: Set[str] = set()

@app.get("/mondaysmart_agent", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def serve_ia_chat_clone_page(request: Request, clone_session: Optional[str] = Cookie(None), username: Optional[str] = Cookie(None)):
    # Check if user is authenticated
    if clone_session and clone_session in ACTIVE_SESSIONS:
        # User is authenticated, serve the chatbot page with username
        # Read the HTML file
        with open("static/mondaysmart_agent.html", "r") as file:
            html_content = file.read()
            
        # Replace the placeholder with actual username
        if username:
            html_content = html_content.replace('<span id="loggedInUsername">User</span>', f'<span id="loggedInUsername">{username}</span>')
        
        return HTMLResponse(content=html_content)
    else:
        # User is not authenticated, redirect to login page
        return FileResponse("static/mondaysmart_agent_login.html")

@app.post("/mondaysmart_agent_auth", response_class=HTMLResponse)
@limiter.limit("30/minute")  # Auth endpoint
async def authenticate_clone_chatbot(request: Request, response: Response, username: str = Form(...), password: str = Form(...)):
    # Check if credentials are valid
    if username in CLONE_CHATBOT_USERS and CLONE_CHATBOT_USERS[username] == password:
        # Generate a simple session ID
        session_id = str(uuid.uuid4())
        # Add to active sessions
        ACTIVE_SESSIONS.add(session_id)
        # Set cookies for session and username
        response.set_cookie(key="clone_session", value=session_id, httponly=True)
        response.set_cookie(key="username", value=username)
        # Redirect to chatbot page
        response.headers["Location"] = "/mondaysmart_agent"
        response.status_code = status.HTTP_302_FOUND
        return response
    else:
        # Invalid credentials, redirect back to login with error
        response.headers["Location"] = "/mondaysmart_agent?error=invalid"
        response.status_code = status.HTTP_302_FOUND
        return response

@app.get("/mondaysmart_agent_logout", response_class=HTMLResponse)
@limiter.limit("10/minute")
async def logout_clone_chatbot(request: Request, response: Response, clone_session: Optional[str] = Cookie(None)):
    # Remove session from active sessions
    if clone_session and clone_session in ACTIVE_SESSIONS:
        ACTIVE_SESSIONS.remove(clone_session)
    
    # Clear cookie and redirect to login page
    response.delete_cookie(key="clone_session")
    # Return the login page directly instead of redirecting
    return FileResponse("static/mondaysmart_agent_login.html")

# Chat route with RAG ID
@app.get("/chat/{rag_id}", response_class=HTMLResponse)
@limiter.limit("60/minute")  # Page serving endpoint
async def serve_chat_page_with_rag_id(request: Request, rag_id: str):
    return FileResponse("static/chat.html")

@app.middleware("http") 
async def auth_middleware(request: Request, call_next):
    # Let CORS preflight requests pass through without auth so CORS middleware can respond
    if request.method == "OPTIONS":
        return await call_next(request)
    # Check if the path is in excluded paths
    path = request.url.path
    if any(path.startswith(excluded) for excluded in EXCLUDED_PATHS):
        return await call_next(request)

    # Check if it's an API route
    if path.startswith("/api/"):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing or invalid authorization header"}
            )
        
        token = auth_header.split("Bearer ")[1]
        try:
            # Verify the token
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            if payload.get("type","") == "guest":
                request.state.user_email = "guest"
                return await call_next(request)
                
            email = payload.get("email")
            if not email:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid token payload"}
                )
            # Store user email in request state for access in route handlers
            request.state.user_email = email
        except jwt.ExpiredSignatureError:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Token has expired"}
            )
        except Exception as e:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": str(e)}
            )

    return await call_next(request)

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Serve sessions login page
@app.get("/sessions-logs", response_class=HTMLResponse)
async def serve_sessions_login(request: Request):
    return templates.TemplateResponse("auth/sessions_login.html", {"request": request})

@app.get("/sessions", response_class=FileResponse)
async def sessions_page():
    """Serve the sessions page."""
    return FileResponse("templates/sessions.html")

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)



# Configuration (replace with your actual values)
SECRET_KEY = "AUTHSECAJUN9544213YR2025"  # Replace with a secure key
ALGORITHM = "HS256"
GOOGLE_CLIENT_ID = "482146741158-q68nmoak32jq9il8moj9e6gtohqkfkjb.apps.googleusercontent.com"


# JWT token generation
def generate_jwt_token(data: dict, days_valid: int = 14):
    expiration_time = datetime.utcnow() + timedelta(days=days_valid)
    payload = {
        **data,
        "exp": expiration_time
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token

# Pydantic model for Google login
class TokenData(BaseModel):
    token: str

# Serve login page
@app.get("/login", response_class=HTMLResponse)
@limiter.limit("30/minute")  # Auth endpoint
async def login(request: Request):
    return FileResponse("static/login.html")
    # return templates.TemplateResponse("login.html", {"request": request})


ALLOW_NEW_USERS = False

# Handle Google OAuth login
@app.post("/glogin")
@limiter.limit("30/minute")
async def glogin(request: Request, data: TokenData, db: AsyncSession = Depends(get_db)):
    try:
        idinfo = id_token.verify_oauth2_token(data.token, grequests.Request(), GOOGLE_CLIENT_ID)

        email = idinfo.get("email")
        name = idinfo.get("name", email.split("@")[0] if email else "User")
        picture = idinfo.get("picture", "")

        if not email:
            raise HTTPException(status_code=400, detail="Google token missing email")

        users_repo = UsersRepository(db)

        # fetch or create
        user = await users_repo.get_by_email(email)
        if not user:
            if ALLOW_NEW_USERS is True:
                default_client_id = "36fcfb10-4761-42dc-bc86-91150636ddad"  # TODO replace
                user = await users_repo.create_user(
                    email=email,
                    name=name,
                    picture=picture,
                    client_id=default_client_id,
                    role="user",
                    is_admin=False,
                )
            else:
                raise HTTPException(status_code=403, detail="You're not authorized to access the studio.")
        else:
            if user.is_deleted:
                raise HTTPException(status_code=403, detail="User account has been deleted")
            if not user.is_active:
                raise HTTPException(status_code=403, detail="User account is deactivated")
            # update profile on login
            user = await users_repo.update_profile(email=email, name=name, picture=picture) or user

        # Build JWT payload
        user_data = {
            "email": user.email,
            "name": user.name,
            "picture": user.picture or "",
            "role": user.role or "user",
            "is_admin": bool(user.is_admin),
            "client_id": user.client_id,
            "sub": idinfo.get("sub"),
            "user_id": str(user.id),
        }
        token = generate_jwt_token(user_data)

        return {
            "status": "success",
            "token": token,
            "detail": "Login successful",
            "name": user.name,
            "picture": user.picture or "",
            "email": user.email,
            "role": user.role or "user",
            "is_admin": bool(user.is_admin),
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid token: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

# JWT dependency for protected routes
def get_current_user(request: Request):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing or invalid"
        )
    
    token = auth_header.split("Bearer ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email = payload.get("email")
        if not email:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        return email
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

# Example protected route
@app.get("/protected", response_class=JSONResponse)
async def protected_route(request: Request, email: str = Depends(get_current_user)):
    return {"message": f"Hello, {email}! You are authenticated."}





# Data model for Agent
class AgentFeatures(BaseModel):
    knowledgeBase: bool = False
    dataQuery: bool = False

class Agent(BaseModel):
    id: str
    name: str
    description: str
    llmProvider: str
    llmModel: str
    apiKey: str
    role: str
    goal: str = ""
    expectedOutput: str = ""
    backstory: str = ""
    instructions: str
    verbose: bool = True
    enable_memory: bool = False
    enable_planning: bool = False
    raw_output: bool = False
    ai_framework: str = "CrewAI"  # "CrewAI" or "LangGraph"
    features: AgentFeatures
    tools: List[str] = []  # List of tool IDs
    advanced_tools: List[str] = []  # List of advanced tool IDs
    sample_user_input: str = ""  # Sample user input for the agent
    knowledge_base_id: Optional[str] = None

class AgentCreate(BaseModel):
    name: str
    description: str
    llmProvider: str
    llmModel: str
    apiKey: str
    role: str
    goal: str = ""
    expectedOutput: str = ""
    backstory: str = ""
    instructions: str
    verbose: bool = True
    enable_memory: bool = False
    enable_planning: bool = False
    raw_output: bool = False
    ai_framework: str = "CrewAI"  # "CrewAI" or "LangGraph"
    features: AgentFeatures
    tools: List[str] = []  # List of tool IDs
    advanced_tools: List[str] = []
    sample_user_input: str = ""  # Sample user input for the agent
    knowledge_base_id: Optional[str] = None

class Tool(BaseModel):
    id: str
    name: str
    description: str
    tags: List[str]
    is_added: bool = False
    data_connector_id: Optional[str] = None
    is_internal: Optional[bool] = False

class OpenAPIServer(BaseModel):
    url: str
    description: str

class OpenAPIInfo(BaseModel):
    title: str
    version: str
    description: str = ""

class OpenAPISchema(BaseModel):
    openapi: str
    info: OpenAPIInfo
    servers: List[OpenAPIServer]
    paths: Dict[str, dict]

class CustomTool(BaseModel):
    name: str
    description: str
    tags: List[str]
    schema: Dict[str, Any]
    is_custom: bool = True
    data_connector_id: Optional[str] = None

# Tool Authentication Models
class ToolAuth(BaseModel):
    type: str  # "api_key", "oauth", "basic", etc.
    headers: Dict[str, str] = {}
    params: Dict[str, str] = {}

class ToolConfig(BaseModel):
    id: str
    schema: Dict[str, Any]
    auth: Optional[ToolAuth] = None

# Notification Models
class Notification(BaseModel):
    id: str
    type: str  # info, success, warning, error
    message: str
    timestamp: str
    read: bool = False

class NotificationUpdate(BaseModel):
    read: bool

# UI Deployment Models
class UIDeployment(BaseModel):
    id: str
    appTitle: str
    agentId: str
    appDescription: Optional[str] = ""
    logoUrl: Optional[str] = ""
    backgroundColor: Optional[str] = "#2563eb"
    textColor: Optional[str] = "#ffffff"
    welcomeMessage: Optional[str] = "Hello! How can I help you today?"
    chatTheme: Optional[str] = "modern"
    placeholderText: Optional[str] = "Type your message here..."
    enableDarkMode: Optional[bool] = True
    enableClearChat: Optional[bool] = True
    enableVoice: Optional[bool] = False
    enableTyping: Optional[bool] = True
    enableTimestamps: Optional[bool] = False
    status: Optional[str] = "active"
    createdAt: Optional[str] = None
    updatedAt: Optional[str] = None

class UIDeploymentCreate(BaseModel):
    appTitle: str
    agentId: str
    appDescription: Optional[str] = ""
    logoUrl: Optional[str] = ""
    backgroundColor: Optional[str] = "#2563eb"
    textColor: Optional[str] = "#ffffff"
    welcomeMessage: Optional[str] = "Hello! How can I help you today?"
    chatTheme: Optional[str] = "modern"
    placeholderText: Optional[str] = "Type your message here..."
    enableDarkMode: Optional[bool] = True
    enableClearChat: Optional[bool] = True
    enableVoice: Optional[bool] = False
    enableTyping: Optional[bool] = True
    enableTimestamps: Optional[bool] = False

class UIDeploymentUpdate(BaseModel):
    appTitle: Optional[str] = None
    agentId: Optional[str] = None
    appDescription: Optional[str] = None
    logoUrl: Optional[str] = None
    backgroundColor: Optional[str] = None
    textColor: Optional[str] = None
    welcomeMessage: Optional[str] = None
    chatTheme: Optional[str] = None
    placeholderText: Optional[str] = None
    enableDarkMode: Optional[bool] = None
    enableClearChat: Optional[bool] = None
    enableVoice: Optional[bool] = None
    enableTyping: Optional[bool] = None
    enableTimestamps: Optional[bool] = None
    status: Optional[str] = None

# Helper function to load notifications
def load_notifications() -> List[dict]:
    try:
        with open('static/data/notifications.json', 'r') as f:
            data = json.load(f)
            return data.get('notifications', [])
    except FileNotFoundError:
        return []

# Helper function to save notifications
def save_notifications(notifications: List[dict]):
    with open('static/data/notifications.json', 'w') as f:
        json.dump({'notifications': notifications}, f, indent=4)

# Helper functions for UI deployments
def load_ui_deployments() -> List[dict]:
    try:
        with open('data/ui_deployments.json', 'r') as f:
            data = json.load(f)
            return data.get('deployments', [])
    except FileNotFoundError:
        return []

def save_ui_deployments(deployments: List[dict]):
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    with open('data/ui_deployments.json', 'w') as f:
        json.dump({'deployments': deployments}, f, indent=4)

async def generate_deployment_files(deployment: dict):
    """Generate HTML, CSS, and JS files for a deployment"""
    deployment_id = deployment["id"]
    deployment_dir = f"static/deployments/{deployment_id}"
    
    # Create deployment directory
    os.makedirs(deployment_dir, exist_ok=True)
    
    # Generate HTML file
    html_content = generate_deployment_html(deployment)
    async with aiofiles.open(f"{deployment_dir}/index.html", 'w') as f:
        await f.write(html_content)
    
    # Generate CSS file
    css_content = generate_deployment_css(deployment)
    async with aiofiles.open(f"{deployment_dir}/style.css", 'w') as f:
        await f.write(css_content)
    
    # Generate JS file
    js_content = generate_deployment_js(deployment)
    async with aiofiles.open(f"{deployment_dir}/script.js", 'w') as f:
        await f.write(js_content)

def generate_deployment_html(deployment: dict) -> str:
    """Generate HTML content for a deployment"""
    # Extract configuration
    enable_dark_mode = deployment.get("enableDarkMode", True)
    enable_clear_chat = deployment.get("enableClearChat", True)
    enable_voice = deployment.get("enableVoice", False)
    placeholder_text = deployment.get("placeholderText", "Type your message here...")
    
    # Build controls HTML
    controls_html = ""
    if enable_clear_chat:
        controls_html += '<button class="chat-control-btn" id="clearChatBtn" title="Clear Chat"><i class="fas fa-trash"></i> Clear Chat</button>'
    if enable_dark_mode:
        controls_html += '<button class="chat-control-btn" id="darkModeBtn" title="Toggle Dark Mode"><i class="fas fa-moon"></i> Dark Mode</button>'
    if enable_voice:
        controls_html += '<button class="chat-control-btn" id="voiceBtn" title="Voice Controls"><i class="fas fa-microphone-slash"></i> Voice Off</button>'
    
    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{deployment["appTitle"]}</title>
    <link rel="stylesheet" href="/static/deployments/{deployment["id"]}/style.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
</head>
<body class="theme-light">
    <div class="chat-wrapper">
        <div class="chat-container">
            <!-- Header -->
            <div class="chat-header">
                <div class="chat-header-content">
                    <div class="chat-logo">
                        {f'<img src="{deployment["logoUrl"]}" alt="Logo">' if deployment.get("logoUrl") else '<i class="fas fa-robot"></i>'}
                    </div>
                    <div class="chat-title-section">
                        <h1 class="chat-title">{deployment["appTitle"]}</h1>
                        {f'<p class="chat-subtitle">{deployment["appDescription"]}</p>' if deployment.get("appDescription") else ''}
                    </div>
                </div>
            </div>
            
            <!-- Messages Area -->
            <div class="chat-messages-container">
                <div class="chat-messages" id="chatMessages">
                    <div class="message bot-message">
                        <div class="message-avatar">
                            {f'<img src="{deployment["logoUrl"]}" alt="Bot">' if deployment.get("logoUrl") else '<i class="fas fa-robot"></i>'}
                        </div>
                        <div class="message-content">
                            <div class="message-text">{deployment.get("welcomeMessage", "Hello! How can I help you today?")}</div>
                            <div class="message-meta">
                                <span class="message-time" id="welcomeTime"></span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Execution Steps Container -->
                <div class="execution-steps" id="executionSteps" style="display: none;">
                    <div class="steps-header">
                        <h4><i class="fas fa-cogs"></i> Execution Steps</h4>
                    </div>
                    <div class="steps-list" id="stepsList">
                        <!-- Steps will be added here dynamically -->
                    </div>
                </div>
                
                <!-- Typing Indicator -->
                <div class="typing-indicator" id="typingIndicator" style="display: none;">
                    <div class="message bot-message">
                        <div class="message-avatar">
                            {f'<img src="{deployment["logoUrl"]}" alt="Bot">' if deployment.get("logoUrl") else '<i class="fas fa-robot"></i>'}
                        </div>
                        <div class="message-content">
                            <div class="typing-dots">
                                <span></span>
                                <span></span>
                                <span></span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Controls -->
            <div class="chat-controls">
                {controls_html}
            </div>
            
            <!-- Input Area -->
            <div class="chat-input-container">
                <div class="chat-input-wrapper">
                    <input type="text" id="chatInput" class="chat-input" placeholder="{placeholder_text}" disabled>
                    <button id="sendButton" class="send-button" disabled title="Send message">
                        <i class="fas fa-paper-plane"></i>
                    </button>
                </div>
                <div class="chat-status" id="chatStatus">Connecting...</div>
            </div>
        </div>
        
        <!-- Powered by link -->
        <div class="powered-by">
            <a href="https://impact-agents.ai" target="_blank" rel="noopener noreferrer">
                Smart Agent Studio
            </a>
        </div>
    </div>
    
    <script src="/static/deployments/{deployment["id"]}/script.js"></script>
</body>
</html>'''

def generate_deployment_css(deployment: dict) -> str:
    """Generate CSS content for a deployment"""
    bg_color = deployment.get("backgroundColor", "#2563eb")
    text_color = deployment.get("textColor", "#ffffff")
    theme = deployment.get("chatTheme", "modern")
    
    return f'''/* Modern Chat Application Styles - Inspired by Professional UI */
:root {{
    --primary-color: {bg_color};
    --primary-text: {text_color};
    --bg-light: #f8fafc;
    --bg-dark: #1a202c;
    --surface-light: #ffffff;
    --surface-dark: #2d3748;
    --text-light: #2d3748;
    --text-dark: #f7fafc;
    --border-light: #e2e8f0;
    --border-dark: #4a5568;
    --shadow-light: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    --shadow-dark: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    --radius: 16px;
    --radius-small: 8px;
}}

* {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}}

body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background: var(--bg-light);
    height: 100vh;
    overflow: hidden;
    transition: all 0.3s ease;
}}

body.theme-dark {{
    background: var(--bg-dark);
}}

.chat-wrapper {{
    height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 20px;
    background: linear-gradient(135deg, var(--primary-color)15, var(--primary-color)05);
}}

.chat-container {{
    width: 100%;
    max-width: 900px;
    height: 700px;
    background: var(--surface-light);
    border-radius: var(--radius);
    box-shadow: var(--shadow-light);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    transition: all 0.3s ease;
}}

.theme-dark .chat-container {{
    background: var(--surface-dark);
    box-shadow: var(--shadow-dark);
}}

/* Header Styles */
.chat-header {{
    background: var(--primary-color);
    color: var(--primary-text);
    padding: 20px 24px;
    border-radius: var(--radius) var(--radius) 0 0;
}}

.chat-header-content {{
    display: flex;
    align-items: center;
    gap: 16px;
}}

.chat-logo {{
    width: 48px;
    height: 48px;
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    flex-shrink: 0;
    overflow: hidden;
}}

.chat-logo img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 12px;
}}

.chat-title-section {{
    flex: 1;
}}

.chat-title {{
    font-size: 20px;
    font-weight: 600;
    margin: 0 0 4px 0;
    line-height: 1.2;
}}

.chat-subtitle {{
    font-size: 14px;
    opacity: 0.9;
    margin: 0;
    line-height: 1.3;
}}

/* Messages Container */
.chat-messages-container {{
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
    min-height: 0; /* Important for flex child scrolling */
}}

.chat-messages {{
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
    scroll-behavior: smooth;
    min-height: 0; /* Important for flex child scrolling */
}}

.chat-messages::-webkit-scrollbar {{
    width: 6px;
}}

.chat-messages::-webkit-scrollbar-track {{
    background: transparent;
}}

.chat-messages::-webkit-scrollbar-thumb {{
    background: #cbd5e0;
    border-radius: 3px;
}}

.theme-dark .chat-messages::-webkit-scrollbar-thumb {{
    background: #4a5568;
}}

/* Message Styles */
.message {{
    display: flex;
    gap: 12px;
    max-width: 85%;
    animation: messageSlideIn 0.3s ease;
}}

.user-message {{
    align-self: flex-end;
    flex-direction: row-reverse;
}}

.message-avatar {{
    width: 36px;
    height: 36px;
    border-radius: 18px;
    background: var(--primary-color);
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--primary-text);
    font-size: 16px;
    flex-shrink: 0;
    overflow: hidden;
}}

.user-message .message-avatar {{
    background: #718096;
    color: white;
}}

.message-avatar img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 18px;
}}

.message-content {{
    background: #f7fafc;
    border: 1px solid var(--border-light);
    border-radius: 18px;
    border-top-left-radius: 6px;
    padding: 12px 16px;
    max-width: 100%;
    word-wrap: break-word;
    transition: all 0.2s ease;
}}

.theme-dark .message-content {{
    background: #4a5568;
    border-color: var(--border-dark);
    color: var(--text-dark);
}}

.user-message .message-content {{
    background: var(--primary-color);
    color: var(--primary-text);
    border-color: var(--primary-color);
    border-top-left-radius: 18px;
    border-top-right-radius: 6px;
}}

.message-text {{
    line-height: 1.5;
    margin: 0;
}}

.message-meta {{
    margin-top: 6px;
    opacity: 0.7;
}}

.message-time {{
    font-size: 11px;
    color: inherit;
}}

/* Execution Steps */
.execution-steps {{
    margin: 16px 24px;
    background: var(--surface-light);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-small);
    overflow: hidden;
}}

.theme-dark .execution-steps {{
    background: var(--surface-dark);
    border-color: var(--border-dark);
}}

.steps-header {{
    background: var(--bg-light);
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-light);
    font-size: 14px;
    font-weight: 600;
    color: var(--text-light);
}}

.theme-dark .steps-header {{
    background: var(--bg-dark);
    border-color: var(--border-dark);
    color: var(--text-dark);
}}

.steps-header i {{
    margin-right: 8px;
    color: var(--primary-color);
}}

.steps-list {{
    max-height: 200px;
    overflow-y: auto;
    overflow-x: hidden;
    scroll-behavior: smooth;
}}

.steps-list::-webkit-scrollbar {{
    width: 6px;
}}

.steps-list::-webkit-scrollbar-track {{
    background: transparent;
}}

.steps-list::-webkit-scrollbar-thumb {{
    background: #cbd5e0;
    border-radius: 3px;
}}

.theme-dark .steps-list::-webkit-scrollbar-thumb {{
    background: #4a5568;
}}

.steps-list-container {{
    flex: 1;
    overflow-y: auto;
    overflow-x: hidden;
    scroll-behavior: smooth;
}}

.steps-list-container::-webkit-scrollbar {{
    width: 6px;
}}

.steps-list-container::-webkit-scrollbar-track {{
    background: transparent;
}}

.steps-list-container::-webkit-scrollbar-thumb {{
    background: #cbd5e0;
    border-radius: 3px;
}}

.theme-dark .steps-list-container::-webkit-scrollbar-thumb {{
    background: #4a5568;
}}

.step-item {{
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border-light);
    transition: background-color 0.2s ease;
}}

.theme-dark .step-item {{
    border-color: var(--border-dark);
}}

.step-item:last-child {{
    border-bottom: none;
}}

.step-item.completed {{
    background: rgba(34, 197, 94, 0.05);
}}

.step-item.processing {{
    background: rgba(59, 130, 246, 0.05);
}}

.step-item.error {{
    background: rgba(239, 68, 68, 0.05);
}}

.step-icon {{
    width: 20px;
    height: 20px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    flex-shrink: 0;
    margin-top: 2px;
}}

.step-item.completed .step-icon {{
    background: #22c55e;
    color: white;
}}

.step-item.processing .step-icon {{
    background: #3b82f6;
    color: white;
}}

.step-item.error .step-icon {{
    background: #ef4444;
    color: white;
}}

.step-item.pending .step-icon {{
    background: #e5e7eb;
    color: #6b7280;
}}

.theme-dark .step-item.pending .step-icon {{
    background: #4b5563;
    color: #9ca3af;
}}

.step-content {{
    flex: 1;
}}

.step-title {{
    font-size: 13px;
    font-weight: 500;
    color: var(--text-light);
    margin-bottom: 4px;
}}

.theme-dark .step-title {{
    color: var(--text-dark);
}}

.step-description {{
    font-size: 12px;
    color: #6b7280;
    line-height: 1.4;
}}

.theme-dark .step-description {{
    color: #9ca3af;
}}

/* Tabbed Response */
.tabbed-response {{
    margin: 16px 24px;
    background: var(--surface-light);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-small);
    overflow: hidden;
    max-height: 500px;
    display: flex;
    flex-direction: column;
}}

.theme-dark .tabbed-response {{
    background: var(--surface-dark);
    border-color: var(--border-dark);
}}

.tab-headers {{
    display: flex;
    background: var(--bg-light);
    border-bottom: 1px solid var(--border-light);
}}

.theme-dark .tab-headers {{
    background: var(--bg-dark);
    border-color: var(--border-dark);
}}

.tab-header {{
    flex: 1;
    padding: 12px 16px;
    font-size: 13px;
    font-weight: 500;
    text-align: center;
    cursor: pointer;
    color: #6b7280;
    border-right: 1px solid var(--border-light);
    transition: all 0.2s ease;
}}

.theme-dark .tab-header {{
    color: #9ca3af;
    border-color: var(--border-dark);
}}

.tab-header:last-child {{
    border-right: none;
}}

.tab-header.active {{
    color: var(--primary-color);
    background: rgba(59, 130, 246, 0.05);
    border-bottom: 2px solid var(--primary-color);
    margin-bottom: -1px;
}}

.tab-header:hover:not(.active) {{
    background: rgba(59, 130, 246, 0.02);
    color: var(--text-light);
}}

.theme-dark .tab-header:hover:not(.active) {{
    color: var(--text-dark);
}}

.tab-content {{
    flex: 1;
    padding: 0;
    overflow: hidden;
    position: relative;
    min-height: 0;
}}

.tab-panel {{
    display: none;
    padding: 16px;
    height: 100%;
    overflow-y: auto;
    overflow-x: hidden;
    scroll-behavior: smooth;
}}

.tab-panel.active {{
    display: block;
}}

.tab-panel::-webkit-scrollbar {{
    width: 6px;
}}

.tab-panel::-webkit-scrollbar-track {{
    background: transparent;
}}

.tab-panel::-webkit-scrollbar-thumb {{
    background: #cbd5e0;
    border-radius: 3px;
}}

.theme-dark .tab-panel::-webkit-scrollbar-thumb {{
    background: #4a5568;
}}

.final-answer {{
    line-height: 1.6;
    color: var(--text-light);
    word-wrap: break-word;
    overflow-wrap: break-word;
}}

.theme-dark .final-answer {{
    color: var(--text-dark);
}}

.execution-summary {{
    color: var(--text-light);
}}

.theme-dark .execution-summary {{
    color: var(--text-dark);
}}

.execution-summary .summary-item {{
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid var(--border-light);
    font-size: 13px;
}}

.theme-dark .execution-summary .summary-item {{
    border-color: var(--border-dark);
}}

.execution-summary .summary-item:last-child {{
    border-bottom: none;
}}

/* Logs Panel */
.logs-content {{
    text-align: center;
    padding: 20px;
}}

.logs-info {{
    margin-bottom: 20px;
}}

.logs-info p {{
    margin-bottom: 8px;
    color: var(--text-light);
}}

.theme-dark .logs-info p {{
    color: var(--text-dark);
}}

.logs-info p strong {{
    font-size: 16px;
    color: var(--primary-color);
}}

.logs-link-container {{
    margin: 20px 0;
}}

.logs-link {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 12px 24px;
    background: var(--primary-color);
    color: var(--primary-text);
    text-decoration: none;
    border-radius: var(--radius-small);
    font-weight: 500;
    transition: all 0.2s ease;
}}

.logs-link:hover {{
    background: var(--primary-color)dd;
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}}

.logs-link i {{
    font-size: 14px;
}}

.logs-url {{
    margin-top: 16px;
    padding: 12px;
    background: var(--bg-light);
    border-radius: var(--radius-small);
    border: 1px solid var(--border-light);
}}

.theme-dark .logs-url {{
    background: var(--bg-dark);
    border-color: var(--border-dark);
}}

.logs-url small {{
    color: #6b7280;
    font-size: 12px;
}}

.theme-dark .logs-url small {{
    color: #9ca3af;
}}

.logs-url code {{
    background: rgba(0, 0, 0, 0.05);
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Monaco', 'Consolas', monospace;
    font-size: 11px;
    word-break: break-all;
}}

.theme-dark .logs-url code {{
    background: rgba(255, 255, 255, 0.05);
}}

/* Typing Indicator */
.typing-indicator {{
    padding: 0 24px;
}}

.typing-dots {{
    display: flex;
    gap: 4px;
    align-items: center;
    padding: 12px 16px;
}}

.typing-dots span {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #a0aec0;
    animation: typingPulse 1.4s infinite;
}}

.typing-dots span:nth-child(2) {{
    animation-delay: 0.2s;
}}

.typing-dots span:nth-child(3) {{
    animation-delay: 0.4s;
}}

/* Controls */
.chat-controls {{
    display: flex;
    gap: 8px;
    padding: 12px 24px;
    border-top: 1px solid var(--border-light);
    background: var(--surface-light);
    flex-wrap: wrap;
}}

.theme-dark .chat-controls {{
    border-color: var(--border-dark);
    background: var(--surface-dark);
}}

.chat-control-btn {{
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 8px 12px;
    background: transparent;
    border: 1px solid var(--border-light);
    border-radius: var(--radius-small);
    color: var(--text-light);
    font-size: 13px;
    cursor: pointer;
    transition: all 0.2s ease;
}}

.theme-dark .chat-control-btn {{
    border-color: var(--border-dark);
    color: var(--text-dark);
}}

.chat-control-btn:hover {{
    background: var(--primary-color)10;
    border-color: var(--primary-color);
}}

.chat-control-btn i {{
    font-size: 12px;
}}

/* Input Container */
.chat-input-container {{
    padding: 20px 24px;
    border-top: 1px solid var(--border-light);
    background: var(--surface-light);
    border-radius: 0 0 var(--radius) var(--radius);
}}

.theme-dark .chat-input-container {{
    border-color: var(--border-dark);
    background: var(--surface-dark);
}}

.chat-input-wrapper {{
    display: flex;
    align-items: center;
    background: var(--bg-light);
    border: 2px solid var(--border-light);
    border-radius: 25px;
    padding: 4px;
    transition: all 0.3s ease;
}}

.theme-dark .chat-input-wrapper {{
    background: var(--bg-dark);
    border-color: var(--border-dark);
}}

.chat-input-wrapper:focus-within {{
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px var(--primary-color)20;
}}

.chat-input {{
    flex: 1;
    border: none;
    background: transparent;
    padding: 12px 16px;
    font-size: 15px;
    outline: none;
    color: var(--text-light);
    line-height: 1.4;
}}

.theme-dark .chat-input {{
    color: var(--text-dark);
}}

.chat-input::placeholder {{
    color: #a0aec0;
}}

.send-button {{
    width: 42px;
    height: 42px;
    border-radius: 21px;
    border: none;
    background: var(--primary-color);
    color: var(--primary-text);
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    transition: all 0.2s ease;
    flex-shrink: 0;
}}

.send-button:hover:not(:disabled) {{
    background: var(--primary-color)dd;
    transform: scale(1.05);
}}

.send-button:disabled {{
    background: #cbd5e0;
    cursor: not-allowed;
    transform: none;
}}

.theme-dark .send-button:disabled {{
    background: #4a5568;
}}

.chat-status {{
    text-align: center;
    font-size: 12px;
    color: #718096;
    margin-top: 8px;
    opacity: 0.8;
}}

/* Powered By */
.powered-by {{
    margin-top: 12px;
    text-align: center;
}}

.powered-by a {{
    color: #718096;
    text-decoration: none;
    font-size: 12px;
    opacity: 0.7;
    transition: opacity 0.2s ease;
}}

.powered-by a:hover {{
    opacity: 1;
}}

/* Animations */
@keyframes messageSlideIn {{
    from {{
        opacity: 0;
        transform: translateY(10px);
    }}
    to {{
        opacity: 1;
        transform: translateY(0);
    }}
}}

@keyframes typingPulse {{
    0%, 60%, 100% {{
        transform: translateY(0);
        opacity: 0.4;
    }}
    30% {{
        transform: translateY(-8px);
        opacity: 1;
    }}
}}

/* Responsive Design */
@media (max-width: 768px) {{
    .chat-wrapper {{
        padding: 10px;
    }}
    
    .chat-container {{
        height: calc(100vh - 20px);
        max-width: none;
        border-radius: 12px;
    }}
    
    .chat-header {{
        padding: 16px 20px;
    }}
    
    .chat-messages {{
        padding: 20px 16px;
    }}
    
    .chat-controls {{
        padding: 12px 20px;
    }}
    
    .chat-input-container {{
        padding: 16px 20px;
    }}
    
    .message {{
        max-width: 95%;
    }}
    
    .chat-title {{
        font-size: 18px;
    }}
    
    .chat-control-btn {{
        font-size: 12px;
        padding: 6px 10px;
    }}
}}

@media (max-width: 480px) {{
    .chat-controls {{
        justify-content: center;
    }}
    
    .chat-control-btn {{
        flex: 1;
        justify-content: center;
        min-width: 0;
    }}
}}'''

def generate_deployment_js(deployment: dict) -> str:
    """Generate JavaScript content for a deployment"""
    agent_id = deployment["agentId"]
    enable_dark_mode = deployment.get("enableDarkMode", True)
    enable_clear_chat = deployment.get("enableClearChat", True)
    enable_voice = deployment.get("enableVoice", False)
    enable_typing = deployment.get("enableTyping", True)
    enable_timestamps = deployment.get("enableTimestamps", False)
    
    return f'''// Modern Chat Application JavaScript
class ModernChatApp {{
    constructor() {{
        // Configuration
        this.agentId = "{agent_id}";
        this.apiKey = "IMPCTUNIWHEOPDSFGWBHKIOHWHEBSZXAPWQDUR";
        this.enableDarkMode = {str(enable_dark_mode).lower()};
        this.enableClearChat = {str(enable_clear_chat).lower()};
        this.enableVoice = {str(enable_voice).lower()};
        this.enableTyping = {str(enable_typing).lower()};
        this.enableTimestamps = {str(enable_timestamps).lower()};
        
        // DOM Elements
        this.chatInput = document.getElementById('chatInput');
        this.sendButton = document.getElementById('sendButton');
        this.chatMessages = document.getElementById('chatMessages');
        this.chatStatus = document.getElementById('chatStatus');
        this.typingIndicator = document.getElementById('typingIndicator');
        this.executionSteps = document.getElementById('executionSteps');
        this.stepsList = document.getElementById('stepsList');
        this.clearChatBtn = document.getElementById('clearChatBtn');
        this.darkModeBtn = document.getElementById('darkModeBtn');
        this.voiceBtn = document.getElementById('voiceBtn');
        
        // State
        this.isConnected = false;
        this.isDarkMode = localStorage.getItem('chatTheme') === 'dark';
        this.isVoiceEnabled = false;
        this.messageHistory = [];
        this.currentSteps = [];
        this.stepCounter = 0;
        this.executionTimeSeconds = 0;
        
        this.init();
    }}
    
    init() {{
        this.bindEvents();
        this.initializeTheme();
        this.setWelcomeTimestamp();
        this.checkConnection();
    }}
    
    bindEvents() {{
        // Message sending
        this.sendButton?.addEventListener('click', () => this.sendMessage());
        
        // Enhanced Enter key handling
        this.chatInput?.addEventListener('keydown', (e) => {{
            if (e.key === 'Enter') {{
                if (e.shiftKey) {{
                    // Allow Shift+Enter for new line
                    return;
                }} else {{
                    // Prevent default and send message
                    e.preventDefault();
                    this.sendMessage();
                }}
            }}
        }});
        
        // Control buttons
        this.clearChatBtn?.addEventListener('click', () => this.clearChat());
        this.darkModeBtn?.addEventListener('click', () => this.toggleDarkMode());
        this.voiceBtn?.addEventListener('click', () => this.toggleVoice());
        
        // Auto-resize input
        this.chatInput?.addEventListener('input', () => this.autoResizeInput());
    }}
    
    initializeTheme() {{
        if (this.isDarkMode) {{
            document.body.classList.add('theme-dark');
            this.updateDarkModeButton(true);
        }}
    }}
    
    setWelcomeTimestamp() {{
        if (this.enableTimestamps) {{
            const welcomeTime = document.getElementById('welcomeTime');
            if (welcomeTime) {{
                welcomeTime.textContent = this.formatTime(new Date());
            }}
        }}
    }}
    
    async checkConnection() {{
        try {{
            this.isConnected = true;
            this.updateStatus('Ready to chat', 'success');
            this.enableInput();
        }} catch (error) {{
            this.updateStatus('Connection failed', 'error');
            console.error('Connection error:', error);
        }}
    }}
    
    updateStatus(message, type = 'info') {{
        if (this.chatStatus) {{
            this.chatStatus.textContent = message;
            this.chatStatus.className = `chat-status status-${{type}}`;
        }}
    }}
    
    enableInput() {{
        if (this.chatInput) {{
            this.chatInput.disabled = false;
            this.chatInput.focus();
        }}
        if (this.sendButton) {{
            this.sendButton.disabled = false;
        }}
    }}
    
    disableInput() {{
        if (this.chatInput) this.chatInput.disabled = true;
        if (this.sendButton) this.sendButton.disabled = true;
    }}
    
    async sendMessage() {{
        const message = this.chatInput?.value.trim();
        if (!message || !this.isConnected) return;
        
        // Add user message
        this.addMessage(message, 'user');
        this.chatInput.value = '';
        this.messageHistory.push({{ type: 'user', content: message, timestamp: new Date() }});
        
        // Update UI state
        this.disableInput();
        this.updateStatus('Thinking...', 'processing');
        
        // Initialize execution steps
        this.initializeSteps();
        
        // Show typing indicator
        if (this.enableTyping) {{
            this.showTypingIndicator();
        }}
        
        try {{
            await this.sendToAgent(message);
        }} catch (error) {{
            this.handleError('Failed to send message', error);
        }} finally {{
            this.hideTypingIndicator();
            this.hideSteps();
            this.enableInput();
            this.updateStatus('Ready to chat', 'success');
        }}
    }}
    
    async sendToAgent(userInput) {{
        const response = await fetch('/api/agent/execute/stream', {{
            method: 'POST',
            headers: {{
                'AGENT-API-KEY': this.apiKey,
                'Content-Type': 'application/json'
            }},
            body: JSON.stringify({{
                agentId: this.agentId,
                userInput: userInput
            }})
        }});
        
        if (!response.ok) {{
            throw new Error(`HTTP error! status: ${{response.status}}`);
        }}
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let finalAnswer = '';
        let logUrl = '';
        let executionSteps = [];
        
        try {{
            while (true) {{
                const {{ done, value }} = await reader.read();
                if (done) break;
                
                const chunk = decoder.decode(value);
                const lines = chunk.split('\\n').filter(line => line.trim());
                
                for (const line of lines) {{
                    try {{
                        const data = JSON.parse(line);
                        
                        if (data.status === 'complete' && data.result) {{
                            finalAnswer = this.extractContent(data.result);
                            logUrl = data.result.log_url || '';
                            this.executionTimeSeconds = data.execution_time_seconds || 0;
                            
                            if (finalAnswer) {{
                                // Complete all remaining steps
                                this.completeAllSteps();
                                
                                // Create tabbed response with logs
                                this.createTabbedResponse(finalAnswer, this.currentSteps, logUrl);
                                
                                this.messageHistory.push({{ 
                                    type: 'bot', 
                                    content: finalAnswer, 
                                    steps: [...this.currentSteps],
                                    logUrl: logUrl,
                                    executionTime: this.executionTimeSeconds,
                                    timestamp: new Date() 
                                }});
                            }}
                        }} else if (data.status === 'processing' && data.message) {{
                            // Update status and add step
                            this.updateStatus(data.message, 'processing');
                            this.addOrUpdateStep(data.message, 'processing');
                        }} else if (data.status === 'starting') {{
                            this.addOrUpdateStep(data.message || 'Starting execution...', 'processing');
                        }} else if (data.status === 'error') {{
                            this.addOrUpdateStep(data.message || 'Error occurred', 'error');
                        }}
                    }} catch (parseError) {{
                        console.warn('Failed to parse response chunk:', parseError);
                    }}
                }}
            }}
        }} catch (error) {{
            console.error('Stream reading error:', error);
            this.addOrUpdateStep('Connection error occurred', 'error');
            this.addMessage('Sorry, I encountered an error processing your request.', 'bot');
        }}
    }}
    
    extractContent(result) {{
        // Prioritize html_content for rendering
        if (result.html_content) {{
            return result.html_content;
        }}
        if (result.content) {{
            return result.content.html_content || result.content.text || '';
        }}
        return typeof result === 'string' ? result : '';
    }}
    
    addMessage(content, type, isHTML = false) {{
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${{type}}-message`;
        
        // Create avatar
        const avatar = this.createAvatar(type);
        messageDiv.appendChild(avatar);
        
        // Create message content
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        const messageText = document.createElement('div');
        messageText.className = 'message-text';
        
        if (isHTML && content.includes('<')) {{
            messageText.innerHTML = this.sanitizeHTML(content);
        }} else {{
            messageText.textContent = content;
        }}
        
        messageContent.appendChild(messageText);
        
        // Add timestamp if enabled
        if (this.enableTimestamps) {{
            const messageMeta = document.createElement('div');
            messageMeta.className = 'message-meta';
            const timeSpan = document.createElement('span');
            timeSpan.className = 'message-time';
            timeSpan.textContent = this.formatTime(new Date());
            messageMeta.appendChild(timeSpan);
            messageContent.appendChild(messageMeta);
        }}
        
        messageDiv.appendChild(messageContent);
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        return messageDiv;
    }}
    
    createAvatar(type) {{
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        
        if (type === 'bot') {{
            const logoUrl = "{deployment.get('logoUrl', '')}";
            if (logoUrl) {{
                avatar.innerHTML = `<img src="${{logoUrl}}" alt="Bot" onerror="this.style.display='none'; this.parentNode.innerHTML='<i class=\\"fas fa-robot\\"></i>';">`;
            }} else {{
                avatar.innerHTML = '<i class="fas fa-robot"></i>';
            }}
        }} else {{
            avatar.innerHTML = '<i class="fas fa-user"></i>';
        }}
        
        return avatar;
    }}
    
    showTypingIndicator() {{
        if (this.typingIndicator) {{
            this.typingIndicator.style.display = 'block';
            this.scrollToBottom();
        }}
    }}
    
    hideTypingIndicator() {{
        if (this.typingIndicator) {{
            this.typingIndicator.style.display = 'none';
        }}
    }}
    
    updateMessageContent(messageElement, content) {{
        const messageText = messageElement.querySelector('.message-text');
        if (messageText) {{
            messageText.textContent = content;
        }}
    }}
    
    clearChat() {{
        if (!this.enableClearChat) return;
        
        if (confirm('Are you sure you want to clear the chat history?')) {{
            // Clear messages except welcome message
            const messages = this.chatMessages.querySelectorAll('.message');
            messages.forEach((message, index) => {{
                if (index > 0) {{ // Keep welcome message (first message)
                    message.remove();
                }}
            }});
            
            this.messageHistory = [];
            this.updateStatus('Chat cleared', 'success');
            
            setTimeout(() => {{
                this.updateStatus('Ready to chat', 'success');
            }}, 2000);
        }}
    }}
    
    toggleDarkMode() {{
        if (!this.enableDarkMode) return;
        
        this.isDarkMode = !this.isDarkMode;
        document.body.classList.toggle('theme-dark', this.isDarkMode);
        localStorage.setItem('chatTheme', this.isDarkMode ? 'dark' : 'light');
        this.updateDarkModeButton(this.isDarkMode);
    }}
    
    updateDarkModeButton(isDark) {{
        if (this.darkModeBtn) {{
            const icon = this.darkModeBtn.querySelector('i');
            const text = this.darkModeBtn.querySelector('text') || this.darkModeBtn.childNodes[1];
            
            if (icon) {{
                icon.className = isDark ? 'fas fa-sun' : 'fas fa-moon';
            }}
            if (text && text.textContent) {{
                text.textContent = isDark ? ' Light Mode' : ' Dark Mode';
            }}
        }}
    }}
    
    toggleVoice() {{
        if (!this.enableVoice) return;
        
        this.isVoiceEnabled = !this.isVoiceEnabled;
        const icon = this.voiceBtn?.querySelector('i');
        const text = this.voiceBtn?.childNodes[1];
        
        if (icon) {{
            icon.className = this.isVoiceEnabled ? 'fas fa-microphone' : 'fas fa-microphone-slash';
        }}
        if (text && text.textContent) {{
            text.textContent = this.isVoiceEnabled ? ' Voice On' : ' Voice Off';
        }}
        
        this.updateStatus(
            this.isVoiceEnabled ? 'Voice enabled' : 'Voice disabled', 
            'info'
        );
    }}
    
    autoResizeInput() {{
        if (this.chatInput) {{
            this.chatInput.style.height = 'auto';
            this.chatInput.style.height = Math.min(this.chatInput.scrollHeight, 120) + 'px';
        }}
    }}
    
    formatTime(date) {{
        return date.toLocaleTimeString([], {{ hour: '2-digit', minute: '2-digit' }});
    }}
    
    sanitizeHTML(html) {{
        // Basic HTML sanitization - in production, use a proper library
        return html
            .replace(/<script[^>]*>.*?<\\/script>/gi, '')
            .replace(/<iframe[^>]*>.*?<\\/iframe>/gi, '')
            .replace(/javascript:/gi, '');
    }}
    
    handleError(message, error) {{
        console.error(message, error);
        this.addMessage(`Error: ${{message}}`, 'bot');
        this.updateStatus('Error occurred', 'error');
    }}
    
    scrollToBottom() {{
        if (this.chatMessages) {{
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }}
    }}
    
    scrollStepsToBottom() {{
        if (this.stepsList) {{
            this.stepsList.scrollTop = this.stepsList.scrollHeight;
        }}
    }}
    
    scrollTabToBottom(tabPanel) {{
        if (tabPanel) {{
            setTimeout(() => {{
                tabPanel.scrollTop = tabPanel.scrollHeight;
            }}, 100); // Small delay to ensure content is rendered
        }}
    }}
    
    // === Step Management Methods ===
    
    initializeSteps() {{
        this.currentSteps = [];
        this.stepCounter = 0;
        if (this.stepsList) {{
            this.stepsList.innerHTML = '';
        }}
        if (this.executionSteps) {{
            this.executionSteps.style.display = 'block';
        }}
        this.scrollToBottom();
    }}
    
    addOrUpdateStep(message, status = 'processing') {{
        if (!message) return;
        
        // Always add new step - don't update existing ones to preserve all steps
        const step = {{
            id: ++this.stepCounter,
            title: this.getStepTitle(message, status),
            description: message,
            status: status,
            timestamp: new Date()
        }};
        
        this.currentSteps.push(step);
        
        // Add to UI if stepsList exists
        if (this.stepsList) {{
            this.addStepElement(step, this.currentSteps.length - 1);
            // Auto-scroll steps list to bottom
            this.scrollStepsToBottom();
        }}
        
        // Mark previous step as completed if this is a new processing step
        if (status === 'processing' && this.currentSteps.length > 1) {{
            const prevStep = this.currentSteps[this.currentSteps.length - 2];
            if (prevStep.status === 'processing') {{
                prevStep.status = 'completed';
                if (this.stepsList) {{
                    this.updateStepElement(this.currentSteps.length - 2, prevStep);
                    this.scrollStepsToBottom();
                }}
            }}
        }}
        
        this.scrollToBottom();
    }}
    
    completeAllSteps() {{
        this.currentSteps.forEach((step, index) => {{
            if (step.status === 'processing') {{
                step.status = 'completed';
                this.updateStepElement(index, step);
            }}
        }});
    }}
    
    getStepTitle(message, status) {{
        if (status === 'error') return 'Error';
        if (message.length > 50) {{
            return message.substring(0, 47) + '...';
        }}
        return message;
    }}
    
    addStepElement(step, index) {{
        if (!this.stepsList) return;
        
        const stepElement = document.createElement('div');
        stepElement.className = `step-item ${{step.status}}`;
        stepElement.id = `step-${{step.id}}`;
        
        stepElement.innerHTML = `
            <div class="step-icon">
                ${{this.getStepIcon(step.status)}}
            </div>
            <div class="step-content">
                <div class="step-title">${{step.title}}</div>
                <div class="step-description">${{step.description}}</div>
            </div>
        `;
        
        this.stepsList.appendChild(stepElement);
    }}
    
    updateStepElement(index, step) {{
        const stepElement = document.getElementById(`step-${{step.id}}`);
        if (!stepElement) return;
        
        stepElement.className = `step-item ${{step.status}}`;
        
        const icon = stepElement.querySelector('.step-icon');
        const title = stepElement.querySelector('.step-title');
        const description = stepElement.querySelector('.step-description');
        
        if (icon) icon.innerHTML = this.getStepIcon(step.status);
        if (title) title.textContent = step.title;
        if (description) description.textContent = step.description;
    }}
    
    getStepIcon(status) {{
        switch (status) {{
            case 'completed': return '✓';
            case 'processing': return '⟳';
            case 'error': return '✗';
            default: return '○';
        }}
    }}
    
    hideSteps() {{
        if (this.executionSteps) {{
            setTimeout(() => {{
                this.executionSteps.style.display = 'none';
            }}, 2000); // Hide after 2 seconds
        }}
    }}
    
    // === Tabbed Response Methods ===
    
    createTabbedResponse(finalAnswer, steps, logUrl = '') {{
        const tabbedContainer = document.createElement('div');
        tabbedContainer.className = 'tabbed-response';
        
        // Create tab headers
        const tabHeaders = document.createElement('div');
        tabHeaders.className = 'tab-headers';
        
        const answerTab = document.createElement('div');
        answerTab.className = 'tab-header active';
        answerTab.textContent = 'Answer';
        answerTab.onclick = () => this.switchTab(tabbedContainer, 'answer');
        
        const stepsTab = document.createElement('div');
        stepsTab.className = 'tab-header';
        stepsTab.textContent = `Steps (${{steps.length}})`;
        stepsTab.onclick = () => this.switchTab(tabbedContainer, 'steps');
        
        tabHeaders.appendChild(answerTab);
        tabHeaders.appendChild(stepsTab);
        
        // Add logs tab if logUrl is available
        if (logUrl) {{
            const logsTab = document.createElement('div');
            logsTab.className = 'tab-header';
            logsTab.textContent = 'Logs';
            logsTab.onclick = () => this.switchTab(tabbedContainer, 'logs');
            tabHeaders.appendChild(logsTab);
        }}
        
        // Create tab content
        const tabContent = document.createElement('div');
        tabContent.className = 'tab-content';
        
        // Answer panel
        const answerPanel = document.createElement('div');
        answerPanel.className = 'tab-panel active';
        answerPanel.id = 'answer-panel';
        
        const answerDiv = document.createElement('div');
        answerDiv.className = 'final-answer';
        // Always render as HTML since we prioritize html_content
        answerDiv.innerHTML = this.sanitizeHTML(finalAnswer);
        answerPanel.appendChild(answerDiv);
        
        // Auto-scroll answer panel to bottom
        setTimeout(() => {{
            this.scrollTabToBottom(answerPanel);
        }}, 200);
        
        // Steps panel
        const stepsPanel = document.createElement('div');
        stepsPanel.className = 'tab-panel';
        stepsPanel.id = 'steps-panel';
        
        const summaryDiv = document.createElement('div');
        summaryDiv.className = 'execution-summary';
        
        const completedSteps = steps.filter(s => s.status === 'completed').length;
        // Use execution_time_seconds from the API response if available
        const totalTime = this.executionTimeSeconds || 
            (steps.length > 0 ? Math.round((steps[steps.length - 1].timestamp - steps[0].timestamp) / 1000) : 0);
        
        summaryDiv.innerHTML = `
            <div class="summary-item">
                <span>Total Steps:</span>
                <span>${{steps.length}}</span>
            </div>
            <div class="summary-item">
                <span>Completed:</span>
                <span>${{completedSteps}}</span>
            </div>
            <div class="summary-item">
                <span>Execution Time:</span>
                <span>${{totalTime}}s</span>
            </div>
        `;
        
        const stepsListDiv = document.createElement('div');
        stepsListDiv.className = 'steps-list-container';
        stepsListDiv.style.marginTop = '16px';
        stepsListDiv.style.flex = '1';
        stepsListDiv.style.overflowY = 'auto';
        stepsListDiv.style.overflowX = 'hidden';
        
        if (steps.length === 0) {{
            stepsListDiv.innerHTML = '<p style="text-align: center; color: #6b7280; padding: 20px;">No execution steps recorded</p>';
        }} else {{
            steps.forEach((step, index) => {{
                const stepDiv = document.createElement('div');
                stepDiv.className = `step-item ${{step.status}}`;
                stepDiv.innerHTML = `
                    <div class="step-icon">${{this.getStepIcon(step.status)}}</div>
                    <div class="step-content">
                        <div class="step-title">${{step.title || step.description}}</div>
                        <div class="step-description">${{step.description}}</div>
                    </div>
                `;
                stepsListDiv.appendChild(stepDiv);
            }});
        }}
        
        stepsPanel.appendChild(summaryDiv);
        stepsPanel.appendChild(stepsListDiv);
        
        // Auto-scroll steps panel to bottom
        setTimeout(() => {{
            this.scrollTabToBottom(stepsListDiv);
        }}, 200);
        
        tabContent.appendChild(answerPanel);
        tabContent.appendChild(stepsPanel);
        
        // Add logs panel if logUrl is available
        if (logUrl) {{
            const logsPanel = document.createElement('div');
            logsPanel.className = 'tab-panel';
            logsPanel.id = 'logs-panel';
            
            const logsContent = document.createElement('div');
            logsContent.className = 'logs-content';
            logsContent.innerHTML = `
                <div class="logs-info">
                    <p><strong>Execution Logs</strong></p>
                    <p>View detailed execution logs for this request:</p>
                </div>
                <div class="logs-link-container">
                    <a href="${{logUrl}}" target="_blank" rel="noopener noreferrer" class="logs-link">
                        <i class="fas fa-external-link-alt"></i>
                        Open Logs in New Tab
                    </a>
                </div>
                <div class="logs-url">
                    <small>Log URL: <code>${{logUrl}}</code></small>
                </div>
            `;
            
            logsPanel.appendChild(logsContent);
            tabContent.appendChild(logsPanel);
        }}
        
        tabbedContainer.appendChild(tabHeaders);
        tabbedContainer.appendChild(tabContent);
        
        // Add to chat messages
        this.chatMessages.appendChild(tabbedContainer);
        this.scrollToBottom();
    }}
    
    switchTab(container, tabName) {{
        // Update tab headers
        const headers = container.querySelectorAll('.tab-header');
        headers.forEach(header => header.classList.remove('active'));
        
        const activeHeader = Array.from(headers).find(h => 
            (tabName === 'answer' && h.textContent === 'Answer') ||
            (tabName === 'steps' && h.textContent.startsWith('Steps')) ||
            (tabName === 'logs' && h.textContent === 'Logs')
        );
        if (activeHeader) activeHeader.classList.add('active');
        
        // Update tab panels
        const panels = container.querySelectorAll('.tab-panel');
        panels.forEach(panel => panel.classList.remove('active'));
        
        const activePanel = container.querySelector(`#${{tabName}}-panel`);
        if (activePanel) {{
            activePanel.classList.add('active');
            // Auto-scroll to bottom when switching tabs
            this.scrollTabToBottom(activePanel);
        }}
    }}
}}

// Initialize the modern chat app
document.addEventListener('DOMContentLoaded', () => {{
    window.chatApp = new ModernChatApp();
}});

// Prevent form submission on Enter in input
document.addEventListener('keydown', (e) => {{
    if (e.key === 'Enter' && e.target.tagName === 'INPUT') {{
        e.preventDefault();
    }}
}});'''

async def cleanup_deployment_files(deployment_id: str):
    """Remove deployment files"""
    deployment_dir = f"static/deployments/{deployment_id}"
    try:
        import shutil
        if os.path.exists(deployment_dir):
            shutil.rmtree(deployment_dir)
    except Exception as e:
        logging.warning(f"Error cleaning up deployment files: {e}")

async def regenerate_all_deployments():
    """Regenerate files for all existing deployments"""
    try:
        deployments = load_ui_deployments()
        for deployment in deployments:
            logging.info(f"Regenerating files for deployment: {deployment['id']}")
            await generate_deployment_files(deployment)
        logging.info(f"Regenerated {len(deployments)} deployments")
    except Exception as e:
        logging.error(f"Error regenerating deployments: {e}")

# File to store agents
AGENTS_FILE = "agents.json"
MULTIAGENTS_FILE = "data/multiagents.json"
UI_DEPLOYMENTS_FILE = "data/ui_deployments.json"
CONNECTORS_FILE = "data/connectors.json"
AGENT_CACHE_FILE = "agent_cache.json"

# Flag to control whether session data is saved to files in the request_logs folder
SAVE_SESSION_FILE = False

# In-memory cache for agent_cache.json
_agent_cache_data = None
_agent_cache_loaded = False

def load_agents():
    # Try to get from cache first
    agents_data = cache.get('agents_data')
    if agents_data:
        logging.info("Agents loaded from cache")
        return json.loads(agents_data)
    
    # Cache miss - load from file
    if os.path.exists(AGENTS_FILE):
        with open(AGENTS_FILE, 'r') as f:
            agents = json.load(f)
            # Cache the loaded data
            cache.setex('agents_data', 604800, json.dumps(agents))
            logging.info("Agents loaded from file and cached")
            return agents
    return []

def save_agents(agents):
    # Save to file for persistence
    with open(AGENTS_FILE, 'w') as f:
        json.dump(agents, f)
    
    # Update cache
    cache.setex('agents_data', 604800, json.dumps(agents))
    logging.info("Agents saved to file and cache")

def load_multi_agents():
    logger.info(f"Attempting to load multi-agents from: {MULTIAGENTS_FILE}")
    try:
        # Try to get from cache first
        multi_agents_data = cache.get('multi_agents_data')
        if multi_agents_data:
            logger.info("Multi-agents loaded from cache")
            result = json.loads(multi_agents_data)
            if not isinstance(result, list):
                logger.error("Cached multi_agents_data is not a list")
                return []
            return result
        
        # Cache miss - load from file
        if os.path.exists(MULTIAGENTS_FILE):
            with open(MULTIAGENTS_FILE, 'r') as f:
                data = json.load(f)
                # Cache the loaded data
                cache.setex('multi_agents_data', 604800, json.dumps(data))
                logger.info("Multi-agents loaded from file and cached")
                return data
        else:
            logger.info(f"File not found: {MULTIAGENTS_FILE}")
            return []
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {MULTIAGENTS_FILE}: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading {MULTIAGENTS_FILE}: {e}")
        return []

def save_multi_agents(multi_agents):
    # Save to file for persistence
    with open(MULTIAGENTS_FILE, 'w') as f:
        json.dump(multi_agents, f, indent=4)
    # Update cache
    cache.setex('multi_agents_data', 604800, json.dumps(multi_agents))
    logger.info("Multi-agents saved to file and cache")

def load_connectors():
    connectors_dir = Path("data")
    connectors_dir.mkdir(parents=True, exist_ok=True)
    connectors_file = connectors_dir / "connectors.json"
    if connectors_file.exists():
        try:
            with open(connectors_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {connectors_file}")
            return []
    return []

def save_connectors(connectors):
    connectors_dir = Path("data")
    connectors_dir.mkdir(parents=True, exist_ok=True)
    connectors_file = connectors_dir / "connectors.json"
    # Save to file for persistence
    with open(connectors_file, 'w') as f:
        json.dump(connectors, f, indent=4)
    # Update cache
    cache.setex('connectors_data', 604800, json.dumps(connectors))
    logging.info("Connectors saved to file and cache")

def load_agent_cache():
    """Load agent cache from JSON file and store in memory"""
    global _agent_cache_data, _agent_cache_loaded
    
    # Return cached data if already loaded
    if _agent_cache_loaded and _agent_cache_data is not None:
        return _agent_cache_data
    
    # Load from file
    if os.path.exists(AGENT_CACHE_FILE):
        try:
            with open(AGENT_CACHE_FILE, 'r') as f:
                _agent_cache_data = json.load(f)
                _agent_cache_loaded = True
                logging.info(f"Agent cache loaded from {AGENT_CACHE_FILE}")
                return _agent_cache_data
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {AGENT_CACHE_FILE}: {e}")
            _agent_cache_data = {}
            _agent_cache_loaded = True
            return _agent_cache_data
        except Exception as e:
            logging.error(f"Error loading agent cache from {AGENT_CACHE_FILE}: {e}")
            _agent_cache_data = {}
            _agent_cache_loaded = True
            return _agent_cache_data
    else:
        logging.info(f"Agent cache file {AGENT_CACHE_FILE} not found, using empty cache")
        _agent_cache_data = {}
        _agent_cache_loaded = True
        return _agent_cache_data

def get_cached_agent_response(agent_id: str, user_input: str):
    """Check if there's a cached response for the given agent_id and user_input (case-insensitive)"""
    agent_cache = load_agent_cache()
    
    # Check if agent_id exists in cache
    if agent_id not in agent_cache:
        return None
    
    # Normalize user_input to lowercase for case-insensitive comparison
    user_input_normalized = user_input.lower().strip() if user_input else ""
    
    # Get the list of cached entries for this agent
    cached_entries = agent_cache[agent_id]
    
    # Check if any entry matches the user_input (case-insensitive)
    for entry in cached_entries:
        cached_user_input = entry.get("userInput", "")
        cached_user_input_normalized = cached_user_input.lower().strip() if cached_user_input else ""
        
        if cached_user_input_normalized == user_input_normalized:
            logging.info(f"Found cached response for agent {agent_id} with input: {user_input[:50]}...")
            return entry.get("result")
    
    return None

def load_tools() -> List[Tool]:
    return []
    # Try to get from cache first
    tools_data = cache.get('built_in_tools')
    if tools_data:
        logging.info("Built-in tools loaded from cache")
        return [Tool(**tool_data) for tool_data in json.loads(tools_data)]
    
    # Cache miss - try loading from file
    try:
        with open('tools.json', 'r') as f:
            tools_data = json.load(f)
            tools = [Tool(**tool) for tool in tools_data]
            # Cache the loaded data
            cache.setex('built_in_tools', 604800, json.dumps([tool.dict() for tool in tools]))
            logging.info("Built-in tools loaded from file and cached")
            return tools
    except FileNotFoundError:
        # File not found - use default tools
        tools = [
            Tool(
                id="github",
                name="GitHub",
                description="Connect your GitHub repositories to automate code analysis, PR reviews, and issue management.",
                tags=["Code", "Version Control", "Automation"]
            ),
            Tool(
                id="slack",
                name="Slack",
                description="Integrate with Slack to receive notifications and interact with your workspace through chat commands.",
                tags=["Communication", "Notifications", "Chat"]
            ),
            Tool(
                id="discord",
                name="Discord",
                description="Connect your Discord server to manage community interactions and automate moderation tasks.",
                tags=["Community", "Chat", "Gaming"]
            ),
            Tool(
                id="clickup",
                name="ClickUp",
                description="Integrate with ClickUp to manage tasks, track progress, and automate project workflows.",
                tags=["Project Management", "Tasks", "Productivity"]
            ),
            Tool(
                id="spotify",
                name="Spotify",
                description="Control Spotify playback and manage playlists through automated commands.",
                tags=["Music", "Entertainment", "Media"]
            ),
            Tool(
                id="twitter",
                name="Twitter",
                description="Automate tweet scheduling, monitoring, and engagement with your Twitter audience.",
                tags=["Social Media", "Marketing", "Automation"]
            ),
            Tool(
                id="notion",
                name="Notion",
                description="Connect with Notion to manage documents, databases, and knowledge bases automatically.",
                tags=["Knowledge Base", "Documentation", "Organization"]
            ),
            Tool(
                id="outlook",
                name="Outlook",
                description="Integrate with Outlook to manage emails, calendar events, and contacts programmatically.",
                tags=["Email", "Calendar", "Communication"]
            )
        ]

def save_tools(tools: List[Tool]):
    # Save to file
    with open('tools.json', 'w') as f:
        json.dump([tool.dict() for tool in tools], f, indent=4)
    
    # Update cache
    cache.setex('built_in_tools', 604800, json.dumps([tool.dict() for tool in tools]))
    logging.info("Built-in tools saved to file and cache")

def load_custom_tools() -> List[Tool]:
    try:
        with open('custom_tools.json', 'r') as f:
            tools_data = json.load(f)
            return [Tool(**tool) for tool in tools_data]
    except FileNotFoundError:
        return []

def save_custom_tools(tools: List[Tool]):
    # Convert tools to dict for serialization
    tools_data = [tool.dict() for tool in tools]
    # Save to file for persistence
    with open('custom_tools.json', 'w') as f:
        json.dump(tools_data, f, indent=2)
    # Update cache
    cache.setex('custom_tools_data', 604800, json.dumps(tools_data))
    logging.info("Custom tools saved to file and cache")

# --- Data Connector Models and APIs ---

class PostgresConnectionConfig(BaseModel):
    uniqueName: str
    vectorStoreUser: str
    vectorStoreHost: str
    vectorStorePassword: str
    vectorStorePort: str
    vectorStoreDBName: str
    connectorType: str = 'postgres'
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class BigQueryConnectionConfig(BaseModel):
    uniqueName: str
    projectId: str
    datasetId: str
    # Optional: when empty, BigQuery client falls back to Application Default Credentials
    # (compute engine / Cloud Run attached identity).
    serviceAccountKey: Dict[str, Any] = Field(default_factory=dict)
    connectorType: str = 'bigquery'
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class PostgresConnectionTest(BaseModel):
    type: str
    config: Dict[str, Any]



RECAPTCHA_SECRET_KEY = "6LeG1VYrAAAAAB2EW7-bv-t3j-SA3ix5b466u5dO"
JWT_SECRET_KEY = "EXTERFGHTDFEREFGAAWEESHUOIFDSD#$QWERTYUIOP"  # change it to cutom token for guests

# "/api/generate_captcha_token"
async def verify_recaptcha(token: str):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            'https://www.google.com/recaptcha/api/siteverify',
            data={
                'secret': RECAPTCHA_SECRET_KEY,
                'response': token
            }
        )
        result = resp.json()
        if not result.get("success") or result.get("score", 0) < 0.5:
            raise HTTPException(status_code=403, detail="reCAPTCHA verification failed")

@app.post("/api/generate_captcha_token")
@limiter.limit("10/minute")
async def submigenerate_captcha_tokent_form(request: Request, data: dict):
    token = data.get('token')
    await verify_recaptcha(token)
    
    # Get client IP
    client_ip = request.client.host
    
    # Generate JWT token with 30-minute expiry
    expiry = datetime.utcnow() + timedelta(minutes=30)
    jwt_token = jwt.encode(
        {
            'ip': client_ip,
            'exp': expiry,
            'type' : 'guest'
        },
        JWT_SECRET_KEY,
        algorithm='HS256'
    )
    
    return {
        "message": "reCAPTCHA passed. Form accepted.",
        "token": jwt_token,
        "expires": expiry.isoformat()
    }

@app.post("/api/data-connectors", status_code=201)
@limiter.limit("10/minute")
async def save_data_connector(request: Request, connector_config: Union[PostgresConnectionConfig, BigQueryConnectionConfig]):
    if connector_config.connectorType not in ['postgres', 'bigquery']:
        raise HTTPException(status_code=400, detail="Only postgres and bigquery connectors are supported currently.")
        
    connectors = load_connectors()
    
    if any(c.get('uniqueName') == connector_config.uniqueName for c in connectors):
        raise HTTPException(status_code=409, detail=f"Connector with name '{connector_config.uniqueName}' already exists.")

    new_connector_dict = connector_config.dict()
    new_connector_dict['id'] = str(uuid.uuid4())
    new_connector_dict['createdAt'] = datetime.now(timezone.utc).isoformat()
    
    connectors.append(new_connector_dict)
    save_connectors(connectors)
    
    return new_connector_dict

@app.get("/api/data-connectors", response_model=List[Union[PostgresConnectionConfig, BigQueryConnectionConfig]])
@limiter.limit("60/minute")
async def get_data_connectors(request: Request):
    connectors = load_connectors()
    return connectors

@app.put("/api/data-connectors/{connector_id}", response_model=Union[PostgresConnectionConfig, BigQueryConnectionConfig])
@limiter.limit("10/minute")
async def update_data_connector(request: Request, connector_id: str, connector_config: Union[PostgresConnectionConfig, BigQueryConnectionConfig]):
    if not connector_id:
        raise HTTPException(status_code=400, detail="Connector ID is required")

    connectors = load_connectors()
    
    connector_index = None
    for i, connector in enumerate(connectors):
        if connector.get('id') == connector_id:
            connector_index = i
            break
    
    if connector_index is None:
        raise HTTPException(status_code=404, detail=f"Connector with ID '{connector_id}' not found.")

    name_conflict = any(
        c['uniqueName'] == connector_config.uniqueName and c['id'] != connector_id 
        for c in connectors
    )
    if name_conflict:
        raise HTTPException(
            status_code=409, 
            detail=f"Connector with name '{connector_config.uniqueName}' already exists."
        )

    existing_connector = connectors[connector_index]
    updated_connector = connector_config.dict()
    updated_connector['id'] = connector_id
    updated_connector['createdAt'] = existing_connector.get('createdAt')

    # For Postgres, preserve password if not provided
    if connector_config.connectorType == 'postgres' and not updated_connector.get('vectorStorePassword'):
        updated_connector['vectorStorePassword'] = existing_connector.get('vectorStorePassword', '')
    # For BigQuery, an empty serviceAccountKey is a legitimate value ("use ADC") — do NOT preserve the old key.

    connectors[connector_index] = updated_connector
    save_connectors(connectors)
    
    if connector_config.connectorType == 'postgres':
        return PostgresConnectionConfig(**updated_connector)
    else:
        return BigQueryConnectionConfig(**updated_connector)

@app.delete("/api/data-connectors/{connector_id}")
@limiter.limit("10/minute")
async def delete_data_connector(request: Request, connector_id: str):
    # Check if any advanced tools are using this connector
    # advanced_tools = load_advanced_tools()
    # tools_using_connector = [tool.name for tool in advanced_tools if tool.data_connector_id == connector_id]
    
    tools_using_connector = []

    # Check if any custom tools are using this connector
    custom_tools = load_custom_tools()
    tools_using_connector.extend([tool.name for tool in custom_tools if tool.data_connector_id == connector_id])
    
    # If tools are using this connector, prevent deletion
    if tools_using_connector:
        tool_names = ", ".join(tools_using_connector)
        raise HTTPException(
            status_code=400,
            detail=f"Cannot delete connector because it is being used by the following tool: {tool_names}"
        )
    
    # If no tools are using the connector, proceed with deletion
    connectors = load_connectors()
    connectors = [c for c in connectors if c["id"] != connector_id]
    save_connectors(connectors)
    return {"message": "Connector deleted successfully"}

if ENABLE_AGENT_RUN:
    @app.post("/api/data-connectors/test")
    @limiter.limit("20/minute")
    async def test_connection(request: Request, connection_data: PostgresConnectionTest):
        if connection_data.type == "postgres":
            config = connection_data.config
            required_fields = ['host', 'port', 'database', 'user']
            missing_fields = [field for field in required_fields if not config.get(field)]
            
            if missing_fields:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required fields: {', '.join(missing_fields)}"
                )

            try:
                try:
                    port = int(config['port'])
                except ValueError:
                    raise HTTPException(status_code=400, detail="Port must be a valid number")

                conn = psycopg2.connect(
                    host=config['host'],
                    port=port,
                    database=config['database'],
                    user=config['user'],
                    password=config.get('password', ''),
                    connect_timeout=10
                )

                try:
                    with conn.cursor() as cur:
                        cur.execute('SELECT version();')
                        version = cur.fetchone()[0]
                
                    conn.close()

                    return {
                        "status": "success",
                        "message": "Connection successful",
                        "details": {
                            "version": version,
                            "connected_to": f"{config['host']}:{config['port']}/{config['database']}"
                        }
                    }

                except PostgresError as e:
                    if not conn.closed:
                        conn.close()
                    raise HTTPException(
                        status_code=400,
                        detail=f"Database query failed: {str(e)}"
                    )

            except PostgresError as e:
                error_message = str(e)
                if "timeout expired" in error_message.lower():
                    error_message = "Connection timed out. Please check if the database is accessible and the host/port are correct."
                elif "password authentication failed" in error_message.lower():
                    error_message = "Authentication failed. Please check your username and password."
                elif "database" in error_message.lower() and "does not exist" in error_message.lower():
                    error_message = f"Database '{config['database']}' does not exist."
                elif "could not connect to server" in error_message.lower():
                    error_message = "Could not connect to server. Please check if the host and port are correct and the server is running."

                raise HTTPException(
                    status_code=400,
                    detail=error_message
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"An unexpected error occurred: {str(e)}"
                )
        elif connection_data.type == "bigquery":
            config = connection_data.config
            required_fields = ['projectId', 'datasetId']
            missing_fields = [field for field in required_fields if not config.get(field)]
            
            if missing_fields:
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required fields: {', '.join(missing_fields)}"
                )

            try:
                service_account_info = config.get('serviceAccountKey')
                # Parse service account key JSON if provided (may be empty when using ADC)
                if service_account_info:
                    try:
                        if isinstance(service_account_info, str):
                            # Handle backward compatibility with string format
                            service_account_info = json.loads(service_account_info)
                    except (json.JSONDecodeError, TypeError) as e:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid service account key format: {str(e)}"
                        )
                else:
                    service_account_info = None

                # Initialize BigQuery client (explicit SA credentials when provided, else ADC)
                try:
                    if service_account_info:
                        credentials = service_account.Credentials.from_service_account_info(service_account_info)
                        client = bigquery.Client(credentials=credentials, project=config['projectId'])
                    else:
                        client = bigquery.Client(project=config['projectId'])
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to initialize BigQuery client: {str(e)}"
                    )

                # Test dataset access
                try:
                    dataset_ref = client.dataset(config['datasetId'])
                    client.get_dataset(dataset_ref)
                    return {
                        "status": "success",
                        "message": "Connection successful",
                        "details": {
                            "project": config['projectId'],
                            "dataset": config['datasetId']
                        }
                    }
                except Exception as e:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to access dataset: {str(e)}"
                    )
                finally:
                    client.close()

            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"An unexpected error occurred: {str(e)}"
                )
        else:
            raise HTTPException(status_code=400, detail="Unsupported connector type")

# --- API Endpoints ---


def delete_session_cookie(response: Response, cookie_name: str = "session_id"):
    """Deletes the session_id cookie."""
    response.set_cookie(
        key=cookie_name,
        value="",
        path="/",
        expires="Thu, 01 Jan 1970 00:00:00 GMT",
        secure=True,
        samesite="Lax",
        httponly=False
    )


@app.get("/reset-session")
@limiter.limit("10/minute")
async def reset_session(request: Request, response: Response):
    session_id = request.cookies.get("session_id") or request.query_params.get("session_id") or request.headers.get("X-Session-ID")
    new_session_id = str(uuid.uuid4())
    if session_id:
        delete_session_cookie(response=response, cookie_name="session_id")
        logger.info(f"Cleared session history for session_id: {session_id}")
        return {"message": "Session reset successfully", "new_session_id": new_session_id}
    else:
        return {"message": "No session_id found", "new_session_id": new_session_id}

@app.get("/api/agents")
@limiter.limit("60/minute")
async def get_agents(request: Request):
    return load_agents()

@app.post("/api/agents")
@limiter.limit("30/minute")
async def create_agent(request: Request, agent: AgentCreate):
    agents = load_agents()
    new_agent = Agent(
        id=str(uuid.uuid4()),
        **agent.dict()
    )
    agents.append(new_agent.dict())
    save_agents(agents)
    return new_agent

@app.get("/api/agents/{agent_id}")
@limiter.limit("60/minute")
async def get_agent(request: Request, agent_id: str, response: Response, reset: Optional[bool] = Query(False)):
    session_id = request.cookies.get("session_id") or request.query_params.get("session_id") or request.headers.get("X-Session-ID")
    if session_id and reset:
        delete_session_cookie(response=response, cookie_name="session_id")
        logger.info(f"Cleared session history for session_id: {session_id}")

    agents = load_agents()
    for agent in agents:
        if agent["id"] == agent_id:
            return agent
    raise HTTPException(status_code=404, detail="Agent not found")

@app.put("/api/agents/{agent_id}")
@limiter.limit("30/minute")
async def update_agent(request: Request, agent_id: str, updated_agent: AgentCreate):
    agents = load_agents()
    for i, agent in enumerate(agents):
        if agent["id"] == agent_id:
            agents[i] = {
                "id": agent_id,
                **updated_agent.dict()
            }
            save_agents(agents)
            return agents[i]
    raise HTTPException(status_code=404, detail="Agent not found")

@app.delete("/api/agents/{agent_id}")
@limiter.limit("30/minute")
async def delete_agent(request: Request, agent_id: str):
    agents = load_agents()
    agents = [agent for agent in agents if agent["id"] != agent_id]
    save_agents(agents)
    return {"message": "Agent deleted"}

# ============================================================================
# DEPRECATED: Store Hub APIs - Moved to DDD Architecture
# ============================================================================
# These routes have been refactored and moved to:
# backend/applications/store_hub/
#
# The new architecture follows Domain-Driven Design principles with:
# - API Layer: applications/store_hub/api/routes.py
# - Application Layer: applications/store_hub/application/store_service.py
# - Domain Layer: applications/store_hub/domain/
# - Infrastructure Layer: applications/store_hub/infrastructure/
#
# The routes are now registered via the store_hub_router in the app setup above.
# ============================================================================

# @app.get("/api/agents/custom/store-hub/graph")
# @limiter.limit("60/minute")
# async def get_store_hub_graph(
#     request: Request,
#     district_id: Optional[str] = Query(None),
#     store_id: Optional[str] = Query(None),
#     db: AsyncSession = Depends(get_db),
#     user: Dict[str, Any] = Depends(get_tenant_auth_user)
# ):
#     """
#     Get store hub dashboard graph data
#     
#     **Requires**: User authentication from log_users table (Bearer token from /api/sessions-login)
#     
#     Args:
#         district_id: The district ID filter (optional)
#         store_id: The store ID filter (optional)
#         user: Authenticated user from log_users table (automatically injected via dependency)
#     
#     Returns:
#         Dashboard data including KPIs, charts, and store information
#     """
#     
    # If district_id is provided, make API call to REVV analysis service
#     request_data = {
#         "db_config": {
#             "db_type": "bigquery",
#             "project_id": "tools-universe-29042025",
#             "database": "petsupplyplus",
#             "credentials_json": get_bigquery_credentials()
#         }
#     }
#     if store_id:
#         request_data["store_id"] = store_id
#     if district_id:
#         request_data["district"] = district_id
#     try:
        # Increase timeout to 120 seconds for heavy analysis operations
#         timeout = httpx.Timeout(120.0, connect=10.0)
#         async with httpx.AsyncClient(timeout=timeout) as client:
#             logger.info(f"Calling REVV analysis API with district_id={district_id}, store_id={store_id}")
#             revv_response = await client.post(
#                 "https://tools-universe.devs.impact-agents.ai/internal/revv_analysis",
#                 headers={
#                     "Content-Type": "application/json",
#                     "Authorization": f"Bearer {TOOLS_UNIVERSE_API_KEY}"
#                 },
#                 json=request_data
#             )
#             logger.info(f"REVV API response status: {revv_response.status_code}")
#             revv_response.raise_for_status()
#             response_data = revv_response.json()
#             logger.info(f"REVV API response received successfully")
#             return response_data
#     except httpx.TimeoutException as e:
#         logger.error(f"Timeout calling REVV analysis API after 120 seconds: {str(e)}")
#         import traceback
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         raise HTTPException(
#             status_code=504,
#             detail=f"REVV analysis API request timed out after 120 seconds. The analysis may still be processing in the background."
#         )
#     except httpx.ConnectError as e:
#         logger.error(f"Connection error calling REVV analysis API: {str(e)}")
#         import traceback
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         raise HTTPException(
#             status_code=503,
#             detail=f"Could not connect to REVV analysis API at localhost:8001. Please ensure the service is running."
#         )
#     except httpx.HTTPStatusError as e:
#         logger.error(f"HTTP error calling REVV analysis API: Status {e.response.status_code}, {str(e)}")
#         logger.error(f"Response text: {e.response.text[:500] if hasattr(e.response, 'text') else 'N/A'}")
#         import traceback
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         raise HTTPException(
#             status_code=e.response.status_code,
#             detail=f"REVV analysis API returned error {e.response.status_code}: {e.response.text[:200] if hasattr(e.response, 'text') else str(e)}"
#         )
#     except httpx.HTTPError as e:
#         logger.error(f"HTTP error calling REVV analysis API: {str(e)}")
#         import traceback
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to fetch REVV analysis data: {str(e)}"
#         )
#     except Exception as e:
#         logger.error(f"Unexpected error calling REVV analysis API: {str(e)}")
#         import traceback
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Unexpected error: {str(e)}"
#         )
#     
    # Hardcoded response data (when no district_id is provided)
#     return {
#         "storeFilter": {
#             "currentSelection": "All Stores",
#             "options": ["All Stores", "PSP Downtown", "Store 1", "Store 2"]
#         },
#         "districtFilter": {
#             "currentSelection": "Miller",
#             "options": ["Miller"]
#         },
#         "kpiCards": [
#             {
#                 "id": "kpi_1",
#                 "title": "Net Sales Comp (QTD)",
#                 "icon": "sales",
#                 "value": "$5.7M",
#                 "trend": {
#                     "value": 30,
#                     "direction": "up",
#                     "label": "Vs LY"
#                 },
#                 "sparklineData": [5, 10, 8, 15, 12, 18, 20]
#             },
#             {
#                 "id": "kpi_2",
#                 "title": "REVV Overall Score",
#                 "icon": "score",
#                 "value": "85/100",
#                 "status": "Good",
#                 "sparklineData": None
#             },
#             {
#                 "id": "kpi_3",
#                 "title": "NRR Audit Average",
#                 "icon": "audit",
#                 "value": "$5.7M",
#                 "trend": {
#                     "value": 22,
#                     "direction": "up",
#                     "label": "Vs LY"
#                 },
#                 "sparklineData": [15, 12, 18, 10, 14, 13, 16]
#             },
#             {
#                 "id": "kpi_4",
#                 "title": "Labor SPH (Sales Per Hour)",
#                 "icon": "labor",
#                 "value": "$5.7M",
#                 "trend": {
#                     "value": 30,
#                     "direction": "up",
#                     "label": "Vs LY"
#                 },
#                 "sparklineData": [8, 9, 12, 11, 14, 10, 15]
#             }
#         ],
#         "topPriorityStores": {
#             "title": "Top Priority Stores",
#             "subtitle": "Action Required",
#             "stores": [
#                 {
#                     "id": "store_A",
#                     "name": "PSP Downtown",
#                     "location": "Downtown, Downtown, U...",
#                     "tags": ["Low Rev Store", "Inventory Issue", "Action Required"]
#                 },
#                 {
#                     "id": "store_B",
#                     "name": "PSP Downtown",
#                     "location": "Downtown, Downtown, U...",
#                     "tags": ["Low Rev Store", "Inventory Issue", "Action Required"]
#                 }
#             ]
#         },
#         "topNegativeThemes": {
#             "title": "Top Negative Customer Theme",
#             "subtitle": "Most frequent customer concern",
#             "items": [
#                 {
#                     "id": "theme_001",
#                     "title": "Out of Stock",
#                     "percentage": 15,
#                     "description": "15% of negative comments",
#                     "tags": ["PSP Downtown", "PSP Downtown"]
#                 },
#                 {
#                     "id": "theme_002",
#                     "title": "Service Issues",
#                     "percentage": 15,
#                     "description": "15% of negative comments",
#                     "tags": ["PSP Downtown"]
#                 }
#             ]
#         },
#         "performanceChart": {
#             "title": "Performance Progress Chart",
#             "subtitle": "Most frequent customer concern",
#             "chart": {
#                 "title": "Sales Comp",
#                 "xAxisLabel": "Category",
#                 "dataPoints": [
#                     {"x": -800000, "y": 40000},
#                     {"x": -700000, "y": -50000},
#                     {"x": -500000, "y": 0},
#                     {"x": -400000, "y": -80000},
#                     {"x": -300000, "y": -30000},
#                     {"x": -200000, "y": 40000},
#                     {"x": 200000, "y": -20000},
#                     {"x": 600000, "y": -10000},
#                     {"x": 700000, "y": -60000},
#                     {"x": 800000, "y": -40000}
#                 ]
#             }
#         }
#     }
# 
# @app.get("/api/agents/custom/store-hub/store/graph")
# @limiter.limit("60/minute")
# async def get_store_hub_graph(
#     request: Request,
#     district_id: Optional[str] = Query(None),
#     store_id: Optional[str] = Query(None),
#     db: AsyncSession = Depends(get_db),
#     user: Dict[str, Any] = Depends(get_tenant_auth_user)
# ):
#     """
#     Get store hub dashboard graph data
#     
#     **Requires**: User authentication from log_users table (Bearer token from /api/sessions-login)
#     
#     Args:
#         district_id: The district ID filter (optional)
#         store_id: The store ID filter (optional)
#         user: Authenticated user from log_users table (automatically injected via dependency)
#     
#     Returns:
#         Dashboard data including KPIs, charts, and store information
#     """
#     
    # If district_id is provided, make API call to REVV analysis service
#     request_data = {
#         "db_config": {
#             "db_type": "bigquery",
#             "project_id": "tools-universe-29042025",
#             "database": "petsupplyplus",
#             "credentials_json": get_bigquery_credentials()
#         }
#     }
#     if store_id:
#         request_data["store_id"] = store_id
#     if district_id:
#         request_data["district"] = district_id
#     try:
        # Increase timeout to 120 seconds for heavy analysis operations
#         timeout = httpx.Timeout(120.0, connect=10.0)
#         async with httpx.AsyncClient(timeout=timeout) as client:
#             logger.info(f"Calling store deep dive API with district_id={district_id}, store_id={store_id}")
#             revv_response = await client.post(
#                 "https://tools-universe.devs.impact-agents.ai/internal/store_deep_dive",
#                 headers={
#                     "Content-Type": "application/json",
#                     "Authorization": f"Bearer {TOOLS_UNIVERSE_API_KEY}"
#                 },
#                 json=request_data
#             )
#             logger.info(f"Store deep dive API response status: {revv_response.status_code}")
#             revv_response.raise_for_status()
#             response_data = revv_response.json()
#             logger.info(f"Store deep dive API response received successfully")
#             return response_data
#     except httpx.TimeoutException as e:
#         logger.error(f"Timeout calling store deep dive API after 120 seconds: {str(e)}")
#         import traceback
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         raise HTTPException(
#             status_code=504,
#             detail=f"Store deep dive API request timed out after 120 seconds. The analysis may still be processing in the background."
#         )
#     except httpx.ConnectError as e:
#         logger.error(f"Connection error calling store deep dive API: {str(e)}")
#         import traceback
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         raise HTTPException(
#             status_code=503,
#             detail=f"Could not connect to store deep dive API. Please ensure the service is running."
#         )
#     except httpx.HTTPStatusError as e:
#         logger.error(f"HTTP error calling store deep dive API: Status {e.response.status_code}, {str(e)}")
#         logger.error(f"Response text: {e.response.text[:500] if hasattr(e.response, 'text') else 'N/A'}")
#         import traceback
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         raise HTTPException(
#             status_code=e.response.status_code,
#             detail=f"Store deep dive API returned error {e.response.status_code}: {e.response.text[:200] if hasattr(e.response, 'text') else str(e)}"
#         )
#     except httpx.HTTPError as e:
#         logger.error(f"HTTP error calling store deep dive API: {str(e)}")
#         import traceback
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to fetch store deep dive data: {str(e)}"
#         )
#     except Exception as e:
#         logger.error(f"Unexpected error calling store deep dive API: {str(e)}")
#         import traceback
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Unexpected error: {str(e)}"
#         )
#     
    # Hardcoded response data (when no district_id is provided)
#     return {
#         "storeFilter": {
#             "currentSelection": "All Stores",
#             "options": ["All Stores", "PSP Downtown", "Store 1", "Store 2"]
#         },
#         "districtFilter": {
#             "currentSelection": "Miller",
#             "options": ["Miller"]
#         },
#         "kpiCards": [
#             {
#                 "id": "kpi_1",
#                 "title": "Net Sales Comp (QTD)",
#                 "icon": "sales",
#                 "value": "$5.7M",
#                 "trend": {
#                     "value": 30,
#                     "direction": "up",
#                     "label": "Vs LY"
#                 },
#                 "sparklineData": [5, 10, 8, 15, 12, 18, 20]
#             },
#             {
#                 "id": "kpi_2",
#                 "title": "REVV Overall Score",
#                 "icon": "score",
#                 "value": "85/100",
#                 "status": "Good",
#                 "sparklineData": None
#             },
#             {
#                 "id": "kpi_3",
#                 "title": "NRR Audit Average",
#                 "icon": "audit",
#                 "value": "$5.7M",
#                 "trend": {
#                     "value": 22,
#                     "direction": "up",
#                     "label": "Vs LY"
#                 },
#                 "sparklineData": [15, 12, 18, 10, 14, 13, 16]
#             },
#             {
#                 "id": "kpi_4",
#                 "title": "Labor SPH (Sales Per Hour)",
#                 "icon": "labor",
#                 "value": "$5.7M",
#                 "trend": {
#                     "value": 30,
#                     "direction": "up",
#                     "label": "Vs LY"
#                 },
#                 "sparklineData": [8, 9, 12, 11, 14, 10, 15]
#             }
#         ],
#         "topPriorityStores": {
#             "title": "Top Priority Stores",
#             "subtitle": "Action Required",
#             "stores": [
#                 {
#                     "id": "store_A",
#                     "name": "PSP Downtown",
#                     "location": "Downtown, Downtown, U...",
#                     "tags": ["Low Rev Store", "Inventory Issue", "Action Required"]
#                 },
#                 {
#                     "id": "store_B",
#                     "name": "PSP Downtown",
#                     "location": "Downtown, Downtown, U...",
#                     "tags": ["Low Rev Store", "Inventory Issue", "Action Required"]
#                 }
#             ]
#         },
#         "topNegativeThemes": {
#             "title": "Top Negative Customer Theme",
#             "subtitle": "Most frequent customer concern",
#             "items": [
#                 {
#                     "id": "theme_001",
#                     "title": "Out of Stock",
#                     "percentage": 15,
#                     "description": "15% of negative comments",
#                     "tags": ["PSP Downtown", "PSP Downtown"]
#                 },
#                 {
#                     "id": "theme_002",
#                     "title": "Service Issues",
#                     "percentage": 15,
#                     "description": "15% of negative comments",
#                     "tags": ["PSP Downtown"]
#                 }
#             ]
#         },
#         "performanceChart": {
#             "title": "Performance Progress Chart",
#             "subtitle": "Most frequent customer concern",
#             "chart": {
#                 "title": "Sales Comp",
#                 "xAxisLabel": "Category",
#                 "dataPoints": [
#                     {"x": -800000, "y": 40000},
#                     {"x": -700000, "y": -50000},
#                     {"x": -500000, "y": 0},
#                     {"x": -400000, "y": -80000},
#                     {"x": -300000, "y": -30000},
#                     {"x": -200000, "y": 40000},
#                     {"x": 200000, "y": -20000},
#                     {"x": 600000, "y": -10000},
#                     {"x": 700000, "y": -60000},
#                     {"x": 800000, "y": -40000}
#                 ]
#             }
#         }
#     }
# 
# 
# @app.get("/api/agents/custom/store-hub/all-stores")
# @limiter.limit("60/minute")
# async def get_all_stores(
#     request: Request,
#     district_id: Optional[str] = Query(None),
#     db: AsyncSession = Depends(get_db),
#     user: Dict[str, Any] = Depends(get_tenant_auth_user)
# ):
#     """
#     Get all stores in a district with their ratings and metrics
#     
#     **Requires**: User authentication from log_users table (Bearer token from /api/sessions-login)
#     
#     Args:
#         district_id: The district ID filter (optional, defaults to "Miller")
#         user: Authenticated user from log_users table (automatically injected via dependency)
#     
#     Returns:
#         All stores data including ratings, tags, KPIs, and sentiment breakdown
#     """
#     
    # Return hardcoded data with only Canton - Cromer store
#     return {
#         "allStores": {
#             "title": "All Stores in Miller District",
#             "subtitle": "Ranked by average rating (ascending)",
#             "stores": [
#                 {
#                     "id": "0049",
#                     "name": "Canton - Cromer",
#                     "location": {
#                         "district": "Miller",
#                         "label": "Canton, 44709-2908"
#                     },
#                     "average_rating": 71.43,
#                     "tags": [
#                         {
#                             "themes": [
#                                 {
#                                     "type": "positive",
#                                     "value": "Staff Interaction / Service Quality"
#                                 }
#                             ]
#                         }
#                     ],
#                     "topIssue": None,
#                     "kpiDeltas": {
#                         "netSales": {
#                             "value": None,
#                             "unit": "percent",
#                             "direction": None
#                         },
#                         "nrr": {
#                             "value": None,
#                             "unit": "points",
#                             "direction": "flat"
#                         },
#                         "revv": {
#                             "value": 71.43,
#                             "unit": "points",
#                             "direction": "down"
#                         }
#                     },
#                     "sentimentBreakdown": {
#                         "positive": 85.71,
#                         "neutral": 0.0,
#                         "negative": 14.29
#                     }
#                 }
#             ]
#         }
#     }



@app.get("/api/tools", response_model=List[Tool])
@limiter.limit("60/minute")
async def get_tools(request: Request):
    built_in_tools = load_tools()
    custom_tools = load_custom_tools()
    return built_in_tools + custom_tools

@app.post("/api/tools/add")
@limiter.limit("30/minute")
async def add_tool(request: Request, tool_id: str):
    tools = load_tools()
    for tool in tools:
        if tool.id == tool_id:
            tool.is_added = True
            break
    save_tools(tools)
    return {"message": "Tool added successfully"}

@app.post("/api/tools/custom")
@limiter.limit("10/minute")
async def create_custom_tool(request: Request, tool: CustomTool):
    custom_tools = load_custom_tools()
    
    if any(t.name.lower() == tool.name.lower() for t in custom_tools):
        raise HTTPException(status_code=400, detail="Tool with this name already exists")
    
    new_tool = Tool(
        id=str(uuid.uuid4()),
        name=tool.name,
        description=tool.description,
        tags=tool.tags,
        is_custom=True,
        is_added=False,
        data_connector_id=tool.data_connector_id
    )

    # Handle schema
    schema_dir = "tool_schemas"
    os.makedirs(schema_dir, exist_ok=True)
    with open(f"{schema_dir}/{new_tool.id}.json", 'w') as f:
        json.dump(tool.schema, f, indent=2)
    
    # Handle metadata (data connector)
    if tool.data_connector_id:
        metadata_dir = "tool_metadata"
        os.makedirs(metadata_dir, exist_ok=True)
        with open(f"{metadata_dir}/{new_tool.id}.json", 'w') as f:
            json.dump({"data_connector_id": tool.data_connector_id}, f, indent=2)
    
    custom_tools.append(new_tool)
    save_custom_tools(custom_tools)
    
    # Invalidate tools cache
    cache.delete('tools_cache')
    
    return new_tool

# Cache management functions
def get_cached_tools():
    """Get both built-in and custom tools from cache or load from files if cache miss"""
    # Try to get both types from cache
    built_in_tools_data = cache.get('built_in_tools_cache')
    custom_tools_data = cache.get('custom_tools_cache')
    
    built_in_tools = None
    custom_tools = None
    
    # Handle built-in tools
    if built_in_tools_data:
        logging.info("Built-in tools loaded from cache")
        built_in_tools = [Tool(**tool_data) for tool_data in json.loads(built_in_tools_data)]
    else:
        built_in_tools = load_tools()
        cache.setex('built_in_tools_cache', 604800, json.dumps([tool.dict() for tool in built_in_tools]))
        logging.info("Built-in tools loaded from files and cached")
    
    # Handle custom tools
    if custom_tools_data:
        logging.info("Custom tools loaded from cache")
        custom_tools = [Tool(**tool_data) for tool_data in json.loads(custom_tools_data)]
    else:
        custom_tools = load_custom_tools()
        cache.setex('custom_tools_cache', 604800, json.dumps([tool.dict() for tool in custom_tools]))
        logging.info("Custom tools loaded from files and cached")
    
    # Combine and return all tools
    return built_in_tools + custom_tools

def get_cached_agents():
    """Get agents from cache or load from files if cache miss"""
    agents_data = cache.get('agents_cache')
    if agents_data:
        logging.info("Agents loaded from cache")
        return json.loads(agents_data)
    
    # Cache miss - load from files
    agents = load_agents()
    cache.setex('agents_cache', 604800, json.dumps([agent.dict() for agent in agents]))
    logging.info("Agents loaded from files and cached")
    return agents

def invalidate_agent_cache(agent_id=None):
    """Invalidate agent cache when changes are made"""
    if agent_id:
        # Invalidate specific agent
        cache.delete(f'agent_{agent_id}')
    # Invalidate full agents list
    cache.delete('agents_cache')

def invalidate_tool_cache(tool_id=None):
    """Invalidate tool cache when changes are made"""
    if tool_id:
        # Invalidate specific tool
        cache.delete(f'tool_{tool_id}')
    # Invalidate full tools list
    cache.delete('tools_cache')

@app.get("/api/tools/{tool_id}/schema")
@limiter.limit("60/minute")
async def get_tool_schema(request: Request, tool_id: str):
    # Try to get from cache first
    cached_schema = cache.get(f'tool_schema_{tool_id}')
    if cached_schema:
        return json.loads(cached_schema)
    
    # Cache miss - load from file
    schema_path = f"tool_schemas/{tool_id}.json"
    if not os.path.exists(schema_path):
        raise HTTPException(status_code=404, detail="Schema not found")
        
    with open(schema_path) as f:
        schema = json.load(f)
        # Cache for 1 hour
        cache.setex(f'tool_schema_{tool_id}', 604800, json.dumps(schema))
        return schema
    
    with open(schema_path, 'r') as f:
        return json.load(f)

@app.put("/api/tools/{tool_id}/auth")
@limiter.limit("10/minute")
async def update_tool_auth(request: Request, tool_id: str, auth: ToolAuth):
    auth_dir = "tool_auth"
    os.makedirs(auth_dir, exist_ok=True)
    
    auth_path = f"{auth_dir}/{tool_id}.json"
    with open(auth_path, 'w') as f:
        json.dump(auth.dict(), f, indent=2)
    
    return {"message": "Tool authentication updated"}

@app.get("/api/tools/{tool_id}")
@limiter.limit("60/minute")
async def get_tool(request: Request, tool_id: str):
    # Try to get from cache first
    cached_tool = cache.get(f'tool_{tool_id}')
    if cached_tool:
        logging.info(f"Tool {tool_id} loaded from cache")
        return json.loads(cached_tool)
    
    # Cache miss - check both built-in and custom tools
    built_in_tools = load_tools()
    custom_tools = load_custom_tools()
    all_tools = built_in_tools + custom_tools
    
    for tool in all_tools:
        if tool.id == tool_id:
            # Cache individual tool for future requests
            cache.setex(f'tool_{tool_id}', 604800, json.dumps(tool.dict()))
            return tool
            
    raise HTTPException(status_code=404, detail="Tool not found")

@app.put("/api/tools/{tool_id}")
@limiter.limit("10/minute")
async def update_tool(request: Request, tool_id: str, updated_tool: CustomTool):
    custom_tools = load_custom_tools()
    
    for i, tool in enumerate(custom_tools):
        if tool.id == tool_id:
            updated = Tool(
                id=tool_id,
                name=updated_tool.name,
                description=updated_tool.description,
                tags=updated_tool.tags,
                is_custom=True,
                data_connector_id=updated_tool.data_connector_id
            )
            
            # Update the tool in storage
            custom_tools[i] = updated
            save_custom_tools(custom_tools)
            
            # Update schema if provided
            if updated_tool.schema:
                schema_dir = "tool_schemas"
                os.makedirs(schema_dir, exist_ok=True)
                with open(f"{schema_dir}/{tool_id}.json", 'w') as f:
                    json.dump(updated_tool.schema, f, indent=2)
                # Invalidate schema cache
                cache.delete(f'tool_schema_{tool_id}')
            
            # Invalidate both individual and list caches
            invalidate_tool_cache(tool_id)
            
            return updated
            
            schema_dir = "tool_schemas"
            with open(f"{schema_dir}/{tool_id}.json", 'w') as f:
                json.dump(updated_tool.schema, f, indent=2)
            
            # Handle metadata
            metadata_dir = "tool_metadata"
            metadata_path = f"{metadata_dir}/{tool_id}.json"
            
            if updated_tool.data_connector_id:
                # Update metadata with connector
                metadata = {"data_connector_id": updated_tool.data_connector_id}
                os.makedirs(metadata_dir, exist_ok=True)
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            else:
                # Remove metadata file if data connector is unselected
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
            
            custom_tools[i] = updated
            save_custom_tools(custom_tools)
            return updated
            
    raise HTTPException(status_code=404, detail="Tool not found")

@app.delete("/api/tools/{tool_id}")
@limiter.limit("10/minute")
async def delete_tool(request: Request, tool_id: str):
    custom_tools = load_custom_tools()
    tool_found = False
    
    for i, tool in enumerate(custom_tools):
        if tool.id == tool_id:
            # Remove tool
            del custom_tools[i]
            save_custom_tools(custom_tools)
            tool_found = True
            break
    
    if not tool_found:
        raise HTTPException(status_code=404, detail="Tool not found")
    
    # Remove schema
    schema_path = f"tool_schemas/{tool_id}.json"
    if os.path.exists(schema_path):
        os.remove(schema_path)
    
    # Remove metadata if exists
    metadata_path = f"tool_metadata/{tool_id}.json"
    if os.path.exists(metadata_path):
        os.remove(metadata_path)
    
    # Update agents that use this tool
    agents = load_agents()
    agents_updated = False
    
    for agent in agents:
        if tool_id in agent.get('tools', []):
            agent['tools'].remove(tool_id)
            agents_updated = True
        if tool_id in agent.get('advanced_tools', []):
            agent['advanced_tools'].remove(tool_id)
            agents_updated = True
    
    if agents_updated:
        save_agents(agents)
        # Invalidate agents cache since tool references were removed
        invalidate_agent_cache()
    
    # Invalidate all tool-related caches
    invalidate_tool_cache(tool_id)  # This invalidates both tool and list cache
    cache.delete(f'tool_schema_{tool_id}')
    
    return {"status": "success", "message": "Tool deleted and all references updated"}

@app.get("/api/notifications")
@limiter.limit("60/minute")
async def get_notifications(request: Request):
    return load_notifications()

@app.post("/api/notifications/{notification_id}/mark-read")
@limiter.limit("10/minute")
async def mark_notification_read(request: Request, notification_id: str):
    notifications = load_notifications()
    for notification in notifications:
        if notification["id"] == notification_id:
            notification["read"] = True
            save_notifications(notifications)
            return {"message": "Notification marked as read"}
    raise HTTPException(status_code=404, detail="Notification not found")

@app.post("/api/notifications/mark-all-read")
@limiter.limit("60/minute")
async def mark_all_notifications_read(request: Request):
    notifications = load_notifications()
    for notification in notifications:
        notification["read"] = True
    save_notifications(notifications)
    return {"message": "All notifications marked as read"}

# UI Deployments API Endpoints
@app.get("/api/ui-deployments", response_model=List[UIDeployment])
@limiter.limit("30/minute")
async def get_ui_deployments(request: Request):
    """Get all UI deployments for the authenticated user"""
    deployments = load_ui_deployments()
    return deployments

@app.post("/api/ui-deployments", response_model=UIDeployment, status_code=201)
@limiter.limit("10/minute")
async def create_ui_deployment(request: Request, deployment_data: UIDeploymentCreate):
    """Create a new UI deployment"""
    deployments = load_ui_deployments()
    
    # Generate unique ID
    deployment_id = str(uuid.uuid4())
    current_time = datetime.now(timezone.utc).isoformat()
    
    # Create deployment object
    new_deployment = {
        "id": deployment_id,
        "appTitle": deployment_data.appTitle,
        "agentId": deployment_data.agentId,
        "appDescription": deployment_data.appDescription,
        "logoUrl": deployment_data.logoUrl,
        "backgroundColor": deployment_data.backgroundColor,
        "textColor": deployment_data.textColor,
        "welcomeMessage": deployment_data.welcomeMessage,
        "chatTheme": deployment_data.chatTheme,
        "placeholderText": deployment_data.placeholderText,
        "enableDarkMode": deployment_data.enableDarkMode,
        "enableClearChat": deployment_data.enableClearChat,
        "enableVoice": deployment_data.enableVoice,
        "enableTyping": deployment_data.enableTyping,
        "enableTimestamps": deployment_data.enableTimestamps,
        "status": "active",
        "createdAt": current_time,
        "updatedAt": current_time
    }
    
    deployments.append(new_deployment)
    save_ui_deployments(deployments)
    
    # Generate dedicated files for the deployment
    await generate_deployment_files(new_deployment)
    
    return new_deployment

@app.get("/api/ui-deployments/{deployment_id}", response_model=UIDeployment)
@limiter.limit("30/minute")
async def get_ui_deployment(request: Request, deployment_id: str):
    """Get a specific UI deployment by ID"""
    deployments = load_ui_deployments()
    deployment = next((d for d in deployments if d["id"] == deployment_id), None)
    
    if not deployment:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    return deployment

@app.put("/api/ui-deployments/{deployment_id}", response_model=UIDeployment)
@limiter.limit("10/minute")
async def update_ui_deployment(request: Request, deployment_id: str, deployment_data: UIDeploymentUpdate):
    """Update a UI deployment"""
    deployments = load_ui_deployments()
    deployment_index = next((i for i, d in enumerate(deployments) if d["id"] == deployment_id), None)
    
    if deployment_index is None:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    # Update deployment
    deployment = deployments[deployment_index]
    update_data = deployment_data.dict(exclude_unset=True)
    deployment.update(update_data)
    deployment["updatedAt"] = datetime.now(timezone.utc).isoformat()
    
    deployments[deployment_index] = deployment
    save_ui_deployments(deployments)
    
    # Regenerate deployment files with updated data
    await generate_deployment_files(deployment)
    
    return deployment

@app.delete("/api/ui-deployments/{deployment_id}")
@limiter.limit("10/minute")
async def delete_ui_deployment(request: Request, deployment_id: str):
    """Delete a UI deployment"""
    deployments = load_ui_deployments()
    deployment_index = next((i for i, d in enumerate(deployments) if d["id"] == deployment_id), None)
    
    if deployment_index is None:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    # Remove deployment files
    await cleanup_deployment_files(deployment_id)
    
    # Remove from list
    deployments.pop(deployment_index)
    save_ui_deployments(deployments)
    
    return {"message": "Deployment deleted successfully"}

@app.post("/api/ui-deployments/regenerate")
@limiter.limit("5/minute")
async def regenerate_deployments(request: Request):
    """Regenerate all deployment files"""
    await regenerate_all_deployments()
    return {"message": "All deployments regenerated successfully"}

# Route to serve standalone chat applications
@app.get("/chat-app/{deployment_id}", response_class=HTMLResponse)
@limiter.limit("60/minute")
async def serve_chat_app(request: Request, deployment_id: str):
    """Serve standalone chat application"""
    deployment_dir = f"static/deployments/{deployment_id}"
    index_file = f"{deployment_dir}/index.html"
    
    # Check if deployment exists
    if not os.path.exists(index_file):
        raise HTTPException(status_code=404, detail="Chat application not found")
    
    return FileResponse(index_file)

# Assets are now served through the static files mount at /static/deployments/{deployment_id}/

@app.get("/pages/{page_name}")
@limiter.limit("10/minute")
async def read_page_fragment(request: Request, page_name: str):
    # Prevent path traversal
    if ".." in page_name or page_name.startswith("/"):
        raise HTTPException(status_code=404, detail="Page not found")
    
    file_path = f"static/pages/{page_name}.html"
    
    # Check if file exists and is a file (asynchronously)
    try:
        file_stat = await aiofiles.os.stat(file_path)
        # if not os.path.isfile(file_path):  # Still need os.path.isfile for stat check
        #     raise HTTPException(status_code=404, detail="Page fragment not found")
        return FileResponse(file_path)
    except FileNotFoundError:
        # Handle default case for empty or "home" page
        if page_name == "" or page_name == "home":
            default_path = "static/pages/home.html"
            try:
                await aiofiles.os.stat(default_path)
                return FileResponse(default_path)
            except FileNotFoundError:
                raise HTTPException(status_code=404, detail="Page fragment not found")
        raise HTTPException(status_code=404, detail="Page fragment not found")

# --- Feedback Model & Endpoint ---

class FeedbackRequest(BaseModel):
    execution_id: str
    feedback: str
    session_id: str

@app.post("/api/session/execution-feedback")
async def submit_feedback(request: Request, feedback_data: FeedbackRequest):
    """
    Submit feedback for a specific execution ID
    """
    try:
        auth_result = verify_and_set_user_email(request, JWT_SECRET_KEY)
        if auth_result:
            return auth_result

        # session_id = request.cookies.get("session_id")
        if not feedback_data.session_id:
            raise HTTPException(status_code=400, detail="Session is not active")

        # Update feedback
        success, error = await session_logs_db.update_feedback(
            session_id=feedback_data.session_id,
            execution_id=feedback_data.execution_id,
            feedback=feedback_data.feedback
        )
        
        if not success:
            raise HTTPException(status_code=500, detail=error)
            
        return {"status": "success", "message": "Feedback submitted successfully"}
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Agent Inference Models & Endpoint ---

class TableData(BaseModel):
    headers: List[str]
    rows: List[List[str]]

class ChartData(BaseModel):
    type: str
    data: Dict[str, Any]

class CodeData(BaseModel):
    language: str
    code: str

class ListData(BaseModel):
    items: List[str]

class TextData(BaseModel):
    text: str

class ErrorData(BaseModel):
    message: str
    details: Optional[str] = None

# Import MessageBlock for MessageResponse
try:
    from langgraph_agent_poc.types import MessageBlock
except ImportError:
    # Fallback if import fails - define a minimal type
    MessageBlock = Dict[str, Any]

class MessageResponse(BaseModel):
    type: str
    content: Union[TextData, ErrorData, TableData, ChartData, CodeData, ListData, Dict[str, Any]]
    execution_id: Optional[str] = None
    log_url: Optional[str] = None
    session_id: Optional[str] = None
    html_content: Optional[str] = None
    user_input: Optional[str] = None
    sql: Optional[str] = None
    more_instructions: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None  # Updated conversation history
    message_blocks: Optional[List[MessageBlock]] = None

class InferenceRequest(BaseModel):
    agentId: str
    userInput: str

ALLOWED_FILE_TYPES = {
    "image/jpeg": "image",
    "image/png": "image",
    "image/gif": "image",
    "text/csv": "csv",
    "application/json": "json",
    "text/plain": "text",
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel": "xlsx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "docx"
}

if ENABLE_AGENT_RUN:
    # Utility to sanitize strings for logging, preserving emojis
    def sanitize_for_logging(text: Any) -> str:
        try:
            # Convert to string and ensure UTF-8 encoding, preserving valid Unicode (e.g., emojis)
            text_str = str(text)
            return text_str.encode("utf-8", errors="replace").decode("utf-8")
        except Exception:
            # Fallback for problematic inputs
            return str(text).encode("utf-8", errors="replace").decode("utf-8")

    # Initialize cache based on configuration
    CACHE_TYPE = os.getenv('CACHE_TYPE', 'diskcache').lower()  # Default to diskcache if not specified
    cache = None

    if CACHE_TYPE == 'redis':
        try:
            logging.info("Attempting to connect to Redis...")
            cache = redis.Redis(
                host='gemrmkjcl.dragonflydb.cloud',
                port=6379,
                password='hmv33lnff0eh',
                decode_responses=True,  # This ensures responses are returned as strings rather than bytes
                socket_timeout=5,  # 5 seconds timeout for operations
                socket_connect_timeout=5,  # 5 seconds timeout for connection
                retry_on_timeout=True,  # Retry once on timeout
                health_check_interval=30  # Check connection health every 30 seconds
            )
            # Test the connection
            response = cache.ping()
            logging.info(f"Successfully connected to Redis. Ping response: {response}")
        except redis.TimeoutError as e:
            logging.error(f"Redis connection timeout: {str(e)}. Please check if the Redis server is accessible.")
            logging.info("Falling back to diskcache...")
            CACHE_TYPE = 'diskcache'
        except redis.AuthenticationError as e:
            logging.error(f"Redis authentication failed: {str(e)}. Please check credentials.")
            logging.info("Falling back to diskcache...")
            CACHE_TYPE = 'diskcache'
        except redis.ConnectionError as e:
            logging.error(f"Failed to connect to Redis: {str(e)}")
            logging.info("Falling back to diskcache...")
            CACHE_TYPE = 'diskcache'
        except Exception as e:
            logging.error(f"Unexpected error connecting to Redis: {str(e)}")
            logging.info("Falling back to diskcache...")
            CACHE_TYPE = 'diskcache'

    class CacheWrapper:
        def __init__(self, cache_instance, cache_type):
            self._cache = cache_instance
            self._type = cache_type

        def setex(self, key: str, time_seconds: int, value: str):
            """Set value with expiration time in seconds"""
            if self._type == 'redis':
                return self._cache.setex(key, time_seconds, value)
            else:  # diskcache
                return self._cache.set(key, value, expire=time_seconds)

        def get(self, key: str) -> str:
            """Get value by key"""
            return self._cache.get(key)

        def delete(self, key: str):
            """Delete value by key"""
            if self._type == 'redis':
                return self._cache.delete(key)
            else:  # diskcache
                return self._cache.delete(key)

    if CACHE_TYPE == 'diskcache':
        # Use diskcache as fallback or default
        import diskcache
        cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
        logging.info(f"Using diskcache at: {cache_dir}")
        _cache_instance = diskcache.Cache(cache_dir)
        cache = CacheWrapper(_cache_instance, 'diskcache')
        logging.info("Successfully initialized diskcache")
    else:
        # Redis was successfully initialized above
        cache = CacheWrapper(cache, 'redis')


    # Helper functions for memory enrichment method
    async def check_if_followup_question(user_input: str, logger: logging.Logger) -> bool:
        """
        Check if the user input is a follow-up question using LLM.
        
        Args:
            user_input: The user's input text
            logger: Logger instance for logging
            
        Returns:
            bool: True if it's a follow-up question, False otherwise
        """
        try:
            prompt = f"""Analyze the following user input and determine if it is a follow-up question that references previous conversation context.

User input: "{user_input}"

A follow-up question typically:
- Contains pronouns or references like "it", "this", "that", "they", "he", "she", "them", "those"
- Uses phrases like "what about", "how about", "tell me more", "explain", "why", "how"
- Is incomplete without previous context
- References something mentioned earlier

Respond with ONLY "yes" or "no" (lowercase, no punctuation, no explanation)."""
            
            response = await llm_chat(prompt, max_tokens=10)
            result = response.strip().lower()
            
            is_followup = result == "yes"
            logger.info(f"Follow-up check result: {is_followup} for input: {sanitize_for_logging(user_input[:100])}")
            
            return is_followup
            
        except Exception as e:
            logger.warning(f"Error checking follow-up question: {sanitize_for_logging(str(e))}. Defaulting to False.")
            return False

    async def enrich_user_input_with_history(user_input: str, history_data: List[Dict[str, str]], logger: logging.Logger) -> str:
        """
        Enrich user input with context from history using LLM.
        
        Args:
            user_input: The user's input text
            history_data: List of previous conversations (last 2 conversations)
            logger: Logger instance for logging
            
        Returns:
            str: Enriched user input with context
        """
        try:
            # Format history for context
            history_context = ""
            for i, entry in enumerate(history_data, 1):
                user_msg = entry.get("input", "")
                agent_msg = entry.get("response", "")
                history_context += f"Conversation {i}:\n"
                history_context += f"User: {user_msg}\n"
                history_context += f"Agent: {agent_msg}\n\n"
            
            # Get the most recent conversation
            most_recent = history_data[-1] if history_data else {}
            most_recent_agent = most_recent.get("response", "")
            
            prompt = f"""You are a context enrichment assistant. Your task is to enrich the user's current input by adding necessary context from the conversation history.

            Conversation History:
            {history_context}

            MOST RECENT Agent Response (HIGHEST PRIORITY):
            {most_recent_agent}

            Current User Input: "{user_input}"

            CRITICAL RULES:
            1. SIMPLE CONFIRMATIONS (yes/no/ok/correct/right/exactly):
            - Analyze what the MOST RECENT agent message is asking or suggesting
            - If the agent is offering a suggestion or asking if user wants something:
                * Example: "Would you like to know about his research?"
                * User: "yes" → Enrich to: "Yes, tell me about his research" or "Yes, I would like to know about his research"
            - If the agent is making a statement or providing information (not asking):
                * User: "yes" → Keep as "yes" (simple acknowledgment)
            - If the agent is asking for clarification about what the user meant:
                * User: "yes" → Keep as "yes" (confirming their previous statement)

            2. CLARIFICATION QUESTIONS vs SUBSTANTIVE RESPONSES:
            - If the most recent agent response is asking for clarification (contains "?", "could you clarify", "I'm not sure", etc.):
                * This is a CLARIFICATION REQUEST
                * The user's next input is likely ANSWERING that clarification
                * Do NOT reference the clarification question itself
            - If the most recent agent response is providing information (no clarification request):
                * This is a SUBSTANTIVE RESPONSE
                * The user's next input is likely a FOLLOW-UP about that information
                * You can add context from that substantive response

            3. PRONOUN RESOLUTION:
            - Replace pronouns (he/she/they/it/there/that) with specific entities from the MOST RECENT SUBSTANTIVE response
            - Skip older clarification requests when resolving pronouns
            - Example: If most recent substantive response mentions "Tamal Chowdhury" and user asks "where did he study?" → "Where did Tamal Chowdhury study?"

            4. MINIMAL ENRICHMENT:
            - Add only what's necessary to make the input clear
            - Preserve the user's original tone and intent
            - If already clear, return as-is

            EXAMPLES:

            Example 1 (Confirmation to informational statement - DO NOT ENRICH):
            History: 
            - User: "where did Tamal do his PhD?"
            - Agent: "Tamal did his PhD at MIT. He focused on computational modeling."
            Current Input: "yes"
            Enriched Output: yes

            Example 2 (Confirmation after clarification request - DO NOT ENRICH):
            History:
            - User: "when did he sleep there?"
            - Agent: "I'm not sure what you mean by 'there'. Could you clarify?"
            Current Input: "yes"
            Enriched Output: yes

            Example 3 (Confirmation to proceed with suggestion - ENRICH):
            History:
            - User: "tell me about Tamal"
            - Agent: "Tamal Chowdhury is a data scientist. Would you like to know more about his research background?"
            Current Input: "yes"
            Enriched Output: Yes, tell me more about Tamal Chowdhury's research background

            Example 4 (Confirmation to proceed with offer - ENRICH):
            History:
            - User: "what did he do?"
            - Agent: "He completed his PhD at MIT. Would you like to know about his current work?"
            Current Input: "yes"
            Enriched Output: Yes, I would like to know about his current work

            Example 5 (Follow-up with pronoun - ENRICH):
            History:
            - User: "tell me about Tamal"
            - Agent: "Tamal Chowdhury is a data scientist..."
            Current Input: "where did he study?"
            Enriched Output: Where did Tamal Chowdhury study?

            Example 6 (Answering clarification - DO NOT OVER-ENRICH):
            History:
            - User: "when did he go there?"
            - Agent: "Could you clarify who 'he' refers to?"
            Current Input: "I meant Tamal"
            Enriched Output: I meant Tamal
            (Note: Do NOT say "I meant Tamal when asking when did he go there")

            Return ONLY the enriched user input without any explanations, quotes, or additional text."""
            
            enriched_input = await llm_chat(prompt, max_tokens=500)
            enriched_input = enriched_input.strip().strip('"').strip("'")
            
            # Additional safety check: if input is a simple confirmation, check if it should be enriched
            simple_confirmations = ['yes', 'no', 'ok', 'okay', 'correct', 'right', 'exactly', 'sure', 'yep', 'nope']
            if user_input.lower().strip() in simple_confirmations:
                # Check if the most recent agent message contains a suggestion/offer (indicated by '?')
                has_question = '?' in most_recent_agent
                # If enrichment added substantial context (more than just confirmation), it's likely correct
                # If it didn't add context but agent was asking something, that might be wrong
                if enriched_input.lower().strip() == user_input.lower().strip() and has_question and 'would you like' in most_recent_agent.lower():
                    logger.warning(f"Confirmation '{user_input}' to a suggestion question was not enriched. This might need manual review.")
                elif len(enriched_input.split()) > 10:
                    # If enrichment made it too verbose, that's suspicious
                    logger.info(f"Enriched confirmation is lengthy. Original: '{user_input}', Enriched: '{enriched_input}'")
            
            logger.info(f"Enriched input: {sanitize_for_logging(enriched_input[:200])}")
            
            return enriched_input
            
        except Exception as e:
            logger.warning(f"Error enriching user input: {sanitize_for_logging(str(e))}. Returning original input.")
            return user_input

    async def check_for_ambiguity(user_input: str, enriched_input: str, history_data: List[Dict[str, str]], logger: logging.Logger) -> Optional[str]:
        """
        Check if there's ambiguity, confusion, or lack of relevance in the follow-up question.
        Also checks if the follow-up question is clearly connected to previous responses.
        
        Args:
            user_input: Original user input
            enriched_input: Enriched user input
            history_data: List of previous conversations
            logger: Logger instance for logging
            
        Returns:
            Optional[str]: Clarification questions if ambiguity/relevance issues detected, None otherwise
        """
        try:
            # Format history for context
            history_context = ""
            for i, entry in enumerate(history_data, 1):
                user_msg = entry.get("input", "")
                agent_msg = entry.get("response", "")
                history_context += f"Conversation {i}:\n"
                history_context += f"User: {user_msg}\n"
                history_context += f"Agent: {agent_msg}\n\n"
            
            # Get the most recent conversation for context
            most_recent = history_data[-1] if history_data else {}
            most_recent_user = most_recent.get("input", "")
            most_recent_agent = most_recent.get("response", "")
            
            prompt = f"""Analyze the CURRENT user input to determine if it is GENUINELY ambiguous or disconnected from the conversation.

            Conversation History (for context):
            {history_context}

            MOST RECENT Conversation (give this HIGHEST priority):
            User: {most_recent_user}
            Agent: {most_recent_agent}

            CURRENT User Input: "{user_input}"
            Enriched User Input: "{enriched_input}"

            CRITICAL RULES - BE VERY LENIENT:
            1. DEFAULT TO "OK": Unless the input is TRULY nonsensical or completely unrelated to the ENTIRE conversation domain, respond with "OK"
            2. SIMPLE CONFIRMATIONS: If the current input is "yes", "no", "correct", "ok", etc. → ALWAYS respond "OK"
            3. DOMAIN RELEVANCE: If the current input is about the same general domain/business area as the conversation (even if asking about different specific details), it's RELEVANT → Respond "OK"
            4. CONTEXTUAL SHIFTS ARE NORMAL: Users often ask follow-up questions that explore different aspects of the same topic. This is NORMAL conversation flow → Respond "OK"

            Evaluate the CURRENT input with MAXIMUM LENIENCY:

            1. Is it a simple confirmation or acknowledgment?
            - "yes", "no", "correct", "ok", "right", "exactly", "thanks", "got it" → ALWAYS respond "OK"
            
            2. Is the CURRENT input within the same GENERAL DOMAIN as the conversation?
            - If discussing stores/templates/business and the question is about stores/locations/details → Respond "OK"
            - If discussing a person and the question is about that person's work/education/life → Respond "OK"
            - If discussing a product and the question is about features/pricing/availability → Respond "OK"
            - Even if the specific detail changes (e.g., from template name to store locations), if it's the same domain → Respond "OK"
            - Natural conversation includes topic shifts within the same domain → Respond "OK"

            3. Can the CURRENT input be reasonably understood?
            - If it's a grammatically correct question, even if it explores a new angle → Respond "OK"
            - If the enriched input makes any sense → Respond "OK"
            - Only if the input is gibberish or truly nonsensical → Ask for clarification

            ONLY ask for clarification if the CURRENT input is:
            - Complete gibberish that cannot be understood at all (e.g., "asdf jkl qwerty")
            - About a COMPLETELY DIFFERENT DOMAIN that has zero connection (e.g., discussing pet stores, then suddenly asking about quantum physics)
            - Genuinely ambiguous with pronouns that have no clear referent in the entire conversation history

            Examples that should ALWAYS get "OK":
            - "yes" → "OK"
            - "Which stores are in Illinois?" (even if previous question was about template names) → "OK" (same domain: stores)
            - "What about pricing?" (even if previous was about features) → "OK" (same domain: product)
            - "where did he do his phd?" (when "he" refers to someone from conversation) → "OK"
            - "tell me more" → "OK"
            - "what are the wage guidelines?" (when discussing store templates) → "OK" (related to business/stores)

            Examples that should ask for clarification (VERY RARE):
            - "asdf jkl qwerty" (complete gibberish) → Ask for clarification
            - Discussing pet stores, then: "What's the weather in Tokyo?" (completely unrelated domain) → Ask for clarification
            - "when did they do that?" (when "they" and "that" have no referent in entire conversation) → Ask for clarification

            CRITICAL OUTPUT FORMAT:
            - If the input is clear and can be processed: Respond with EXACTLY "OK" and nothing else
            - If clarification is needed: Provide ONLY the chatbot message that will be shown directly to the user
            * DO NOT include any meta-commentary like "Therefore, the response should be:", "The response is:", etc.
            * DO NOT explain your reasoning or analysis
            * Just write the EXACT message the chatbot should say to the user
            * Keep it SHORT, FRIENDLY, and CONVERSATIONAL

            GOOD EXAMPLES of clarification messages (ONLY the message, no meta-text):
            - "I'm not quite sure what you're asking. Could you rephrase that?"
            - "Could you clarify what you mean by that?"

            BAD EXAMPLES (DO NOT do this):
            - "The current input is unrelated. Therefore, the response should be: I notice your question seems..." ❌
            - "I need to ask for clarification. The response is: Could you clarify..." ❌

            Remember: BE LENIENT. Only block truly problematic inputs. Your output will be used DIRECTLY as the chatbot's response. Only output "OK" or the exact user-facing message."""
            
            response = await llm_chat(prompt, max_tokens=300)
            response = response.strip()
            
            # Clean up any meta-commentary using a second LLM call
            if response.upper() != "OK":
                cleanup_prompt = f"""Extract ONLY the direct user-facing message from the following text.

                Input text: {response}

                Rules:
                - If the text contains meta-commentary (like "Therefore, the response should be:", "I should ask:", etc.), remove it and return ONLY the actual message
                - Remove any quotes surrounding the message
                - Return ONLY the clean message that should be shown to the user
                - If there's no meta-commentary, return the text as-is
                - Do not add or modify the actual message content

                Examples:
                Input: "Therefore, the response should be: I notice your question seems off-topic."
                Output: I notice your question seems off-topic.

                Input: "I notice your question seems off-topic."
                Output: I notice your question seems off-topic.

                Input: "The response is: 'Could you clarify what you mean?'"
                Output: Could you clarify what you mean?

                Now extract the clean message:"""
                
                try:
                    response = await llm_chat(cleanup_prompt, max_tokens=200)
                    response = response.strip()
                except Exception as cleanup_error:
                    logger.warning(f"Error in cleanup step: {sanitize_for_logging(str(cleanup_error))}. Using original response.")
            
            if response.upper() == "OK":
                logger.info("Follow-up question is clear, relevant, and connected to conversation")
                return None
            else:
                logger.warning(f"Relevance/clarity issue detected: {sanitize_for_logging(response)}")
                return response
                
        except Exception as e:
            logger.warning(f"Error checking for ambiguity/relevance: {sanitize_for_logging(str(e))}. Proceeding with enriched input.")
            return None


    @memory_monitor
    async def run_agent(request: Request, agentId: str, userInput: str, file: Optional[UploadFile] = None, response: Response = None, session_id: str = None, status_queue: Optional[asyncio.Queue] = None, ai_mode: str = None, use_streaming_executor: bool = False):
        # Get user email from request state
        user_email = getattr(request.state, "user_email", "External User")
        
        # Ensure executor and file-related variables always exist for the finally block
        executor = None
        file_path = None
        file_type = None

        # Start timing the execution
        start_time = time.time()
        
        # Generate execution ID
        execution_uuid = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        execution_id = f"{execution_uuid}_{timestamp}"
        log_filename = f"agent_execution_{execution_id}.log"
        log_file = os.path.join(LOG_DIR, log_filename)
        
        # Generate log URL
        log_url = f"{BASE_URL}/api/logs/{execution_id}"
        
        # Configure logger
        logger = logging.getLogger(f"agent_infer_{execution_id}")
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.error(f"Failed to create log file handler: {sanitize_for_logging(e)}")
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Log user email if available
        if user_email:
            logger.info(f"Request from user: {user_email}")
        else:
            user_email = "agent@impactanalytics.co"
            logger.info("Request from unauthenticated user")

        # Check content type to determine if it's JSON or form data
        content_type = request.headers.get("content-type", "")
        logger.info(f"Request content type: {content_type}")
        
        # Handle JSON request
        if "application/json" in content_type:
            try:
                json_data = await request.json()
                if agentId is None:
                    agentId = json_data.get("agentId")
                userInput = json_data.get("userInput")
                file = None  # File uploads not supported in JSON mode
                logger.info(f"Processed JSON request: agentId={agentId}, userInput={sanitize_for_logging(userInput)}")
            except Exception as e:
                logger.error(f"Error parsing JSON request: {sanitize_for_logging(str(e))}")
                return MessageResponse(
                    type="error",
                    content=ErrorData(
                        message="Invalid JSON payload",
                        details=str(e)
                    ),
                    execution_id=execution_id,
                    log_url=log_url,
                    session_id=session_id
                )
        
        # Validate required parameters
        if not agentId:
            logger.error("Missing agentId parameter")
            return MessageResponse(
                type="error",
                content=ErrorData(
                    message="Missing parameters",
                    details="agentId is required"
                ),
                execution_id=execution_id,
                log_url=log_url,
                session_id=session_id
            )
            
        if not userInput:
            userInput = ""  # Allow empty input
        
        # Retrieve chat history from diskcache
        # cache_key = f"history:{agentId}:{user_email or session_id}"
        

        cache_key = f"history:{agentId}:{session_id}"
        
        
        # Log request details
        sanitized_user_input = sanitize_for_logging(userInput)
        sanitized_file_name = sanitize_for_logging(file.filename if file else "None")
        logger.debug(f"Execution ID: {execution_id}")
        logger.debug(f"Log URL: {log_url}")

        try:
            agents = load_agents()
            agent = next((a for a in agents if a["id"] == agentId), None)

            # Determine memory method (default: "default")
            # memory_method = agent.get("memory_method", "default") if agent else "default"
            memory_method = "enrichment"

            # Variable to track if we should pass history to executor
            pass_history_to_executor = True
            # Variable to track original user input (for enrichment method)
            original_user_input = userInput
            # Variable to store clarification questions if any clarity issues detected
            clarity_issue = None
            
            if agent and agent.get("enable_memory", False):
                history_data = cache.get(cache_key)
                history = json.loads(history_data) if history_data else []

                sanitized_history = sanitize_for_logging(json.dumps(history)) if history else "None"
                logger.info(f"Received request: agentId={agentId}, userInput={sanitized_user_input}, history={sanitized_history}, file={sanitized_file_name}, memory_method={memory_method}")
                
                if memory_method == "enrichment":
                    # Enrichment method: Store last 2 conversations without trimming content
                    if len(history) > 2:
                        history = history[-2:]
                        cache.setex(cache_key, 1800, json.dumps(history))
                    
                    logger.info(f"Using enrichment method. Retrieved history from cache for key {cache_key}: {len(history)} entries")
                    
                    # Check if user input is a follow-up question
                    if history:  # Only check if there's history
                        is_followup = await check_if_followup_question(userInput, logger)
                        
                        if is_followup:
                            logger.info("User input detected as follow-up question. Enriching with history context.")
                            
                            # Enrich user input with history context
                            enriched_input = await enrich_user_input_with_history(userInput, history, logger)
                            
                            # Check for ambiguity, relevance, or clarity issues
                            clarity_issue = await check_for_ambiguity(userInput, enriched_input, history, logger)
                            
                            if clarity_issue:
                                logger.info(f"Relevance/clarity issue detected in follow-up question. Returning clarification questions without running agent.")
                                
                                # Update history with clarification response
                                if agent.get("enable_memory", False):
                                    memory_method = agent.get("memory_method", "default")
                                    history_input = original_user_input if memory_method == "enrichment" else userInput
                                    
                                    history.append({
                                        "input": history_input,
                                        "response": clarity_issue,
                                        "timestamp": datetime.now(timezone.utc).isoformat()
                                    })
                                    
                                    if memory_method == "enrichment":
                                        # Keep only last 2 conversations without trimming content
                                        if len(history) > 2:
                                            history = history[-2:]
                                    else:
                                        # Default method: Keep only last 3 items
                                        if len(history) > 3:
                                            history = history[-3:]
                                    
                                    # Use setex with a 30-minute expiry
                                    cache.setex(cache_key, 1800, json.dumps(history))
                                    logger.info(f"Updated history in cache for key {cache_key}: {len(history)} entries (method: {memory_method})")
                                
                                # Return clarification questions as the agent response
                                html_content = await generate_html_content(clarity_issue)
                                
                                return MessageResponse(
                                    type="text",
                                    content=TextData(text=clarity_issue),
                                    execution_id=execution_id,
                                    log_url=log_url,
                                    session_id=session_id,
                                    html_content=html_content,
                                    user_input=userInput
                                )
                            
                            # Use enriched input instead of original
                            userInput = enriched_input
                            logger.info(f"Using enriched input: {sanitize_for_logging(userInput[:200])}")
                            
                            # Don't pass history to executor - we've already enriched the input
                            pass_history_to_executor = False
                        else:
                            logger.info("User input is not a follow-up question. Proceeding with original input.")
                            # Not a follow-up, don't pass history to executor
                            pass_history_to_executor = False
                    else:
                        # No history available, proceed normally
                        pass_history_to_executor = False
                else:
                    # Default method: Keep only the last 3 items (existing behavior)
                    if len(history) > 3:
                        history = history[-3:]
                        # Use setex with a 30-minute expiry
                        cache.setex(cache_key, 1800, json.dumps(history))

                    logger.info(f"Using default method. Retrieved history from cache for key {cache_key}: {len(history)} entries")
                    pass_history_to_executor = True
            else:
                history = []
                pass_history_to_executor = False
            
            if not agent:
                logger.error(f"Agent not found: {agentId}")
                return MessageResponse(
                    type="error",
                    content=ErrorData(
                        message="Agent not found",
                        details=f"No agent found with ID: {agentId}"
                    ),
                    execution_id=execution_id,
                    log_url=log_url,
                    session_id=session_id
                )

            # Send status update if queue is available
            if status_queue:
                await status_queue.put(f"Connecting tools to {agent.get('name', 'Unknown')} Agent")

            file_info = None
            # file_path and file_type are initialized earlier for finally cleanup
            if file:
                contents = await file.read()
                file_size = len(contents)
                logger.debug(f"Processing file: {sanitized_file_name}, size: {file_size} bytes")
                if file_size > 10 * 1024 * 1024:
                    logger.error("File too large: exceeds 10MB")
                    return MessageResponse(
                        type="error",
                        content=ErrorData(
                            message="File too large",
                            details="Maximum file size is 10MB"
                        ),
                        execution_id=execution_id,
                        log_url=log_url,
                        session_id=session_id
                    )

                if file.content_type not in ALLOWED_FILE_TYPES:
                    logger.error(f"Unsupported file type: {file.content_type}")
                    return MessageResponse(
                        type="error",
                        content=ErrorData(
                            message="Unsupported file type",
                            details=f"Only CSV, JSON, TXT, PDF, and image files supported. Got: {file.content_type}"
                        ),
                        execution_id=execution_id,
                        log_url=log_url,
                        session_id=session_id
                    )

                file_extension = os.path.splitext(file.filename)[1]
                unique_filename = f"{uuid.uuid4()}{file_extension}"
                file_path = UPLOAD_DIR / unique_filename

                async with aiofiles.open(file_path, "wb") as buffer:
                    await buffer.write(contents)

                file_info = {
                    "original_name": sanitized_file_name,
                    "saved_name": sanitize_for_logging(unique_filename),
                    "size": file_size,
                    "type": file.content_type,
                    "path": str(file_path),
                    "uploaded_at": datetime.now(timezone.utc).isoformat()
                }
                file_type = file.content_type
                logger.info(f"File saved: {file_path}")

                # Send status update if queue is available
                if status_queue:
                    await status_queue.put(f"Processing file: {file.filename}")
            elif hasattr(request.state, "pre_saved_file_info"):
                # Use file that was pre-saved by streaming endpoint
                file_info = request.state.pre_saved_file_info
                file_path = pathlib.Path(file_info.get("path")) if file_info.get("path") else None
                file_type = file_info.get("type")
                sanitized_file_name = sanitize_for_logging(file_info.get("original_name", "pre_saved_file"))
                logger.info(f"Using pre-saved uploaded file: {file_path}")
                if status_queue and file_info.get("original_name"):
                    await status_queue.put(f"Processing file: {file_info.get('original_name')}")

            tools_config = []
            all_tools = load_custom_tools()
            connectors = load_connectors()
            
            for tool_id in agent.get("tools", []):
                schema_path = f"tool_schemas/{tool_id}.json"
                auth_path = f"tool_auth/{tool_id}.json"
                
                if os.path.exists(schema_path):
                    tool_config = {"id": tool_id}
                    tool = next((t for t in all_tools if t.id == tool_id), None)
                    if tool and tool.data_connector_id:
                        connector = next((c for c in connectors if c["id"] == tool.data_connector_id), None)
                        if connector:
                            tool_config["data_connector"] = connector
                    
                    with open(schema_path, "r", encoding="utf-8") as f:
                        tool_config["schema"] = json.load(f)
                    if os.path.exists(auth_path):
                        with open(auth_path, "r", encoding="utf-8") as f:
                            tool_config["auth"] = json.load(f)
                    tools_config.append(tool_config)
                    logger.debug(f"Loaded tool: {tool_id}")

            # Send status update if queue is available
            if status_queue:
                await status_queue.put(f"Loaded {len(tools_config)} tools")

            # Load advanced tools
            advanced_tools_config = []
            for tool_id in agent.get("advanced_tools", []):
                schema_path = f"tool_schemas/{tool_id}.json"
                if os.path.exists(schema_path):
                    with open(schema_path, "r", encoding="utf-8") as f:
                        advanced_tools_config.append({"id": tool_id, "schema": json.load(f)})
                    logger.debug(f"Loaded advanced tool: {tool_id}")

            # Create task executor
            executor = None
            try:
                # Choose executor based on agent's ai_framework configuration or ai_mode parameter
                # Priority: ai_mode parameter > agent's ai_framework > default (CrewAI)
                agent_framework = agent.get("ai_framework", "CrewAI")
                if ai_mode == 'LF':
                    use_langraph = True
                elif ai_mode:
                    use_langraph = False
                else:
                    # Use agent's configured framework
                    use_langraph = agent_framework == "LangGraph"
                
                executor_type = "LangGraph" if use_langraph else "CrewAI"
                
                logger.info(f"Creating {executor_type} task executor (configured: {agent_framework}, override: {ai_mode or 'None'}, streaming: {use_streaming_executor})")
                if status_queue:
                    await status_queue.put(f"Initializing {executor_type} task execution module..")
                
                agent_config = {
                    "role": agent.get("role", "AI Assistant"),
                    "goal": agent.get("goal", "Help the user with their request"),
                    "backstory": agent.get("backstory", ""),
                    "knowledge_base_id": agent.get("knowledge_base_id"),
                    "name": agent.get("name", "Unknown Agent"),
                    "id": agent.get("id", "Unknown Agent ID")
                }
                
                if use_langraph and use_streaming_executor:
                    # Use LangGraphExecutorStream only for WebSocket streaming
                    executor = await LangGraphExecutorStream.create(
                        agent_config=agent_config,
                        tools_config=tools_config,
                        log_file=log_file,
                        user_email=user_email,
                        enable_planning=agent.get("enable_planning", False),
                        enable_memory=agent.get("enable_memory", False),
                        verbose=agent.get("verbose", True),
                        raw_output=agent.get("raw_output", False),
                        execution_id=execution_id
                    )
                elif use_langraph:
                    # Use LangGraphExecutor for non-streaming LangGraph executions
                    executor = await LangGraphExecutor.create(
                        agent_config=agent_config,
                        tools_config=tools_config,
                        log_file=log_file,
                        user_email=user_email,
                        enable_planning=agent.get("enable_planning", False),
                        enable_memory=agent.get("enable_memory", False),
                        verbose=agent.get("verbose", True),
                        raw_output=agent.get("raw_output", False),
                        execution_id=execution_id
                    )
                else:
                    executor = await TaskExecutor.create(
                        agent_config=agent_config,
                        tools_config=tools_config,
                        log_file=log_file,
                        user_email=user_email,
                        enable_planning=agent.get("enable_planning", False),
                        enable_memory=agent.get("enable_memory", False),
                        verbose=agent.get("verbose", True),
                        raw_output=agent.get("raw_output", False),
                        execution_id=execution_id  # CRITICAL FIX: Pass execution_id for isolation
                    )
                
                logger.info(f"{executor_type} task executor created successfully")
                
                # Execute the task
                logger.info("Executing task")
                if status_queue:
                    await status_queue.put("Validating the request..")
                
                # Prepare history parameter based on memory method
                executor_history = history if pass_history_to_executor else []
                
                task_result = await executor.execute_task(
                    description=agent.get("instructions", "Help the user with their request"),
                    expected_output=agent.get("expectedOutput", "A helpful response"),
                    task_name="Agent Execution",
                    file_path=file_path.as_posix() if file_path else None,
                    file_type=file_type,
                    input=userInput,
                    user_email=user_email,
                    history=executor_history,
                    status_queue=status_queue  # Pass the status queue to execute_task
                )
                
                # Update history in cache
                if agent.get("enable_memory", False):
                    memory_method = agent.get("memory_method", "default")
                    
                    # Use original user input for history storage (not enriched)
                    history_input = original_user_input if memory_method == "enrichment" else userInput
                    
                    if task_result and task_result.strip():
                        history.append({
                            "input": history_input,
                            "response": task_result,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                    else:
                        logger.warning("Not adding empty response to history")

                    
                    if memory_method == "enrichment":
                        # Keep only last 2 conversations without trimming content
                        if len(history) > 2:
                            history = history[-2:]
                    else:
                        # Default method: Keep only last 3 items
                        if len(history) > 3:
                            history = history[-3:]
                    
                    # Use setex with a 30-minute expiry
                    cache.setex(cache_key, 1800, json.dumps(history))
                    logger.info(f"Updated history in cache for key {cache_key}: {len(history)} entries (method: {memory_method})")
                
                # Calculate execution time
                execution_time = time.time() - start_time
                logger.info(f"Task executed successfully in {execution_time:.2f} seconds")
                
                # Check if the result is JSON
                try:
                    result_json = json.loads(task_result)
                    if isinstance(result_json, dict):
                        # Generate HTML content if needed
                        html_content = await generate_html_content(result_json)
                        
                        # Return MessageResponse with the JSON result
                        result = MessageResponse(
                            type="text",
                            content=result_json,
                            execution_id=execution_id,
                            log_url=log_url,
                            session_id=session_id,
                            html_content=html_content,
                            user_input=userInput
                        )
                        
                        return result
                except json.JSONDecodeError:
                    # Not JSON, continue with text processing
                    pass
                
                # Generate HTML content if needed
                html_content = await generate_html_content(task_result)
                
                # Return MessageResponse with the text result
                result = MessageResponse(
                    type="text",
                    content=TextData(text=task_result),
                    execution_id=execution_id,
                    log_url=log_url,
                    session_id=session_id,
                    html_content=html_content,
                    user_input=userInput
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Error executing task: {sanitize_for_logging(str(e))}")
                return MessageResponse(
                    type="error",
                    content=ErrorData(
                        message="Error executing task",
                        details=str(e)
                    ),
                    execution_id=execution_id,
                    log_url=log_url,
                    session_id=session_id,
                    user_input=userInput
                )
        except Exception as e:
            logger.error(f"Error running agent: {sanitize_for_logging(str(e))}")
            return MessageResponse(
                type="error",
                content=ErrorData(
                    message="Error running agent",
                    details=str(e)
                ),
                execution_id=execution_id,
                log_url=log_url,
                session_id=session_id,
                user_input=userInput
            )
        finally:
            # ENHANCED CLEANUP: Comprehensive memory management
            initial_cleanup_memory = get_memory_usage()
            
            if executor:
                try:
                    # Add executor to tracking before cleanup
                    _active_executors.add(executor)
                    
                    # Enhanced cleanup with memory monitoring
                    cleanup_executor_resources(executor)
                    # logger.debug(f"[{execution_id}] Executor cleanup completed")
                except Exception as cleanup_error:
                    logger.warning(f"[{execution_id}] Error during executor cleanup: {cleanup_error}")
                finally:
                    executor = None  # Clear reference for garbage collection
            
            # Clean up any uploaded files to free disk space
            if 'file_path' in locals() and file_path and os.path.exists(file_path):
                try:
                    await aiofiles.os.remove(file_path)
                    logger.debug(f"[{execution_id}] Cleaned up uploaded file: {file_path}")
                except Exception as e:
                    logger.warning(f"[{execution_id}] Failed to clean up file: {e}")
            
            # Force garbage collection for this request if memory usage is high
            current_memory = get_memory_usage()
            if current_memory > MEMORY_THRESHOLD_MB:
                collected = gc.collect()
                final_memory = get_memory_usage()
                logger.info(f"[{execution_id}] Post-request GC: freed {collected} objects, "
                           f"memory: {initial_cleanup_memory:.1f}MB -> {final_memory:.1f}MB")
            
            # Log memory statistics
            logger.debug(f"[{execution_id}] Final memory usage: {current_memory:.1f}MB, "
                        f"Active executors: {len(_active_executors)}")


    # import json
    # import re
    # from typing import Any, Set



    async def generate_table_preview(url: str) -> str:
        try:
            # Normalize URL: convert http:// to https:// if it's the same domain (avoids redirects)
            normalized_url = url
            if url.startswith('http://') and not url.startswith('http://localhost') and not url.startswith('http://127.0.0.1'):
                # Try https version first to avoid redirects
                https_url = url.replace('http://', 'https://', 1)
                normalized_url = https_url
            
            # Use async httpx with redirect following enabled
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = None
                last_error = None
                
                # Try normalized (https) URL first, then fall back to original
                for attempt_url in [normalized_url, url]:
                    if attempt_url == url and normalized_url == url:
                        # Only one URL to try
                        break
                    try:
                        response = await client.get(attempt_url)
                        response.raise_for_status()
                        # Success, break out of loop
                        break
                    except (httpx.HTTPStatusError, httpx.RequestError) as e:
                        last_error = e
                        # If this was the https attempt and it failed, try http version
                        if attempt_url == normalized_url and normalized_url != url:
                            continue
                        # Otherwise, we'll raise the error
                        if attempt_url == url:
                            raise
                
                if response is None:
                    raise Exception(f"Failed to fetch URL: {last_error}" if last_error else "Unknown error")

                # Parse file based on extension
                try:
                    if url.lower().endswith('.csv'):
                        df = pd.read_csv(StringIO(response.text))
                    elif url.lower().endswith('.xlsx'):
                        df = pd.read_excel(BytesIO(response.content))
                    else:
                        return f"<p style='color:red;'>Unsupported file type for: {url}</p>"
                except Exception as parse_error:
                    logger.error(f"Error parsing spreadsheet {url}: {str(parse_error)}")
                    return f"<p style='color:red;'>Could not parse spreadsheet from {url}: {str(parse_error)}</p>"

                if df is None or df.empty:
                    return f"<p>Empty spreadsheet from: <a href='{url}' target='_blank'>{url}</a></p>"

                original_cols = len(df.columns)
                
                # Remove columns where all values are identical (but keep at least some columns if possible)
                cols_with_variation = df.loc[:, df.nunique(dropna=False) > 1]
                if not cols_with_variation.empty:
                    df = cols_with_variation
                elif len(df.columns) > 0:
                    # Even if all values are identical in a column, show it if that's all we have
                    # Just take first few columns to avoid showing everything
                    df = df.iloc[:, :min(5, len(df.columns))]

                # Show only top 10 rows
                df = df.head(10)
                
                removed_cols = original_cols - len(df.columns)

                if df.empty:
                    return f"<p>No data to preview from: <a href='{url}' target='_blank'>{url}</a> (file may be empty or all columns identical)</p>"

                # Convert to styled HTML table
                table_html = df.to_html(index=False, escape=True, border=1, classes="table", justify="left")

                preview_note = ""
                if removed_cols > 0:
                    preview_note = f"Showing top 10 rows ({removed_cols} column(s) with identical values removed)"
                else:
                    preview_note = "Showing top 10 rows"

                return f'''
                    <div style="margin-bottom: 25px;">
                        <p style="margin-bottom: 8px;">
                            <a href="{url}" target="_blank" style="font-weight: bold; color: #007bff;">{url}</a><br>
                            <span style="font-size: 12px; color: #555;">{preview_note}</span>
                        </p>
                        <div style="overflow-x:auto;">{table_html}</div>
                    </div>
                '''
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code} for {url}"
            if e.response.status_code == 404:
                error_msg += " (File not found)"
            logger.error(f"{error_msg}: {str(e)}")
            return f"<p style='color:red;'>Could not fetch spreadsheet from {url}: {error_msg}</p>"
        except httpx.RequestError as e:
            logger.error(f"Network error fetching {url}: {str(e)}")
            return f"<p style='color:red;'>Could not fetch spreadsheet from {url}: Network error - {str(e)}</p>"
        except Exception as e:
            logger.error(f"Unexpected error rendering spreadsheet from {url}: {str(e)}", exc_info=True)
            return f"<p style='color:red;'>Could not render spreadsheet from {url}: {str(e)}</p>"

    async def generate_html_content(data: Any, title: str = "") -> str:
        url_pattern = r'https?://[^\s<>"\'()]+(?:\([^\s<>"\'()]*\))?[^\s<>"\'().,;!?]'
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg', '.ico'}
        spreadsheet_extensions = {'.csv', '.xlsx'}

        def extract_urls_from_text(text: str) -> Set[str]:
            return set(re.findall(url_pattern, text))

        def extract_urls_from_data(obj: Any) -> Set[str]:
            urls = set()
            if isinstance(obj, str):
                urls.update(extract_urls_from_text(obj))
            elif isinstance(obj, dict):
                for value in obj.values():
                    urls.update(extract_urls_from_data(value))
            elif isinstance(obj, list):
                for item in obj:
                    urls.update(extract_urls_from_data(item))
            return urls

        def is_image_url(url: str) -> bool:
            return any(url.lower().endswith(ext) for ext in image_extensions)

        def is_spreadsheet_url(url: str) -> bool:
            return any(url.lower().endswith(ext) for ext in spreadsheet_extensions)

        def wrap_urls_in_text(text: str) -> str:
            def replace_url(match):
                url = match.group(0)
                return f'<a href="{url}" target="_blank" rel="noopener noreferrer" style="color: #007bff; text-decoration: underline;">{url}</a>'
            return re.sub(url_pattern, replace_url, text)

        def wrap_urls_in_json(obj: Any) -> Any:
            if isinstance(obj, str):
                return wrap_urls_in_text(obj)
            elif isinstance(obj, dict):
                return {key: wrap_urls_in_json(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [wrap_urls_in_json(item) for item in obj]
            return obj

        def json_to_html(obj: Any, indent: int = 0) -> str:
            indent_str = "&nbsp;" * (indent * 4)
            if isinstance(obj, dict):
                if not obj: return "{}"
                html_parts = ["{<br>"]
                for i, (key, value) in enumerate(obj.items()):
                    comma = "," if i < len(obj) - 1 else ""
                    html_parts.append(f'{indent_str}&nbsp;&nbsp;&nbsp;&nbsp;<span style=" font-weight: bold;">"{key}"</span>: {json_to_html(value, indent + 1)}{comma}<br>')
                html_parts.append(f"{indent_str}}}")
                return "".join(html_parts)
            elif isinstance(obj, list):
                if not obj: return "[]"
                html_parts = ["[<br>"]
                for i, item in enumerate(obj):
                    comma = "," if i < len(obj) - 1 else ""
                    html_parts.append(f'{indent_str}&nbsp;&nbsp;&nbsp;&nbsp;{json_to_html(item, indent + 1)}{comma}<br>')
                html_parts.append(f"{indent_str}]")
                return "".join(html_parts)
            elif isinstance(obj, str):
                return f'<span style="">"{obj}"</span>'
            elif isinstance(obj, bool):
                return f'<span style="font-weight: bold;">{str(obj).lower()}</span>'
            elif obj is None:
                return '<span style="font-weight: bold;">null</span>'
            else:
                return f'<span style="">{obj}</span>'

        # Collect URLs
        all_urls = extract_urls_from_data(data)
        image_urls = [url for url in all_urls if is_image_url(url)]
        sheet_urls = [url for url in all_urls if is_spreadsheet_url(url)]

        # Begin HTML
        base_styles = (
            "padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); font-family: Arial, sans-serif;"
        )
        html_parts = [f'<div style="{base_styles}">']

        if title:
            html_parts.append(f'<h2 style="font-size: 1.5em; margin-top: 0;">{title}</h2>')

        if isinstance(data, (dict, list)):
            wrapped_data = wrap_urls_in_json(data)
            json_html = json_to_html(wrapped_data)
            pre_styles = (
                "border: 1px solid #ddd; padding: 15px; border-radius: 5px; overflow-x: auto; "
                "font-family: 'Courier New', monospace; font-size: 14px; line-height: 1.6; margin: 10px 0;"
            )
            html_parts.append(f'<div style="{pre_styles}">{json_html}</div>')
        elif isinstance(data, (int, float)):
            html_parts.append(f'<p style="font-weight: bold; margin: 10px 0; text-align: center; font-size: 1.2em;">{data}</p>')
        elif isinstance(data, str):
            try:
                parsed_json = json.loads(data)
                wrapped_data = wrap_urls_in_json(parsed_json)
                json_html = json_to_html(wrapped_data)
                pre_styles = (
                    "border: 1px solid #ddd; padding: 15px; border-radius: 5px; overflow-x: auto; "
                    "font-family: 'Courier New', monospace; font-size: 14px; line-height: 1.6; margin: 10px 0;"
                )
                html_parts.append(f'<div style="{pre_styles}">{json_html}</div>')
            except json.JSONDecodeError:
                wrapped_text = wrap_urls_in_text(data)
                escaped_text = wrapped_text.replace('\n', '<br>')
                html_parts.append(f'<div style="line-height: 1.6; margin: 10px 0; word-wrap: break-word;">{escaped_text}</div>')
        else:
            wrapped_text = wrap_urls_in_text(str(data)).replace('\n', '<br>')
            html_parts.append(f'<div style="line-height: 1.6; margin: 10px 0; word-wrap: break-word;">{wrapped_text}</div>')

        # Image Preview
        if image_urls:
            html_parts.append('<hr><h3>Image Previews:</h3><div style="display: flex; flex-wrap: wrap; gap: 15px;">')
            for img_url in image_urls:
                html_parts.append(f'''
                    <div style="border: 1px solid #ddd; border-radius: 8px; padding: 10px; max-width: 290px;">
                        <a href="{img_url}" target="_blank">
                            <img src="{img_url}" alt="Preview" style="width: 100%; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);" 
                                onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                            <div style="display: none; font-size: 12px; color: #666;">Image failed to load</div>
                        </a>
                        <a href="{img_url}" target="_blank" style="font-size: 12px; display: block; color: #007bff; text-decoration: underline;">
                            {img_url if len(img_url) <= 50 else img_url[:50] + '...'}
                        </a>
                    </div>
                ''')
            html_parts.append('</div>')

        # Spreadsheet Preview
        if sheet_urls:
            html_parts.append('<hr><h3>Spreadsheet Previews:</h3><div>')
            for url in sheet_urls:
                preview_html = await generate_table_preview(url)
                html_parts.append(preview_html)
            html_parts.append('</div>')

        html_parts.append('</div>')  # Close main

        return ''.join(html_parts)



    def generate_table_html(data: list, headers: list = None) -> str:
        """
        Generates HTML table content for displaying tabular data.
        
        Args:
            data (list): List of dictionaries or list of lists representing table data
            headers (list): Optional list of column headers
        
        Returns:
            str: HTML div containing a styled table
        """
        
        if not data:
            return generate_html_content("No data to display")
        
        base_styles = (
            "padding: 20px; "
            "border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); "
            "font-family: Arial, sans-serif;"
        )
        
        table_styles = (
            "width: 100%; border-collapse: collapse; margin-top: 10px;"
        )
        
        th_styles = (
            "background-color: #f5f5f5; padding: 12px; text-align: left; "
            "border-bottom: 2px solid #ddd; font-weight: bold;"
        )
        
        td_styles = (
            "padding: 10px 12px; border-bottom: 1px solid #eee;"
        )
        
        html_parts = [f'<div style="{base_styles}">']
        html_parts.append(f'<table style="{table_styles}">')
        
        # Handle headers
        if isinstance(data[0], dict):
            # If data is list of dictionaries, use keys as headers
            headers = headers or list(data[0].keys())
        elif headers is None:
            # If no headers provided for list of lists, create generic ones
            headers = [f"Column {i+1}" for i in range(len(data[0]) if data else 0)]
        
        # Add header row
        if headers:
            html_parts.append('<thead><tr>')
            for header in headers:
                html_parts.append(f'<th style="{th_styles}">{header}</th>')
            html_parts.append('</tr></thead>')
        
        # Add data rows
        html_parts.append('<tbody>')
        for row in data:
            html_parts.append('<tr>')
            if isinstance(row, dict):
                for header in headers:
                    value = row.get(header, '')
                    html_parts.append(f'<td style="{td_styles}">{value}</td>')
            else:
                for cell in row:
                    html_parts.append(f'<td style="{td_styles}">{cell}</td>')
            html_parts.append('</tr>')
        
        html_parts.append('</tbody></table></div>')
        
        return ''.join(html_parts)
        

    @app.post("/api/agent/infer")
    @limiter.limit("10/minute")
    async def agent_infer(
        request: Request,
        agentId: Optional[str] = Form(None),
        userInput: Optional[str] = Form(None),
        file: Optional[UploadFile] = File(None),
        session_id: Optional[str] = Form(None),
        ai_mode: Optional[str] = Form(None),  # Optional ai_mode parameter ('LF' for LangGraph)
        response: Response = None  # Add Response for setting cookies
    ):
        try:
            # Generate or retrieve session_id for unauthenticated users
            session_id = request.cookies.get("session_id") or session_id or str(uuid.uuid4())
            
            # Configure response headers for long-running request to prevent network suspension
            response.headers["Connection"] = "keep-alive"
            response.headers["Keep-Alive"] = "timeout=1200, max=1000"  # 20 minutes timeout, max 1000 requests
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            
            # Add headers to prevent browser from suspending the connection
            response.headers["X-Accel-Buffering"] = "no"  # Disable nginx buffering if behind nginx
            response.headers["X-Content-Type-Options"] = "nosniff"
            
            # Run agent without timeout constraint
            agent_result = await run_agent(request, agentId, userInput, file, response, session_id, None, ai_mode)
            return agent_result
            
        except asyncio.TimeoutError:
            logger.error("Agent inference timed out")
            return MessageResponse(
                type="error",
                content=ErrorData(
                    message="Request timed out",
                    details="The agent took too long to process your request. Please try again."
                ),
                execution_id=str(uuid.uuid4()),
                log_url=None
            )
        except Exception as e:
            logger.error(f"Error in agent_infer: {str(e)}")
            return MessageResponse(
                type="error",
                content=ErrorData(
                    message="Error processing request",
                    details=str(e)
                ),
                execution_id=str(uuid.uuid4()),
                log_url=None
            )
    

    @app.post("/api/agent/infer/stream")
    @limiter.limit("10/minute")
    async def agent_infer_stream(
        request: Request,
        agentId: Optional[str] = Form(None),
        userInput: Optional[str] = Form(None),
        file: Optional[UploadFile] = File(None),
        session_id: Optional[str] = Form(None),
        ai_mode: Optional[str] = Form(None),  # Optional ai_mode parameter ('LF' for LangGraph)
        response: Response = None  # Add Response for setting cookies
    ):
        try:
            # Generate or retrieve session_id for unauthenticated users
            session_id = request.cookies.get("session_id") or session_id or str(uuid.uuid4())
            
            # Configure response headers for long-running request to prevent network suspension
            response.headers["Connection"] = "keep-alive"
            response.headers["Keep-Alive"] = "timeout=1200, max=1000"  # 20 minutes timeout, max 1000 requests
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            
            # Add headers to prevent browser from suspending the connection
            response.headers["X-Accel-Buffering"] = "no"  # Disable nginx buffering if behind nginx
            response.headers["X-Content-Type-Options"] = "nosniff"
            
            # Start timing the execution
            start_time = time.time()
            
            # Create a queue for status updates
            status_queue = asyncio.Queue()
            
            # If a file is uploaded, read and persist it BEFORE starting streaming
            # to avoid 'I/O operation on closed file' when processing in background.
            if file is not None:
                try:
                    contents = await file.read()
                    file_size = len(contents)
                    file_extension = os.path.splitext(file.filename)[1]
                    unique_filename = f"{uuid.uuid4()}{file_extension}"
                    pre_saved_path = UPLOAD_DIR / unique_filename
                    async with aiofiles.open(pre_saved_path, "wb") as buffer:
                        await buffer.write(contents)
                    # Stash saved file info on request.state for downstream processing
                    request.state.pre_saved_file_info = {
                        "original_name": sanitize_for_logging(file.filename),
                        "saved_name": sanitize_for_logging(unique_filename),
                        "size": file_size,
                        "type": file.content_type,
                        "path": str(pre_saved_path),
                        "uploaded_at": datetime.now(timezone.utc).isoformat()
                    }
                    # Do not pass the UploadFile object further; it may be closed after return
                    file = None
                except Exception as e:
                    logger.error(f"Failed to pre-save uploaded file: {sanitize_for_logging(str(e))}")
                    # Proceed without file if saving failed
                    file = None

            # Start a background task to run the agent
            background_task = asyncio.create_task(
                run_agent_with_status_updates(request, agentId, userInput, file, response, session_id, status_queue, ai_mode)
            )
            
            # Define the streaming response generator
            async def stream_response():
                try:
                    # First yield a status update that we're starting
                    yield json.dumps({"status": "starting", "message": "Processing your request.."}) + "\n"
                    
                    # Stream status updates as they come in
                    last_heartbeat_time = 0
                    while True:
                        try:
                            # Get status update with a timeout
                            status_update = await asyncio.wait_for(status_queue.get(), timeout=300.0)
                            
                            if status_update == "DONE":
                                # Get the final result from the background task
                                agent_result = await background_task
                                
                                # Extract execution_id from the result
                                execution_id = agent_result.execution_id if hasattr(agent_result, 'execution_id') else None
                                
                                # Calculate execution time
                                execution_time = time.time() - start_time
                                
                                # Get agent name for logging
                                agents = load_agents()
                                agent_name = None
                                for agent in agents:
                                    if agent["id"] == agentId:
                                        agent_name = agent.get("name", "Unknown Agent")
                                        break
                                
                                # Convert agent_result to dict and add session_id
                                result_dict = agent_result.dict() if hasattr(agent_result, 'dict') else dict(agent_result)
                                result_dict['session_id'] = session_id
                                
                                # Log the request and response with explicit argument naming
                                await log_request_to_json(
                                    request=request,
                                    user_input=userInput if userInput else "",
                                    response_data=result_dict,
                                    session_id=session_id,
                                    execution_time=execution_time,
                                    agent_id=agentId,
                                    agent_name=agent_name,
                                    execution_id=execution_id
                                )
                                
                                # Yield the final result with execution time and session_id
                                yield json.dumps({"status": "complete", "result": result_dict, "execution_time_seconds": execution_time}) + "\n"
                                break
                            else:
                                # Yield status update
                                yield json.dumps({"status": "processing", "message": status_update}) + "\n"
                                
                        except asyncio.TimeoutError:
                            # Throttle heartbeats (max once every 10s) to avoid spamming
                            now = time.time()
                            if now - last_heartbeat_time >= 10:
                                yield json.dumps({"status": "processing", "message": "Processing continues.."}) + "\n"
                                last_heartbeat_time = now
                            continue
                        except Exception as e:
                            logger.error(f"Error in stream_response: {str(e)}")
                            yield json.dumps({"status": "error", "message": f"Error: {str(e)}"}) + "\n"
                            break
                            
                except Exception as e:
                    logger.error(f"Error in stream_response generator: {str(e)}")
                    yield json.dumps({"status": "error", "message": f"Stream error: {str(e)}"}) + "\n"
            
            # Return streaming response
            from fastapi.responses import StreamingResponse
            return StreamingResponse(
                stream_response(),
                media_type="application/x-ndjson",
                headers={
                    "Transfer-Encoding": "chunked",
                    "Connection": "keep-alive"
                }
            )
            
        except Exception as e:
            logger.error(f"Error in agent_infer_stream: {str(e)}")
            # Return error as streaming response for consistency
            async def error_stream():
                yield json.dumps({
                    "status": "error", 
                    "message": f"Error: {str(e)}"
                }) + "\n"
            
            from fastapi.responses import StreamingResponse
            return StreamingResponse(
                error_stream(),
                media_type="application/x-ndjson"
            )
    

    async def log_request_to_json(request: Request, user_input: str, response_data: dict, session_id: str, execution_time: float, agent_id: str = None, agent_name: str = None, execution_id: str = None, user: str = None):
        timestamp_utc = datetime.now(timezone.utc)
        log_entry = {
            "timestamp": timestamp_utc.isoformat(),
            "ip_address": request.client.host,
            "user_agent": request.headers.get("user-agent", "Unknown"),
            "request_headers": dict(request.headers),
            "user_input": user_input,
            "response": response_data,
            "execution_time_seconds": execution_time,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "endpoint": str(request.url),
            "method": request.method,
            "execution_id": execution_id
        }

        if SAVE_SESSION_FILE:
            log_dir = "request_logs"
            os.makedirs(log_dir, exist_ok=True)
            ist = pytz.timezone('Asia/Kolkata')
            current_date = datetime.now(ist).strftime("%Y-%m-%d")
            log_file = f"{log_dir}/session_logs_{current_date}.json"

            logs = {}
            if os.path.exists(log_file):
                async with aiofiles.open(log_file, 'r') as f:
                    content = await f.read()
                    if content.strip():
                        logs = json.loads(content)

            if session_id not in logs:
                logs[session_id] = []
            logs[session_id].append(log_entry)

            async with aiofiles.open(log_file, 'w') as f:
                await f.write(json.dumps(logs, indent=2, ensure_ascii=False))

        # Database logging (ensure SessionLogsDB is async)
        try:
            success, error = await session_logs_db.log_session(
                session_id=session_id,
                timestamp=timestamp_utc,
                ip_address=request.client.host,
                user_agent=request.headers.get("user-agent", "Unknown"),
                request_headers=dict(request.headers),
                user_input=user_input,
                response=response_data,
                execution_time_seconds=execution_time,
                agent_id=agent_id,
                agent_name=agent_name,
                endpoint=str(request.url),
                method=request.method,
                execution_id=execution_id,
                user_email=user  # Fixed: use user_email parameter name
            )
            if not success:
                logger.error(f"Failed to store session log in database for session {session_id}: {error}")
                # Log but don't fail the request - session logging is non-critical
        except Exception as e:
            logger.error(f"Exception storing session log in database for session {session_id}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            # Log but don't fail the request - session logging is non-critical

    @app.post("/api/agent/execute")
    @limiter.limit("15/minute")
    @memory_monitor
    async def execute_agent(
        request: Request,
        response: Response = None
    ):
        try:
            # return {
            #     "type": "text",
            #     "content": {
            #         "text": "\"Product Name\": Play On Cat Toy Silvervine Sucker\n\"Description\": Hunt, Swat & Pounce: Hours of interactive play\n\"Segment\" : Specialty Pet Owner\n\"Image URL\": https://delivery-petsuppliesplus.stylelabs.cloud/api/public/content/750683139725_1a\n\"Event Theme\" : Summer Travel and Safety for New Pet Owner\n\"Reason To Choose\": This product was selected based on product-related information (90% weight) and user input (10% weight). The match achieved a similarity score of 0.951. Key matching factors include the product name 'Play On Cat Toy Silvervine Sucker' and description that align well with the search criteria.\n\"Headline\": Summer Fun & Safety for Your Feline!\n\"Pitch\": Keep your new cat entertained and secure on every adventure with the Play On Silvervine Sucker!\n\"Platform\" : Meta\n\"matched_keyword\" : cat treat sampler\n\nGenerated banner Image: \"https://tools.impact-agents.ai/generated_images/6458befd.png\""
            #     },
            #     "execution_id": "b3f7117c-f3be-46c0-b026-8d14bd128a28_20250827_141749",
            #     "log_url": "http://localhost:8000/api/logs/b3f7117c-f3be-46c0-b026-8d14bd128a28_20250827_141749",
            #     "session_id": "a4b4039a-8018-4f80-a959-805aceaa68c3",
            #     "html_content": "<div style=\"padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); font-family: Arial, sans-serif;\"><div style=\"line-height: 1.6; margin: 10px 0; word-wrap: break-word;\">\"Product Name\": Play On Cat Toy Silvervine Sucker<br>\"Description\": Hunt, Swat & Pounce: Hours of interactive play<br>\"Segment\" : Specialty Pet Owner<br>\"Image URL\": <a href=\"https://delivery-petsuppliesplus.stylelabs.cloud/api/public/content/750683139725_1a\" target=\"_blank\" rel=\"noopener noreferrer\" style=\"color: #007bff; text-decoration: underline;\">https://delivery-petsuppliesplus.stylelabs.cloud/api/public/content/750683139725_1a</a><br>\"Event Theme\" : Summer Travel and Safety for New Pet Owner<br>\"Reason To Choose\": This product was selected based on product-related information (90% weight) and user input (10% weight). The match achieved a similarity score of 0.951. Key matching factors include the product name 'Play On Cat Toy Silvervine Sucker' and description that align well with the search criteria.<br>\"Headline\": Summer Fun & Safety for Your Feline!<br>\"Pitch\": Keep your new cat entertained and secure on every adventure with the Play On Silvervine Sucker!<br>\"Platform\" : Meta<br>\"matched_keyword\" : cat treat sampler<br><br>Generated banner Image: \"<a href=\"https://tools.impact-agents.ai/generated_images/6458befd.png\" target=\"_blank\" rel=\"noopener noreferrer\" style=\"color: #007bff; text-decoration: underline;\">https://tools.impact-agents.ai/generated_images/6458befd.png</a>\"</div><hr><h3>Image Previews:</h3><div style=\"display: flex; flex-wrap: wrap; gap: 15px;\">\n                    <div style=\"border: 1px solid #ddd; border-radius: 8px; padding: 10px; max-width: 290px;\">\n                        <a href=\"https://tools.impact-agents.ai/generated_images/6458befd.png\" target=\"_blank\">\n                            <img src=\"https://tools.impact-agents.ai/generated_images/6458befd.png\" alt=\"Preview\" style=\"width: 100%; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);\" \n                                onerror=\"this.style.display='none'; this.nextElementSibling.style.display='block';\">\n                            <div style=\"display: none; font-size: 12px; color: #666;\">Image failed to load</div>\n                        </a>\n                        <a href=\"https://tools.impact-agents.ai/generated_images/6458befd.png\" target=\"_blank\" style=\"font-size: 12px; display: block; color: #007bff; text-decoration: underline;\">\n                            https://tools.impact-agents.ai/generated_images/64...\n                        </a>\n                    </div>\n                </div></div>",
            #     "user_input": "Generate a banner image for Play On Cat Toy Silvervine Sucker for Summer Travel and Safety for New Pet Owner"
            # }
        
            if request.headers.get("AGENT-API-KEY", "") == AGENT_API_KEY:
                # Parse JSON with error handling
                try:
                    json_data = await request.json()
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in execute request: {str(e)}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid JSON in request body: {str(e)}. Please ensure your JSON is properly formatted with double quotes."
                    )
                except Exception as e:
                    logger.error(f"Error parsing request JSON: {str(e)}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Error parsing request body: {str(e)}"
                    )
                
                agentId = json_data.get("agentId")
                userInput = json_data.get("userInput")
                user = json_data.get("user")  # Optional user parameter
                ai_mode = json_data.get("ai_mode")  # Optional ai_mode parameter ('LF' for LangGraph)
                
                # Generate or retrieve session_id for unauthenticated users
                session_id = request.cookies.get("session_id") or json_data.get("session_id") or str(uuid.uuid4())
                
                # Check for cached response before running the agent
                cached_response = get_cached_agent_response(agentId, userInput)
                if cached_response is not None:
                    logging.info(f"Returning cached response for agent {agentId}")
                    
                    # Add intentional delay for cached responses (5-6 seconds)
                    await asyncio.sleep(5.5)
                    
                    # Update session_id in cached response if needed
                    cached_response["session_id"] = session_id
                    if userInput:
                        cached_response["user_input"] = userInput
                    
                    # Convert cached response to MessageResponse
                    # Handle content field - it might be a dict with "text" key or already a dict
                    content_data = cached_response.get("content", {})
                    if isinstance(content_data, dict) and "text" in content_data:
                        # Create TextData object
                        content_obj = TextData(text=content_data["text"])
                    else:
                        # Fallback: use content_data as-is
                        content_obj = content_data
                    
                    agent_result = MessageResponse(
                        type=cached_response.get("type", "text"),
                        content=content_obj,
                        execution_id=cached_response.get("execution_id"),
                        log_url=cached_response.get("log_url"),
                        session_id=session_id,
                        html_content=cached_response.get("html_content"),
                        user_input=userInput if userInput else cached_response.get("user_input"),
                        sql=cached_response.get("sql"),
                        more_instructions=cached_response.get("more_instructions"),
                        messages=cached_response.get("messages"),
                        message_blocks=cached_response.get("message_blocks")
                    )
                    
                    # Log the cached request
                    agents = load_agents()
                    agent_name = None
                    for agent in agents:
                        if agent["id"] == agentId:
                            agent_name = agent.get("name", "Unknown Agent")
                            break
                    
                    await log_request_to_json(
                        request=request,
                        user_input=userInput if userInput else "",
                        response_data=agent_result.dict() if hasattr(agent_result, 'dict') else dict(agent_result),
                        session_id=session_id,
                        execution_time=5.5,  # Cached response with intentional delay
                        agent_id=agentId,
                        agent_name=agent_name,
                        execution_id=cached_response.get("execution_id"),
                        user=user
                    )
                    
                    return agent_result
                
                # Configure response headers for long-running request to prevent network suspension
                response.headers["Connection"] = "keep-alive"
                response.headers["Keep-Alive"] = "timeout=1200, max=1000"  # 20 minutes timeout, max 1000 requests
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
                
                # Add headers to prevent browser from suspending the connection
                response.headers["X-Accel-Buffering"] = "no"  # Disable nginx buffering if behind nginx
                response.headers["X-Content-Type-Options"] = "nosniff"
                
                # Start timing the execution
                start_time = time.time()
                
                # Run the agent
                agent_result = await run_agent(request, agentId, userInput, None, response, session_id, None, ai_mode)
                
                # Extract execution_id from the result
                execution_id = agent_result.execution_id if hasattr(agent_result, 'execution_id') else None
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Get agent name for logging
                agents = load_agents()
                agent_name = None
                for agent in agents:
                    if agent["id"] == agentId:
                        agent_name = agent.get("name", "Unknown Agent")
                        break
                
                # Log the request and response with explicit argument naming
                await log_request_to_json(
                    request=request,
                    user_input=userInput if userInput else "",
                    response_data=agent_result.dict() if hasattr(agent_result, 'dict') else dict(agent_result),
                    session_id=session_id,
                    execution_time=execution_time,
                    agent_id=agentId,
                    agent_name=agent_name,
                    execution_id=execution_id,
                    user=user
                )
                
                return agent_result
            else:
                raise HTTPException(status_code=401, detail="Unauthorized access. Please provide a valid Agent API key and try again.")
        except asyncio.TimeoutError:
            logger.error("Agent execution timed out")
            raise HTTPException(
                status_code=408, 
                detail="Request timed out. The agent took too long to process your request. Please try again."
            )
        except Exception as e:
            logger.error(f"Error in execute_agent: {str(e)}")
            raise

    @app.post("/api/agent/langgraph/execute")
    @limiter.limit("15/minute")
    @memory_monitor
    async def execute_langgraph_agent(
        request: Request,
        response: Response = None
    ):
        """
        Execute LangGraph agent (Text-to-SQL agent).
        Accepts the same input format as /api/agent/execute but uses LangGraph instead of CrewAI.
        """
        try:
            if request.headers.get("AGENT-API-KEY", "") == AGENT_API_KEY:
                # Parse JSON with error handling
                try:
                    json_data = await request.json()
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in langgraph execute request: {str(e)}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid JSON in request body: {str(e)}. Please ensure your JSON is properly formatted with double quotes."
                    )
                except Exception as e:
                    logger.error(f"Error parsing request JSON: {str(e)}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Error parsing request body: {str(e)}"
                    )
                
                agentId = json_data.get("agentId")
                userInput = json_data.get("userInput")
                user = json_data.get("user")  # Optional user parameter
                guardrails_req = bool(json_data.get("guardrails_req", False))
                messages = json_data.get("messages", [])  # Optional conversation history
                
                # Generate agentId if not provided from frontend
                if not agentId:
                    agentId = str(uuid.uuid4())
                    logger.info(f"[LangGraph] No agentId provided in request, generated new agentId: {agentId}")
                else:
                    logger.info(f"[LangGraph] Using agentId from request: {agentId}")
                
                # Generate or retrieve session_id for unauthenticated users
                # Priority: 1. Cookie, 2. JSON body, 3. Generate new UUID
                session_id = request.cookies.get("session_id") or json_data.get("session_id") or str(uuid.uuid4())
                
                # If messages are not explicitly provided, try to fetch from session logs
                if not messages or len(messages) == 0:
                    try:
                        # Fetch conversation history from session logs
                        conversation_data, error = await session_logs_db.get_session_conversation(session_id)
                        
                        if not error and conversation_data and conversation_data.get("messages"):
                            # Convert session log format to messages format
                            # Session logs have: user_input, response
                            # Messages format needs: [{"type": "human", "content": ...}, {"type": "ai", "content": ...}]
                            session_messages = conversation_data.get("messages", [])
                            messages = []
                            
                            for log_entry in session_messages:
                                # Add human message (user_input)
                                if log_entry.get("user_input"):
                                    messages.append({
                                        "type": "human",
                                        "content": str(log_entry["user_input"])
                                    })
                                
                                # Add AI message (response)
                                if log_entry.get("response"):
                                    # Response might be a JSON string or dict
                                    response_content = log_entry["response"]
                                    if isinstance(response_content, str):
                                        try:
                                            # Try to parse as JSON to extract text content
                                            response_dict = json.loads(response_content)
                                            # Extract text from response (could be in content.text or just text)
                                            if isinstance(response_dict, dict):
                                                if "content" in response_dict:
                                                    content_obj = response_dict["content"]
                                                    if isinstance(content_obj, dict) and "text" in content_obj:
                                                        ai_text = content_obj["text"]
                                                    elif isinstance(content_obj, str):
                                                        ai_text = content_obj
                                                    else:
                                                        ai_text = str(response_dict.get("text", response_content))
                                                else:
                                                    ai_text = str(response_dict.get("text", response_content))
                                            else:
                                                ai_text = str(response_content)
                                        except (json.JSONDecodeError, TypeError):
                                            # If not JSON, use as-is
                                            ai_text = response_content
                                    elif isinstance(response_content, dict):
                                        # Already a dict, extract text
                                        if "content" in response_content:
                                            content_obj = response_content["content"]
                                            if isinstance(content_obj, dict) and "text" in content_obj:
                                                ai_text = content_obj["text"]
                                            elif isinstance(content_obj, str):
                                                ai_text = content_obj
                                            else:
                                                ai_text = str(response_content.get("text", ""))
                                        else:
                                            ai_text = str(response_content.get("text", ""))
                                    else:
                                        ai_text = str(response_content)
                                    
                                    if ai_text and ai_text.strip():
                                        messages.append({
                                            "type": "ai",
                                            "content": ai_text.strip()
                                        })
                            
                            logger.info(f"[LangGraph] Fetched {len(messages)} messages from session logs for session_id: {session_id}")
                        else:
                            logger.info(f"[LangGraph] No conversation history found for session_id: {session_id}")
                    except Exception as e:
                        logger.warning(f"[LangGraph] Error fetching messages from session logs: {str(e)}")
                        messages = []
                
                # Validate messages format
                if messages and isinstance(messages, list):
                    # Ensure messages are in correct format: [{"type": "human"|"ai", "content": str}, ...]
                    validated_messages = []
                    for msg in messages:
                        if isinstance(msg, dict) and "type" in msg and "content" in msg:
                            msg_type = msg.get("type")
                            if msg_type in ["human", "ai"]:
                                validated_messages.append({
                                    "type": msg_type,
                                    "content": str(msg.get("content", ""))
                                })
                    messages = validated_messages
                else:
                    messages = []
                
                # Configure response headers for long-running request to prevent network suspension
                response.headers["Connection"] = "keep-alive"
                response.headers["Keep-Alive"] = "timeout=1200, max=1000"  # 20 minutes timeout, max 1000 requests
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
                
                # Add headers to prevent browser from suspending the connection
                response.headers["X-Accel-Buffering"] = "no"  # Disable nginx buffering if behind nginx
                response.headers["X-Content-Type-Options"] = "nosniff"
                
                # Generate execution ID (same format as run_agent)
                execution_uuid = str(uuid.uuid4())
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                execution_id = f"{execution_uuid}_{timestamp}"
                log_filename = f"langgraph_agent_execution_{execution_id}.log"
                log_file = os.path.join(LOG_DIR, log_filename)
                
                # Configure logger using centralized logger manager
                # logger = logger
                # configure_file_logger(
                #     logger_name=f"langgraph_agent_{execution_id}",
                #     log_file=log_file,
                #     level=logging.DEBUG,
                #     unique_identifier=execution_id
                # )
                
                # Start timing the execution
                start_time = time.time()
                
                # Import LangGraph agent
                from langgraph_agent_poc.main import build_text_to_sql_graph
                from langgraph_agent_poc.types import AgentState
                
                # Build and run the LangGraph agent
                graph = build_text_to_sql_graph()
                initial_state: AgentState = {
                    "user_query": userInput if userInput else "",
                    "messages": messages,  # Pass conversation history
                    "agent_id": agentId,  # Pass agent_id for fetching agent-specific tools
                }
                
                logger.info(f"[LangGraph] Executing agent {agentId} with query: {userInput[:100] if userInput else 'empty'}...")
                logger.info(f"[LangGraph] Execution ID: {execution_id}")
                
                # Note: guardrails_req is accepted but not implemented for LangGraph agent yet
                if guardrails_req:
                    logger.warning("[LangGraph] Guardrails requested but not yet implemented for LangGraph agent")
                
                final_state: AgentState = await graph.ainvoke(initial_state)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Debug: Log the entire final state to see what's there
                logger.info(f"[LangGraph] Final state keys: {list(final_state.keys())}")
                logger.info(f"[LangGraph] Final state preview: {json.dumps({k: str(v)[:200] if isinstance(v, (str, dict, list)) else v for k, v in list(final_state.items())[:5]}, default=str)}")
                
                # Extract and log history_context for debugging
                history_context = final_state.get("history_context")
                messages = final_state.get("messages", [])
                messages_count = len(messages)
                
                # Print last few messages for context
                if messages_count > 0:
                    print(f"[HISTORY CONTEXT DEBUG] Last {min(4, messages_count)} messages:")
                    for i, msg in enumerate(messages[-4:], start=max(1, messages_count-3)):
                        msg_type = msg.get("type", "unknown")
                        content = msg.get("content", "")[:100]  # Truncate for readability
                        print(f"  [{i}] {msg_type.upper()}: {content}...")
                else:
                    print("  → No previous messages in conversation history")
                
                print(f"[HISTORY CONTEXT DEBUG] History Context:")
                if history_context:
                    print(f"  {history_context}")
                else:
                    print("  → No history context extracted (not a follow-up question or no previous messages)")
                print("=" * 80)
                
                logger.info(f"[LangGraph] Session {session_id} - History Context: {history_context if history_context else 'None (not a follow-up)'}")
                logger.info(f"[LangGraph] Session {session_id} - Messages count: {messages_count}")
                if messages_count > 0:
                    logger.info(f"[LangGraph] Session {session_id} - Last message types: {[msg.get('type') for msg in messages[-4:]]}")
                
                # Extract final answer from state
                # The smart_chart_rendering_node stores result in final_answer
                final_answer = final_state.get("final_answer", {})
                
                # Extract SQL and more_instructions from state
                sql_query = final_state.get("sql", "")
                more_instructions = final_state.get("more_instructions", "")
                
                logger.info(f"[LangGraph] final_answer type: {type(final_answer)}")
                logger.info(f"[LangGraph] final_answer content: {json.dumps(final_answer, default=str)[:1000] if isinstance(final_answer, dict) else str(final_answer)[:500]}")
                logger.info(f"[LangGraph] SQL query length: {len(sql_query)}, more_instructions length: {len(more_instructions)}")
                
                if isinstance(final_answer, dict):
                    answer_text = final_answer.get("text", "")
                    html_content = final_answer.get("html_content", "")
                    message_blocks = final_answer.get("blocks", [])  # Extract blocks from final_answer
                    logger.info(f"[LangGraph] Extracted from final_answer - text length: {len(answer_text)}, html_content length: {len(html_content)}, blocks count: {len(message_blocks)}")
                else:
                    answer_text = ""
                    html_content = ""
                    message_blocks = []
                    logger.warning(f"[LangGraph] final_answer is not a dict, using empty values")
                
                # Fallback: if no final_answer, try to get answer from state
                if not answer_text:
                    answer_text = final_state.get("answer", "")
                    logger.info(f"[LangGraph] Using fallback answer from state, length: {len(answer_text)}")
                
                # Generate HTML content if not present but we have text
                # IMPORTANT: Only generate HTML if html_content is truly empty (not just whitespace)
                # This preserves the chart HTML from smart_chart_rendering_node
                if not html_content or (isinstance(html_content, str) and len(html_content.strip()) == 0):
                    if answer_text:
                        logger.info(f"[LangGraph] html_content is empty or whitespace, generating from text")
                        html_content = await generate_html_content(answer_text)
                else:
                    logger.info(f"[LangGraph] Using html_content from final_answer (preserving chart HTML with {len(html_content)} chars)")
                    # Log a preview to verify it's the chart HTML
                    if len(html_content) > 200:
                        logger.info(f"[LangGraph] html_content preview: {html_content[:200]}...")
                    else:
                        logger.info(f"[LangGraph] html_content: {html_content}")
                    
                if not answer_text and not html_content:
                    # Last resort: use summary or query info
                    logger.warning(f"[LangGraph] Both answer_text and html_content are empty, using last resort")
                    tool_responses = final_state.get("tool_responses", {})
                    raw_response = tool_responses.get("raw_response", {})
                    if isinstance(raw_response, dict):
                        answer_text = raw_response.get("summary", "Query executed successfully.")
                    else:
                        answer_text = "Query executed successfully."
                    html_content = await generate_html_content(answer_text)
                
                # Get agent name for logging (try to get from agents list, fallback to default)
                agents = load_agents()
                agent_name = "LangGraph Text-to-SQL Agent"  # Default name
                for agent in agents:
                    if agent["id"] == agentId:
                        agent_name = agent.get("name", "LangGraph Text-to-SQL Agent")
                        break
                
                # Generate log URL
                log_url = f"{BASE_URL}/api/logs/{execution_id}"
                
                # Clean HTML content to remove any escaped characters
                # This ensures we don't have double-escaped quotes or newlines
                if html_content:
                    # Import the clean function
                    from langgraph_agent_poc.response_formatter_node import clean_html_content
                    original_length = len(html_content)
                    # Check if HTML contains escaped quotes
                    has_escaped_quotes = '\\"' in html_content
                    logger.info(f"[LangGraph] Before cleaning - length: {original_length}, has escaped quotes: {has_escaped_quotes}")
                    html_content = clean_html_content(html_content)
                    logger.info(f"[LangGraph] After cleaning - length: {len(html_content)}, still has escaped quotes: {'\\\"' in html_content}")
                    if len(html_content) > 300:
                        logger.info(f"[LangGraph] HTML preview (first 300 chars): {html_content[:300]}")
                
                # Get updated messages from final state (includes new user query and AI response)
                updated_messages = final_state.get("messages", messages)
                
                # CRITICAL: Serialize dict fields in messages to JSON strings for MessageResponse
                # MessageResponse expects messages to be List[Dict[str, str]], but we store data and final_answer as dicts
                if updated_messages:
                    serialized_messages = []
                    for msg in updated_messages:
                        if isinstance(msg, dict):
                            serialized_msg = {
                                "type": msg.get("type", ""),
                                "content": msg.get("content", "")
                            }
                            # Serialize data and final_answer to JSON strings if they exist
                            if "data" in msg and msg["data"]:
                                if isinstance(msg["data"], dict):
                                    serialized_msg["data"] = json.dumps(msg["data"])
                                else:
                                    serialized_msg["data"] = str(msg["data"])
                            if "final_answer" in msg and msg["final_answer"]:
                                if isinstance(msg["final_answer"], dict):
                                    serialized_msg["final_answer"] = json.dumps(msg["final_answer"])
                                else:
                                    serialized_msg["final_answer"] = str(msg["final_answer"])
                            serialized_messages.append(serialized_msg)
                        else:
                            serialized_messages.append(msg)
                    updated_messages = serialized_messages
                
                # CRITICAL: Convert message_blocks dicts to Pydantic models before validation
                # This ensures Chart objects have series=None and only highchartsSeries set
                validated_blocks = None
                if message_blocks:
                    from langgraph_agent_poc.types import HtmlBlock, ProductCarouselBlock, ChartBlock, KPIBlock, DividerBlock, Chart, HighchartsSeries
                    validated_blocks = []
                    for block_dict in message_blocks:
                        if not isinstance(block_dict, dict):
                            validated_blocks.append(block_dict)
                            continue
                        
                        block_type = block_dict.get("type")
                        
                        # Handle ChartBlock specially - keep series and highchartsSeries as dicts
                        if block_type == "chart":
                            try:
                                chart_dict = block_dict.get("chart", {})
                                # Chart dict should have 'series' field (final output format)
                                # Keep both series and highchartsSeries as dicts (not Pydantic objects)
                                if "series" in chart_dict:
                                    # Get series list and ensure they are dicts (not Pydantic objects)
                                    series_list = chart_dict.get("series", [])
                                    
                                    # Convert any Pydantic objects to dicts
                                    series_dicts = []
                                    for s in series_list:
                                        if isinstance(s, dict):
                                            series_dicts.append(s)
                                        elif hasattr(s, 'dict'):
                                            # It's a Pydantic model, convert to dict
                                            series_dicts.append(s.dict(exclude_none=True))
                                        elif hasattr(s, 'model_dump'):
                                            # Pydantic v2
                                            series_dicts.append(s.model_dump(exclude_none=True))
                                        else:
                                            # Fallback: try to convert to dict
                                            series_dicts.append(dict(s) if hasattr(s, '__dict__') else s)
                                    
                                    # Clean yAxis to remove None values from each axis config
                                    y_axis_cleaned = None
                                    y_axis_raw = chart_dict.get("yAxis")
                                    if y_axis_raw:
                                        if isinstance(y_axis_raw, list):
                                            y_axis_cleaned = []
                                            for axis in y_axis_raw:
                                                if isinstance(axis, dict):
                                                    # Remove None values from axis dict
                                                    cleaned_axis = {k: v for k, v in axis.items() if v is not None}
                                                    y_axis_cleaned.append(cleaned_axis)
                                                else:
                                                    y_axis_cleaned.append(axis)
                                        elif isinstance(y_axis_raw, dict):
                                            y_axis_cleaned = {k: v for k, v in y_axis_raw.items() if v is not None}
                                        else:
                                            y_axis_cleaned = y_axis_raw
                                    
                                    # Create chart dict with only defined fields (no None values)
                                    chart_data = {}
                                    if chart_dict.get("type"):
                                        chart_data["type"] = chart_dict.get("type")
                                    if chart_dict.get("chart"):
                                        chart_data["chart"] = chart_dict.get("chart")
                                    if chart_dict.get("title"):
                                        chart_data["title"] = chart_dict.get("title")
                                    if chart_dict.get("xAxis"):
                                        chart_data["xAxis"] = chart_dict.get("xAxis")
                                    if y_axis_cleaned:
                                        chart_data["yAxis"] = y_axis_cleaned
                                    if chart_dict.get("tooltip"):
                                        chart_data["tooltip"] = chart_dict.get("tooltip")
                                    if chart_dict.get("legend"):
                                        chart_data["legend"] = chart_dict.get("legend")
                                    if chart_dict.get("plotOptions"):
                                        chart_data["plotOptions"] = chart_dict.get("plotOptions")
                                    chart_data["series"] = series_dicts
                                    # Don't include highchartsSeries - we only use series
                                    
                                    # Recursively remove None values from chart_data
                                    def remove_none_values_recursive(obj):
                                        if isinstance(obj, dict):
                                            cleaned = {}
                                            for k, v in obj.items():
                                                if v is not None:
                                                    cleaned[k] = remove_none_values_recursive(v)
                                            return cleaned
                                        elif isinstance(obj, list):
                                            return [remove_none_values_recursive(item) for item in obj if item is not None]
                                        else:
                                            return obj
                                    
                                    # Keep chart as dict to avoid Pydantic adding None defaults
                                    cleaned_chart_data = remove_none_values_recursive(chart_data)
                                    
                                    # Keep as dict instead of converting to Pydantic model
                                    chart_block_dict = {
                                        "id": block_dict.get("id", f"chart_{uuid.uuid4().hex[:8]}"),
                                        "type": "chart",
                                        "order": block_dict.get("order", 0),
                                        "chart": cleaned_chart_data
                                    }
                                    validated_blocks.append(chart_block_dict)
                                else:
                                    # Keep as dict if missing series field
                                    logger.warning(f"[execute_langgraph_agent] Chart block {block_dict.get('id')} missing series field, keeping as dict")
                                    validated_blocks.append(block_dict)
                            except Exception as e:
                                logger.error(f"[execute_langgraph_agent] Error processing chart block {block_dict.get('id')}: {e}, keeping as dict")
                                validated_blocks.append(block_dict)
                        elif block_type == "html":
                            try:
                                validated_blocks.append(HtmlBlock(**block_dict))
                            except Exception as e:
                                logger.error(f"[execute_langgraph_agent] Error processing HTML block {block_dict.get('id')}: {e}, keeping as dict")
                                validated_blocks.append(block_dict)
                        elif block_type == "product_carousel":
                            try:
                                validated_blocks.append(ProductCarouselBlock(**block_dict))
                            except Exception as e:
                                logger.error(f"[execute_langgraph_agent] Error processing carousel block {block_dict.get('id')}: {e}, keeping as dict")
                                validated_blocks.append(block_dict)
                        elif block_type == "kpi":
                            try:
                                validated_blocks.append(KPIBlock(**block_dict))
                            except Exception as e:
                                logger.error(f"[execute_langgraph_agent] Error processing KPI block {block_dict.get('id')}: {e}, keeping as dict")
                                validated_blocks.append(block_dict)
                        elif block_type == "divider":
                            try:
                                validated_blocks.append(DividerBlock(**block_dict))
                            except Exception as e:
                                logger.error(f"[execute_langgraph_agent] Error processing divider block {block_dict.get('id')}: {e}, keeping as dict")
                                validated_blocks.append(block_dict)
                        else:
                            # Unknown block type - keep as dict instead of skipping
                            logger.warning(f"[execute_langgraph_agent] Unknown block type: {block_type}, keeping as dict")
                            validated_blocks.append(block_dict)
                
                # Clean validated_blocks to remove None values before creating MessageResponse
                # This prevents Pydantic from adding None defaults when validating chart blocks
                cleaned_validated_blocks = None
                if validated_blocks:
                    def clean_block_for_response(block):
                        """Clean block to remove None values before Pydantic validation."""
                        if isinstance(block, dict):
                            def remove_none_recursive(obj):
                                if isinstance(obj, dict):
                                    return {k: remove_none_recursive(v) for k, v in obj.items() if v is not None}
                                elif isinstance(obj, list):
                                    return [remove_none_recursive(item) for item in obj if item is not None]
                                else:
                                    return obj
                            return remove_none_recursive(block)
                        elif hasattr(block, 'model_dump'):
                            block_dict = block.model_dump(exclude_none=True)
                            def remove_none_recursive(obj):
                                if isinstance(obj, dict):
                                    return {k: remove_none_recursive(v) for k, v in obj.items() if v is not None}
                                elif isinstance(obj, list):
                                    return [remove_none_recursive(item) for item in obj if item is not None]
                                else:
                                    return obj
                            return remove_none_recursive(block_dict)
                        elif hasattr(block, 'dict'):
                            block_dict = block.dict(exclude_none=True)
                            def remove_none_recursive(obj):
                                if isinstance(obj, dict):
                                    return {k: remove_none_recursive(v) for k, v in obj.items() if v is not None}
                                elif isinstance(obj, list):
                                    return [remove_none_recursive(item) for item in obj if item is not None]
                                else:
                                    return obj
                            return remove_none_recursive(block_dict)
                        return block
                    
                    cleaned_validated_blocks = [clean_block_for_response(block) for block in validated_blocks]
                
                # Create MessageResponse
                # Note: Pydantic will try to validate dicts against MessageBlock types
                # For chart blocks, we keep them as dicts to avoid None defaults
                result = MessageResponse(
                    type="text",
                    content=TextData(text=answer_text),
                    execution_id=execution_id,
                    log_url=log_url,
                    session_id=session_id,
                    html_content=html_content,
                    user_input=userInput,
                    sql=sql_query if sql_query else None,
                    more_instructions=more_instructions if more_instructions else None,
                    messages=updated_messages if updated_messages else None,  # Include updated conversation history
                    message_blocks=cleaned_validated_blocks if cleaned_validated_blocks else None
                )
                
                # Prepare response data for logging - ensure blocks are included for consistency
                # Use model_dump() if available (Pydantic v2), otherwise use dict()
                # Exclude None values to prevent them from being added
                if hasattr(result, 'model_dump'):
                    response_data_dict = result.model_dump(exclude_none=True)
                elif hasattr(result, 'dict'):
                    response_data_dict = result.dict(exclude_none=True)
                else:
                    response_data_dict = dict(result)
                
                # Convert message_blocks to list of dicts if they're Pydantic models
                # Clean None values from blocks
                def clean_block_dict(block):
                    """Convert block to dict and remove None values."""
                    if hasattr(block, 'model_dump'):
                        block_dict = block.model_dump(exclude_none=True)
                    elif hasattr(block, 'dict'):
                        block_dict = block.dict(exclude_none=True)
                    elif isinstance(block, dict):
                        block_dict = block
                    else:
                        # Fallback: try to convert to dict
                        block_dict = dict(block) if hasattr(block, '__dict__') else block
                    
                    # Recursively remove None values from dict
                    def remove_none_recursive(obj):
                        if isinstance(obj, dict):
                            return {k: remove_none_recursive(v) for k, v in obj.items() if v is not None}
                        elif isinstance(obj, list):
                            return [remove_none_recursive(item) for item in obj if item is not None]
                        else:
                            return obj
                    
                    if isinstance(block_dict, dict):
                        return remove_none_recursive(block_dict)
                    return block_dict
                
                blocks_for_save = None
                if message_blocks:
                    blocks_for_save = []
                    for block in message_blocks:
                        block_dict = clean_block_dict(block)
                        
                        # Skip chart blocks - chart rendering is disabled
                        if isinstance(block_dict, dict) and block_dict.get("type") == "chart":
                            print(f"[execute_langgraph_agent] Skipping chart block {block_dict.get('id')} - chart rendering is disabled")
                            continue
                        
                        blocks_for_save.append(block_dict)
                
                # Also add 'blocks' key for consistency with final_answer structure
                # This ensures backward compatibility and easier extraction
                if blocks_for_save and 'blocks' not in response_data_dict:
                    response_data_dict['blocks'] = blocks_for_save
                
                # Ensure message_blocks is also set (might already be in response_data_dict from result.dict())
                if blocks_for_save:
                    response_data_dict['message_blocks'] = blocks_for_save
                
                # Also ensure text and html_content are at top level for easy access
                if 'text' not in response_data_dict and answer_text:
                    response_data_dict['text'] = answer_text
                if 'html_content' not in response_data_dict and html_content:
                    response_data_dict['html_content'] = html_content
                
                # Recursively remove None/null values from message_blocks and response data
                def remove_none_values_recursive(obj):
                    """
                    Recursively remove None/null values from nested dictionaries and lists.
                    This ensures clean JSON responses without null clutter.
                    """
                    if obj is None:
                        return None
                    elif isinstance(obj, dict):
                        cleaned = {}
                        for k, v in obj.items():
                            if v is not None:
                                cleaned_value = remove_none_values_recursive(v)
                                # Only add non-None values
                                if cleaned_value is not None:
                                    cleaned[k] = cleaned_value
                        return cleaned if cleaned else None
                    elif isinstance(obj, list):
                        cleaned_list = []
                        for item in obj:
                            if item is not None:
                                cleaned_item = remove_none_values_recursive(item)
                                if cleaned_item is not None:
                                    cleaned_list.append(cleaned_item)
                        return cleaned_list if cleaned_list else None
                    else:
                        return obj
                
                # Clean message_blocks before assigning to result
                if cleaned_validated_blocks:
                    cleaned_blocks = remove_none_values_recursive(cleaned_validated_blocks)
                    # Update result's message_blocks with cleaned version
                    result.message_blocks = cleaned_blocks if cleaned_blocks else None
                
                # Also clean response_data_dict for logging
                response_data_dict = remove_none_values_recursive(response_data_dict)
                
                # Log the request and response to session logs with agent ID (from request or generated)
                await log_request_to_json(
                    request=request,
                    user_input=userInput if userInput else "",
                    response_data=response_data_dict,
                    session_id=session_id,
                    execution_time=execution_time,
                    agent_id=agentId,  # Use agentId from request or generated UUID
                    agent_name=agent_name,
                    execution_id=execution_id,
                    user=user
                )
                
                logger.info(f"[LangGraph] Agent execution completed in {execution_time:.2f} seconds")
                
                # Final cleanup: Convert result to dict, clean recursively, and return as JSONResponse
                # This ensures all null values are removed at every level before returning
                if hasattr(result, 'model_dump'):
                    result_dict = result.model_dump(exclude_none=True)
                elif hasattr(result, 'dict'):
                    result_dict = result.dict(exclude_none=True)
                else:
                    result_dict = dict(result)
                
                # Clean the entire result dict recursively to remove all None values
                cleaned_result_dict = remove_none_values_recursive(result_dict)
                
                # Return cleaned result as JSONResponse to ensure null values are removed
                return JSONResponse(content=cleaned_result_dict)
            else:
                raise HTTPException(status_code=401, detail="Unauthorized access. Please provide a valid Agent API key and try again.")
        except asyncio.TimeoutError:
            logger.error("LangGraph agent execution timed out")
            raise HTTPException(
                status_code=408, 
                detail="Request timed out. The agent took too long to process your request. Please try again."
            )
        except Exception as e:
            logger.error(f"Error in execute_langgraph_agent: {str(e)}")
            raise

    @app.post("/api/agent/image-extraction/execute")
    @limiter.limit("15/minute")
    @memory_monitor
    async def execute_image_extraction_agent(
        request: Request,
        response: Response = None
    ):
        """
        Execute LangGraph image extraction agent.
        Extracts product ID from image URL and calls extract-image-details-v2 tool.
        
        Request Body:
        {
            "agentId": "optional-agent-id",
            "userInput": "http://example.com/image.png",  # Image URL
            "session_id": "optional-session-id"
        }
        """
        try:
            if request.headers.get("AGENT-API-KEY", "") == AGENT_API_KEY:
                # Parse JSON with error handling
                try:
                    json_data = await request.json()
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON in image extraction execute request: {str(e)}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid JSON in request body: {str(e)}. Please ensure your JSON is properly formatted with double quotes."
                    )
                except Exception as e:
                    logger.error(f"Error parsing request JSON: {str(e)}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Error parsing request body: {str(e)}"
                    )
                
                agentId = json_data.get("agentId")
                userInput = json_data.get("userInput")
                user = json_data.get("user")  # Optional user parameter
                messages = json_data.get("messages", [])  # Optional conversation history
                
                # Generate agentId if not provided
                if not agentId:
                    agentId = str(uuid.uuid4())
                    logger.info(f"[ImageExtraction] No agentId provided in request, generated new agentId: {agentId}")
                else:
                    logger.info(f"[ImageExtraction] Using agentId from request: {agentId}")
                
                # Generate or retrieve session_id for unauthenticated users
                # Priority: 1. Cookie, 2. JSON body, 3. Generate new UUID
                session_id = request.cookies.get("session_id") or json_data.get("session_id") or str(uuid.uuid4())
                
                # If messages are not explicitly provided, try to fetch from session logs
                if not messages or len(messages) == 0:
                    try:
                        # Fetch conversation history from session logs
                        conversation_data, error = await session_logs_db.get_session_conversation(session_id)
                        
                        if not error and conversation_data and conversation_data.get("messages"):
                            # Convert session log format to messages format
                            # Session logs have: user_input, response
                            # Messages format needs: [{"type": "human", "content": ...}, {"type": "ai", "content": ...}]
                            session_messages = conversation_data.get("messages", [])
                            messages = []
                            
                            for log_entry in session_messages:
                                # Add human message (user_input)
                                if log_entry.get("user_input"):
                                    messages.append({
                                        "type": "human",
                                        "content": str(log_entry["user_input"])
                                    })
                                
                                # Add AI message (response)
                                if log_entry.get("response"):
                                    # Response might be a JSON string or dict
                                    response_content = log_entry["response"]
                                    if isinstance(response_content, str):
                                        try:
                                            # Try to parse as JSON to extract text content
                                            response_dict = json.loads(response_content)
                                            # Extract text from response (could be in content.text or just text)
                                            if isinstance(response_dict, dict):
                                                if "content" in response_dict:
                                                    content_obj = response_dict["content"]
                                                    if isinstance(content_obj, dict) and "text" in content_obj:
                                                        ai_text = content_obj["text"]
                                                    elif isinstance(content_obj, str):
                                                        ai_text = content_obj
                                                    else:
                                                        ai_text = str(response_dict.get("text", response_content))
                                                else:
                                                    ai_text = str(response_dict.get("text", response_content))
                                            else:
                                                ai_text = str(response_content)
                                        except (json.JSONDecodeError, TypeError):
                                            # If not JSON, use as-is
                                            ai_text = response_content
                                    elif isinstance(response_content, dict):
                                        # Already a dict, extract text
                                        if "content" in response_content:
                                            content_obj = response_content["content"]
                                            if isinstance(content_obj, dict) and "text" in content_obj:
                                                ai_text = content_obj["text"]
                                            elif isinstance(content_obj, str):
                                                ai_text = content_obj
                                            else:
                                                ai_text = str(response_content.get("text", ""))
                                        else:
                                            ai_text = str(response_content.get("text", ""))
                                    else:
                                        ai_text = str(response_content)
                                    
                                    if ai_text and ai_text.strip():
                                        messages.append({
                                            "type": "ai",
                                            "content": ai_text.strip()
                                        })
                            
                            logger.info(f"[ImageExtraction] Fetched {len(messages)} messages from session logs for session_id: {session_id}")
                        else:
                            logger.info(f"[ImageExtraction] No conversation history found for session_id: {session_id}")
                    except Exception as e:
                        logger.warning(f"[ImageExtraction] Error fetching messages from session logs: {str(e)}")
                        messages = []
                
                # Validate messages format
                if messages and isinstance(messages, list):
                    # Ensure messages are in correct format: [{"type": "human"|"ai", "content": str}, ...]
                    validated_messages = []
                    for msg in messages:
                        if isinstance(msg, dict) and "type" in msg and "content" in msg:
                            msg_type = msg.get("type")
                            if msg_type in ["human", "ai"]:
                                validated_messages.append({
                                    "type": msg_type,
                                    "content": str(msg.get("content", ""))
                                })
                    messages = validated_messages
                else:
                    messages = []
                
                # Configure response headers
                response.headers["Connection"] = "keep-alive"
                response.headers["Keep-Alive"] = "timeout=1200, max=1000"
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
                response.headers["X-Accel-Buffering"] = "no"
                response.headers["X-Content-Type-Options"] = "nosniff"
                
                # Generate execution ID
                execution_uuid = str(uuid.uuid4())
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                execution_id = f"{execution_uuid}_{timestamp}"
                log_filename = f"image_extraction_agent_execution_{execution_id}.log"
                log_file = os.path.join(LOG_DIR, log_filename)
                
                # Start timing
                start_time = time.time()
                
                # Import image extraction agent
                from langgraph_agent_poc.image_extraction_agent import build_image_extraction_graph
                from langgraph_agent_poc.types import ImageExtractionAgentState
                
                # Build and run the image extraction agent
                graph = build_image_extraction_graph()
                initial_state: ImageExtractionAgentState = {
                    "user_query": userInput if userInput else "",
                    "messages": messages,  # Pass conversation history
                    "agent_id": agentId,
                }
                
                logger.info(f"[ImageExtraction] Executing agent {agentId} with image URL: {userInput[:100] if userInput else 'empty'}...")
                logger.info(f"[ImageExtraction] Execution ID: {execution_id}")
                
                final_state: ImageExtractionAgentState = await graph.ainvoke(initial_state)
                
                # Calculate execution time
                execution_time = time.time() - start_time
                
                # Check for errors
                if final_state.get("stop_execution") and final_state.get("error"):
                    error_msg = final_state.get("error", "Unknown error occurred")
                    logger.error(f"[ImageExtraction] Error: {error_msg}")
                    raise HTTPException(
                        status_code=500,
                        detail=error_msg
                    )
                
                # Extract final answer from state (set by nodes)
                logger.info(f"[ImageExtraction] ===== Extracting response from final state =====")
                logger.info(f"[ImageExtraction] Final state keys: {list(final_state.keys())}")
                
                answer_text = ""
                html_content = ""
                
                # Get product_id first for logging and fallback checks
                product_id = final_state.get("product_id")
                logger.info(f"[ImageExtraction] Product ID from state: {product_id}")
                
                # Check final_answer from state (set by nodes)
                final_answer = final_state.get("final_answer", {})
                logger.info(f"[ImageExtraction] final_answer type: {type(final_answer)}")
                logger.info(f"[ImageExtraction] final_answer value preview: {str(final_answer)[:500] if final_answer else 'None'}")
                
                if isinstance(final_answer, dict):
                    answer_text = final_answer.get("text", "")
                    html_content = final_answer.get("html_content", "")
                    logger.info(f"[ImageExtraction] Extracted from final_answer - text length: {len(answer_text)}, html_content length: {len(html_content)}")
                    
                    # If text is empty but html_content exists, use html_content
                    if not answer_text and html_content:
                        logger.info(f"[ImageExtraction] text is empty but html_content exists, using html_content for text")
                        answer_text = html_content
                else:
                    answer_text = final_state.get("answer", "")
                    html_content = ""
                    logger.info(f"[ImageExtraction] final_answer is not dict, using answer from state: {len(answer_text)} chars")
                
                # If no answer from final_answer, check for mapped HTML directly using product_id
                if not answer_text:
                    if product_id:
                        logger.info(f"[ImageExtraction] No answer_text from final_answer, checking for mapped HTML for product_id: {product_id}")
                        from langgraph_agent_poc.image_extraction_agent import get_mapped_html_for_product_id
                        mapped_html = get_mapped_html_for_product_id(product_id)
                        if mapped_html:
                            logger.info(f"[ImageExtraction] ✓ Found mapped HTML for product_id {product_id}, using it (length: {len(mapped_html)})")
                            answer_text = mapped_html
                            html_content = mapped_html
                        else:
                            logger.info(f"[ImageExtraction] No mapped HTML found for product_id {product_id}")
                    else:
                        logger.warning(f"[ImageExtraction] No product_id found in state, cannot check mapped HTML")
                
                # Also check tool_response as fallback
                if not answer_text:
                    tool_response = final_state.get("tool_response")
                    if tool_response:
                        logger.info(f"[ImageExtraction] No answer_text found, checking tool_response...")
                        # Generate HTML from tool_response if available
                        from langgraph_agent_poc.image_extraction_agent import generate_html_from_tool_response
                        product_id = final_state.get("product_id")
                        image_url = final_state.get("image_url")
                        answer_text = generate_html_from_tool_response(tool_response, product_id, image_url)
                        html_content = answer_text
                        logger.info(f"[ImageExtraction] Generated HTML from tool_response, length: {len(answer_text)}")
                
                # Fallback: Try to extract product ID from user input if not in state
                if not answer_text and not product_id:
                    logger.info(f"[ImageExtraction] No product_id in state, trying to extract from user input")
                    user_input_str = str(userInput) if userInput else ""
                    # Try to find product ID patterns in user input
                    import re
                    # Look for numeric product IDs (4-6 digits)
                    numeric_match = re.search(r'\b(\d{4,6})\b', user_input_str)
                    if numeric_match:
                        potential_product_id = numeric_match.group(1)
                        logger.info(f"[ImageExtraction] Found potential product ID in user input: {potential_product_id}")
                        from langgraph_agent_poc.image_extraction_agent import get_mapped_html_for_product_id
                        mapped_html = get_mapped_html_for_product_id(potential_product_id)
                        if mapped_html:
                            logger.info(f"[ImageExtraction] ✓ Found mapped HTML for potential product_id {potential_product_id}")
                            answer_text = mapped_html
                            html_content = mapped_html
                
                # Fallback: if no answer, use error message
                if not answer_text:
                    error_msg = final_state.get("error", "No response generated")
                    logger.warning(f"[ImageExtraction] No answer found, using error/fallback: {error_msg}")
                    answer_text = error_msg
                    html_content = await generate_html_content(answer_text)
                
                # Generate HTML content if not present
                if not html_content or (isinstance(html_content, str) and len(html_content.strip()) == 0):
                    if answer_text:
                        html_content = await generate_html_content(answer_text)
                
                # Get agent name
                agents = load_agents()
                agent_name = "Image Extraction Agent"
                for agent in agents:
                    if agent["id"] == agentId:
                        agent_name = agent.get("name", "Image Extraction Agent")
                        break
                
                # Generate log URL
                log_url = f"{BASE_URL}/api/logs/{execution_id}"
                
                # Get updated messages from final state (includes new user query and AI response)
                updated_messages = final_state.get("messages", messages)
                
                # CRITICAL: Serialize dict fields in messages to JSON strings for MessageResponse
                # MessageResponse expects messages to be List[Dict[str, str]], but we store data and final_answer as dicts
                if updated_messages:
                    serialized_messages = []
                    for msg in updated_messages:
                        if isinstance(msg, dict):
                            serialized_msg = {
                                "type": msg.get("type", ""),
                                "content": msg.get("content", "")
                            }
                            # Serialize data and final_answer to JSON strings if they exist
                            if "data" in msg and msg["data"]:
                                if isinstance(msg["data"], dict):
                                    serialized_msg["data"] = json.dumps(msg["data"])
                                else:
                                    serialized_msg["data"] = str(msg["data"])
                            if "final_answer" in msg and msg["final_answer"]:
                                if isinstance(msg["final_answer"], dict):
                                    serialized_msg["final_answer"] = json.dumps(msg["final_answer"])
                                else:
                                    serialized_msg["final_answer"] = str(msg["final_answer"])
                            serialized_messages.append(serialized_msg)
                        else:
                            serialized_messages.append(msg)
                    updated_messages = serialized_messages
                
                # Create MessageResponse
                result = MessageResponse(
                    type="text",
                    content=TextData(text=answer_text),
                    execution_id=execution_id,
                    log_url=log_url,
                    session_id=session_id,
                    html_content=html_content,
                    user_input=userInput,
                    messages=updated_messages if updated_messages else None,  # Include updated conversation history
                    message_blocks=None
                )
                
                # Prepare response data for logging
                if hasattr(result, 'model_dump'):
                    response_data_dict = result.model_dump(exclude_none=True)
                elif hasattr(result, 'dict'):
                    response_data_dict = result.dict(exclude_none=True)
                else:
                    response_data_dict = dict(result)
                
                # Add text and html_content at top level
                if 'text' not in response_data_dict and answer_text:
                    response_data_dict['text'] = answer_text
                if 'html_content' not in response_data_dict and html_content:
                    response_data_dict['html_content'] = html_content
                
                # Recursively remove None/null values from message_blocks and response data
                def remove_none_values_recursive(obj):
                    """
                    Recursively remove None/null values from nested dictionaries and lists.
                    This ensures clean JSON responses without null clutter.
                    """
                    if obj is None:
                        return None
                    elif isinstance(obj, dict):
                        cleaned = {}
                        for k, v in obj.items():
                            if v is not None:
                                cleaned_value = remove_none_values_recursive(v)
                                # Only add non-None values
                                if cleaned_value is not None:
                                    cleaned[k] = cleaned_value
                        return cleaned if cleaned else None
                    elif isinstance(obj, list):
                        cleaned_list = []
                        for item in obj:
                            if item is not None:
                                cleaned_item = remove_none_values_recursive(item)
                                if cleaned_item is not None:
                                    cleaned_list.append(cleaned_item)
                        return cleaned_list if cleaned_list else None
                    else:
                        return obj
                
                # Clean response_data_dict for logging
                response_data_dict = remove_none_values_recursive(response_data_dict)
                
                # Log the request and response
                await log_request_to_json(
                    request=request,
                    user_input=userInput if userInput else "",
                    response_data=response_data_dict,
                    session_id=session_id,
                    execution_time=execution_time,
                    agent_id=agentId,
                    agent_name=agent_name,
                    execution_id=execution_id,
                    user=user
                )
                
                logger.info(f"[ImageExtraction] Agent execution completed in {execution_time:.2f} seconds")
                
                # Final cleanup: Convert result to dict, clean recursively, and return as JSONResponse
                # This ensures all null values are removed at every level before returning
                if hasattr(result, 'model_dump'):
                    result_dict = result.model_dump(exclude_none=True)
                elif hasattr(result, 'dict'):
                    result_dict = result.dict(exclude_none=True)
                else:
                    result_dict = dict(result)
                
                # Clean the entire result dict recursively to remove all None values
                cleaned_result_dict = remove_none_values_recursive(result_dict)
                
                # Return cleaned result as JSONResponse to ensure null values are removed
                return JSONResponse(content=cleaned_result_dict)
            else:
                raise HTTPException(status_code=401, detail="Unauthorized access. Please provide a valid Agent API key and try again.")
        except asyncio.TimeoutError:
            logger.error("Image extraction agent execution timed out")
            raise HTTPException(
                status_code=408, 
                detail="Request timed out. The agent took too long to process your request. Please try again."
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in execute_image_extraction_agent: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )

    @app.post("/api/agent/execute/stream")
    @limiter.limit("15/minute")
    @memory_monitor
    async def execute_agent_stream(
        request: Request,
        response: Response = None
    ):
        try:
            if request.headers.get("AGENT-API-KEY", "") == AGENT_API_KEY:
                json_data = await request.json()
                agentId = json_data.get("agentId")
                userInput = json_data.get("userInput")
                
                # Generate or retrieve session_id for unauthenticated users
                session_id = request.cookies.get("session_id") or json_data.get("session_id") or str(uuid.uuid4())
                
                # Configure response headers for long-running request to prevent network suspension
                response.headers["Connection"] = "keep-alive"
                response.headers["Keep-Alive"] = "timeout=1200, max=1000"  # 20 minutes timeout, max 1000 requests
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
                
                # Add headers to prevent browser from suspending the connection
                response.headers["X-Accel-Buffering"] = "no"  # Disable nginx buffering if behind nginx
                response.headers["X-Content-Type-Options"] = "nosniff"
                
                # Start timing the execution
                start_time = time.time()
                
                # Create a queue for status updates
                status_queue = asyncio.Queue()
                
                # Start a background task to run the agent
                background_task = asyncio.create_task(
                    run_agent_with_status_updates(request, agentId, userInput, None, response, session_id, status_queue)
                )
                
                # Define the streaming response generator
                async def stream_response():
                    try:
                        # First yield a status update that we're starting
                        yield json.dumps({"status": "starting", "message": "Processing your request.."}) + "\n"
                        
                        # Stream status updates as they come in
                        last_heartbeat_time = 0
                        while True:
                            try:
                                # Get status update with a timeout
                                status_update = await asyncio.wait_for(status_queue.get(), timeout=300.0)
                                
                                if status_update == "DONE":
                                    # Get the final result from the background task
                                    agent_result = await background_task
                                    
                                    # Extract execution_id from the result
                                    execution_id = agent_result.execution_id if hasattr(agent_result, 'execution_id') else None
                                    
                                    # Calculate execution time
                                    execution_time = time.time() - start_time
                                    
                                    # Get agent name for logging
                                    agents = load_agents()
                                    agent_name = None
                                    for agent in agents:
                                        if agent["id"] == agentId:
                                            agent_name = agent.get("name", "Unknown Agent")
                                            break
                                    
                                    # Log the request and response with explicit argument naming
                                    await log_request_to_json(
                                        request=request,
                                        user_input=userInput if userInput else "",
                                        response_data=agent_result.dict() if hasattr(agent_result, 'dict') else dict(agent_result),
                                        session_id=session_id,
                                        execution_time=execution_time,
                                        agent_id=agentId,
                                        agent_name=agent_name,
                                        execution_id=execution_id
                                    )
                                    
                                    # Yield the final result with execution time
                                    yield json.dumps({"status": "complete", "result": agent_result.dict() if hasattr(agent_result, 'dict') else dict(agent_result), "execution_time_seconds": execution_time}) + "\n"
                                    break
                                else:
                                    # Yield the status update
                                    yield json.dumps({"status": "processing", "message": status_update}) + "\n"
                                
                            except asyncio.TimeoutError:
                                # Throttle heartbeats (max once every 10s) to avoid spamming
                                now = time.time()
                                if now - last_heartbeat_time >= 10:
                                    yield json.dumps({"status": "processing", "message": "Processing continues.."}) + "\n"
                                    last_heartbeat_time = now
                                continue
                    except Exception as e:
                        logger.error(f"Error in stream_response: {str(e)}")
                        yield json.dumps({"status": "error", "message": str(e)}) + "\n"
                        
                # Return the streaming response
                return StreamingResponse(
                    stream_response(),
                    media_type="application/x-ndjson"
                )
            else:
                raise HTTPException(status_code=401, detail="Unauthorized access. Please provide a valid Agent API key and try again.")
        except asyncio.TimeoutError:
            logger.error("Agent execution stream timed out")
            raise HTTPException(
                status_code=408, 
                detail="Request timed out. The agent took too long to process your request. Please try again."
            )
        except Exception as e:
            logger.error(f"Error in execute_agent_stream: {str(e)}")
            raise

    async def run_agent_with_status_updates(request, agentId, userInput, file, response, session_id, status_queue, ai_mode=None):
        """
        Run the agent with status updates sent to the queue
        """
        try:
            
            # Get the agent info to display more detailed status
            agents = load_agents()
            agent = next((a for a in agents if a["id"] == agentId), None)
            
            if agent:
                await status_queue.put(f"{agent.get('name', 'Unknown')} Agent's requirements are loading..")
                if agent.get("tools", []):
                    await status_queue.put(f"Validating {len(agent.get('tools', []))} connected Tools..")
                if agent.get("knowledge_base_id"):
                    await status_queue.put("Loading knowledge base..")
            
            # Run the actual agent with status updates
            agent_result = await run_agent(request, agentId, userInput, file, response, session_id, status_queue, ai_mode)
            
            # Signal that we're done
            await status_queue.put("DONE")
            
            return agent_result
        except Exception as e:
            logger.error(f"Error in run_agent_with_status_updates: {str(e)}")
            await status_queue.put(f"Error: {str(e)}")
            raise

    @app.websocket("/api/agent/execute/ws")
    async def langgraph_agent_ws(websocket: WebSocket):
        """
        WebSocket endpoint for streaming agent thoughts and status updates.
        Specifically for LangGraph agents.
        Supports multi-turn conversation.
        """
        await websocket.accept()
        try:
            while True:
                # Receive configuration/input
                try:
                    data = await websocket.receive_json()
                except WebSocketDisconnect:
                    break
                except Exception:
                    await websocket.send_json({"status": "error", "message": "Invalid configuration format. Expected JSON."})
                    continue

                agent_id = data.get("agentId")
                user_input = data.get("userInput")
                # Handle both common variations of the key
                agent_api_key = data.get("agentApiKey") or data.get("apiKey")
                user_email = data.get("userEmail", "External User")
                
                # Auth check - support both private and public agent keys
                valid_keys = [k for k in [AGENT_API_KEY, PUBLIC_AGENT_API_KEY] if k]
                if valid_keys and agent_api_key not in valid_keys:
                    await websocket.send_json({"status": "error", "message": "Unauthorized: Invalid API Key"})
                    continue

                if not agent_id or not user_input:
                    await websocket.send_json({"status": "error", "message": "agentId and userInput are required"})
                    continue

                # Setup status queue
                status_queue = asyncio.Queue()
                session_id = data.get("sessionId") or str(uuid.uuid4())
                
                # Mock request/response for run_agent
                class MockState:
                    def __init__(self, email):
                        self.user_email = email
                
                class MockRequest:
                    def __init__(self, email):
                        self.state = MockState(email)
                        self.cookies = {}
                        self.headers = {"AGENT-API-KEY": AGENT_API_KEY}
                
                mock_request = MockRequest(user_email)
                
                # Start running the agent in the background
                # ai_mode='LF' ensures it uses LangGraph
                # use_streaming_executor=True ensures it uses LangGraphExecutorStream for WebSocket streaming
                execution_task = asyncio.create_task(
                    run_agent(mock_request, agent_id, user_input, None, None, session_id, status_queue, ai_mode='LF', use_streaming_executor=True)
                )
                
                # Stream status updates from the queue to the client
                loop_done = False
                while not loop_done:
                    try:
                        # Wait for a status update from the queue with a timeout
                        status_msg = await asyncio.wait_for(status_queue.get(), timeout=1.0)
                        
                        if status_msg == "DONE":
                            loop_done = True
                            continue
                        
                        # Send the status update to the client
                        await websocket.send_json({
                            "status": "processing",
                            "message": status_msg
                        })
                        
                    except asyncio.TimeoutError:
                        if execution_task.done():
                            loop_done = True
                        else:
                            await websocket.send_json({"status": "heartbeat", "message": "Thinking..."})
                    except Exception as e:
                        logger.error(f"Error in WebSocket status streaming: {str(e)}")
                        loop_done = True
                
                # Once we're out of the loop, get the final result
                try:
                    final_result = await execution_task
                    
                    # Convert MessageResponse to dict
                    if hasattr(final_result, "dict"):
                        result_data = final_result.dict()
                    else:
                        result_data = final_result
                        
                    await websocket.send_json({
                        "status": "complete",
                        "result": result_data
                    })
                except Exception as e:
                    logger.error(f"Agent execution failed in WebSocket: {str(e)}")
                    await websocket.send_json({
                        "status": "error",
                        "message": f"Execution failed: {str(e)}"
                    })
            
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Top-level WebSocket error: {str(e)}")
            try:
                # websocket might be closed
                pass
            except:
                pass

    @app.get("/api/system/memory")
    @limiter.limit("100/minute")
    async def get_system_stats(request: Request):
        """Get comprehensive system statistics including memory, disk, and temporary folder usage"""
        try:
            # Memory statistics
            current_memory = get_memory_usage()
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # System memory statistics
            system_memory = psutil.virtual_memory()
            
            # Disk usage statistics
            disk_usage = psutil.disk_usage('/')
            tmp_disk_usage = None
            try:
                tmp_disk_usage = psutil.disk_usage('temp')
            except:
                pass
            
            # Import cleanup function from task_executor
            from task_executor import cleanup_old_tmp_directories
            
            # Get temporary folder statistics (dry run to avoid deletion)
            tmp_stats = cleanup_old_tmp_directories(max_age_hours=24, dry_run=True)
            
            # Count current temporary folders by pattern
            import glob
            tmp_folder_counts = {
                "cs_folders": len(glob.glob('temp/cs_*')),
                "cm_folders": len(glob.glob('temp/cm_*')),
                "cc_folders": len(glob.glob('temp/cc_*')),
                "crew_folders": len(glob.glob('temp/crew*')),
                "crewai_folders": len(glob.glob('temp/crewai*')),
                "total_temp_folders": len(glob.glob('temp/c*_*')) + len(glob.glob('temp/crew*'))
            }
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Load average (Unix systems)
            load_avg = None
            try:
                load_avg = psutil.getloadavg()
            except:
                pass
            
            return {
                # Process Memory Statistics
                "memory": {
                    "process_usage_mb": current_memory,
                    "process_percent": process.memory_percent(),
                    "rss_mb": memory_info.rss / 1024 / 1024,
                    "vms_mb": memory_info.vms / 1024 / 1024,
                    "active_executors": len(_active_executors),
                    "request_count": _request_counter
                },
                
                # System Memory Statistics
                "system_memory": {
                    "total_gb": system_memory.total / 1024 / 1024 / 1024,
                    "available_gb": system_memory.available / 1024 / 1024 / 1024,
                    "used_gb": system_memory.used / 1024 / 1024 / 1024,
                    "percent_used": system_memory.percent,
                    "free_gb": system_memory.free / 1024 / 1024 / 1024
                },
                
                # Disk Usage Statistics
                "disk": {
                    "root": {
                        "total_gb": disk_usage.total / 1024 / 1024 / 1024,
                        "used_gb": disk_usage.used / 1024 / 1024 / 1024,
                        "free_gb": disk_usage.free / 1024 / 1024 / 1024,
                        "percent_used": (disk_usage.used / disk_usage.total) * 100
                    },
                    "tmp": {
                        "total_gb": tmp_disk_usage.total / 1024 / 1024 / 1024 if tmp_disk_usage else None,
                        "used_gb": tmp_disk_usage.used / 1024 / 1024 / 1024 if tmp_disk_usage else None,
                        "free_gb": tmp_disk_usage.free / 1024 / 1024 / 1024 if tmp_disk_usage else None,
                        "percent_used": (tmp_disk_usage.used / tmp_disk_usage.total) * 100 if tmp_disk_usage else None
                    } if tmp_disk_usage else None
                },
                
                # CPU Statistics
                "cpu": {
                    "percent_used": cpu_percent,
                    "core_count": cpu_count,
                    "load_average": list(load_avg) if load_avg else None
                },
                
                # Temporary Folder Statistics
                "temp_folders": {
                    "counts": tmp_folder_counts,
                    "cleanup_stats": {
                        "scanned": tmp_stats.get('scanned', 0),
                        "deletable": tmp_stats.get('deleted', 0),
                        "preserved": tmp_stats.get('preserved', 0),
                        "total_size_mb": tmp_stats.get('total_size_mb', 0),
                        "patterns_found": tmp_stats.get('patterns_found', {})
                    }
                },
                
                # Garbage Collection Statistics
                "gc_stats": {
                    "generation_0": gc.get_count()[0],
                    "generation_1": gc.get_count()[1],
                    "generation_2": gc.get_count()[2],
                    "total_collections": sum(gc.get_stats()[i]['collections'] for i in range(3))
                },
                
                # Timestamp
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "uptime_seconds": time.time() - psutil.Process().create_time()
            }
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {"error": str(e)}

    @app.post("/api/system/gc")
    @limiter.limit("5/minute")
    async def force_garbage_collection(request: Request):
        """Force garbage collection and return memory stats"""
        try:
            initial_memory = get_memory_usage()
            collected = gc.collect()
            final_memory = get_memory_usage()
            
            return {
                "initial_memory_mb": initial_memory,
                "final_memory_mb": final_memory,
                "memory_freed_mb": initial_memory - final_memory,
                "objects_collected": collected,
                "active_executors": len(_active_executors)
            }
        except Exception as e:
            logger.error(f"Error during forced GC: {str(e)}")
            return {"error": str(e)}

    @app.post("/api/system/cleanup-temp")
    @limiter.limit("10/minute")
    async def cleanup_temp_folders(request: Request, max_age_hours: int = Query(24, description="Delete folders older than this many hours"), dry_run: bool = Query(False, description="Preview cleanup without actually deleting")):
        """Clean up old temporary folders created by agent execution"""
        try:
            from task_executor import cleanup_old_tmp_directories
            
            # Run cleanup with specified parameters
            stats = cleanup_old_tmp_directories(max_age_hours=max_age_hours, dry_run=dry_run)
            
            return {
                "success": True,
                "dry_run": dry_run,
                "max_age_hours": max_age_hours,
                "stats": stats,
                "message": f"{'Would delete' if dry_run else 'Deleted'} {stats.get('deleted', 0)} items, freed {stats.get('total_size_mb', 0):.1f}MB",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Error during temp folder cleanup: {str(e)}")
            return {"error": str(e), "success": False}

    @app.get("/api/system/temp-stats")
    @limiter.limit("60/minute")
    async def get_temp_folder_stats(request: Request):
        """Get statistics about temporary folders without running cleanup"""
        try:
            import glob
            from task_executor import cleanup_old_tmp_directories
            
            # Get cleanup statistics (dry run)
            cleanup_stats = cleanup_old_tmp_directories(max_age_hours=24, dry_run=True)
            
            # Count current temporary folders by pattern
            folder_counts = {
                "cs_folders": len(glob.glob('temp/cs_*')),
                "cm_folders": len(glob.glob('temp/cm_*')),
                "cc_folders": len(glob.glob('temp/cc_*')),
                "crew_folders": len(glob.glob('temp/crew*')),
                "crewai_folders": len(glob.glob('temp/crewai*')),
                "total_temp_folders": len(glob.glob('temp/c*_*')) + len(glob.glob('temp/crew*'))
            }
            
            # Get sizes of different folder types
            folder_sizes = {}
            for pattern, count in folder_counts.items():
                if count > 0:
                    pattern_glob = pattern.replace('_folders', '_*')
                    if pattern == 'crew_folders':
                        pattern_glob = 'temp/crew*'
                    elif pattern == 'crewai_folders':
                        pattern_glob = 'temp/crewai*'
                    elif pattern == 'total_temp_folders':
                        continue
                    else:
                        pattern_glob = f'temp/{pattern.split("_")[0]}_*'
                    
                    total_size = 0
                    try:
                        for folder_path in glob.glob(pattern_glob):
                            if os.path.isdir(folder_path):
                                for dirpath, dirnames, filenames in os.walk(folder_path):
                                    for filename in filenames:
                                        filepath = os.path.join(dirpath, filename)
                                        try:
                                            total_size += os.path.getsize(filepath)
                                        except (OSError, IOError):
                                            pass
                    except Exception:
                        pass
                    folder_sizes[pattern] = total_size / 1024 / 1024  # Convert to MB
            
            return {
                "folder_counts": folder_counts,
                "folder_sizes_mb": folder_sizes,
                "cleanup_preview": {
                    "deletable_count": cleanup_stats.get('deleted', 0),
                    "deletable_size_mb": cleanup_stats.get('total_size_mb', 0),
                    "preserved_count": cleanup_stats.get('preserved', 0),
                    "patterns_found": cleanup_stats.get('patterns_found', {})
                },
                "recommendations": {
                    "should_cleanup": cleanup_stats.get('total_size_mb', 0) > 50,  # Recommend if >50MB
                    "cleanup_urgency": "high" if cleanup_stats.get('total_size_mb', 0) > 100 else "medium" if cleanup_stats.get('total_size_mb', 0) > 25 else "low"
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting temp folder stats: {str(e)}")
            return {"error": str(e)}

    def verify_and_set_user_email(request: Request, jwt_secret_key: str):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Missing or invalid authorization header"}
            )

        token = auth_header.split("Bearer ")[1]

        try:
            payload = jwt.decode(token, jwt_secret_key, algorithms=['HS256'])

            if payload.get("type", "") == "guest":
                request.state.user_email = "guest"
            else:
                email = payload.get("email")
                if not email:
                    return JSONResponse(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        content={"detail": "Invalid token payload"}
                    )
                request.state.user_email = email

            return None  # Indicate successful verification

        except jwt.ExpiredSignatureError:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Token has expired"}
            )
        except jwt.InvalidTokenError:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid token"}
            )


    @app.post("/api/ia_chatbot")
    @limiter.limit("25/minute")
    async def ia_chatbot_agent(
        request: Request,
        response: Response = None
    ):
        
        auth_result = verify_and_set_user_email(request, JWT_SECRET_KEY)
        if auth_result:
            return auth_result

            
        try:
            json_data = await request.json()
            userInput = json_data.get("userInput")
            session_id = json_data.get("session_id") or str(uuid.uuid4())

            agentId = IA_CHATBOT_AGENT_ID_PROD
            #"b30e7e49-a784-4863-adcc-b91fe2f15c8e"
            
            
            # Start timing the execution
            start_time = time.time()
            
            # Run the agent
            agent_result = await run_agent(request, agentId, userInput, None, response, session_id)
            
            # Extract execution_id from the result
            execution_id = agent_result.execution_id if hasattr(agent_result, 'execution_id') else None
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            agent_name = "IA Chatbot"
            
            # Log the request and response with explicit argument naming
            await log_request_to_json(
                request=request,
                user_input=userInput if userInput else "",
                response_data=agent_result.dict() if hasattr(agent_result, 'dict') else dict(agent_result),
                session_id=session_id,
                execution_time=execution_time,
                agent_id=agentId,
                agent_name=agent_name,
                execution_id=execution_id
            )
            
            return agent_result
            
        except Exception as e:
            logger.error(f"Error in execute_agent: {str(e)}")
            raise


    # List of keys to mask in JSON objects
    MASKED_KEYS = ["db_config", "api_key", "password", "credentials_json"]
    

    # Utility to mask sensitive keys in a JSON object
    def mask_sensitive_data(data: Any, masked_keys: List[str]) -> Any:
        if isinstance(data, dict):
            return {
                key: "XXXXXXXXXXXXXXX" if key.lower() in [k.lower() for k in masked_keys] else mask_sensitive_data(value, masked_keys)
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [mask_sensitive_data(item, masked_keys) for item in data]
        return data

    # Utility to reconstruct JSON strings that may span multiple lines
    def reconstruct_json_lines(lines: List[str]) -> List[str]:
        reconstructed_lines = []
        json_buffer = ""
        in_json = False
        brace_count = 0

        for line in lines:
            if not in_json:
                # Look for the start of a JSON object
                if "{" in line:
                    in_json = True
                    json_buffer = line
                    brace_count = line.count("{") - line.count("}")
                    if brace_count == 0:
                        # Complete JSON in one line
                        in_json = False
                        reconstructed_lines.append(json_buffer)
                        json_buffer = ""
                    continue
            else:
                json_buffer += line
                brace_count += line.count("{") - line.count("}")
                if brace_count == 0:
                    # Complete JSON object
                    in_json = False
                    reconstructed_lines.append(json_buffer)
                    json_buffer = ""
                    continue
            if not in_json:
                reconstructed_lines.append(line)

        if json_buffer:
            # Handle incomplete JSON
            reconstructed_lines.append(json_buffer)
        return reconstructed_lines


    @app.get("/api/logs/{execution_id}", response_class=HTMLResponse)
    @limiter.limit("10/minute")
    async def get_log_file(execution_id: str, request: Request):
        log_file = os.path.join(LOG_DIR, f"agent_execution_{execution_id}.log")
        
        if not os.path.exists(log_file):
            raise HTTPException(status_code=404, detail="Log file not found")
        
        log_lines = []
        warning = ""
        logger = logging.getLogger(f"log_reader_{execution_id}")
        
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                log_lines = f.readlines()
        except UnicodeDecodeError as e:
            warning = f"Warning: Some characters in the log file could not be decoded and were replaced. Error: {sanitize_for_logging(str(e))}"
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                log_lines = f.readlines()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading log file: {sanitize_for_logging(str(e))}")

        processed_lines = []
        # json_regex = r'\{(?:[^"{}]+|"(?:\\.|[^"\\])*"|\{[^{}]*\}|\[[^\[\]]*\])*\}'
        json_regex = r'\{.*?\}'
        
        for line in log_lines:
            processed_line = line
            json_matches = re.finditer(json_regex, line, re.DOTALL)
            for match in json_matches:
                json_str = match.group(0)
                logger.debug(f"Detected JSON in line: {sanitize_for_logging(json_str[:100])}...")
                try:
                    json_obj = json.loads(json_str)
                    def has_sensitive_keys(obj):
                        if isinstance(obj, dict):
                            return any(
                                key.lower() in [k.lower() for k in MASKED_KEYS] or has_sensitive_keys(value)
                                for key, value in obj.items()
                            )
                        elif isinstance(obj, list):
                            return any(has_sensitive_keys(item) for item in obj)
                        return False

                    if has_sensitive_keys(json_obj):
                        logger.debug(f"Found sensitive keys in JSON: {sanitize_for_logging(json_str[:100])}...")
                    masked_json_obj = mask_sensitive_data(json_obj, MASKED_KEYS)
                    masked_json_str = json.dumps(masked_json_obj, ensure_ascii=False)
                    processed_line = processed_line.replace(json_str, masked_json_str, 1)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON in line: {sanitize_for_logging(json_str[:100])}... Error: {sanitize_for_logging(str(e))}")
                    continue
            processed_lines.append(processed_line)
        
        log_content = "".join(processed_lines)
        
        # Escape log content for safe HTML rendering
        log_content = log_content.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        
        # Get current time for the futuristic UI
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Render the template with dynamic data
        return page_templates.TemplateResponse(
            "logs.html",
            {
                "request": request,
                "execution_id": execution_id,
                "warning": warning,
                "log_content": log_content,
                "current_time": current_time
            }
        )

    class MultiAgentInferenceRequest(BaseModel):
        multi_agent_id: str
        user_input: str

    
    

    @app.post("/api/multi_agent/infer")
    @limiter.limit("20/minute")
    async def multi_agent_infer(
        request: Request,
        multi_agent_id: Optional[str] = Form(None),
        user_input: Optional[str] = Form(None),
        file: Optional[UploadFile] = File(None),
        response: Response = None  # Add Response for setting cookies
    ):
        # Initialize variables for cleanup and error handling
        file_path = None
        multi_agent_config = None
        
        # Generate execution ID outside try block for error handling
        execution_uuid = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        execution_id = f"{execution_uuid}_{timestamp}"
        log_filename = f"multi_agent_execution_{execution_id}.log"
        log_file = os.path.join(LOG_DIR, log_filename)
        
        # Generate log URL
        log_url = f"{BASE_URL}/api/multi_agent/logs/{execution_id}"
        
        # Configure logger early
        logger = logging.getLogger(f"multi_agent_infer_{execution_id}")
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # Fallback if file handler fails
            pass
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Start timing the execution
        start_time = time.time()
        
        # Get user email from request state
        user_email = getattr(request.state, "user_email", "External User")
        if not user_email or user_email == "External User":
            user_email = "agent@impactanalytics.co"
            logger.info("Request from unauthenticated user")
        else:
            logger.info(f"Request from user: {user_email}")

        # Initialize session_id early for error responses
        session_id = request.cookies.get("session_id") or str(uuid.uuid4())
        
        try:
            # Configure response headers for long-running request to prevent network suspension
            response.headers["Connection"] = "keep-alive"
            response.headers["Keep-Alive"] = "timeout=1200, max=1000"  # 20 minutes timeout, max 1000 requests
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            
            # Add headers to prevent browser from suspending the connection
            response.headers["X-Accel-Buffering"] = "no"  # Disable nginx buffering if behind nginx
            response.headers["X-Content-Type-Options"] = "nosniff"
            
            # Check content type to determine if it's JSON or form data
            content_type = request.headers.get("content-type", "")
            logger.info(f"Request content type: {content_type}")
            
            # Parse request based on content type
            if "application/json" in content_type:
                try:
                    json_data = await request.json()
                    if multi_agent_id is None:
                        multi_agent_id = json_data.get("multi_agent_id")
                    user_input = json_data.get("user_input")
                    session_id = request.cookies.get("session_id") or json_data.get("session_id") or session_id
                    file = None  # File uploads not supported in JSON mode
                    logger.info(f"Processed JSON request: multi_agent_id={multi_agent_id}, user_input={sanitize_for_logging(user_input)}")
                except Exception as e:
                    logger.error(f"Error parsing JSON request: {sanitize_for_logging(str(e))}")
                    return JSONResponse(
                        content={
                            "type": "error",
                            "content": {
                                "message": "Invalid JSON payload",
                                "details": str(e)
                            },
                            "execution_id": execution_id,
                            "log_url": log_url,
                            "session_id": session_id
                        },
                        status_code=400
                    )
            else:
                # Form data mode - values already in parameters
                session_id = request.cookies.get("session_id") or session_id
                logger.info(f"Processed form request: multi_agent_id={multi_agent_id}, user_input={sanitize_for_logging(user_input)}")
            
            # Validate required parameters
            if not multi_agent_id:
                logger.error("Missing multi_agent_id parameter")
                return JSONResponse(
                    content={
                        "type": "error",
                        "content": {
                            "message": "Missing parameters",
                            "details": "multi_agent_id is required"
                        },
                        "execution_id": execution_id,
                        "log_url": log_url,
                        "session_id": session_id
                    },
                    status_code=400
                )
            
            if not user_input or user_input.strip() == "":
                logger.error("Empty or invalid user input provided")
                return JSONResponse(
                    content={
                        "type": "error",
                        "content": {
                            "message": "User input cannot be empty",
                            "details": "Please provide valid user input"
                        },
                        "execution_id": execution_id,
                        "log_url": log_url,
                        "session_id": session_id
                    },
                    status_code=400
                )
            
            sanitized_user_input = sanitize_for_logging(user_input)
            logger.info(f"Received multi-agent infer request for ID: {multi_agent_id}, userInput={sanitized_user_input}")
            logger.debug(f"Execution ID: {execution_id}")
            logger.debug(f"Log URL: {log_url}")

            # Handle file upload
            file_info = None
            file_path = None
            file_type = None
            if file:
                contents = await file.read()
                file_size = len(contents)
                logger.debug(f"Processing file: {sanitize_for_logging(file.filename)}, size: {file_size} bytes")
                if file_size > 10 * 1024 * 1024:
                    logger.error("File too large: exceeds 10MB")
                    return JSONResponse(
                        content={
                            "type": "error",
                            "content": {
                                "message": "File too large",
                                "details": "Maximum file size is 10MB"
                            },
                            "execution_id": execution_id,
                            "log_url": log_url,
                            "session_id": session_id
                        },
                        status_code=400
                    )

                if file.content_type not in ALLOWED_FILE_TYPES:
                    logger.error(f"Unsupported file type: {file.content_type}")
                    return JSONResponse(
                        content={
                            "type": "error",
                            "content": {
                                "message": "Unsupported file type",
                                "details": f"Only CSV, JSON, TXT, PDF, and image files supported. Got: {file.content_type}"
                            },
                            "execution_id": execution_id,
                            "log_url": log_url,
                            "session_id": session_id
                        },
                        status_code=400
                    )

                file_extension = os.path.splitext(file.filename)[1]
                unique_filename = f"{uuid.uuid4()}{file_extension}"
                file_path = os.path.join(UPLOAD_DIR, unique_filename)

                async with aiofiles.open(file_path, "wb") as buffer:
                    await buffer.write(contents)

                file_info = {
                    "original_name": sanitize_for_logging(file.filename),
                    "saved_name": sanitize_for_logging(unique_filename),
                    "size": file_size,
                    "type": file.content_type,
                    "path": str(file_path),
                    "uploaded_at": datetime.now(timezone.utc).isoformat()
                }
                file_type = file.content_type
                logger.info(f"File saved: {file_path}")

            multi_agents = load_multi_agents()
            multi_agent_config = next((ma for ma in multi_agents if ma["id"] == multi_agent_id), None)
            
            if not multi_agent_config:
                logger.error(f"Multi-Agent not found: {multi_agent_id}")
                return JSONResponse(
                    content={
                        "type": "error",
                        "content": {
                            "message": "Multi-Agent not found",
                            "details": f"No multi-agent found with ID: {multi_agent_id}"
                        },
                        "execution_id": execution_id,
                        "log_url": log_url,
                        "session_id": session_id
                    },
                    status_code=404
                )

            multi_agent_config.setdefault("role", "Coordinator")
            multi_agent_config.setdefault("goal", "Efficiently manage and delegate tasks.")
            multi_agent_config.setdefault("backstory", "Orchestrator for connected agents.")
            multi_agent_config.setdefault("description", "Coordinate the processing of the user request by delegating to worker agents.")

            multi_agent_config.setdefault("expected_output", (
                "Agent Outputs:\n"
                "<agent_name> Output: <output from agent>\n"
                "(repeated for each agent in the sequence)\n"
            ))
            all_agents = load_agents()
            connected_agent_ids = multi_agent_config.get("agent_ids", [])
            worker_agent_configs = []

            all_tools = load_custom_tools()
            connectors = load_connectors()

            for agent_id in connected_agent_ids:
                agent_data = next((a for a in all_agents if a["id"] == agent_id), None)
                if not agent_data:
                    logger.warning(f"Agent with ID {agent_id} not found, skipping.")
                    continue

                worker_tools_config = []
                for tool_id in agent_data.get("tools", []):
                    schema_path = f"tool_schemas/{tool_id}.json"
                    auth_path = f"tool_auth/{tool_id}.json"
                    tool_cfg = {"id": tool_id}

                    if os.path.exists(schema_path):
                        try:
                            with open(schema_path, 'r') as f:
                                tool_cfg["schema"] = json.load(f)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON in schema file {schema_path} for agent {agent_id}: {e}")
                            continue
                    else:
                        logger.warning(f"Schema file not found for tool {tool_id} in agent {agent_id}")
                        continue

                    if os.path.exists(auth_path):
                        try:
                            with open(auth_path, 'r') as f:
                                tool_cfg["auth"] = json.load(f)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON in auth file {auth_path} for agent {agent_id}: {e}")
                            tool_cfg["auth"] = {}

                    tool = next((t for t in all_tools if t.id == tool_id), None)
                    if tool and tool.data_connector_id:
                        connector = next((c for c in connectors if c["id"] == tool.data_connector_id), None)
                        if connector:
                            tool_cfg["data_connector"] = connector

                    worker_tools_config.append(tool_cfg)

                worker_config = {
                    "id": agent_data["id"],
                    "name": agent_data.get("name", agent_data["role"]),
                    "role": agent_data["role"],
                    "goal": agent_data["goal"],
                    "backstory": agent_data["backstory"],
                    "instructions": agent_data.get("instructions", f"Perform tasks as {agent_data['role']}"),
                    "expectedOutput": agent_data.get("expectedOutput", "A contribution to the overall goal"),
                    "tools": worker_tools_config
                }
                worker_agent_configs.append(worker_config)
                logger.info(f"Loaded config for worker agent {agent_id} ({worker_config['name']})")

            MIN_AGENTSFOR_MULTI = 2
            if len(worker_agent_configs) < MIN_AGENTSFOR_MULTI:
                logger.error("At least two worker agents are required.")
                return JSONResponse(
                    content={
                        "type": "error",
                        "content": {
                            "message": "Insufficient worker agents",
                            "details": "Multi-agent requires at least two worker agents"
                        },
                        "execution_id": execution_id,
                        "log_url": log_url,
                        "session_id": session_id
                    },
                    status_code=400
                )

            if user_input:
                default_description = multi_agent_config["description"]
                if "{{input}}" in default_description:
                    multi_agent_config["description"] = default_description.replace("{{input}}", user_input)
                else:
                    multi_agent_config["description"] = f"{default_description}\nInput to process: {user_input}"

            # Retrieve chat history from diskcache
            cache_key = f"history:{multi_agent_id}:{session_id or user_email}"
            history_data = cache.get(cache_key)
            history = json.loads(history_data) if history_data else []
            if history:
                logger.info(f"Retrieved history from cache for key {cache_key}: {len(history)} entries")
                if len(history) > 5:
                    history = history[-5:]
                    cache.setex(cache_key, 3600, json.dumps(history))
            else:
                logger.info(f"No history found in cache for key {cache_key}")

            # Create multi-agent executor
            logger.info("Initializing multi-agent executor")
            executor = MultiAgentExecutor(
                multi_agent_config=multi_agent_config,
                worker_agent_configs=worker_agent_configs,
                execution_id=execution_id,
                log_file=log_file,
                log_url=log_url,
                user_email=user_email
            )
            logger.info("Multi-agent executor initialized successfully")

            # Execute the task
            logger.info("Starting multi-agent task execution")
            result = await executor.execute_task(
                user_input=user_input,
                file_path=file_path,
                file_type=file_type,
                history=history
            )
            
            # Calculate execution time
            execution_time = time.time() - start_time
            logger.info(f"Multi-agent task completed successfully in {execution_time:.2f} seconds: {sanitize_for_logging(result[:100])}{'...' if len(result) > 100 else ''}")

            # Update chat history
            history.append({
                "input": user_input,
                "response": result,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            cache.setex(cache_key, 3600, json.dumps(history))
            logger.info(f"Updated history in cache for key {cache_key}: {len(history)} entries")

            # Log the request and response
            try:
                agent_name = multi_agent_config.get("name", "Multi-Agent System") if multi_agent_config else "Multi-Agent System"
                await log_request_to_json(
                    request=request,
                    user_input=user_input,
                    response_data={
                        "type": "text",
                        "content": {
                            "response": result,
                            "sender_agent_name": "Manager Agent"
                        },
                        "execution_id": execution_id,
                        "log_url": log_url,
                        "session_id": session_id
                    },
                    session_id=session_id,
                    execution_time=execution_time,
                    agent_id=multi_agent_id,
                    agent_name=agent_name,
                    execution_id=execution_id,
                    user=user_email
                )
            except Exception as log_error:
                logger.warning(f"Failed to log request to JSON: {sanitize_for_logging(str(log_error))}")

            return JSONResponse(
                content={
                    "type": "text",
                    "content": {
                        "response": result,
                        "sender_agent_name": "Manager Agent"
                    },
                    "execution_id": execution_id,
                    "log_url": log_url,
                    "session_id": session_id
                },
                status_code=200
            )

        except HTTPException as http_exc:
            logger.error(f"HTTP error in multi_agent_infer: {sanitize_for_logging(http_exc.detail)}")
            return JSONResponse(
                content={
                    "type": "error",
                    "content": {
                        "message": http_exc.detail,
                        "details": str(http_exc)
                    },
                    "execution_id": execution_id,
                    "log_url": log_url,
                    "session_id": session_id
                },
                status_code=http_exc.status_code
            )
        except Exception as e:
            logger.error(f"Error in multi_agent_infer: {sanitize_for_logging(str(e))}", exc_info=True)
            return JSONResponse(
                content={
                    "type": "error",
                    "content": {
                        "message": "Internal server error",
                        "details": str(e)
                    },
                    "execution_id": execution_id,
                    "log_url": log_url,
                    "session_id": session_id
                },
                status_code=500
            )
        finally:
            # Clean up any uploaded files to free disk space
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"[{execution_id}] Cleaned up uploaded file: {file_path}")
                except Exception as e:
                    logger.warning(f"[{execution_id}] Failed to clean up file: {e}")
        


    @app.post("/api/multi-agent/execute")
    @limiter.limit("20/minute")
    async def multi_agent_execute(
        request: Request,
        response: Response = None
    ):
        try:
            if request.headers.get("AGENT-API-KEY", "") == AGENT_API_KEY:
                # Generate execution ID
                execution_uuid = str(uuid.uuid4())
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                execution_id = f"{execution_uuid}_{timestamp}"
                log_filename = f"multi_agent_execution_{execution_id}.log"
                log_file = os.path.join(LOG_DIR, log_filename)
                
                # Extract values from the request body
                json_data = await request.json()
                multi_agent_id = json_data.get("agentId")
                user_input = json_data.get("userInput")
                
                session_id = request.cookies.get("session_id") or json_data.get("session_id") or str(uuid.uuid4())
                
                # Configure response headers for long-running request to prevent network suspension
                response.headers["Connection"] = "keep-alive"
                response.headers["Keep-Alive"] = "timeout=1200, max=1000"  # 20 minutes timeout, max 1000 requests
                response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
                
                # Add headers to prevent browser from suspending the connection
                response.headers["X-Accel-Buffering"] = "no"  # Disable nginx buffering if behind nginx
                response.headers["X-Content-Type-Options"] = "nosniff"
                
                # Generate log URL
                log_url = f"{BASE_URL}/api/multi_agent/logs/{execution_id}"
                
                # Configure logger
                logger = logging.getLogger(f"multi_agent_execute_{execution_id}")
                logger.setLevel(logging.DEBUG)
                formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
                try:
                    file_handler = logging.FileHandler(log_file, encoding="utf-8")
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)
                except Exception as e:
                    logger.error(f"Failed to create log file handler: {sanitize_for_logging(e)}")
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                logger.addHandler(console_handler)

                # Get user email from request state
                user_email = getattr(request.state, "user_email", "External User")
                if not user_email:
                    user_email = "agent@impactanalytics.co"
                    logger.info("Request from unauthenticated user")

                try:
                    sanitized_user_input = sanitize_for_logging(user_input)
                    logger.info(f"Received multi-agent execute request for ID: {multi_agent_id}, userInput={sanitized_user_input}")
                    logger.debug(f"Execution ID: {execution_id}")
                    logger.debug(f"Log URL: {log_url}")

                    if not user_input or user_input.strip() == "":
                        logger.error("Empty or invalid user input provided.")
                        return JSONResponse(
                            content={
                                "type": "error",
                                "content": {
                                    "message": "User input cannot be empty",
                                    "details": "Please provide valid user input"
                                },
                                "execution_id": execution_id,
                                "log_url": log_url,
                                "session_id": session_id
                            },
                            status_code=400
                        )

                    multi_agents = load_multi_agents()
                    multi_agent_config = next((ma for ma in multi_agents if ma["id"] == multi_agent_id), None)
                    
                    if not multi_agent_config:
                        logger.error(f"Multi-Agent not found: {multi_agent_id}")
                        return JSONResponse(
                            content={
                                "type": "error",
                                "content": {
                                    "message": "Multi-Agent not found",
                                    "details": f"No multi-agent found with ID: {multi_agent_id}"
                                },
                                "execution_id": execution_id,
                                "log_url": log_url,
                                "session_id": session_id
                            },
                            status_code=404
                        )

                    multi_agent_config.setdefault("role", "Coordinator")
                    multi_agent_config.setdefault("goal", "Efficiently manage and delegate tasks.")
                    multi_agent_config.setdefault("backstory", "Orchestrator for connected agents.")
                    multi_agent_config.setdefault("description", "Coordinate the processing of the user request by delegating to worker agents.")

                    multi_agent_config.setdefault("expected_output", (
                        "Agent Outputs:\n"
                        "<agent_name> Output: <output from agent>\n"
                        "(repeated for each agent in the sequence)\n"
                    ))
                    all_agents = load_agents()
                    connected_agent_ids = multi_agent_config.get("agent_ids", [])
                    worker_agent_configs = []

                    all_tools = load_custom_tools()
                    connectors = load_connectors()

                    for agent_id in connected_agent_ids:
                        agent_data = next((a for a in all_agents if a["id"] == agent_id), None)
                        if not agent_data:
                            logger.warning(f"Agent with ID {agent_id} not found, skipping.")
                            continue

                        worker_tools_config = []
                        for tool_id in agent_data.get("tools", []):
                            schema_path = f"tool_schemas/{tool_id}.json"
                            auth_path = f"tool_auth/{tool_id}.json"
                            tool_cfg = {"id": tool_id}

                            if os.path.exists(schema_path):
                                try:
                                    with open(schema_path, 'r') as f:
                                        tool_cfg["schema"] = json.load(f)
                                except json.JSONDecodeError as e:
                                    logger.warning(f"Invalid JSON in schema file {schema_path} for agent {agent_id}: {e}")
                                    continue
                            else:
                                logger.warning(f"Schema file not found for tool {tool_id} in agent {agent_id}")
                                continue

                            if os.path.exists(auth_path):
                                try:
                                    with open(auth_path, 'r') as f:
                                        tool_cfg["auth"] = json.load(f)
                                except json.JSONDecodeError as e:
                                    logger.warning(f"Invalid JSON in auth file {auth_path} for agent {agent_id}: {e}")
                                    tool_cfg["auth"] = {}

                            tool = next((t for t in all_tools if t.id == tool_id), None)
                            if tool and tool.data_connector_id:
                                connector = next((c for c in connectors if c["id"] == tool.data_connector_id), None)
                                if connector:
                                    tool_cfg["data_connector"] = connector

                            worker_tools_config.append(tool_cfg)

                        worker_config = {
                            "id": agent_data["id"],
                            "name": agent_data.get("name", agent_data["role"]),
                            "role": agent_data["role"],
                            "goal": agent_data["goal"],
                            "backstory": agent_data["backstory"],
                            "instructions": agent_data.get("instructions", f"Perform tasks as {agent_data['role']}"),
                            "expectedOutput": agent_data.get("expectedOutput", "A contribution to the overall goal"),
                            "tools": worker_tools_config
                        }
                        worker_agent_configs.append(worker_config)
                        logger.info(f"Loaded config for worker agent {agent_id} ({worker_config['name']})")

                    MIN_AGENTSFOR_MULTI = 2
                    if len(worker_agent_configs) < MIN_AGENTSFOR_MULTI:
                        logger.error("At least two worker agents are required.")
                        return JSONResponse(
                            content={
                                "type": "error",
                                "content": {
                                    "message": "Insufficient worker agents",
                                    "details": "Multi-agent requires at least two worker agents"
                                },
                                "execution_id": execution_id,
                                "log_url": log_url,
                                "session_id": session_id
                            },
                            status_code=400
                        )

                    if user_input:
                        default_description = multi_agent_config["description"]
                        if "{{input}}" in default_description:
                            multi_agent_config["description"] = default_description.replace("{{input}}", user_input)
                        else:
                            multi_agent_config["description"] = f"{default_description}\nInput to process: {user_input}"

                    # Retrieve chat history from diskcache
                    cache_key = f"history:{multi_agent_id}:{session_id or user_email}"
                    history_data = cache.get(cache_key)
                    history = json.loads(history_data) if history_data else []
                    if history:
                        logger.info(f"Retrieved history from cache for key {cache_key}: {len(history)} entries")
                        if len(history) > 5:
                            history = history[-5:]
                            cache.setex(cache_key, 3600, json.dumps(history))

                    executor = MultiAgentExecutor(
                        multi_agent_config=multi_agent_config,
                        worker_agent_configs=worker_agent_configs,
                        execution_id=execution_id,
                        log_file=log_file,
                        log_url=log_url,
                        user_email=user_email
                    )

                    # Start timing the execution
                    start_time = time.time()

                    result = await executor.execute_task(
                        user_input=user_input,
                        file_path=None,  # API key authenticated requests don't support file uploads
                        file_type=None,
                        history=history
                    )
                    logger.info(f"Multi-agent task completed successfully: {sanitize_for_logging(result[:100])}{'...' if len(result) > 100 else ''}")

                    # Calculate execution time
                    execution_time = time.time() - start_time

                    # Generate HTML content using existing function
                    html_content = await generate_html_content(result)

                    # Get multi-agent name for logging
                    multi_agent_name = multi_agent_config.get("name", "Unknown Multi-Agent")

                    # Prepare response data
                    response_data = {
                        "type": "text",
                        "content": {
                            "response": result,
                            "sender_agent_name": "Manager Agent"
                        },
                        "execution_id": execution_id,
                        "log_url": log_url,
                        "session_id": session_id,
                        "html_content": html_content
                    }

                    # Log the request and response with explicit argument naming
                    await log_request_to_json(
                        request=request,
                        user_input=user_input if user_input else "",
                        response_data=response_data,
                        session_id=session_id,
                        execution_time=execution_time,
                        agent_id=multi_agent_id,
                        agent_name=multi_agent_name,
                        execution_id=execution_id
                    )

                    # Update chat history
                    history.append({
                        "input": user_input,
                        "response": result,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    cache.setex(cache_key, 3600, json.dumps(history))
                    logger.info(f"Updated history in cache for key {cache_key}: {len(history)} entries")

                    return JSONResponse(
                        content=response_data,
                        status_code=200
                    )

                except HTTPException as http_exc:
                    logger.error(f"HTTP error in multi_agent_execute: {sanitize_for_logging(http_exc.detail)}")
                    return JSONResponse(
                        content={
                            "type": "error",
                            "content": {
                                "message": http_exc.detail,
                                "details": str(http_exc)
                            },
                            "execution_id": execution_id,
                            "log_url": log_url,
                            "session_id": session_id
                        },
                        status_code=http_exc.status_code
                    )
                except Exception as e:
                    logger.error(f"Error in multi_agent_execute: {sanitize_for_logging(str(e))}", exc_info=True)
                    return JSONResponse(
                        content={
                            "type": "error",
                            "content": {
                                "message": "Internal server error",
                                "details": str(e)
                            },
                            "execution_id": execution_id,
                            "log_url": log_url,
                            "session_id": session_id
                        },
                        status_code=500
                    )
            else:
                raise HTTPException(status_code=401, detail="Unauthorized access. Please provide a valid Agent API key and try again.")
        except asyncio.TimeoutError:
            logger.error("Multi-agent execution timed out")
            raise HTTPException(
                status_code=408, 
                detail="Request timed out. The multi-agent took too long to process your request. Please try again."
            )
        except Exception as e:
            logger.error(f"Error in multi_agent_execute: {str(e)}")
            raise



    
    @app.post("/api/multi_agent/execute")
    @limiter.limit("20/minute")
    async def multi_agent_execute_v2(
        request: Request,
        response: Response = None
    ):
        """
        Execute multi-agent with AGENT-API-KEY authentication.
        Clone of /api/multi_agent/infer but uses AGENT-API-KEY header instead of token.
        
        Request Body:
        {
            "multi_agent_id": "uuid-of-multi-agent",
            "user_input": "Your input here",
            "session_id": "optional-session-id"
        }
        """
        # Initialize variables for cleanup and error handling
        file_path = None
        multi_agent_config = None
        
        # Generate execution ID outside try block for error handling
        execution_uuid = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        execution_id = f"{execution_uuid}_{timestamp}"
        log_filename = f"multi_agent_execution_{execution_id}.log"
        log_file = os.path.join(LOG_DIR, log_filename)
        
        # Generate log URL
        log_url = f"{BASE_URL}/api/multi_agent/logs/{execution_id}"
        
        # Configure logger early
        logger = logging.getLogger(f"multi_agent_execute_v2_{execution_id}")
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # Fallback if file handler fails
            pass
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Start timing the execution
        start_time = time.time()
        
        # Initialize session_id early for error responses
        session_id = str(uuid.uuid4())

        try:
            # Validate AGENT-API-KEY
            if request.headers.get("AGENT-API-KEY", "") != AGENT_API_KEY:
                raise HTTPException(status_code=401, detail="Unauthorized access. Please provide a valid Agent API key and try again.")
            
            # Parse JSON with error handling
            try:
                json_data = await request.json()
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in multi_agent execute request: {str(e)}")
                return JSONResponse(
                    content={
                        "type": "error",
                        "content": {
                            "message": "Invalid JSON in request body",
                            "details": str(e)
                        },
                        "execution_id": execution_id,
                        "log_url": log_url,
                        "session_id": session_id
                    },
                    status_code=400
                )
            except Exception as e:
                logger.error(f"Error parsing request JSON: {str(e)}")
                return JSONResponse(
                    content={
                        "type": "error",
                        "content": {
                            "message": "Error parsing request body",
                            "details": str(e)
                        },
                        "execution_id": execution_id,
                        "log_url": log_url,
                        "session_id": session_id
                    },
                    status_code=400
                )
            
            multi_agent_id = json_data.get("multi_agent_id")
            user_input = json_data.get("user_input")
            user = json_data.get("user")  # Optional user parameter
            
            # Get session_id from request
            session_id = request.cookies.get("session_id") or json_data.get("session_id") or session_id
            
            # Configure response headers for long-running request to prevent network suspension
            response.headers["Connection"] = "keep-alive"
            response.headers["Keep-Alive"] = "timeout=1200, max=1000"  # 20 minutes timeout, max 1000 requests
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            
            # Add headers to prevent browser from suspending the connection
            response.headers["X-Accel-Buffering"] = "no"  # Disable nginx buffering if behind nginx
            response.headers["X-Content-Type-Options"] = "nosniff"
            
            # Get user email from request state or use default
            user_email = user or getattr(request.state, "user_email", "External User")
            if not user_email or user_email == "External User":
                user_email = "agent@impactanalytics.co"
                logger.info("Request from unauthenticated user")
            else:
                logger.info(f"Request from user: {user_email}")
            
            # Validate required parameters
            if not multi_agent_id:
                logger.error("Missing multi_agent_id parameter")
                return JSONResponse(
                    content={
                        "type": "error",
                        "content": {
                            "message": "Missing parameters",
                            "details": "multi_agent_id is required"
                        },
                        "execution_id": execution_id,
                        "log_url": log_url,
                        "session_id": session_id
                    },
                    status_code=400
                )
            
            if not user_input or user_input.strip() == "":
                logger.error("Empty or invalid user input provided")
                return JSONResponse(
                    content={
                        "type": "error",
                        "content": {
                            "message": "User input cannot be empty",
                            "details": "Please provide valid user input"
                        },
                        "execution_id": execution_id,
                        "log_url": log_url,
                        "session_id": session_id
                    },
                    status_code=400
                )
            
            sanitized_user_input = sanitize_for_logging(user_input)
            logger.info(f"Received multi-agent execute request for ID: {multi_agent_id}, userInput={sanitized_user_input}")
            logger.debug(f"Execution ID: {execution_id}")
            logger.debug(f"Log URL: {log_url}")

            multi_agents = load_multi_agents()
            multi_agent_config = next((ma for ma in multi_agents if ma["id"] == multi_agent_id), None)
            
            if not multi_agent_config:
                logger.error(f"Multi-Agent not found: {multi_agent_id}")
                return JSONResponse(
                    content={
                        "type": "error",
                        "content": {
                            "message": "Multi-Agent not found",
                            "details": f"No multi-agent found with ID: {multi_agent_id}"
                        },
                        "execution_id": execution_id,
                        "log_url": log_url,
                        "session_id": session_id
                    },
                    status_code=404
                )

            multi_agent_config.setdefault("role", "Coordinator")
            multi_agent_config.setdefault("goal", "Efficiently manage and delegate tasks.")
            multi_agent_config.setdefault("backstory", "Orchestrator for connected agents.")
            multi_agent_config.setdefault("description", "Coordinate the processing of the user request by delegating to worker agents.")

            multi_agent_config.setdefault("expected_output", (
                "Agent Outputs:\n"
                "<agent_name> Output: <output from agent>\n"
                "(repeated for each agent in the sequence)\n"
            ))
            all_agents = load_agents()
            connected_agent_ids = multi_agent_config.get("agent_ids", [])
            worker_agent_configs = []

            all_tools = load_custom_tools()
            connectors = load_connectors()

            for agent_id in connected_agent_ids:
                agent_data = next((a for a in all_agents if a["id"] == agent_id), None)
                if not agent_data:
                    logger.warning(f"Agent with ID {agent_id} not found, skipping.")
                    continue

                worker_tools_config = []
                for tool_id in agent_data.get("tools", []):
                    schema_path = f"tool_schemas/{tool_id}.json"
                    auth_path = f"tool_auth/{tool_id}.json"
                    tool_cfg = {"id": tool_id}

                    if os.path.exists(schema_path):
                        try:
                            with open(schema_path, 'r') as f:
                                tool_cfg["schema"] = json.load(f)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON in schema file {schema_path} for agent {agent_id}: {e}")
                            continue
                    else:
                        logger.warning(f"Schema file not found for tool {tool_id} in agent {agent_id}")
                        continue

                    if os.path.exists(auth_path):
                        try:
                            with open(auth_path, 'r') as f:
                                tool_cfg["auth"] = json.load(f)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Invalid JSON in auth file {auth_path} for agent {agent_id}: {e}")
                            tool_cfg["auth"] = {}

                    tool = next((t for t in all_tools if t.id == tool_id), None)
                    if tool and tool.data_connector_id:
                        connector = next((c for c in connectors if c["id"] == tool.data_connector_id), None)
                        if connector:
                            tool_cfg["data_connector"] = connector

                    worker_tools_config.append(tool_cfg)

                worker_config = {
                    "id": agent_data["id"],
                    "name": agent_data.get("name", agent_data["role"]),
                    "role": agent_data["role"],
                    "goal": agent_data["goal"],
                    "backstory": agent_data["backstory"],
                    "instructions": agent_data.get("instructions", f"Perform tasks as {agent_data['role']}"),
                    "expectedOutput": agent_data.get("expectedOutput", "A contribution to the overall goal"),
                    "tools": worker_tools_config
                }
                worker_agent_configs.append(worker_config)
                logger.info(f"Loaded config for worker agent {agent_id} ({worker_config['name']})")

            MIN_AGENTSFOR_MULTI = 2
            if len(worker_agent_configs) < MIN_AGENTSFOR_MULTI:
                logger.error("At least two worker agents are required.")
                return JSONResponse(
                    content={
                        "type": "error",
                        "content": {
                            "message": "Insufficient worker agents",
                            "details": "Multi-agent requires at least two worker agents"
                        },
                        "execution_id": execution_id,
                        "log_url": log_url,
                        "session_id": session_id
                    },
                    status_code=400
                )

            if user_input:
                default_description = multi_agent_config["description"]
                if "{{input}}" in default_description:
                    multi_agent_config["description"] = default_description.replace("{{input}}", user_input)
                else:
                    multi_agent_config["description"] = f"{default_description}\nInput to process: {user_input}"

            # Retrieve chat history from diskcache
            cache_key = f"history:{multi_agent_id}:{session_id or user_email}"
            history_data = cache.get(cache_key)
            history = json.loads(history_data) if history_data else []
            if history:
                logger.info(f"Retrieved history from cache for key {cache_key}: {len(history)} entries")
                if len(history) > 5:
                    history = history[-5:]
                    cache.setex(cache_key, 3600, json.dumps(history))
            else:
                logger.info(f"No history found in cache for key {cache_key}")

            # Create multi-agent executor
            logger.info("Initializing multi-agent executor")
            executor = MultiAgentExecutor(
                multi_agent_config=multi_agent_config,
                worker_agent_configs=worker_agent_configs,
                execution_id=execution_id,
                log_file=log_file,
                log_url=log_url,
                user_email=user_email
            )
            logger.info("Multi-agent executor initialized successfully")

            # Execute the task
            logger.info("Starting multi-agent task execution")
            result = await executor.execute_task(
                user_input=user_input,
                file_path=None,  # API key authenticated requests don't support file uploads
                file_type=None,
                history=history
            )
            
            # Calculate execution time
            execution_time = time.time() - start_time
            logger.info(f"Multi-agent task completed successfully in {execution_time:.2f} seconds: {sanitize_for_logging(result[:100])}{'...' if len(result) > 100 else ''}")

            # Generate HTML content using existing function
            html_content = await generate_html_content(result)

            # Update chat history
            history.append({
                "input": user_input,
                "response": result,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            cache.setex(cache_key, 3600, json.dumps(history))
            logger.info(f"Updated history in cache for key {cache_key}: {len(history)} entries")

            # Prepare response data
            response_data = {
                "type": "text",
                "content": {
                    "response": result,
                    "sender_agent_name": "Manager Agent"
                },
                "execution_id": execution_id,
                "log_url": log_url,
                "session_id": session_id,
                "html_content": html_content
            }

            # Log the request and response with explicit argument naming
            try:
                agent_name = multi_agent_config.get("name", "Multi-Agent System") if multi_agent_config else "Multi-Agent System"
                await log_request_to_json(
                    request=request,
                    user_input=user_input,
                    response_data=response_data,
                    session_id=session_id,
                    execution_time=execution_time,
                    agent_id=multi_agent_id,
                    agent_name=agent_name,
                    execution_id=execution_id,
                    user=user_email
                )
            except Exception as log_error:
                logger.warning(f"Failed to log request to JSON: {sanitize_for_logging(str(log_error))}")

            return JSONResponse(
                content=response_data,
                status_code=200
            )

        except HTTPException as http_exc:
            logger.error(f"HTTP error in multi_agent_execute_v2: {sanitize_for_logging(http_exc.detail)}")
            return JSONResponse(
                content={
                    "type": "error",
                    "content": {
                        "message": http_exc.detail,
                        "details": str(http_exc)
                    },
                    "execution_id": execution_id,
                    "log_url": log_url,
                    "session_id": session_id
                },
                status_code=http_exc.status_code
            )
        except asyncio.TimeoutError:
            logger.error("Multi-agent execution timed out")
            return JSONResponse(
                content={
                    "type": "error",
                    "content": {
                        "message": "Request timed out",
                        "details": "The multi-agent took too long to process your request. Please try again."
                    },
                    "execution_id": execution_id,
                    "log_url": log_url,
                    "session_id": session_id
                },
                status_code=408
            )
        except Exception as e:
            logger.error(f"Error in multi_agent_execute_v2: {sanitize_for_logging(str(e))}", exc_info=True)
            return JSONResponse(
                content={
                    "type": "error",
                    "content": {
                        "message": "Internal server error",
                        "details": str(e)
                    },
                    "execution_id": execution_id,
                    "log_url": log_url,
                    "session_id": session_id
                },
                status_code=500
            )


    # Shared log reading and masking function
    async def read_and_mask_log_file(execution_id: str, log_prefix: str) -> tuple[str, str, str]:
        log_file = os.path.join(LOG_DIR, f"{log_prefix}_{execution_id}.log")
        
        if not os.path.exists(log_file):
            raise HTTPException(status_code=404, detail="Log file not found")
        
        log_lines = []
        warning = ""
        logger = logging.getLogger(f"log_reader_{execution_id}")
        
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                log_lines = f.readlines()
        except UnicodeDecodeError as e:
            warning = f"Warning: Some characters in the log file could not be decoded and were replaced. Error: {sanitize_for_logging(str(e))}"
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                log_lines = f.readlines()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading log file: {sanitize_for_logging(str(e))}")

        processed_lines = []
        json_regex = r'\{(?:[^"{}]+|"(?:\\.|[^"\\])*"|\{[^{}]*\}|\[[^\[\]]*\])*\}'
        
        for line in log_lines:
            processed_line = line
            json_matches = re.finditer(json_regex, line, re.DOTALL)
            for match in json_matches:
                json_str = match.group(0)
                logger.debug(f"Detected JSON in line: {sanitize_for_logging(json_str[:100])}...")
                try:
                    json_obj = json.loads(json_str)
                    def has_sensitive_keys(obj):
                        if isinstance(obj, dict):
                            return any(
                                key.lower() in [k.lower() for k in MASKED_KEYS] or has_sensitive_keys(value)
                                for key, value in obj.items()
                            )
                        elif isinstance(obj, list):
                            return any(has_sensitive_keys(item) for item in obj)
                        return False

                    if has_sensitive_keys(json_obj):
                        logger.debug(f"Found sensitive keys in JSON: {sanitize_for_logging(json_str[:100])}...")
                    masked_json_obj = mask_sensitive_data(json_obj, MASKED_KEYS)
                    masked_json_str = json.dumps(masked_json_obj, ensure_ascii=False)
                    processed_line = processed_line.replace(json_str, masked_json_str, 1)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON in line: {sanitize_for_logging(json_str[:100])}... Error: {sanitize_for_logging(str(e))}")
                    continue
            processed_lines.append(processed_line)
        
        log_content = "".join(processed_lines)
        log_content = log_content.replace("<", "<").replace(">", ">").replace("\n", "<br>")
        
        return log_content, warning, log_file

    @app.get("/api/multi_agent/logs/{execution_id}", response_class=HTMLResponse)
    @limiter.limit("10/minute")
    async def get_multi_agent_log_file(execution_id: str, request: Request):
        log_content, warning, log_file = await read_and_mask_log_file(execution_id, "multi_agent_execution")
        
        # Render the template with dynamic data
        return page_templates.TemplateResponse(
            "multiagent_logs.html",
            {
                "request": request,
                "execution_id": execution_id,
                "warning": warning,
                "log_content": log_content
            }
        )

def check_in_sentence(sentence="", input_to_check="{{input}}"):
    sentence_lower = sentence.lower()
    input_lower = input_to_check.lower()
    
    if input_lower in sentence_lower:
        return sentence
    else:
        return sentence + "\n\n\ninput: " + input_to_check

@app.post("/api/agent/upload")
@limiter.limit("30/minute")  # File upload endpoint
async def upload_file(request: Request, file: UploadFile = File(...), agentId: str = None):
    try:
        if not agentId:
            return MessageResponse(
                type="error",
                content=ErrorData(
                    message="Agent ID is required",
                    details="No agent ID provided"
                )
            )

        file_size = 0
        contents = await file.read()
        file_size = len(contents)
        if file_size > 10 * 1024 * 1024:
            return MessageResponse(
                type="error",
                content=ErrorData(
                    message="File too large",
                    details="Maximum file size is 10MB"
                )
            )

        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = UPLOAD_DIR / unique_filename

        with open(file_path, "wb") as buffer:
            buffer.write(contents)

        file_info = {
            "original_name": file.filename,
            "saved_name": unique_filename,
            "size": file_size,
            "type": file.content_type,
            "uploaded_at": datetime.now().isoformat()
        }

        if file.content_type.startswith('image/'):
            return MessageResponse(
                type="image",
                content={
                    "url": f"/static/uploads/{unique_filename}",
                    "info": file_info
                }
            )
        else:
            return MessageResponse(
                type="file",
                content={
                    "info": file_info,
                    "preview": None
                }
            )

    except Exception as e:
        return MessageResponse(
            type="error",
            content=ErrorData(
                message="File upload failed",
                details=str(e)
            )
        )

def _transcribe_audio_sync(audio_bytes: bytes, model, is_faster_whisper: bool = False) -> dict:
    """
    Synchronous transcription function to run in thread pool.
    Optimized for performance with faster-whisper and optimized parameters.
    """
    import tempfile
    
    temp_file_path = None
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir='temp') as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name
        
        # Use faster-whisper API if available (much faster)
        if is_faster_whisper:
            # faster-whisper optimized parameters
            segments, info = model.transcribe(
                temp_file_path,
                beam_size=5,  # Balance between speed and accuracy (default: 5)
                best_of=5,    # Number of candidates (default: 5)
                patience=1.0, # Beam search patience (lower = faster)
                length_penalty=1.0,
                temperature=0.0,  # Deterministic (faster)
                compression_ratio_threshold=2.4,  # Skip if too repetitive
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=True,
                initial_prompt=None,
                word_timestamps=False,  # Disable for speed
                prepend_punctuations="\"'\u00BF([{-",  # \u00BF is inverted question mark ¿
                append_punctuations="\"'.\u3002\uFF0C\uFF0C!\uFF01?\uFF1F:\uFF1A)\uFF1D]\uFF1D}\u3001",  # Unicode escapes for punctuation
                vad_filter=True,  # Voice activity detection (faster)
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    max_speech_duration_s=float("inf"),
                    threshold=0.5
                )
            )
            
            # Collect segments
            text_segments = []
            for segment in segments:
                text_segments.append(segment.text)
            
            text = " ".join(text_segments).strip()
            
            return {
                "text": text,
                "language": info.language if hasattr(info, 'language') else "unknown"
            }
        else:
            # OpenAI Whisper with optimized parameters
            result = model.transcribe(
                temp_file_path,
                fp16=False,  # Use FP32 for CPU, FP16 for GPU
                verbose=False,  # Disable verbose output for speed
                temperature=0.0,  # Deterministic (faster)
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=True,
                initial_prompt=None,
                word_timestamps=False,  # Disable for speed
                prepend_punctuations="\"'\u00BF([{-",  # \u00BF is inverted question mark ¿
                append_punctuations="\"'.\u3002\uFF0C\uFF0C!\uFF01?\uFF1F:\uFF1A)\uFF1D]\uFF1D}\u3001",  # Unicode escapes for punctuation
            )
            
            return {
                "text": result.get("text", ""),
                "language": result.get("language", "unknown")
            }
    finally:
        # Ensure temp file is always cleaned up
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass

@app.post("/api/transcribe")
@limiter.limit("30/minute")  # Transcription endpoint
async def transcribe_audio(request: Request, file: UploadFile = File(...)):
    """
    Transcribe audio file using Whisper model (optimized for production).
    
    Optimizations:
    - Pre-loaded model (loaded once at startup)
    - faster-whisper library (CTranslate2 backend, 4x faster)
    - GPU acceleration (if available)
    - INT8 quantization for faster inference
    - Thread pool execution (non-blocking)
    - Optimized transcription parameters
    - VAD (Voice Activity Detection) filtering
    
    Accepts audio files in various formats (wav, mp3, m4a, etc.)
    """
    global whisper_model, whisper_executor
    
    # Check if Whisper model is available
    if whisper_model is None or whisper_executor is None:
        raise HTTPException(
            status_code=503,
            detail="Transcription service is not available. Whisper model failed to load."
        )
    
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an audio file"
            )
        
        # Read file contents into memory
        contents = await file.read()
        file_size = len(contents)
        
        # Limit file size to 25MB (reasonable for audio)
        if file_size > 25 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size is 25MB"
            )
        
        # Use the global flag to determine which library is loaded
        global is_faster_whisper_loaded
        is_faster_whisper = is_faster_whisper_loaded
        
        # Double-check by trying to import and check instance type
        if is_faster_whisper:
            try:
                from faster_whisper import WhisperModel
                if not isinstance(whisper_model, WhisperModel):
                    is_faster_whisper = False
            except:
                is_faster_whisper = False
        
        # Run transcription in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            whisper_executor,
            _transcribe_audio_sync,
            contents,
            whisper_model,
            is_faster_whisper
        )
        
        return JSONResponse({
            "success": True,
            "text": result["text"],
            "language": result.get("language", "unknown")
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )

@app.post("/api/whisper/transcribe")
@limiter.limit("30/minute")  # Alternative transcription endpoint
async def transcribe_audio_whisper(request: Request, file: UploadFile = File(...)):
    """
    Alternative endpoint for Whisper transcription.
    Same functionality as /api/transcribe but with a more organized path structure.
    """
    # Reuse the same logic by calling the original function
    return await transcribe_audio(request, file)

class MultiAgentCreate(BaseModel):
    name: str
    description: str
    agent_ids: List[str]
    role: Optional[str] = "Coordinator"
    goal: Optional[str] = "Efficiently manage and delegate tasks to connected agents based on user requests."
    backstory: Optional[str] = "I am a manager agent responsible for orchestrating multiple specialized agents to achieve complex goals."
    expected_output: str

class MultiAgent(MultiAgentCreate):
    id: str

@app.get("/api/multi-agents")
@limiter.limit("60/minute")
async def get_multi_agents(request: Request):
    return load_multi_agents()

@app.post("/api/multi-agents")
@limiter.limit("30/minute")
async def create_multi_agent(request: Request, multi_agent: MultiAgentCreate):
    multi_agents = load_multi_agents()
    agent_data = multi_agent.dict(exclude_unset=False)
    new_multi_agent = MultiAgent(
        id=str(uuid.uuid4()),
        **agent_data
    )
    multi_agents.append(new_multi_agent.dict())
    save_multi_agents(multi_agents)
    return new_multi_agent

@app.get("/api/multi-agents/{multi_agent_id}")
@limiter.limit("60/minute")
async def get_multi_agent(request: Request, multi_agent_id: str):
    multi_agents = load_multi_agents()
    for ma in multi_agents:
        if ma["id"] == multi_agent_id:
            ma.setdefault("role", "Coordinator")
            ma.setdefault("goal", "Efficiently manage and delegate tasks.")
            ma.setdefault("backstory", "Orchestrator for connected agents.")
            return ma
    raise HTTPException(status_code=404, detail="Multi-Agent not found")

@app.put("/api/multi-agents/{multi_agent_id}")
@limiter.limit("30/minute")
async def update_multi_agent(request: Request, multi_agent_id: str, updated_multi_agent: MultiAgentCreate):
    multi_agents = load_multi_agents()
    for i, ma in enumerate(multi_agents):
        if ma["id"] == multi_agent_id:
            updated_data = updated_multi_agent.dict(exclude_unset=False)
            multi_agents[i] = {
                "id": multi_agent_id,
                **updated_data
            }
            save_multi_agents(multi_agents)
            return multi_agents[i]
    raise HTTPException(status_code=404, detail="Multi-Agent not found")

@app.delete("/api/multi-agents/{multi_agent_id}")
@limiter.limit("30/minute")
async def delete_multi_agent(request: Request, multi_agent_id: str):
    multi_agents = load_multi_agents()
    multi_agents = [ma for ma in multi_agents if ma["id"] != multi_agent_id]
    save_multi_agents(multi_agents)
    return {"message": "Multi-Agent deleted"}

class TimeRequest(BaseModel):
    hour: Optional[int] = None



class TextRequest(BaseModel):
    text: str


# --- Advanced Tool Models ---
class AdvancedTool(BaseModel):
    id: str
    name: str
    description: str
    tags: List[str]
    schema: Dict[str, Any]
    data_connector_id: Optional[str] = None
    connector_uniqueName: Optional[str] = None
    connectorType: Optional[str] = None
    is_added: bool = False

class AdvancedToolCreate(BaseModel):
    name: str
    description: str
    tags: List[str]
    schema: Dict[str, Any]
    data_connector_id: Optional[str] = None

def load_advanced_tools() -> List[AdvancedTool]:
    try:
        with open('advanced_tools.json', 'r') as f:
            tools_data = json.load(f)
            return [AdvancedTool(**tool) for tool in tools_data]
    except FileNotFoundError:
        return []

def save_advanced_tools(tools: List[AdvancedTool]):
    with open('advanced_tools.json', 'w') as f:
        json.dump([tool.dict() for tool in tools], f, indent=2)

# --- Advanced Tool Endpoints ---
@app.get("/api/advanced-tools", response_model=List[AdvancedTool])
@limiter.limit("10/minute")
async def get_advanced_tools(request: Request):
    logger.info("Fetching advanced tools")
    tools = load_advanced_tools()
    connectors = load_connectors()
    
    # Create a mapping of connector IDs to their names and types
    connector_map = {
        connector['id']: {
            'uniqueName': connector['uniqueName'],
            'connectorType': connector['connectorType']
        } for connector in connectors
    }
    
    # Add connector name and type to each tool
    for tool in tools:
        if tool.data_connector_id and tool.data_connector_id in connector_map:
            tool.connector_uniqueName = connector_map[tool.data_connector_id]['uniqueName']
            tool.connectorType = connector_map[tool.data_connector_id]['connectorType']
        else:
            tool.connector_uniqueName = None
            tool.connectorType = None
            
    logger.info(f"Returning {len(tools)} advanced tools")
    return tools

@app.post("/api/advanced-tools")
@limiter.limit("10/minute")
async def create_advanced_tool(request: Request, tool: AdvancedToolCreate):
    if tool.data_connector_id:
        connectors = load_connectors()
        if not any(c["id"] == tool.data_connector_id for c in connectors):
            raise HTTPException(status_code=400, detail="Invalid data connector ID")

    tools = load_advanced_tools()
    
    if any(t.name.lower() == tool.name.lower() for t in tools):
        raise HTTPException(status_code=400, detail="Tool with this name already exists")
    
    new_tool = AdvancedTool(
        id=str(uuid.uuid4()),
        name=tool.name,
        description=tool.description,
        tags=tool.tags,
        schema=tool.schema,
        data_connector_id=tool.data_connector_id,
        is_added=False
    )
    
    schema_dir = "tool_schemas"
    os.makedirs(schema_dir, exist_ok=True)
    with open(f"{schema_dir}/{new_tool.id}.json", 'w') as f:
        json.dump(tool.schema, f, indent=2)
    
    tools.append(new_tool)
    save_advanced_tools(tools)
    
    return new_tool

@app.get("/api/advanced-tools/{tool_id}")
@limiter.limit("10/minute")
async def get_advanced_tool(request: Request, tool_id: str):
    tools = load_advanced_tools()
    for tool in tools:
        if tool.id == tool_id:
            return tool
    raise HTTPException(status_code=404, detail="Advanced tool not found")

@app.put("/api/advanced-tools/{tool_id}")
@limiter.limit("10/minute")
async def update_advanced_tool(request: Request, tool_id: str, updated_tool: AdvancedToolCreate):
    tools = load_advanced_tools()
    
    if updated_tool.data_connector_id:
        connectors = load_connectors()
        if not any(c["id"] == updated_tool.data_connector_id for c in connectors):
            raise HTTPException(status_code=400, detail="Invalid data connector ID")
    
    for i, tool in enumerate(tools):
        if tool.id == tool_id:
            if any(t.name.lower() == updated_tool.name.lower() and t.id != tool_id for t in tools):
                raise HTTPException(status_code=400, detail="Tool with this name already exists")
            
            updated = AdvancedTool(
                id=tool_id,
                name=updated_tool.name,
                description=updated_tool.description,
                tags=updated_tool.tags,
                schema=updated_tool.schema,
                data_connector_id=updated_tool.data_connector_id,
                is_added=tool.is_added
            )
            
            schema_dir = "tool_schemas"
            with open(f"{schema_dir}/{tool_id}.json", 'w') as f:
                json.dump(updated_tool.schema, f, indent=2)
            
            tools[i] = updated
            save_advanced_tools(tools)
            return updated
            
    raise HTTPException(status_code=404, detail="Advanced tool not found")

@app.delete("/api/advanced-tools/{tool_id}")
@limiter.limit("10/minute")
async def delete_advanced_tool(request: Request, tool_id: str):
    tools = load_advanced_tools()
    tools = [tool for tool in tools if tool.id != tool_id]
    save_advanced_tools(tools)
    
    schema_path = f"tool_schemas/{tool_id}.json"
    if os.path.exists(schema_path):
        os.remove(schema_path)
    
    # Remove references from agents
    agents = load_agents()
    for agent in agents:
        if tool_id in agent.get('advanced_tools', []):
            agent['advanced_tools'].remove(tool_id)
    save_agents(agents)
    
    return {"message": "Advanced tool deleted"}

@app.post("/api/advanced-tools/{tool_id}/add")
@limiter.limit("30/minute")
async def add_advanced_tool(request: Request, tool_id: str):
    tools = load_advanced_tools()
    for tool in tools:
        if tool.id == tool_id:
            tool.is_added = True
            save_advanced_tools(tools)
            return {"message": "Advanced tool added successfully"}
    raise HTTPException(status_code=404, detail="Advanced tool not found")

@app.get("/api/tools/{tool_id}/metadata")
@limiter.limit("20/minute")
async def get_tool_metadata(request: Request, tool_id: str):
    metadata_path = f"tool_metadata/{tool_id}.json"
    if not os.path.exists(metadata_path):
        return {}  # Return empty metadata if none exists
    
    with open(metadata_path, 'r') as f:
        return json.load(f)


# Database health check endpoint
@app.get("/api/db/health", response_model=Dict[str, str])
@limiter.limit("20/minute")
async def db_health_check(request: Request, db: AsyncSession = Depends(get_db)):
    try:
        conn = db
        
        # Check if connection is active
        if conn and not conn.closed:
            with conn.cursor() as cursor:
                # Get PostgreSQL version
                cursor.execute("SELECT version();")
                version = cursor.fetchone()[0]
                
                # Get connection details
                cursor.execute("SELECT current_database(), current_user;")
                db_info = cursor.fetchone()
                
                return {
                    "status": "connected",
                    "database": db_info[0] if db_info else "unknown",
                    "user": db_info[1] if db_info else "unknown",
                    "version": version,
                    "connection_type": "persistent" if conn is db_connection else "pooled"
                }
        else:
            return {
                "status": "disconnected",
                "message": "Database connection is closed or not initialized"
            }
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/sessions-logs/dates")
@limiter.limit("20/minute")
async def get_available_dates(request: Request):
    """Get list of dates for which logs are available"""
    try:
        # Use SessionLogsDB to get available dates from database
        from DB.async_session_logs import AsyncSessionLogsDB
        session_logs_db = AsyncSessionLogsDB()
        
        dates, error = await session_logs_db.get_available_dates()
        
        if error:
            logger.error(f"Error getting available dates from database: {error}")
            # Fallback to file-based approach if database query fails
            log_dir = "request_logs"
            if not os.path.exists(log_dir):
                return {"dates": []}
            
            file_dates = []
            for filename in os.listdir(log_dir):
                if filename.startswith("session_logs_") and filename.endswith(".json"):
                    date_str = filename.replace("session_logs_", "").replace(".json", "")
                    try:
                        # Validate the date format
                        datetime.strptime(date_str, "%Y-%m-%d")
                        file_dates.append(date_str)
                    except ValueError:
                        continue
            
            # Sort dates in descending order (newest first)
            file_dates.sort(reverse=True)
            return {"dates": file_dates}
        
        # Return dates from database
        return {"dates": dates}
    except Exception as e:
        logger.error(f"Error getting available dates: {str(e)}")
        return {"dates": [], "error": str(e)}

@app.get("/api/sessions-logs")
@limiter.limit("20/minute")
async def get_sessions(request: Request, date: Optional[str] = None):
    """Get all sessions with summary data for a specific date or the latest date"""
    try:
        # Use SessionLogsDB to get sessions from database
        from DB.async_session_logs import AsyncSessionLogsDB
        session_logs_db = AsyncSessionLogsDB()
        
        # If no date is provided, get the latest date
        if not date:
            dates, error = await session_logs_db.get_available_dates()
            if error or not dates:
                return {
                    "sessions": [],
                    "total": 0,
                    "message": "No sessions found"
                }
            date = dates[0]  # Get the most recent date
        
        # Get sessions for the specified date
        sessions, error = await session_logs_db.get_sessions_by_date(date)
        
        if error:
            logger.error(f"Error getting sessions from database: {error}")
            # Fallback to file-based approach if database query fails
            log_dir = "request_logs"
            log_file = f"{log_dir}/session_logs_{date}.json"
            
            if not os.path.exists(log_file):
                return {
                    "sessions": [],
                    "total": 0,
                    "message": f"No sessions found for date {date}"
                }
            
            # Read the logs from file
            with open(log_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    return {
                        "sessions": [],
                        "total": 0,
                        "message": "Log file is empty"
                    }
                all_logs = json.loads(content)
            
            # Create session summaries from file data
            file_sessions = []
            for session_id, session_logs in all_logs.items():
                if not session_logs:
                    continue
                    
                # Sort by timestamp to get first and last
                sorted_logs = sorted(session_logs, key=lambda x: x.get("timestamp", ""))
                first_log = sorted_logs[0]
                last_log = sorted_logs[-1]
                
                # Get unique agents used in this session
                agents_used = list(set(log.get("agent_name", "Unknown") for log in session_logs if log.get("agent_name")))
                
                # Calculate total execution time
                total_execution_time = sum(log.get("execution_time_seconds", 0) for log in session_logs)
                
                session_summary = {
                    "session_id": session_id,
                    "conversation_count": len(session_logs),
                    "agents_used": agents_used,
                    "first_interaction": first_log.get("timestamp"),
                    "last_interaction": last_log.get("timestamp"),
                    "total_execution_time": total_execution_time,
                    "ip_address": first_log.get("ip_address", "Unknown"),
                    "user_agent": first_log.get("user_agent", "Unknown")
                }
                file_sessions.append(session_summary)
            
            # Sort sessions by last interaction (newest first)
            file_sessions.sort(key=lambda x: x.get("last_interaction", ""), reverse=True)
            
            return {
                "sessions": file_sessions,
                "total": len(file_sessions),
                "date": date,
                "source": "file"
            }
        
        # Return sessions from database
        return {
            "sessions": sessions,
            "total": len(sessions),
            "date": date,
            "source": "database"
        }
        
    except Exception as e:
        logger.error(f"Error getting sessions: {str(e)}")
        return {
            "sessions": [],
            "total": 0,
            "error": str(e)
        }

@app.get("/api/sessions-logs/stats")
@limiter.limit("20/minute")
async def get_session_stats(request: Request, date: Optional[str] = None):
    """Get statistics about session logs for a specific date or all dates"""
    try:
        # Use SessionLogsDB to get session stats from database
        from DB.async_session_logs import AsyncSessionLogsDB
        session_logs_db = AsyncSessionLogsDB()
        
        # Get stats from database
        stats, error = await session_logs_db.get_session_stats(date)
        
        if error:
            logger.error(f"Error getting session stats from database: {error}")
            # Fallback to file-based approach if database query fails
            log_dir = "request_logs"
            total_sessions = 0
            total_requests = 0
            agents_used = set()
            
            # Get list of files to process
            if date:
                files_to_process = [f"{log_dir}/session_logs_{date}.json"]
            else:
                # Process all available dates
                dates = (await get_available_dates())["dates"]
                files_to_process = [f"{log_dir}/session_logs_{d}.json" for d in dates]
            
            # Process each file
            for log_file in files_to_process:
                if not os.path.exists(log_file):
                    continue
                
                with open(log_file, 'r') as f:
                    content = f.read().strip()
                    if not content:
                        continue
                    all_logs = json.loads(content)
                    
                    total_sessions += len(all_logs)
                    total_requests += sum(len(session_logs) for session_logs in all_logs.values())
                    
                    for session_logs in all_logs.values():
                        for log_entry in session_logs:
                            if log_entry.get("agent_name"):
                                agents_used.add(log_entry.get("agent_name"))
            
            return {
                "total_sessions": total_sessions,
                "total_requests": total_requests,
                "agents_used": list(agents_used),
                "date": date if date else "all",
                "source": "file"
            }
        
        # Return stats from database
        stats["source"] = "database"
        return stats
        
    except Exception as e:
        logger.error(f"Error getting session stats: {str(e)}")
        return {
            "total_sessions": 0,
            "total_requests": 0,
            "agents_used": [],
            "error": str(e)
        }

@app.get("/api/sessions-logs/{session_id}/conversation")
@limiter.limit("20/minute")
async def get_session_conversation(request: Request, session_id: str, date: Optional[str] = None):
    """Get all conversations for a specific session"""
    try:
        # Use SessionLogsDB to get session conversation from database
        from DB.async_session_logs import AsyncSessionLogsDB
        session_logs_db = AsyncSessionLogsDB()
        
        # Get conversation from database
        conversation_data, error = await session_logs_db.get_session_conversation(session_id, date)
        
        if error:
            logger.error(f"Error getting session conversation from database: {error}")
            # Fallback to file-based approach if database query fails
            log_dir = "request_logs"
            
            # If no date is provided, search in all files
            if not date:
                dates = (await get_available_dates())["dates"]
                # Search for the session in all date files
                for d in dates:
                    log_file = f"{log_dir}/session_logs_{d}.json"
                    if os.path.exists(log_file):
                        with open(log_file, 'r') as f:
                            content = f.read().strip()
                            if content:
                                all_logs = json.loads(content)
                                if session_id in all_logs:
                                    date = d
                                    break
                
                if not date:
                    return {
                        "conversation": [],
                        "session_info": {},
                        "message": "Session not found in any date"
                    }
            
            log_file = f"{log_dir}/session_logs_{date}.json"
            
            if not os.path.exists(log_file):
                return {
                    "conversation": [],
                    "session_info": {},
                    "message": f"No logs found for date {date}"
                }
            
            # Read the logs
            with open(log_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    return {
                        "conversation": [],
                        "session_info": {},
                        "message": "Log file is empty"
                    }
                all_logs = json.loads(content)
            
            # Get conversation for the specific session
            session_logs = all_logs.get(session_id, [])
            if not session_logs:
                return {
                    "conversation": [],
                    "session_info": {},
                    "message": "Session not found"
                }
            
            # Sort by timestamp
            sorted_logs = sorted(session_logs, key=lambda x: x.get("timestamp", ""))
            
            # Process logs to extract text, html_content, and message_blocks from response
            processed_logs = []
            for log in sorted_logs:
                # Parse response JSON if it's a string
                response_data = log.get('response', {})
                if isinstance(response_data, str):
                    try:
                        response_data = json.loads(response_data)
                    except (json.JSONDecodeError, TypeError):
                        response_data = {}
                elif response_data is None:
                    response_data = {}
                
                # Extract text, html_content, and message_blocks from response
                # Check both 'blocks' and 'message_blocks' for backward compatibility
                text = response_data.get('text') if isinstance(response_data, dict) else None
                html_content = response_data.get('html_content') if isinstance(response_data, dict) else None
                # Try 'blocks' first (from final_answer), then 'message_blocks' (from MessageResponse)
                message_blocks = None
                if isinstance(response_data, dict):
                    message_blocks = response_data.get('blocks') or response_data.get('message_blocks')
                
                # Create processed log entry
                processed_log = {
                    **log,  # Keep all original fields
                    'response': response_data,  # Keep full response for backward compatibility
                    'text': text,  # Extract text
                    'html_content': html_content,  # Extract html_content
                    'message_blocks': message_blocks  # Extract message_blocks
                }
                processed_logs.append(processed_log)
            
            # Create session info
            first_log = processed_logs[0] if processed_logs else {}
            agents_used = list(set(log.get("agent_name", "Unknown") for log in session_logs if log.get("agent_name")))
            total_execution_time = sum(log.get("execution_time_seconds", 0) for log in session_logs)
            
            session_info = {
                "session_id": session_id,
                "conversation_count": len(session_logs),
                "agents_used": agents_used,
                "total_execution_time": total_execution_time,
                "ip_address": first_log.get("ip_address", "Unknown"),
                "user_agent": first_log.get("user_agent", "Unknown"),
                "first_interaction": first_log.get("timestamp"),
                "last_interaction": processed_logs[-1].get("timestamp") if processed_logs else None,
                "date": date
            }
            
            return {
                "conversation": processed_logs,
                "session_info": session_info,
                "source": "file"
            }
        
        # Return conversation from database
        conversation_data["source"] = "database"
        return conversation_data
        
    except Exception as e:
        logger.error(f"Error getting session conversation: {str(e)}")
        return {
            "conversation": [],
            "session_info": {},
            "error": str(e)
        }

# Reports Dashboard API Endpoints
@app.get("/api/reports/agent-logs")
@limiter.limit("20/minute")
async def get_agent_execution_logs(
    request: Request,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    agent_name: Optional[str] = None,
    limit: Optional[int] = 100,
    offset: Optional[int] = 0
):
    """Get agent execution logs with filtering for reports dashboard"""
    try:
        from DB.async_session_logs import AsyncSessionLogsDB
        session_logs_db = AsyncSessionLogsDB()
        
        # Convert date strings to datetime objects if provided
        from_datetime = None
        to_datetime = None
        
        if from_date:
            try:
                from_datetime = datetime.strptime(from_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid from_date format. Use YYYY-MM-DD")
        
        if to_date:
            try:
                # Add 23:59:59 to include the entire day
                to_datetime = datetime.strptime(to_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid to_date format. Use YYYY-MM-DD")
        
        # Get filtered logs from database
        logs, error = await session_logs_db.get_filtered_logs(
            from_date=from_datetime,
            to_date=to_datetime,
            agent_name=agent_name,
            limit=limit,
            offset=offset
        )
        
        if error:
            logger.error(f"Error getting filtered logs: {error}")
            return {
                "logs": [],
                "total": 0,
                "error": error
            }
        
        # Get total count for pagination
        total_count, count_error = await session_logs_db.get_filtered_logs_count(
            from_date=from_datetime,
            to_date=to_datetime,
            agent_name=agent_name
        )
        
        if count_error:
            logger.warning(f"Error getting total count: {count_error}")
            total_count = len(logs)
        
        return {
            "logs": logs,
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "filters": {
                "from_date": from_date,
                "to_date": to_date,
                "agent_name": agent_name
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent execution logs: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/reports/agent-names")
@limiter.limit("20/minute")
async def get_available_agent_names(request: Request):
    """Get list of available agent names for dropdown"""
    try:
        from DB.async_session_logs import AsyncSessionLogsDB
        session_logs_db = AsyncSessionLogsDB()
        
        agent_names, error = await session_logs_db.get_unique_agent_names()
        
        if error:
            logger.error(f"Error getting agent names: {error}")
            return {
                "agent_names": [],
                "error": error
            }
        
        return {
            "agent_names": agent_names
        }
        
    except Exception as e:
        logger.error(f"Error getting agent names: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/reports/stats")
@limiter.limit("20/minute")
async def get_reports_stats(
    request: Request,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    agent_name: Optional[str] = None
):
    """Get statistics for reports dashboard"""
    try:
        from DB.async_session_logs import AsyncSessionLogsDB
        session_logs_db = AsyncSessionLogsDB()
        
        # Convert date strings to datetime objects if provided
        from_datetime = None
        to_datetime = None
        
        if from_date:
            try:
                from_datetime = datetime.strptime(from_date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid from_date format. Use YYYY-MM-DD")
        
        if to_date:
            try:
                to_datetime = datetime.strptime(to_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid to_date format. Use YYYY-MM-DD")
        
        # Get statistics from database
        stats, error = await session_logs_db.get_reports_stats(
            from_date=from_datetime,
            to_date=to_datetime,
            agent_name=agent_name
        )
        
        if error:
            logger.error(f"Error getting reports stats: {error}")
            return {
                "total_executions": 0,
                "unique_sessions": 0,
                "unique_agents": 0,
                "avg_execution_time": 0,
                "total_execution_time": 0,
                "error": error
            }
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting reports stats: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


# Pydantic model for input validation
class AgentInput(BaseModel):
    details: str

# Pydantic model for response structure
class AgentOutput(BaseModel):
    role: str
    goal: str
    backstory: str
    expected_output: str
    agent_name: str
    title: str
    description: str
    sample_input: str

def handle_truncated_json(text: str) -> dict:
    """Handle truncated JSON responses by attempting to complete them with defaults."""
    try:
        # Try to find the start of JSON
        json_start = text.find('{')
        if json_start == -1:
            return None
        
        json_text = text[json_start:]
        
        # Count braces to see if JSON is complete
        open_braces = json_text.count('{')
        close_braces = json_text.count('}')
        
        if open_braces > close_braces:
            # JSON appears truncated, try to fix it
            # Find the last complete field by looking for complete key-value pairs
            # Use regex to find complete "key": "value" patterns
            field_pattern = r'"([^"]+)":\s*"([^"]*)"(?=\s*[,}])'
            matches = list(re.finditer(field_pattern, json_text))
            
            if matches:
                # Find the position after the last complete field
                last_match = matches[-1]
                last_complete_pos = last_match.end()
                
                # Get the text up to the last complete field
                complete_part = json_text[:last_complete_pos]
                # Remove trailing comma if present
                complete_part = complete_part.rstrip().rstrip(',')
                complete_json = complete_part + '}'
                
                # Try to parse the cleaned JSON
                parsed = json.loads(complete_json)
                
                # Fill in missing required fields with defaults
                required_fields = {
                    'role': 'Assistant',
                    'goal': 'Help users with their requests',
                    'backstory': 'An AI assistant ready to help',
                    'expected_output': 'Helpful responses and solutions',
                    'agent_name': 'Assistant',
                    'title': 'AI Assistant',
                    'description': 'General purpose AI assistant',
                    'sample_input': 'How can I help you?'
                }
                
                for field, default_value in required_fields.items():
                    if field not in parsed:
                        parsed[field] = default_value
                
                return parsed
                    
        return None
    except Exception as e:
        logger.error(f"Error in handle_truncated_json: {str(e)}")
        return None

async def llm_chat(prompt: str, max_tokens: int = 200) -> str:
    """
    Makes a synchronous call to the Gemini API with a given prompt, with retry logic and error handling.
    
    Args:
        prompt (str): The input prompt for the Gemini model.
        max_tokens (int): Maximum number of tokens to generate. Default is 200.
        
    Returns:
        str: The generated text from the Gemini model.
        
    Raises:
        ValueError: If the API key is not set or the response is empty.
        httpx.HTTPStatusError: If the API call fails after retries.
        Exception: For unexpected errors during the API call or response parsing.
    """
    api_key = os.getenv("INTERNAL_GEMINI_API_KEY", "")
    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


    if not api_key:
        error_msg = "INTERNAL_GEMINI_API_KEY environment variable is not set"
        logger.error(error_msg)
        raise ValueError(error_msg)

    @retry(
        stop=stop_after_attempt(3),  # Retry up to 3 times
        wait=wait_exponential(multiplier=1, min=4, max=10),  # Exponential backoff: 4s, 8s, 10s
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),  # Retry on network/HTTP errors
        before_sleep=lambda retry_state: logger.info(f"Retrying ({retry_state.attempt_number}/3) after {retry_state.idle_for}s...")
    )
    def make_api_call():
        with httpx.Client(timeout=30.0) as client:
            response = client.post(
                f"{endpoint}?key={api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{
                        "parts": [{"text": prompt}]
                    }],
                    "generationConfig": {
                        "maxOutputTokens": 200,
                        "temperature": 0.7
                    }
                }
            )
            response.raise_for_status()
            return response.json()

    try:
        result = make_api_call()

        # Extract generated text
        generated_text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        if not generated_text:
            error_msg = "Empty response from Gemini API: no text generated"
            logger.error(error_msg)
            raise ValueError(error_msg)

        return generated_text

    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        error_msg = f"Failed to get response from Gemini API after retries: {str(e)}"
        logger.error(error_msg)
        raise
    except Exception as e:
        error_msg = f"Unexpected error during Gemini API call: {str(e)}"
        logger.error(error_msg)
        raise

# Reusable LLM function to generate JSON
async def call_llm(prompt: str) -> dict:
    try:
        # Call llm_chat to get raw text response
        response_text = await llm_chat(prompt)
        
        # Clean up the response text
        # First try to extract JSON from markdown code blocks
        if "```" in response_text:
            # Extract content between code fences
            pattern = r"```(?:json)?\n?([\s\S]*?)```"
            matches = re.findall(pattern, response_text)
            if matches:
                cleaned_text = matches[0].strip()
            else:
                cleaned_text = response_text.strip("```json\n").strip("```").strip()
        else:
            cleaned_text = response_text.strip()
            
        try:
            # Try to parse the cleaned text as JSON
            return json.loads(cleaned_text)
        except json.JSONDecodeError as je:
            # If JSON parsing fails, try to fix common issues
            logger.warning(f"Initial JSON parsing failed: {str(je)}. Attempting to fix...")
            
            # Replace any unescaped newlines in strings
            cleaned_text = re.sub(r'(?<!\\)\n', '\\n', cleaned_text)
            # Fix unescaped quotes in strings
            cleaned_text = re.sub(r'(?<!\\)"(?=.*":)', '\\"', cleaned_text)
            
            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError as je2:
                logger.error(f"Failed to parse JSON even after cleanup: {str(je2)}")
                logger.error(f"Problematic text: {cleaned_text}")
                raise
                
    except Exception as e:
        logger.error(f"Failed to process LLM response: {str(e)}")
        logger.error(f"Raw response: {response_text if 'response_text' in locals() else 'No response received'}")
        raise

# Endpoint to generate agent details
@app.post("/api/generate-agent", response_model=AgentOutput)
@limiter.limit("20/minute")
async def generate_agent(request: Request, agent_input: AgentInput):
    try:
        prompt = f"""
        Create a CrewAI agent based on these details:
        {agent_input.details}

        Return a JSON object with these fields for CrewAI integration:
        - role: The specific role/expertise of the agent (e.g., "Data Analyst", "Research Scientist")
        - goal: The primary objective the agent works to achieve
        - backstory: Brief background explaining the agent's expertise and motivation
        - expected_output: What the agent will produce/deliver when completing tasks
        - agent_name: A descriptive name for the agent (e.g., "DataMaster", "ResearchPro")
        - title: Professional title that describes the agent's position
        - description: A concise summary of what the agent does
        - sample_input: A simple example input the agent could process (MUST be a string, not a list or array)

        Format your response as valid JSON without any additional text, code blocks, or explanations.
        Example format:
        {{"role": "Data Analyst", "goal": "Extract insights from complex datasets", "backstory": "Trained on thousands of datasets to identify patterns and anomalies", "expected_output": "Data visualizations and statistical analysis", "agent_name": "DataMaster", "title": "Senior Data Analyst", "description": "Specialized in finding patterns in large datasets", "sample_input": "sales_data.csv"}}
        """
        
        # Use increased max tokens to ensure complete response
        response_text = await llm_chat(prompt, max_tokens=1500)
        
        # Log response details for debugging
        logger.info(f"LLM response length: {len(response_text)} characters")
        if len(response_text) > 1400:  # Close to token limit
            logger.warning("Response approaching token limit, may be truncated")
        
        # Check for common truncation indicators
        if response_text.endswith(("'", '"', ",")) and not response_text.rstrip().endswith('}'):
            logger.warning("Response appears to be truncated (ends with quote/comma but no closing brace)")
        
        # Enhanced JSON parsing with multiple fallback strategies
        result = None
        
        # Try parsing strategies in sequence
        parsing_strategies = [
            # Strategy 1: Direct JSON parsing
            lambda text: json.loads(text.strip()),
            
            # Strategy 2: Extract from code blocks
            lambda text: json.loads(re.findall(r"```(?:json)?\n?([\s\S]*?)```", text)[0].strip()) 
                if "```" in text and re.findall(r"```(?:json)?\n?([\s\S]*?)```", text) else None,
            
            # Strategy 3: Extract JSON from multiline code blocks (improved)
            lambda text: json.loads(re.search(r"```(?:json)?\s*\n(.*?)(?:\n```|$)", text, re.DOTALL).group(1).strip())
                if re.search(r"```(?:json)?\s*\n(.*?)(?:\n```|$)", text, re.DOTALL) else None,
            
            # Strategy 4: Find complete JSON object with nested braces
            lambda text: json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0))
                if re.search(r"\{.*\}", text, re.DOTALL) else None,
                
            # Strategy 5: Clean and fix common JSON issues
            lambda text: json.loads(text.replace("'", "\"").strip()),
            
            # Strategy 6: Handle truncated JSON by attempting to complete it
            lambda text: handle_truncated_json(text)
        ]
        
        # Try each strategy until one works
        strategy_names = [
            "Direct JSON parsing",
            "Extract from code blocks (simple)",
            "Extract from multiline code blocks",
            "Find complete JSON object",
            "Clean and fix common issues",
            "Handle truncated JSON"
        ]
        
        for i, strategy in enumerate(parsing_strategies):
            try:
                potential_result = strategy(response_text)
                if potential_result:
                    result = potential_result
                    logger.info(f"Successfully parsed JSON using strategy: {strategy_names[i]}")
                    break
            except (json.JSONDecodeError, IndexError, AttributeError) as e:
                logger.debug(f"Strategy '{strategy_names[i]}' failed: {str(e)}")
                continue
        
        if not result:
            logger.error(f"All JSON parsing strategies failed. Raw response: {response_text}")
            raise ValueError("Could not parse valid JSON from response after multiple attempts")
        
        # Ensure all fields are strings, especially sample_input which might be returned as a list
        for key, value in result.items():
            if key == 'sample_input' and not isinstance(value, str):
                # Convert lists, dicts, or other types to string representation
                if isinstance(value, (list, dict)):
                    result[key] = json.dumps(value)
                else:
                    result[key] = str(value)
                    
            # Also ensure other fields are strings
            elif not isinstance(value, str):
                result[key] = str(value)
                
        return AgentOutput(**result)
    except Exception as e:
        logger.error(f"Failed to generate agent details: {str(e)}")
        logger.error(f"Raw response: {response_text if 'response_text' in locals() else 'No response received'}")
        raise HTTPException(status_code=500, detail="Failed to generate agent details")








STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
UPLOADS_DIR = os.path.join(os.path.dirname(__file__), "knowledge")
DB_FILE = os.path.join(os.path.dirname(__file__), "knowledge_base.json")

def generate_timestamped_filename(filename: str) -> str:
    """Generates a new filename with a unix timestamp."""
    name, ext = os.path.splitext(filename)
    # Sanitize the name part of the filename
    name = "".join(c if c.isalnum() or c in ('_','-') else '_' for c in name)
    timestamp = int(time.time())
    return f"{name}_{timestamp}{ext}"

os.makedirs(UPLOADS_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class KnowledgeSourceBase(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str
    content: Optional[str] = None
    filename: Optional[str] = None
    path: Optional[str] = None

class KnowledgeCard(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    sources: List[KnowledgeSourceBase] = []

async def read_db() -> List[KnowledgeCard]:
    if not os.path.exists(DB_FILE):
        return []
    async with aiofiles.open(DB_FILE, "r") as f:
        try:
            data = json.loads(await f.read())
            return [KnowledgeCard(**card) for card in data]
        except json.JSONDecodeError:
            return []

async def write_db(cards: List[KnowledgeCard]):
    async with aiofiles.open(DB_FILE, "w") as f:
        await f.write(json.dumps([card.dict() for card in cards], indent=4))

@app.get("/")
async def read_root():
    return FileResponse(os.path.join(STATIC_DIR, 'index.html'))

@app.get("/api/knowledge_cards", response_model=List[KnowledgeCard])
async def get_knowledge_cards():
    return await read_db()

@app.post("/api/knowledge_cards", response_model=KnowledgeCard)
async def create_knowledge_card(
    title: str = Form(...),
    sources: str = Form(...),
    files: List[UploadFile] = File([])
):
    cards = await read_db()
    
    new_title = title.strip()
    if not new_title:
        raise HTTPException(status_code=422, detail="Title cannot be empty.")
        
    if any(card.title.lower() == new_title.lower() for card in cards):
        raise HTTPException(
            status_code=409, # Conflict
            detail=f"A knowledge card with the title '{new_title}' already exists."
        )

    new_card = KnowledgeCard(title=new_title)
    source_list = json.loads(sources)
    file_map = {file.filename: file for file in files}

    for source_data in source_list:
        source_type = source_data['type']
        content = source_data.get('content')
        
        # Use the ID from the client if available, otherwise generate a new one
        source_id = source_data.get('id', str(uuid.uuid4()))
        
        new_source = KnowledgeSourceBase(id=source_id, type=source_type, content=content)

        if source_type == 'file' and content in file_map:
            file = file_map[content]
            new_filename = generate_timestamped_filename(file.filename)
            file_path = os.path.join(UPLOADS_DIR, new_filename)
            try:
                async with aiofiles.open(file_path, "wb") as buffer:
                    await buffer.write(await file.read())
                new_source.filename = new_filename
                new_source.path = file_path
                new_source.content = new_filename
            except Exception as e:
                # Handle file writing error
                print(f"Error saving file {file.filename}: {e}")
                continue # Skip this source if file can't be saved
        
        new_card.sources.append(new_source)

    cards.append(new_card)
    await write_db(cards)
    return new_card


@app.put("/api/knowledge_cards/{card_id}", response_model=KnowledgeCard)
async def update_knowledge_card(
    card_id: str,
    title: str = Form(...),
    sources: str = Form(...),
    files: List[UploadFile] = File([])
):
    cards = await read_db()
    card_to_update = next((card for card in cards if card.id == card_id), None)
    if not card_to_update:
        raise HTTPException(status_code=404, detail="Card not found")

    new_title = title.strip()
    if not new_title:
        raise HTTPException(status_code=422, detail="Title cannot be empty.")

    if any(card.title.lower() == new_title.lower() and card.id != card_id for card in cards):
        raise HTTPException(
            status_code=409, # Conflict
            detail=f"A knowledge card with the title '{new_title}' already exists."
        )

    card_to_update.title = new_title
    
    form_sources = json.loads(sources)
    form_source_ids = {s['id'] for s in form_sources}
    file_map = {file.filename: file for file in files}

    # Delete files for sources that were removed by the user
    original_sources = card_to_update.sources
    for original_source in original_sources:
        if original_source.id not in form_source_ids:
            if original_source.type == 'file' and original_source.path and os.path.exists(original_source.path):
                await aiofiles.os.remove(original_source.path)

    new_sources_list = []
    for source_data in form_sources:
        source_id = source_data['id']
        source_type = source_data['type']
        content = source_data.get('content')
        
        original_source = next((s for s in original_sources if s.id == source_id), None)
        
        new_source = KnowledgeSourceBase(id=source_id, type=source_type, content=content)

        if source_type == 'file':
            # A new file is uploaded for this source
            if content in file_map:
                file = file_map[content]
                new_filename = generate_timestamped_filename(file.filename)
                file_path = os.path.join(UPLOADS_DIR, new_filename)

                # If this is an update to an existing file source, delete the old file
                if original_source and original_source.path and os.path.exists(original_source.path):
                    await aiofiles.os.remove(original_source.path)

                async with aiofiles.open(file_path, "wb") as buffer:
                    await buffer.write(await file.read())
                new_source.filename = new_filename
                new_source.path = file_path
                new_source.content = new_filename # Update content to be the new name
            # No new file uploaded, keep the old one's data
            elif original_source:
                new_source.filename = original_source.filename
                new_source.path = original_source.path
                new_source.content = original_source.filename
        
        new_sources_list.append(new_source)

    card_to_update.sources = new_sources_list
    
    card_index = next((i for i, card in enumerate(cards) if card.id == card_id), -1)
    if card_index != -1:
        cards[card_index] = card_to_update

    await write_db(cards)
    return card_to_update

@app.delete("/api/knowledge_cards/{card_id}", status_code=204)
async def delete_knowledge_card(card_id: str):
    cards = await read_db()
    card_to_delete = None
    for card in cards:
        if card.id == card_id:
            card_to_delete = card
            break
    
    if not card_to_delete:
        raise HTTPException(status_code=404, detail="Card not found")

    # Optional: Delete associated files
    for source in card_to_delete.sources:
        if source.type == 'file' and source.path and os.path.exists(source.path):
            await aiofiles.os.remove(source.path)

    cards.remove(card_to_delete)
    await write_db(cards)
    return




# --- Catch-all Route for SPA (MUST BE LAST) ---
@app.get("/{full_path:path}")
async def serve_frontend(request: Request):
    full_path = request.path_params.get("full_path", "")
    logger.info(f"Catch-all route matched for path: {full_path}")

    # Define reserved prefixes that should not be handled here
    reserved_prefixes = ["/api/", "/static/", "/pages/", "/login", "/glogin", "/api/logs/*", "/chat", "/ia_chatbot", "/reset-session", "/api/agent/execute", "/api/agent/langgraph/execute", "/api/generate_captcha_token", "/ia_chatbot_embed", "/api/ia_chatbot", "/api/session/execution-feedback", "/session-login", "/ia_digital_agent", "/leslies-pool-agent", "/api/multi-agent/execute", "/psp-agent", "/label-compliance-agent", "/api/multi_agent/execute"]
    
    # Check if this is a reserved path
    if any(full_path.startswith(prefix.replace("/*", "")) for prefix in reserved_prefixes):
        logger.warning(f"Reserved path accessed via catch-all: {full_path}")
        return page_templates.TemplateResponse("404.html", {"request": request}, status_code=404)

    # Special cases for SPA routes
    spa_routes = [
        "multi-agents", 
        "agents", 
        "tools", 
        "models", 
        "create-agent", 
        "edit-agent", 
        "marketplace", 
        "data-connectors", 
        "launch-agent",
        "launch-multi-agent",
        "workflow-editor",
        "session-logs",
        "settings",
        "profile",
        "home",
        "notifications",
        "settings",
        "profile",
        "home",
        "api-keys",
        "reports",
        "statistics",
        "users",
        "feedback",
        "support",
        "about",
        "contact",
        "terms",
        "privacy",
        "help",
        "knowledge-base",
        "ui-deployments"
    ]
    
    if full_path in spa_routes or full_path.startswith("agents/") or full_path.startswith("launch-agent/"):
        return FileResponse("static/index.html")

    # Check if file exists directly in static directory
    potential_file_path = f"static/{full_path}"
    if not os.path.exists(potential_file_path) or os.path.isdir(potential_file_path):
        if full_path not in ["favicon.ico"]:
            # Handle non-root paths that don't exist as files
            if full_path not in ["", "/"]:
                # Return 404 for invalid paths
                return page_templates.TemplateResponse("404.html", {"request": request}, status_code=404)
            # For root, serve the index
            return FileResponse("static/index.html")

    # Serve existing static files
    return FileResponse("static/index.html")

@app.post("/api/knowledge_cards/download/{card_id}/{filename}")
async def download_knowledge_file(request: Request, card_id: str, filename: str, email: str = Depends(get_current_user)):
    # Read the knowledge base
    cards = await read_db()
    
    # Find the card
    card = next((c for c in cards if c.id == card_id), None)
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")
    
    # Find the source with the matching filename
    source = next((s for s in card.sources if s.filename == filename), None)
    if not source:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Get the file path
    file_path = source.path
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Return the file as a download
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )



# --- API Endpoints ---
from fastapi import Body

class DDLogsRequest(BaseModel):
    from_iso: str
    to_iso: str
    query: str
    limit: int = 200

@app.post("/api/datadog/logs")
@limiter.limit("30/minute")
async def datadog_logs_proxy(request: Request, req: DDLogsRequest):
    url = "https://api.us5.datadoghq.com/api/v2/logs/events/search"
    DD_API_KEY = os.getenv("DD_API_KEY", "49a83119f72ef42c76852939a29d5ed2")
    DD_APP_KEY = os.getenv("DD_APP_KEY", "7068800e1435e1240f4e3831d7bd37b6601c2b6f")

    headers = {
        "Content-Type": "application/json",
        "DD-API-KEY": DD_API_KEY,
        "DD-APPLICATION-KEY": DD_APP_KEY,
    }

    payload = {
        "filter": {
            "from": req.from_iso,
            "to": req.to_iso,
            "query": req.query,
        },
        "sort": "desc",
        "page": {"limit": req.limit},
    }

    results = []
    async with httpx.AsyncClient(timeout=30, verify=False) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            return JSONResponse(status_code=r.status_code, content={"detail": r.text})
        data = r.json()
        results.extend(data.get("data", []))
        # pagination
        next_link = data.get("links", {}).get("next")
        while next_link :  # safety cap
            await asyncio.sleep(0.15)
            r = await client.get(next_link, headers=headers)
            if r.status_code >= 400:
                break
            data = r.json()
            results.extend(data.get("data", []))
            next_link = data.get("links", {}).get("next")
            await asyncio.sleep(0.5)

    return {"data": results}

_geo_cache = {}

@app.get("/api/geo/location/{ip}")
@limiter.limit("60/minute")
async def geo_location(request: Request, ip: str):
    try:
        now = time.time()
        cached = _geo_cache.get(f"loc:{ip}")
        if cached and (now - cached[0]) < 600:
            return cached[1]
        url = f"https://ipinfo.io/{ip}/json"
        async with httpx.AsyncClient(timeout=10, verify=False) as client:
            r = await client.get(url)
            if r.status_code == 429:
                await asyncio.sleep(0.5)
                r = await client.get(url)
            if r.status_code >= 400:
                return JSONResponse(status_code=r.status_code, content={"detail": r.text})
            data = r.json()
            result = {
                "ip": data.get("ip"),
                "city": data.get("city"),
                "region": data.get("region"),
                "country": data.get("country"),
                "org": data.get("org"),
                "loc": data.get("loc")
            }
            _geo_cache[f"loc:{ip}"] = (now, result)
            return result
    except Exception:
        return {"ip": ip, "city": None, "region": None, "country": None, "org": None, "loc": None}

@app.get("/api/geo/location")
@limiter.limit("60/minute")
async def geo_location_query(request: Request, ip: str = Query(...)):
    return await geo_location(request, ip)

# --- Catch-all Route for SPA (MUST BE LAST) ---
@app.get("/{full_path:path}")
async def serve_frontend(request: Request):
    full_path = request.path_params.get("full_path", "")
    logger.info(f"Catch-all route matched for path: {full_path}")

    # Define reserved prefixes that should not be handled here
    reserved_prefixes = ["/api/", "/static/", "/pages/", "/login", "/glogin", "/api/logs/*", "/chat", "/ia_chatbot", "/reset-session", "/api/agent/execute", "/api/agent/langgraph/execute", "/api/generate_captcha_token", "/ia_chatbot_embed", "/api/ia_chatbot", "/api/session/execution-feedback", "/session-login", "/ia_digital_agent", "/leslies-pool-agent", "/api/multi-agent/execute", "/psp-agent", "/label-compliance-agent"]
    
    # Check if this is a reserved path
    if any(full_path.startswith(prefix.replace("/*", "")) for prefix in reserved_prefixes):
        logger.warning(f"Reserved path accessed via catch-all: {full_path}")
        return page_templates.TemplateResponse("404.html", {"request": request}, status_code=404)

    # Special cases for SPA routes
    spa_routes = [
        "multi-agents",
        "agents",
        "tools",
        "models",
        "create-agent",
        "edit-agent",
        "marketplace",
        "data-connectors",
        "launch-agent",
        "launch-multi-agent",
        "workflow-editor",
        "session-logs",
        "settings",
        "profile",
        "home",
        "dashboard",
        "notifications",
        "settings",
        "profile",
        "home",
        "api-keys",
        "reports",
        "statistics",
        "users",
        "feedback",
        "support",
        "about",
        "contact",
        "terms",
        "privacy",
        "help",
        "knowledge-base"
    ]

    if full_path in spa_routes or full_path.startswith("agents/") or full_path.startswith("launch-agent/"):
        return FileResponse("static/index.html")

    # Check if file exists directly in static directory
    potential_file_path = f"static/{full_path}"
    if not os.path.exists(potential_file_path) or os.path.isdir(potential_file_path):
        if full_path not in ["favicon.ico"]:
            # Handle non-root paths that don't exist as files
            if full_path not in ["", "/"]:
                # Return 404 for invalid paths
                return page_templates.TemplateResponse("404.html", {"request": request}, status_code=404)
            # For root, serve the index
            return FileResponse("static/index.html")

    # Serve existing static files
    return FileResponse("static/index.html")
