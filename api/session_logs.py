import os
import json
import logging
import uuid
from datetime import datetime, time
from typing import Any, Dict, List, Optional, Union, Tuple
from pydantic import validator

import psycopg2
import psycopg2.extras
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from services.auth_service import AuthService, oauth2_scheme, get_auth_service
from services.session_log_service import SessionLogService
from DB.runtime import get_db as get_async_db
from api.tenants import get_tenant_auth_user

logger = logging.getLogger(__name__)

def row_to_dict(row) -> Dict[str, Any]:
    """Convert SQL row to dictionary"""
    if row is None:
        return None
    return dict(row)

router = APIRouter()


class SessionLogMessage(BaseModel):
    """Model for individual session log message"""

    id: int
    timestamp: str
    user_input: Optional[str] = None
    response: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    feedback: Optional[str] = None
    ip_address: Optional[str] = None
    execution_id: Optional[str] = None


class SessionLogSession(BaseModel):
    """Model representing a session with its messages"""

    session_id: str
    first_timestamp: str
    last_timestamp: str
    interaction_count: int
    messages: List[SessionLogMessage] = []


class SessionLogsFilter(BaseModel):
    """Model for session logs filter parameters"""

    session_id: Optional[str] = None
    from_date: Optional[datetime] = None
    to_date: Optional[datetime] = None
    agent_id: Optional[str] = None


def get_ordinal_suffix(day: int) -> str:
    """Get the ordinal suffix for a day number"""
    if 11 <= (day % 100) <= 13:
        return 'th'
    return {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')

class FilteredSessionLogResponse(BaseModel):
    """Response model for session logs"""

    session_id: str
    id: int
    timestamp: Optional[str] = None
    user_input: Optional[str] = None
    response: Optional[str] = None
    execution_id: Optional[str] = None
    feedback: Optional[str] = None
    ip_address: Optional[str] = None
    
    @validator('timestamp', pre=True)
    def format_timestamp(cls, v):
        if v is None:
            return None
            
        if isinstance(v, str):
            # If it's already a string, try to parse it
            try:
                dt = datetime.fromisoformat(v.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                return v
        elif hasattr(v, 'isoformat'):
            # If it's a datetime object
            dt = v
        else:
            return str(v)
            
        # Format the day with ordinal suffix
        day = dt.day
        suffix = get_ordinal_suffix(day)
        # Format as "10th June, 2024 18:00:00"
        return dt.strftime(f"%d{suffix} %B, %Y %H:%M:%S")


@router.get("/filtered-logs", response_model=List[FilteredSessionLogResponse])
async def get_filtered_session_logs(
    session_id: Optional[str] = None,
    from_date: Optional[datetime] = None,
    to_date: Optional[datetime] = None,
    agent_id: Optional[str] = None,
    db: psycopg2.extensions.connection = Depends(get_db),
    token: str = Depends(oauth2_scheme),
):
    """
    Get filtered session logs based on the provided parameters.
    All parameters are optional. If no filters are provided, returns all logs.
    """
    try:
        # Validate date range
        if from_date and to_date and from_date > to_date:
            raise HTTPException(
                status_code=400, detail="from_date cannot be after to_date"
            )

        # Get logs from service
        session_log_service = SessionLogService(db)
        logs = session_log_service.get_session_logs(
            session_id=session_id,
            from_date=from_date,
            to_date=to_date,
            agent_id=agent_id,
        )
        return logs

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_filtered_session_logs: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An error occurred while fetching logs"
        )


class SessionLogResponse(BaseModel):
    """Response model containing agent data and sessions"""

    agent_id: str
    agent_name: Optional[str] = None
    sessions: List[SessionLogSession] = []


@router.get("/fetch_session_logs", response_model=SessionLogResponse)
async def fetch_session_logs(
    agent_id: Optional[str] = None,
    query_all: bool = True,
    db: psycopg2.extensions.connection = Depends(get_db),
    current_user: dict = Depends(oauth2_scheme),
):
    """
    Fetch session logs for a specific agent with nested session and message structure

    Args:
        agent_id (Optional[str]): UUID string of the agent. If None, uses agent_id from environment variable
        query_all (bool): If True, returns all columns, otherwise returns grouped data

    Returns:
        SessionLogResponse: Nested JSON structure containing agent data, sessions, and messages
    """
    # If agent_id is not provided, use the one from environment variable
    if not agent_id:
        agent_id = os.getenv("IA_CHATBOT_AGENT_ID_PROD")
        if not agent_id:
            raise HTTPException(
                status_code=400,
                detail="No agent_id provided and no default agent_id found in environment variables",
            )

    try:
        service = SessionLogService(db)
        # First get all logs for the agent
        results = service.query_session_logs(agent_id, query_all=True)

        # Group results by session_id
        session_dict = {}
        for result in results:
            session_id = str(result['session_id'])
            if session_id not in session_dict:
                session_dict[session_id] = {
                    "session_id": session_id,
                    "first_timestamp": str(result.get('timestamp')),
                    "last_timestamp": str(result.get('timestamp')),
                    "interaction_count": 1,
                    "messages": [],
                }
            else:
                session_dict[session_id]["last_timestamp"] = str(result.get('timestamp'))
                session_dict[session_id]["interaction_count"] += 1

            # Safely get response text
            response = result.get('response', {})
            response_content = response.get('content', {}) if isinstance(response, dict) else {}
            response_text = response_content.get('text') if isinstance(response_content, dict) else None

            # Add the message to the session
            session_dict[session_id]["messages"].append(
                {
                    "id": result.get('id'),
                    "timestamp": str(result.get('timestamp')),
                    "user_input": result.get('user_input'),
                    "response": response_text,
                    "execution_time_seconds": result.get('execution_time_seconds'),
                    "feedback": result.get('feedback'),
                    "ip_address": result.get('ip_address'),
                    "execution_id": result.get('execution_id'),
                }
            )

        # Convert session_dict to list of SessionLogSession models
        sessions = []
        for session_data in session_dict.values():
            messages = [SessionLogMessage(**msg) for msg in session_data["messages"]]
            sessions.append(
                SessionLogSession(
                    session_id=session_data["session_id"],
                    first_timestamp=session_data["first_timestamp"],
                    last_timestamp=session_data["last_timestamp"],
                    interaction_count=session_data["interaction_count"],
                    messages=messages,
                )
            )

        # Get agent name from any result (they should all be the same)
        agent_name = None
        if results and isinstance(results[0], dict):
            agent_name = results[0].get('agent_name')

        return SessionLogResponse(
            agent_id=agent_id, agent_name=agent_name, sessions=sessions
        )

    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        import traceback
        print(f"Error in fetch_session_logs: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500, 
            detail=f"Error fetching session logs: {str(e)}"
        )


class UserSessionSummary(BaseModel):
    """Model for user session summary"""
    session_id: str
    first_request: str
    last_request: str
    request_count: int
    avg_execution_time: float
    agents_used: str
    session_title: Optional[str] = None  # Last user input message


class UserSessionsResponse(BaseModel):
    """Response model for user sessions"""
    user_email: str
    total_sessions: int
    sessions: List[UserSessionSummary] = []


@router.get("/user-sessions", response_model=UserSessionsResponse)
async def get_user_sessions(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of sessions to return"),
    offset: int = Query(0, ge=0, description="Number of sessions to skip"),
    agent_id: Optional[str] = Query(None, description="Optional agent ID to filter sessions by agent"),
    db: AsyncSession = Depends(get_async_db),
    user: Dict[str, Any] = Depends(get_tenant_auth_user),
):
    """
    Get all sessions for the currently authenticated user
    
    This endpoint returns all sessions associated with the authenticated user.
    Sessions are identified by extracting user_email from JWT tokens in request_headers.
    If agent_id is provided, sessions are filtered to only include those using that agent.
    
    **Requires**: User authentication from log_users table (Bearer token from /api/sessions-login)
    
    Query Parameters:
    - limit: Maximum number of sessions to return (default: 100, max: 1000)
    - offset: Number of sessions to skip for pagination (default: 0)
    - agent_id: Optional agent ID (UUID) to filter sessions by agent (default: None)
    
    Headers:
    - Authorization: Bearer <access_token> (from /api/sessions-login)
    
    Returns:
    - UserSessionsResponse: List of sessions with conversation details
    """
    try:
        # Get user email from authenticated user (from token)
        user_email = user.get('email') if isinstance(user, dict) else None
        
        if not user_email:
            raise HTTPException(
                status_code=400,
                detail="User email not found in token"
            )
        
        # Validate agent_id format if provided (should be UUID)
        agent_id_uuid = None
        if agent_id:
            try:
                agent_id_uuid = uuid.UUID(agent_id)
                logger.info(f"Filtering sessions by agent_id: {agent_id_uuid} for user: {user_email}")
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid agent_id format: {agent_id}, error: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid agent_id format. Expected UUID, got: {agent_id}"
                )
        else:
            logger.info(f"Fetching all sessions for user: {user_email} (no agent_id filter)")
        
        # Use AsyncSessionLogsDB for async operations
        from DB.async_session_logs import AsyncSessionLogsDB
        session_logs_db = AsyncSessionLogsDB()
        
        # Get sessions for this user using SQLAlchemy ORM
        sessions, total_count, error = await session_logs_db.get_sessions_by_user_email(
            db=db,
            user_email=user.get('email'),
            limit=limit,
            offset=offset,
            agent_id=agent_id_uuid
        )
        
        if error:
            logger.error(f"Error fetching user sessions: {error}")
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching sessions: {error}"
            )
        
        logger.info(f"Found {total_count} sessions for user {user_email}" + (f" with agent_id {agent_id_uuid}" if agent_id_uuid else ""))
        
        # Convert to response model (without conversation data)
        session_summaries = []
        for session in sessions or []:
            session_summaries.append(UserSessionSummary(
                session_id=session['session_id'],
                first_request=session['first_request'],
                last_request=session['last_request'],
                request_count=session['request_count'],
                avg_execution_time=session['avg_execution_time'],
                agents_used=session['agents_used'],
                session_title=session.get('session_title')
            ))
        
        return UserSessionsResponse(
            user_email=user_email,
            total_sessions=total_count or 0,
            sessions=session_summaries
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Error in get_user_sessions: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching user sessions: {str(e)}"
        )


class AIMessageContent(BaseModel):
    """Content of AI message"""
    text: Optional[str] = None
    html: Optional[str] = None


class AIMessage(BaseModel):
    """AI message structure"""
    type: Optional[str] = None
    content: Optional[AIMessageContent] = None
    message_blocks: Optional[List[Dict[str, Any]]] = None  # Message blocks array
    display_metadata: Optional[Dict[str, Any]] = None


class ConversationMessage(BaseModel):
    """Model for conversation message"""
    id: int
    timestamp: str
    user_input: Optional[str] = None
    ai_message: Optional[AIMessage] = None
    log_url: Optional[str] = None
    agent_name: Optional[str] = None
    execution_time: Optional[float] = None
    execution_id: Optional[str] = None
    feedback: Optional[str] = None


class ConversationResponse(BaseModel):
    """Response model for session conversation"""
    session_id: str
    total_messages: int
    messages: List[ConversationMessage] = []


@router.get("/user-sessions/session/{session_id}/conversation", response_model=ConversationResponse)
async def get_session_conversation(
    session_id: str,
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of messages to return"),
    offset: int = Query(0, ge=0, description="Number of messages to skip"),
    db: AsyncSession = Depends(get_async_db),
    user: Dict[str, Any] = Depends(get_tenant_auth_user),
):
    """
    Get paginated conversation messages for a specific session
    
    This endpoint returns conversation messages for a given session_id with pagination support.
    
    **Requires**: User authentication from log_users table (Bearer token from /api/sessions-login)
    
    Path Parameters:
    - session_id: The session ID to get conversation for
    
    Query Parameters:
    - limit: Maximum number of messages to return (default: 100, max: 1000)
    - offset: Number of messages to skip for pagination (default: 0)
    
    Headers:
    - Authorization: Bearer <access_token> (from /api/sessions-login)
    
    Returns:
    - ConversationResponse: Paginated conversation messages with total count
    """
    try:
        # Use AsyncSessionLogsDB for async operations
        from DB.async_session_logs import AsyncSessionLogsDB
        session_logs_db = AsyncSessionLogsDB()
        
        # Get conversation messages for this session using SQLAlchemy ORM
        messages, total_count, error = await session_logs_db.get_session_conversation_paginated(
            db=db,
            session_id=session_id,
            limit=limit,
            offset=offset
        )
        
        if error:
            logger.error(f"Error fetching session conversation: {error}")
            raise HTTPException(
                status_code=500,
                detail=f"Error fetching conversation: {error}"
            )
        
        if messages is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
        
        # Convert to response model with parsed AI message
        conversation_messages = []
        for msg in messages:
            # Parse the response JSON string if it exists
            ai_message = None
            log_url = None
            message_blocks = None
            
            # First, check if message_blocks was already extracted (from get_session_conversation_paginated)
            if msg.get('message_blocks'):
                message_blocks_raw = msg['message_blocks']
                # Handle case where message_blocks might be a JSON string
                if isinstance(message_blocks_raw, str):
                    try:
                        message_blocks = json.loads(message_blocks_raw)
                    except (json.JSONDecodeError, TypeError):
                        message_blocks = None
                else:
                    message_blocks = message_blocks_raw
            
            if msg.get('response'):
                try:
                    # If response is already a dict, use it directly; otherwise parse JSON string
                    if isinstance(msg['response'], dict):
                        response_data = msg['response']
                    else:
                        response_data = json.loads(msg['response'])
                    
                    # Extract AI message fields
                    ai_message_type = response_data.get('type')
                    ai_content = response_data.get('content', {})
                    ai_html = response_data.get('html_content') or response_data.get('html')
                    
                    # Extract log_url if present
                    log_url = response_data.get('log_url')
                    
                    # Extract message_blocks from response if not already extracted
                    if not message_blocks:
                        # Try 'blocks' first (from final_answer), then 'message_blocks' (from MessageResponse)
                        message_blocks_raw = response_data.get('blocks') or response_data.get('message_blocks')
                        # Handle case where message_blocks might be a JSON string
                        if isinstance(message_blocks_raw, str):
                            try:
                                message_blocks = json.loads(message_blocks_raw)
                            except (json.JSONDecodeError, TypeError):
                                message_blocks = None
                        else:
                            message_blocks = message_blocks_raw
                    
                    # Build AI message structure with html inside content
                    if ai_message_type or ai_content or ai_html:
                        # Extract text from content if it's a dict
                        ai_text = None
                        if isinstance(ai_content, dict):
                            ai_text = ai_content.get('text')
                        # Also check top-level text if not in content
                        if not ai_text:
                            ai_text = response_data.get('text')
                        
                        # Build content object with both text and html
                        content_obj = None
                        if ai_text or ai_html:
                            content_obj = AIMessageContent(
                                text=ai_text,
                                html=ai_html
                            )
                        
                        ai_message = AIMessage(
                            type=ai_message_type,
                            content=content_obj,
                            message_blocks=message_blocks,  # Include message_blocks
                            display_metadata=response_data.get('display_metadata')
                        )
                except (json.JSONDecodeError, TypeError, AttributeError) as e:
                    logger.warning(f"Failed to parse response JSON for message {msg.get('id')}: {e}")
                    # If parsing fails, create a basic AI message with the raw response as text
                    ai_message = AIMessage(
                        type="text",
                        content=AIMessageContent(text=msg['response'], html=None),
                        message_blocks=message_blocks  # Still include message_blocks if available
                    )
            
            conversation_messages.append(ConversationMessage(
                id=msg['id'],
                timestamp=msg['timestamp'],
                user_input=msg['user_input'],
                ai_message=ai_message,
                log_url=log_url,
                agent_name=msg['agent_name'],
                execution_time=msg['execution_time'],
                execution_id=msg['execution_id'],
                feedback=msg['feedback']
            ))
        
        return ConversationResponse(
            session_id=session_id,
            total_messages=total_count or 0,
            messages=conversation_messages
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Error in get_session_conversation: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching session conversation: {str(e)}"
        )
