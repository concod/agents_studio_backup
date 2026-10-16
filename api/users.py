from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel, field_validator
import re
from typing import Optional, List
from DB.users import UsersDB
from services.mailgun_service import MailgunService
import logging
from DB.runtime import get_db
from sqlalchemy.ext.asyncio import AsyncSession
from repositories.user_repository import UsersRepository

logger = logging.getLogger(__name__)

router = APIRouter()

class InviteUserRequest(BaseModel):
    email: str
    name: str
    role: str = 'user'

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, v):
            raise ValueError('Invalid email format')
        return v.lower()

class CreateUserRequest(BaseModel):
    email: str
    name: str
    role: str = 'user'

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, v):
            raise ValueError('Invalid email format')
        return v.lower()

class AcceptInvitationRequest(BaseModel):
    invitation_token: str

class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    picture: Optional[str] = None
    role: str
    is_active: bool
    is_admin: bool  
    invitation_status: Optional[str] = None
    invited_at: Optional[str] = None
    invited_by: Optional[str] = None
    created_at: str
    updated_at: str

def get_current_user_email(request: Request) -> str:
    """Get the current user's email from the request state"""
    user_email = getattr(request.state, "user_email", None)
    if not user_email or user_email == "guest":
        raise HTTPException(status_code=401, detail="Authentication required")
    return user_email

@router.get("/users")
async def get_all_users(db: AsyncSession = Depends(get_db)):
    """
    Get all users from the database
    """
    try:
        users_repo = UsersRepository(db)
        users = await users_repo.get_all_users()
        
        if not users:
            return []
            
        # Convert SQLAlchemy User objects to dictionaries for JSON serialization
        formatted_users = []
        for user in users:
            formatted_user = {
                "id": user.id,
                "email": user.email,
                "name": user.name,
                "picture": user.picture,
                "role": user.role,
                "is_active": bool(user.is_active),
                "is_admin": bool(user.is_admin),
                "invitation_status": None,  # Users table doesn't have invitation fields
                "invited_at": None,
                "invited_by": None,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "updated_at": user.updated_at.isoformat() if user.updated_at else None,
            }
            formatted_users.append(formatted_user)
            
        return formatted_users
        
    except Exception as e:
        logger.error(f"Unexpected error in get_all_users: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/users")
async def create_user(
    request: Request,
    user_data: CreateUserRequest,
    current_user: str = Depends(get_current_user_email),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new user directly (without invitation). ORM only.
    """
    try:
        users_repo = UsersRepository(db)

        # Check if user already exists
        existing = await users_repo.get_by_email(user_data.email)
        if existing:
            raise HTTPException(status_code=400, detail="User with this email already exists")

        # Create with defaults you used in glogin
        default_client_id = "36fcfb10-4761-42dc-bc86-91150636ddad"  # keep consistent with glogin
        is_admin = (user_data.role == "admin")

        user = await users_repo.create_user(
            email=user_data.email,
            name=user_data.name,
            picture="",           # default
            role=user_data.role,
            is_admin=is_admin,
            client_id=default_client_id,
        )

        formatted_user = {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "picture": user.picture,
            "role": user.role,
            "is_active": bool(user.is_active),
            "is_admin": bool(user.is_admin),
            "invitation_status": None,
            "invited_at": None,
            "invited_by": None,
            "created_at": user.created_at.isoformat() if getattr(user, "created_at", None) else None,
            "updated_at": user.updated_at.isoformat() if getattr(user, "updated_at", None) else None,
        }

        return {
            "message": "User created successfully",
            "user": formatted_user
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in create_user: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/users/invite")
async def invite_user(request: Request, invite_data: InviteUserRequest, current_user: str = Depends(get_current_user_email)):
    """
    Invite a new user to the platform
    """
    try:
        users_db = UsersDB()
        
        # Create the invitation
        user, error = users_db.create_invitation(
            email=invite_data.email,
            name=invite_data.name,
            invited_by=current_user,
            role=invite_data.role
        )
        
        if error:
            logger.error(f"Error creating invitation: {error}")
            if "already exists" in error.lower():
                raise HTTPException(status_code=400, detail="User already exists")
            raise HTTPException(status_code=500, detail=f"Database error: {error}")
        
        if not user:
            raise HTTPException(status_code=500, detail="Failed to create invitation")
            
        # Send invitation email
        mailgun_service = MailgunService()
        email_sent = mailgun_service.send_invitation_email(
            recipient_email=invite_data.email,
            recipient_name=invite_data.name,
            invitation_token=user['invitation_token'],
            invited_by=current_user
        )
        
        if not email_sent:
            logger.warning(f"Invitation created but email failed to send to {invite_data.email}")
            # Don't fail the request if email fails, just log it
            
        # Format response
        formatted_user = dict(user)
        for field in ['created_at', 'updated_at', 'invited_at']:
            if formatted_user.get(field):
                formatted_user[field] = formatted_user[field].isoformat()
                
        return {
            "message": "Invitation sent successfully",
            "user": formatted_user,
            "email_sent": email_sent
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in invite_user: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/users/accept-invitation")
async def accept_invitation(accept_data: AcceptInvitationRequest):
    """
    Accept an invitation and activate the user account
    """
    try:
        users_db = UsersDB()
        
        # First, get the user by invitation token to validate it exists
        invited_user, error = users_db.get_user_by_invitation_token(accept_data.invitation_token)
        if error:
            logger.error(f"Error fetching user by invitation token: {error}")
            raise HTTPException(status_code=500, detail="Database error")
            
        if not invited_user:
            raise HTTPException(status_code=400, detail="Invalid or expired invitation token")
            
        if invited_user.get('invitation_status') != 'pending':
            raise HTTPException(status_code=400, detail="Invitation has already been processed")
        
        # Accept the invitation
        user, error = users_db.accept_invitation(accept_data.invitation_token)
        
        if error:
            logger.error(f"Error accepting invitation: {error}")
            raise HTTPException(status_code=500, detail=f"Database error: {error}")
        
        if not user:
            raise HTTPException(status_code=400, detail="Invalid or expired invitation token")
            
        # Send welcome email
        mailgun_service = MailgunService()
        email_sent = mailgun_service.send_welcome_email(
            recipient_email=user['email'],
            recipient_name=user['name']
        )
        
        if not email_sent:
            logger.warning(f"Welcome email failed to send to {user['email']}")
            
        # Format response
        formatted_user = dict(user)
        for field in ['created_at', 'updated_at']:
            if formatted_user.get(field):
                formatted_user[field] = formatted_user[field].isoformat()
                
        return {
            "message": "Invitation accepted successfully",
            "user": formatted_user,
            "welcome_email_sent": email_sent
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in accept_invitation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/users/invitation/{invitation_token}")
async def get_invitation_details(invitation_token: str):
    """
    Get invitation details by token (for the invitation acceptance page)
    """
    try:
        users_db = UsersDB()
        
        user, error = users_db.get_user_by_invitation_token(invitation_token)
        if error:
            logger.error(f"Error fetching invitation details: {error}")
            raise HTTPException(status_code=500, detail="Database error")
            
        if not user:
            raise HTTPException(status_code=404, detail="Invitation not found")
            
        if user.get('invitation_status') != 'pending':
            raise HTTPException(status_code=400, detail="Invitation has already been processed or expired")
            
        return {
            "email": user['email'],
            "name": user['name'],
            "invited_by": user['invited_by'],
            "invited_at": user['invited_at'].isoformat() if user.get('invited_at') else None,
            "role": user.get('role', 'user')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in get_invitation_details: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error") 