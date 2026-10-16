from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from DB.runtime import get_db
from services.auth_service import AuthService, oauth2_scheme

router = APIRouter()


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    role: str = "user"
    is_admin: bool = False
    district: Optional[str] = None


class UserResponse(BaseModel):
    id: str
    email: str
    name: str
    role: str
    is_active: bool
    is_admin: bool
    picture: Optional[str] = None
    avatar: Optional[str] = None
    district: Optional[str] = None
    district_name: Optional[str] = None
    created_at: str
    updated_at: str


class Token(BaseModel):
    access_token: str
    signed_url: str
    token_type: str
    user_name: str
    user_email: str
    user_picture: Optional[str] = None
    district: Optional[str] = None
    district_name: Optional[str] = None


@router.post("/sessions-login", response_model=Token)
async def login_for_access_token(
    login_data: LoginRequest, db: AsyncSession = Depends(get_db)
):
    """
    Login endpoint that returns JWT token and user info if credentials are valid
    Uses async repository pattern with proper database lifecycle management
    """
    try:
        # Create AuthService with async session
        auth_service = AuthService(db)
        
        # Authenticate user using repository pattern
        user = await auth_service.authenticate_user(login_data.email, login_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Create JWT token
        access_token_expires = timedelta(minutes=30)
        access_token = auth_service.create_access_token(
            data={"sub": user['email']}, expires_delta=access_token_expires
        )
        
        # Create signed URL token (24 hour expiry, separate secret)
        signed_url_expires = timedelta(hours=24)
        signed_url_token = auth_service.create_signed_url_token(
            data={"sub": user['email']}, expires_delta=signed_url_expires
        )
        
        return {
            "access_token": access_token,
            "signed_url": signed_url_token,
            "token_type": "bearer",
            "user_name": user['name'],
            "user_email": user['email'],
            "user_picture": user['picture'],
            "district": user.get('district'),
            "district_name": user.get('district_name'),  # district_name is same as district
        }
    except HTTPException:
        # Re-raise HTTP exceptions (like 401)
        raise
    except Exception as e:
        # Log and handle unexpected errors
        print(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )


@router.post("/register", response_model=UserResponse)
async def register_user(
    user_data: RegisterRequest, db: AsyncSession = Depends(get_db)
):
    """
    Register a new user with tenant domain validation
    
    - **username**: User's display name
    - **email**: User's email address (domain must exist as an active tenant)
    - **password**: User's password (will be hashed with bcrypt)
    - **role**: User role (default: "user")
    - **is_admin**: Admin privileges (default: false)
    
    The user's email domain will be validated against existing tenants.
    Only users with email domains that match active tenant domains can register.
    """
    try:
        # Create AuthService with async session
        auth_service = AuthService(db)
        
        # Check if user already exists
        existing_user = await auth_service.get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Extract domain for validation
        domain = user_data.email.split('@')[1].lower() if '@' in user_data.email else ""
        
        # Create new user with tenant validation
        user = await auth_service.create_user(
            username=user_data.username,
            email=user_data.email,
            password=user_data.password,
            name=user_data.username,  # Use username as name
            role=user_data.role,
            is_admin=user_data.is_admin,
            district=user_data.district,
            validate_tenant=True,
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Registration failed. The domain '{domain}' is not registered as a tenant or is inactive. Please contact administrator."
            )
        
        return UserResponse(
            id=user['id'],
            email=user['email'],
            name=user['name'],
            role=user['role'],
            is_active=user['is_active'],
            is_admin=user['is_admin'],
            picture=user['picture'],
            avatar=user['avatar'],
            district=user['district'],
            district_name=user['district_name'],
            created_at=user['created_at'].isoformat() if user['created_at'] else "",
            updated_at=user['updated_at'].isoformat() if user['updated_at'] else "",
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Log and handle unexpected errors
        print(f"Registration error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User registration service error"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    db: AsyncSession = Depends(get_db),
    token: str = Depends(oauth2_scheme)
):
    """
    Get current user information from JWT token
    Uses repository pattern with proper authentication flow
    """
    try:
        # Create AuthService with async session
        auth_service = AuthService(db)
        
        # Get current user using the token
        user = await auth_service.get_current_user(token)
        
        return UserResponse(
            id=user['id'],
            email=user['email'],
            name=user['name'],
            role=user['role'],
            is_active=user['is_active'],
            is_admin=user['is_admin'],
            picture=user['picture'],
            avatar=user['avatar'],
            district=user['district'],
            district_name=user['district_name'],
            created_at=user['created_at'].isoformat() if user['created_at'] else "",
            updated_at=user['updated_at'].isoformat() if user['updated_at'] else "",
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (like 401)
        raise
    except Exception as e:
        # Log and handle unexpected errors
        print(f"Get user info error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User info service error"
        )


@router.get("/available-domains")
async def get_available_domains(db: AsyncSession = Depends(get_db)):
    """
    Get list of available domains for user registration
    
    Returns all active tenant domains that users can register with.
    This helps users know what email domains are accepted for registration.
    """
    try:
        from services.tenant_service import TenantService
        
        tenant_service = TenantService(db)
        tenants = await tenant_service.get_all_tenants(active_only=True)
        
        domains = [
            {
                "domain": tenant['domain'],
                "tenant_name": tenant['tenant_name'],
                "description": f"Users with @{tenant['domain']} emails can register"
            }
            for tenant in tenants
        ]
        
        return {
            "message": "Available domains for registration",
            "total_domains": len(domains),
            "domains": domains
        }
        
    except Exception as e:
        print(f"Get available domains error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch available domains"
        )
