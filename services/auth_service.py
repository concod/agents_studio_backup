import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from repositories.log_user_repository import LogUserRepository
from repositories.tenant_repository import TenantRepository
from DB.runtime import get_db
from config import settings

import jwt

# Get JWT configuration from settings
SECRET_KEY = settings.JWT_SECRET_KEY
ALGORITHM = settings.JWT_ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
SIGNED_URL_SECRET_KEY = settings.JWT_SIGNED_URL_SECRET_KEY

# Setup logger
logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


class AuthService:
    def __init__(self, db_session: AsyncSession):
        """Initialize AuthService with async database session"""
        self.db = db_session
        self.log_user_repo = LogUserRepository(db_session)
        self.tenant_repo = TenantRepository(db_session)

    async def authenticate_user(self, email: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate user by checking email and password using repository pattern
        
        Args:
            email: The user's email address
            password: The user's password
            
        Returns:
            User dict if authentication succeeds, None otherwise
        """
        try:
            user = await self.log_user_repo.authenticate_user(email, password)
            if not user:
                return None
            
            # Convert SQLAlchemy object to dict for compatibility with existing code
            return {
                'id': str(user.id),
                'email': user.email,
                'name': user.name,
                'picture': user.picture,
                'role': user.role,
                'is_active': user.is_active,
                'is_admin': user.is_admin,
                'avatar': user.avatar,
                'preferences': user.preferences,
                'client_id': user.client_id,
                'tenant_id': str(user.tenant_id) if user.tenant_id else None,
                'district': user.district,
                'district_name': user.district_name,
                'created_at': user.created_at,
                'updated_at': user.updated_at,
            }
        except Exception as e:
            logger.error(f"Error authenticating user: {str(e)}", exc_info=True)
            return None

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Get a user by email using repository pattern
        
        Args:
            email: User's email
            
        Returns:
            User dict if found, None otherwise
        """
        try:
            user = await self.log_user_repo.get_by_email(email)
            if not user:
                return None
            
            # Convert SQLAlchemy object to dict
            return {
                'id': str(user.id),
                'email': user.email,
                'name': user.name,
                'picture': user.picture,
                'role': user.role,
                'is_active': user.is_active,
                'is_admin': user.is_admin,
                'avatar': user.avatar,
                'preferences': user.preferences,
                'client_id': user.client_id,
                'tenant_id': str(user.tenant_id) if user.tenant_id else None,
                'district': user.district,
                'district_name': user.district_name,
                'created_at': user.created_at,
                'updated_at': user.updated_at,
            }
        except Exception as e:
            logger.error(f"Error getting user by email: {str(e)}", exc_info=True)
            return None

    async def validate_domain_against_tenants(self, email: str) -> Optional[str]:
        """
        Validate email domain against existing tenants
        
        Args:
            email: User's email address
            
        Returns:
            Tenant ID if domain exists and is active, None otherwise
        """
        try:
            # Extract domain from email (everything after @)
            if '@' not in email:
                return None
            
            domain = email.split('@')[1].lower()
            
            # Check if tenant with this domain exists and is active
            tenant = await self.tenant_repo.get_by_domain(domain)
            if tenant and tenant.is_active and not tenant.is_deleted:
                return str(tenant.id)
            
            return None
        except Exception as e:
            logger.error(f"Error validating domain: {str(e)}", exc_info=True)
            return None

    async def create_user(
        self,
        *,
        email: str,
        password: str,
        name: str,
        username: Optional[str] = None,  # Add username support
        role: str = "user",
        is_admin: bool = False,
        client_id: Optional[str] = None,
        picture: Optional[str] = None,
        avatar: Optional[str] = None,
        preferences: Optional[str] = None,
        district: Optional[str] = None,
        district_name: Optional[str] = None,
        validate_tenant: bool = True,  # Flag to control tenant validation
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new user using repository pattern with tenant domain validation
        
        Args:
            email: User's email address
            password: Plain text password (will be hashed)
            name: User's name
            username: Username (optional, can be same as name)
            role: User's role (default: "user")
            is_admin: Whether user is admin (default: False)
            client_id: Client ID (optional)
            picture: Profile picture URL (optional)
            avatar: Avatar URL (optional)
            preferences: User preferences JSON string (optional)
            district: User's district/location (optional)
            district_name: User's district name (optional)
            validate_tenant: Whether to validate domain against tenants (default: True)
            
        Returns:
            Created user dict or None if creation failed
        """
        try:
            tenant_id = None
            
            # Validate domain against tenants if required
            if validate_tenant:
                tenant_id = await self.validate_domain_against_tenants(email)
                if not tenant_id:
                    logger.warning(f"Domain validation failed for email: {email}")
                    return None
            
            # Use username if provided, otherwise use name
            display_name = username if username else name
            
            user = await self.log_user_repo.create_user(
                email=email,
                password=password,
                name=display_name,
                role=role,
                is_admin=is_admin,
                client_id=client_id,
                picture=picture,
                avatar=avatar,
                preferences=preferences,
                tenant_id=tenant_id,
                district=district,
                district_name=district_name,
            )
            
            # Convert SQLAlchemy object to dict
            return {
                'id': str(user.id),
                'email': user.email,
                'name': user.name,
                'picture': user.picture,
                'role': user.role,
                'is_active': user.is_active,
                'is_admin': user.is_admin,
                'avatar': user.avatar,
                'preferences': user.preferences,
                'client_id': user.client_id,
                'tenant_id': str(user.tenant_id) if user.tenant_id else None,
                'district': user.district,
                'district_name': user.district_name,
                'created_at': user.created_at,
                'updated_at': user.updated_at,
            }
        except ValueError as e:
            logger.warning(f"User creation failed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error creating user: {str(e)}", exc_info=True)
            return None

    def create_access_token(
        self, data: dict, expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token
        
        Args:
            data: Token payload data
            expires_delta: Token expiration time
            
        Returns:
            JWT token string
        """
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    def create_signed_url_token(
        self, data: dict, expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create a signed URL JWT token using a separate secret key.
        Default expiry is 24 hours.
        
        Args:
            data: Token payload data
            expires_delta: Token expiration time (default: 24 hours)
            
        Returns:
            Signed URL JWT token string
        """
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=24)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SIGNED_URL_SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt

    async def get_current_user(self, token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
        """
        Get current user from JWT token
        
        Args:
            token: JWT token from request
            
        Returns:
            User dict if token is valid
            
        Raises:
            HTTPException: If token is invalid or user not found
        """
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_email: str = payload.get("sub")
            if user_email is None:
                raise credentials_exception
        except Exception:
            raise credentials_exception

        user = await self.get_user_by_email(user_email)
        if user is None:
            raise credentials_exception
        return user


# Factory function to create AuthService with database dependency
async def get_auth_service(db: AsyncSession = Depends(get_db)) -> AuthService:
    """
    Factory function to create AuthService with proper dependency injection
    
    Args:
        db: Async database session from dependency injection
        
    Returns:
        AuthService instance
    """
    return AuthService(db)