from typing import List, Optional, Dict, Any
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, validator
from sqlalchemy.ext.asyncio import AsyncSession

from DB.runtime import get_db
from services.tenant_service import TenantService
from services.auth_service import AuthService, get_auth_service

# Setup logging for debugging
logger = logging.getLogger(__name__)

router = APIRouter()


# ========== TENANT-SPECIFIC AUTHENTICATION ==========

from services.auth_service import oauth2_scheme
from config import settings
import jwt

# Tenant authentication uses log_users table only (separate from Google OAuth)
# Using same JWT configuration as AuthService for compatibility
TENANT_SECRET_KEY = settings.JWT_SECRET_KEY
TENANT_ALGORITHM = settings.JWT_ALGORITHM

async def get_tenant_auth_user(
    db: AsyncSession = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> Dict[str, Any]:
    """
    Get current authenticated user from JWT token - TENANT SPECIFIC
    Validates against log_users table only (not Google OAuth users)
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials for tenant access",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    logger.debug(f"🔐 [TENANT AUTH] Starting authentication check")
    logger.debug(f"🔐 [TENANT AUTH] Token received: {token[:20]}...{token[-20:] if len(token) > 40 else token}")
    logger.debug(f"🔐 [TENANT AUTH] Token length: {len(token)}")
    logger.debug(f"🔐 [TENANT AUTH] JWT_ALGORITHM: {TENANT_ALGORITHM}")
    
    # Log the secret key being used (masked for security but show enough to identify)
    secret_key_preview = f"{TENANT_SECRET_KEY[:10]}...{TENANT_SECRET_KEY[-10:]}" if len(TENANT_SECRET_KEY) > 20 else "[SHORT_KEY]"
    secret_key_length = len(TENANT_SECRET_KEY)
    logger.debug(f"🔐 [TENANT AUTH] JWT_SECRET_KEY in use: {secret_key_preview} (length: {secret_key_length})")
    
    try:
        # Decode JWT token
        logger.debug(f"🔐 [TENANT AUTH] Attempting to decode JWT token...")
        payload = jwt.decode(token, TENANT_SECRET_KEY, algorithms=[TENANT_ALGORITHM])
        logger.debug(f"✅ [TENANT AUTH] JWT decoded successfully")
        
        # Support both 'sub' (Session Login) and 'email' (MS SSO/OAuth) claims
        user_email: str = payload.get("email") or payload.get("sub")
        logger.debug(f"🔐 [TENANT AUTH] Extracted user_email: {user_email}")
        
        if user_email is None:
            logger.error(f"❌ [TENANT AUTH] No email or sub claim found in token")
            raise credentials_exception
            
    except jwt.ExpiredSignatureError as e:
        logger.error(f"❌ [TENANT AUTH] Token has expired - {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError as e:
        logger.error(f"❌ [TENANT AUTH] Invalid token error - {str(e)}")
        raise credentials_exception
    except Exception as e:
        logger.error(f"❌ [TENANT AUTH] Unexpected error during JWT decode - {type(e).__name__}: {str(e)}")
        raise credentials_exception

    # Validate user exists in log_users table specifically
    logger.debug(f"🔐 [TENANT AUTH] Looking up user in database: {user_email}")
    auth_service = AuthService(db)
    try:
        user = await auth_service.get_user_by_email(user_email)
        if user is None:
            logger.error(f"❌ [TENANT AUTH] User not found in log_users table: {user_email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found in tenant system"
            )
        logger.debug(f"✅ [TENANT AUTH] User found in database: {user.get('email')}")
        logger.debug(f"🔐 [TENANT AUTH] User details - ID: {user.get('id')}, is_active: {user.get('is_active')}, is_admin: {user.get('is_admin')}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [TENANT AUTH] Error during user lookup - {type(e).__name__}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error validating user"
        )
    
    # Check if user is active
    is_active = user.get('is_active', False)
    logger.debug(f"🔐 [TENANT AUTH] Checking if user is active: {is_active}")
    if not is_active:
        logger.error(f"❌ [TENANT AUTH] User account is deactivated: {user_email}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account is deactivated"
        )
    
    logger.debug(f"✅ [TENANT AUTH] Authentication successful for user: {user_email}")
    return user


async def get_tenant_admin_user(
    current_user: Dict[str, Any] = Depends(get_tenant_auth_user)
) -> Dict[str, Any]:
    """
    Get current user and ensure admin privileges - TENANT SPECIFIC
    Requires user to be admin in log_users table
    """
    if not current_user.get('is_admin', False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required for tenant operations"
        )
    return current_user


# ========== REQUEST/RESPONSE MODELS ==========

class CreateTenantRequest(BaseModel):
    domain: str
    tenant_name: str
    
    @validator('domain')
    def validate_domain(cls, v):
        if not v or not v.strip():
            raise ValueError('Domain cannot be empty')
        
        # Basic domain validation
        domain = v.lower().strip()
        if len(domain) < 3:
            raise ValueError('Domain must be at least 3 characters long')
        
        # Remove common protocol prefixes if present
        for prefix in ['https://', 'http://', 'www.']:
            if domain.startswith(prefix):
                domain = domain[len(prefix):]
        
        return domain
    
    @validator('tenant_name')
    def validate_tenant_name(cls, v):
        if not v or not v.strip():
            raise ValueError('Tenant name cannot be empty')
        
        tenant_name = v.strip()
        if len(tenant_name) < 2:
            raise ValueError('Tenant name must be at least 2 characters long')
        
        return tenant_name


class UpdateTenantRequest(BaseModel):
    domain: Optional[str] = None
    tenant_name: Optional[str] = None
    
    @validator('domain')
    def validate_domain(cls, v):
        if v is not None:
            if not v.strip():
                raise ValueError('Domain cannot be empty')
            
            # Basic domain validation
            domain = v.lower().strip()
            if len(domain) < 3:
                raise ValueError('Domain must be at least 3 characters long')
            
            # Remove common protocol prefixes if present
            for prefix in ['https://', 'http://', 'www.']:
                if domain.startswith(prefix):
                    domain = domain[len(prefix):]
            
            return domain
        return v
    
    @validator('tenant_name')
    def validate_tenant_name(cls, v):
        if v is not None:
            if not v.strip():
                raise ValueError('Tenant name cannot be empty')
            
            tenant_name = v.strip()
            if len(tenant_name) < 2:
                raise ValueError('Tenant name must be at least 2 characters long')
            
            return tenant_name
        return v


class TenantResponse(BaseModel):
    id: str
    domain: str
    tenant_name: str
    is_active: bool
    is_deleted: bool
    created_at: str
    updated_at: str


class TenantListResponse(BaseModel):
    tenants: List[TenantResponse]
    total: int
    skip: int
    limit: int


class DomainCheckResponse(BaseModel):
    domain: str
    available: bool
    message: str


# ========== API ENDPOINTS ==========

@router.post("/tenants", response_model=TenantResponse, status_code=status.HTTP_201_CREATED)
async def create_tenant(
    tenant_data: CreateTenantRequest,
    db: AsyncSession = Depends(get_db),
    current_admin: Dict[str, Any] = Depends(get_tenant_admin_user)
):
    """
    Create a new tenant (Admin Only)
    
    - **domain**: Unique domain for the tenant (e.g., "acme-corp", "company.com")
    - **tenant_name**: Display name for the tenant (e.g., "ACME Corporation")
    
    **Requires**: Admin authentication from log_users table (Bearer token from /api/sessions-login)
    """
    try:
        tenant_service = TenantService(db)
        
        # Check if domain is available
        is_available = await tenant_service.is_domain_available(tenant_data.domain)
        if not is_available:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Domain '{tenant_data.domain}' is already taken"
            )
        
        # Create tenant
        tenant = await tenant_service.create_tenant(
            domain=tenant_data.domain,
            tenant_name=tenant_data.tenant_name,
        )
        
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create tenant"
            )
        
        return TenantResponse(
            id=tenant['id'],
            domain=tenant['domain'],
            tenant_name=tenant['tenant_name'],
            is_active=tenant['is_active'],
            is_deleted=tenant['is_deleted'],
            created_at=tenant['created_at'].isoformat() if tenant['created_at'] else "",
            updated_at=tenant['updated_at'].isoformat() if tenant['updated_at'] else "",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/tenants", response_model=TenantListResponse)
async def get_tenants(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    active_only: bool = Query(True, description="Filter only active tenants"),
    search: Optional[str] = Query(None, description="Search by tenant name or domain"),
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_tenant_auth_user)
):
    """
    Get all tenants with pagination and search (Authenticated Users)
    
    - **skip**: Number of records to skip for pagination
    - **limit**: Maximum number of records to return (max 1000)
    - **active_only**: Whether to return only active tenants
    - **search**: Optional search query for name or domain
    
    **Requires**: User authentication from log_users table (Bearer token from /api/sessions-login)
    """
    try:
        tenant_service = TenantService(db)
        
        if search:
            # Use search functionality
            tenants = await tenant_service.search_tenants(
                query=search.strip(),
                skip=skip,
                limit=limit
            )
            # For search, we return the count of found results
            total = len(tenants) + skip  # Approximation
        else:
            # Get all tenants with pagination
            tenants = await tenant_service.get_all_tenants(
                skip=skip,
                limit=limit,
                active_only=active_only
            )
            # Get total count
            total = await tenant_service.count_tenants(active_only=active_only)
        
        tenant_responses = []
        for tenant in tenants:
            tenant_responses.append(TenantResponse(
                id=tenant['id'],
                domain=tenant['domain'],
                tenant_name=tenant['tenant_name'],
                is_active=tenant['is_active'],
                is_deleted=tenant['is_deleted'],
                created_at=tenant['created_at'].isoformat() if tenant['created_at'] else "",
                updated_at=tenant['updated_at'].isoformat() if tenant['updated_at'] else "",
            ))
        
        return TenantListResponse(
            tenants=tenant_responses,
            total=total,
            skip=skip,
            limit=limit
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/tenants/{tenant_id}", response_model=TenantResponse)
async def get_tenant(
    tenant_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_tenant_auth_user)
):
    """
    Get a specific tenant by ID (Authenticated Users)
    
    - **tenant_id**: UUID of the tenant
    
    **Requires**: User authentication from log_users table (Bearer token from /api/sessions-login)
    """
    try:
        tenant_service = TenantService(db)
        tenant = await tenant_service.get_tenant_by_id(tenant_id)
        
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        return TenantResponse(
            id=tenant['id'],
            domain=tenant['domain'],
            tenant_name=tenant['tenant_name'],
            is_active=tenant['is_active'],
            is_deleted=tenant['is_deleted'],
            created_at=tenant['created_at'].isoformat() if tenant['created_at'] else "",
            updated_at=tenant['updated_at'].isoformat() if tenant['updated_at'] else "",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/tenants/by-domain/{domain}", response_model=TenantResponse)
async def get_tenant_by_domain(
    domain: str,
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_tenant_auth_user)
):
    """
    Get a specific tenant by domain (Authenticated Users)
    
    - **domain**: Domain of the tenant
    
    **Requires**: User authentication from log_users table (Bearer token from /api/sessions-login)
    """
    try:
        tenant_service = TenantService(db)
        tenant = await tenant_service.get_tenant_by_domain(domain)
        
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        return TenantResponse(
            id=tenant['id'],
            domain=tenant['domain'],
            tenant_name=tenant['tenant_name'],
            is_active=tenant['is_active'],
            is_deleted=tenant['is_deleted'],
            created_at=tenant['created_at'].isoformat() if tenant['created_at'] else "",
            updated_at=tenant['updated_at'].isoformat() if tenant['updated_at'] else "",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.put("/tenants/{tenant_id}", response_model=TenantResponse)
async def update_tenant(
    tenant_id: str,
    tenant_data: UpdateTenantRequest,
    db: AsyncSession = Depends(get_db),
    current_admin: Dict[str, Any] = Depends(get_tenant_admin_user)
):
    """
    Update a tenant's information (Admin Only)
    
    - **tenant_id**: UUID of the tenant to update
    - **domain**: New domain (optional)
    - **tenant_name**: New tenant name (optional)
    
    **Requires**: Admin authentication from log_users table (Bearer token from /api/sessions-login)
    """
    try:
        tenant_service = TenantService(db)
        
        # Check if domain is available (if provided)
        if tenant_data.domain:
            is_available = await tenant_service.is_domain_available(
                tenant_data.domain, 
                exclude_tenant_id=tenant_id
            )
            if not is_available:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Domain '{tenant_data.domain}' is already taken"
                )
        
        # Update tenant
        tenant = await tenant_service.update_tenant(
            tenant_id=tenant_id,
            domain=tenant_data.domain,
            tenant_name=tenant_data.tenant_name,
        )
        
        if not tenant:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        return TenantResponse(
            id=tenant['id'],
            domain=tenant['domain'],
            tenant_name=tenant['tenant_name'],
            is_active=tenant['is_active'],
            is_deleted=tenant['is_deleted'],
            created_at=tenant['created_at'].isoformat() if tenant['created_at'] else "",
            updated_at=tenant['updated_at'].isoformat() if tenant['updated_at'] else "",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.patch("/tenants/{tenant_id}/activate")
async def activate_tenant(
    tenant_id: str,
    db: AsyncSession = Depends(get_db),
    current_admin: Dict[str, Any] = Depends(get_tenant_admin_user)
):
    """
    Activate a tenant (Admin Only)
    
    - **tenant_id**: UUID of the tenant to activate
    
    **Requires**: Admin authentication from log_users table (Bearer token from /api/sessions-login)
    """
    try:
        tenant_service = TenantService(db)
        success = await tenant_service.activate_tenant(tenant_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        return {"message": "Tenant activated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.patch("/tenants/{tenant_id}/deactivate")
async def deactivate_tenant(
    tenant_id: str,
    db: AsyncSession = Depends(get_db),
    current_admin: Dict[str, Any] = Depends(get_tenant_admin_user)
):
    """
    Deactivate a tenant (Admin Only)
    
    - **tenant_id**: UUID of the tenant to deactivate
    
    **Requires**: Admin authentication from log_users table (Bearer token from /api/sessions-login)
    """
    try:
        tenant_service = TenantService(db)
        success = await tenant_service.deactivate_tenant(tenant_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        return {"message": "Tenant deactivated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.delete("/tenants/{tenant_id}")
async def delete_tenant(
    tenant_id: str,
    db: AsyncSession = Depends(get_db),
    current_admin: Dict[str, Any] = Depends(get_tenant_admin_user)
):
    """
    Soft delete a tenant (Admin Only)
    
    - **tenant_id**: UUID of the tenant to delete
    
    **Requires**: Admin authentication from log_users table (Bearer token from /api/sessions-login)
    """
    try:
        tenant_service = TenantService(db)
        success = await tenant_service.delete_tenant(tenant_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tenant not found"
            )
        
        return {"message": "Tenant deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )


@router.get("/tenants/check-domain/{domain}", response_model=DomainCheckResponse)
async def check_domain_availability(
    domain: str,
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_tenant_auth_user)
):
    """
    Check if a domain is available for tenant registration (Authenticated Users)
    
    - **domain**: Domain to check availability
    
    **Requires**: User authentication from log_users table (Bearer token from /api/sessions-login)
    """
    try:
        tenant_service = TenantService(db)
        
        # Normalize domain
        normalized_domain = domain.lower().strip()
        
        # Remove common protocol prefixes if present
        for prefix in ['https://', 'http://', 'www.']:
            if normalized_domain.startswith(prefix):
                normalized_domain = normalized_domain[len(prefix):]
        
        is_available = await tenant_service.is_domain_available(normalized_domain)
        
        return DomainCheckResponse(
            domain=normalized_domain,
            available=is_available,
            message="Domain is available" if is_available else "Domain is already taken"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )
