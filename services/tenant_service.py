import logging
from typing import Any, Dict, List, Optional

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from repositories.tenant_repository import TenantRepository
from DB.runtime import get_db

logger = logging.getLogger(__name__)


class TenantService:
    def __init__(self, db_session: AsyncSession):
        """Initialize TenantService with async database session"""
        self.db = db_session
        self.tenant_repo = TenantRepository(db_session)

    async def get_tenant_by_id(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """
        Get tenant by ID using repository pattern
        
        Args:
            tenant_id: Tenant's UUID
            
        Returns:
            Tenant dict if found, None otherwise
        """
        logger.debug(f"Getting tenant by ID: {tenant_id}")
        try:
            tenant = await self.tenant_repo.get_by_id(tenant_id)
            if not tenant:
                logger.info(f"Tenant not found with ID: {tenant_id}")
                return None
            
            logger.info(f"Successfully retrieved tenant: {tenant_id}")
            return self._tenant_to_dict(tenant)
        except Exception as e:
            logger.error(f"Error getting tenant by ID {tenant_id}: {str(e)}")
            return None

    async def get_tenant_by_domain(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        Get tenant by domain using repository pattern
        
        Args:
            domain: Tenant's domain
            
        Returns:
            Tenant dict if found, None otherwise
        """
        logger.debug(f"Getting tenant by domain: {domain}")
        try:
            tenant = await self.tenant_repo.get_by_domain(domain)
            logger.debug(f"Tenant retrieved: {tenant}")
            if not tenant:
                logger.info(f"Tenant not found with domain: {domain}")
                return None
            
            logger.info(f"Successfully retrieved tenant for domain: {domain}")
            return self._tenant_to_dict(tenant)
        except Exception as e:
            logger.error(f"Error getting tenant by domain {domain}: {str(e)}")
            return None

    async def get_all_tenants(
        self, 
        skip: int = 0, 
        limit: int = 100, 
        active_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get all tenants with pagination
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            active_only: Whether to return only active tenants
            
        Returns:
            List of tenant dicts
        """
        logger.debug(f"Getting all tenants with skip={skip}, limit={limit}, active_only={active_only}")
        try:
            tenants = await self.tenant_repo.get_all_tenants(
                skip=skip, 
                limit=limit, 
                active_only=active_only
            )
            logger.info(f"Successfully retrieved {len(tenants)} tenants")
            return [self._tenant_to_dict(tenant) for tenant in tenants]
        except Exception as e:
            logger.error(f"Error getting all tenants: {str(e)}")
            return []

    async def search_tenants(
        self, 
        query: str, 
        skip: int = 0, 
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search tenants by name or domain
        
        Args:
            query: Search query string
            skip: Number of records to skip
            limit: Maximum number of records to return
            
        Returns:
            List of matching tenant dicts
        """
        logger.debug(f"Searching tenants with query='{query}', skip={skip}, limit={limit}")
        try:
            tenants = await self.tenant_repo.search_tenants(
                query=query, 
                skip=skip, 
                limit=limit
            )
            logger.info(f"Found {len(tenants)} tenants matching query '{query}'")
            return [self._tenant_to_dict(tenant) for tenant in tenants]
        except Exception as e:
            logger.error(f"Error searching tenants with query '{query}': {str(e)}")
            return []

    async def count_tenants(self, active_only: bool = True) -> int:
        """
        Count total tenants
        
        Args:
            active_only: Whether to count only active tenants
            
        Returns:
            Total count of tenants
        """
        logger.debug(f"Counting tenants with active_only={active_only}")
        try:
            count = await self.tenant_repo.count_tenants(active_only=active_only)
            logger.info(f"Total tenant count: {count} (active_only={active_only})")
            return count
        except Exception as e:
            logger.error(f"Error counting tenants: {str(e)}")
            return 0

    async def create_tenant(
        self,
        *,
        domain: str,
        tenant_name: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Create a new tenant using repository pattern
        
        Args:
            domain: Tenant's domain (must be unique)
            tenant_name: Tenant's display name
            
        Returns:
            Created tenant dict or None if creation failed
        """
        logger.info(f"Creating new tenant with domain='{domain}', name='{tenant_name}'")
        try:
            tenant = await self.tenant_repo.create_tenant(
                domain=domain,
                tenant_name=tenant_name,
            )
            
            logger.info(f"Successfully created tenant '{tenant_name}' with ID: {tenant.id}")
            return self._tenant_to_dict(tenant)
        except ValueError as e:
            logger.warning(f"Tenant creation failed for domain '{domain}': {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error creating tenant with domain '{domain}': {str(e)}")
            return None

    async def update_tenant(
        self,
        *,
        tenant_id: str,
        domain: Optional[str] = None,
        tenant_name: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Update tenant information
        
        Args:
            tenant_id: Tenant's UUID
            domain: New domain (optional)
            tenant_name: New tenant name (optional)
            
        Returns:
            Updated tenant dict or None if update failed
        """
        updates = []
        if domain:
            updates.append(f"domain='{domain}'")
        if tenant_name:
            updates.append(f"tenant_name='{tenant_name}'")
        logger.info(f"Updating tenant {tenant_id} with: {', '.join(updates) if updates else 'no changes'}")
        
        try:
            tenant = await self.tenant_repo.update_tenant(
                tenant_id=tenant_id,
                domain=domain,
                tenant_name=tenant_name,
            )
            
            if not tenant:
                logger.warning(f"Tenant not found for update: {tenant_id}")
                return None
                
            logger.info(f"Successfully updated tenant: {tenant_id}")
            return self._tenant_to_dict(tenant)
        except ValueError as e:
            logger.warning(f"Tenant update failed for {tenant_id}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error updating tenant {tenant_id}: {str(e)}")
            return None

    async def activate_tenant(self, tenant_id: str) -> bool:
        """
        Activate tenant account
        
        Args:
            tenant_id: Tenant's UUID
            
        Returns:
            True if activation succeeded, False otherwise
        """
        logger.info(f"Activating tenant: {tenant_id}")
        try:
            result = await self.tenant_repo.activate_tenant(tenant_id)
            if result:
                logger.info(f"Successfully activated tenant: {tenant_id}")
            else:
                logger.warning(f"Failed to activate tenant (not found?): {tenant_id}")
            return result
        except Exception as e:
            logger.error(f"Error activating tenant {tenant_id}: {str(e)}")
            return False

    async def deactivate_tenant(self, tenant_id: str) -> bool:
        """
        Deactivate tenant account
        
        Args:
            tenant_id: Tenant's UUID
            
        Returns:
            True if deactivation succeeded, False otherwise
        """
        logger.info(f"Deactivating tenant: {tenant_id}")
        try:
            result = await self.tenant_repo.deactivate_tenant(tenant_id)
            if result:
                logger.info(f"Successfully deactivated tenant: {tenant_id}")
            else:
                logger.warning(f"Failed to deactivate tenant (not found?): {tenant_id}")
            return result
        except Exception as e:
            logger.error(f"Error deactivating tenant {tenant_id}: {str(e)}")
            return False

    async def delete_tenant(self, tenant_id: str) -> bool:
        """
        Soft delete tenant (mark as deleted)
        
        Args:
            tenant_id: Tenant's UUID
            
        Returns:
            True if deletion succeeded, False otherwise
        """
        logger.info(f"Soft deleting tenant: {tenant_id}")
        try:
            result = await self.tenant_repo.soft_delete_tenant(tenant_id)
            if result:
                logger.info(f"Successfully soft deleted tenant: {tenant_id}")
            else:
                logger.warning(f"Failed to soft delete tenant (not found?): {tenant_id}")
            return result
        except Exception as e:
            logger.error(f"Error deleting tenant {tenant_id}: {str(e)}")
            return False

    async def is_domain_available(
        self, 
        domain: str, 
        exclude_tenant_id: Optional[str] = None
    ) -> bool:
        """
        Check if a domain is available for use
        
        Args:
            domain: Domain to check
            exclude_tenant_id: Tenant ID to exclude from check (for updates)
            
        Returns:
            True if domain is available, False otherwise
        """
        logger.debug(f"Checking domain availability: '{domain}' (excluding tenant: {exclude_tenant_id})")
        try:
            is_available = await self.tenant_repo.is_domain_available(
                domain=domain, 
                exclude_tenant_id=exclude_tenant_id
            )
            logger.debug(f"Domain '{domain}' availability: {is_available}")
            return is_available
        except Exception as e:
            logger.error(f"Error checking domain availability for '{domain}': {str(e)}")
            return False

    async def validate_tenant_exists(self, tenant_id: str) -> bool:
        """
        Check if tenant exists and is active
        
        Args:
            tenant_id: Tenant's UUID
            
        Returns:
            True if tenant exists and is active, False otherwise
        """
        logger.debug(f"Validating tenant exists: {tenant_id}")
        try:
            is_valid = await self.tenant_repo.validate_tenant_exists(tenant_id)
            logger.debug(f"Tenant {tenant_id} validation result: {is_valid}")
            return is_valid
        except Exception as e:
            logger.error(f"Error validating tenant {tenant_id}: {str(e)}")
            return False

    def _tenant_to_dict(self, tenant) -> Dict[str, Any]:
        """Convert SQLAlchemy Tenant object to dictionary"""
        return {
            'id': str(tenant.id),
            'domain': tenant.domain,
            'tenant_name': tenant.tenant_name,
            'is_active': tenant.is_active,
            'is_deleted': tenant.is_deleted,
            'created_at': tenant.created_at,
            'updated_at': tenant.updated_at,
        }


# Factory function to create TenantService with database dependency
async def get_tenant_service(db: AsyncSession = Depends(get_db)) -> TenantService:
    """
    Factory function to create TenantService with proper dependency injection
    
    Args:
        db: Async database session from dependency injection
        
    Returns:
        TenantService instance
    """
    logger.debug("Creating new TenantService instance")
    return TenantService(db)
