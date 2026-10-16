from __future__ import annotations
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from models.tenant import Tenant


class TenantRepository:
    def __init__(self, db: AsyncSession):
        """Initialize repository with async database session"""
        self.db = db

    # ========== READ OPERATIONS ==========
    
    async def get_by_id(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID (excludes soft-deleted tenants)"""
        stmt = select(Tenant).where(
            Tenant.id == tenant_id,
            Tenant.is_deleted.is_(False)
        )
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def get_by_domain(self, domain: str) -> Optional[Tenant]:
        """Get tenant by domain (excludes soft-deleted tenants)"""
        stmt = select(Tenant).where(
            Tenant.domain == domain,
            Tenant.is_deleted.is_(False)
        )
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def get_by_name(self, tenant_name: str) -> Optional[Tenant]:
        """Get tenant by name (excludes soft-deleted tenants)"""
        stmt = select(Tenant).where(
            Tenant.tenant_name == tenant_name,
            Tenant.is_deleted.is_(False)
        )
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def get_all_tenants(
        self, 
        skip: int = 0, 
        limit: int = 100,
        active_only: bool = True
    ) -> List[Tenant]:
        """Get all tenants with pagination (excludes soft-deleted by default)"""
        stmt = select(Tenant).where(Tenant.is_deleted.is_(False))
        
        if active_only:
            stmt = stmt.where(Tenant.is_active.is_(True))
            
        stmt = stmt.order_by(Tenant.created_at.desc()).offset(skip).limit(limit)
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def count_tenants(self, active_only: bool = True) -> int:
        """Count total tenants"""
        stmt = select(func.count(Tenant.id)).where(Tenant.is_deleted.is_(False))
        
        if active_only:
            stmt = stmt.where(Tenant.is_active.is_(True))
            
        result = await self.db.execute(stmt)
        return result.scalar() or 0

    async def search_tenants(
        self, 
        query: str, 
        skip: int = 0, 
        limit: int = 50
    ) -> List[Tenant]:
        """Search tenants by name or domain"""
        search_pattern = f"%{query}%"
        stmt = select(Tenant).where(
            Tenant.is_deleted.is_(False),
            Tenant.is_active.is_(True),
            (
                Tenant.tenant_name.ilike(search_pattern) |
                Tenant.domain.ilike(search_pattern)
            )
        ).order_by(Tenant.tenant_name.asc()).offset(skip).limit(limit)
        
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # ========== CREATE OPERATIONS ==========
    
    async def create_tenant(
        self,
        *,
        domain: str,
        tenant_name: str,
    ) -> Tenant:
        """
        Create a new tenant
        
        Args:
            domain: Tenant's domain (must be unique)
            tenant_name: Tenant's display name
            
        Returns:
            Created Tenant object
            
        Raises:
            ValueError: If domain already exists
            IntegrityError: If database constraints are violated
        """
        # Normalize domain to lowercase
        domain = domain.lower().strip()
        tenant_name = tenant_name.strip()
        
        tenant = Tenant(
            domain=domain,
            tenant_name=tenant_name,
            is_active=True,
            is_deleted=False,
        )
        
        try:
            self.db.add(tenant)
            await self.db.commit()
            await self.db.refresh(tenant)  # Load generated fields (id, timestamps)
            return tenant
        except IntegrityError:
            await self.db.rollback()
            # Handle unique constraint violations
            existing = await self.get_by_domain(domain)
            if existing:
                raise ValueError(f"Tenant with domain '{domain}' already exists")
            raise

    # ========== UPDATE OPERATIONS ==========
    
    async def update_tenant(
        self,
        *,
        tenant_id: str,
        domain: Optional[str] = None,
        tenant_name: Optional[str] = None,
    ) -> Optional[Tenant]:
        """Update tenant information"""
        tenant = await self.get_by_id(tenant_id)
        if not tenant:
            return None

        # Update only provided fields
        if domain is not None:
            # Normalize domain
            domain = domain.lower().strip()
            # Check if new domain is already taken by another tenant
            existing = await self.get_by_domain(domain)
            if existing and existing.id != tenant.id:
                raise ValueError(f"Domain '{domain}' is already taken by another tenant")
            tenant.domain = domain
            
        if tenant_name is not None:
            tenant.tenant_name = tenant_name.strip()

        try:
            await self.db.commit()
            await self.db.refresh(tenant)  # Refresh to get updated timestamps
            return tenant
        except SQLAlchemyError:
            await self.db.rollback()
            raise

    # ========== STATUS OPERATIONS ==========
    
    async def activate_tenant(self, tenant_id: str) -> bool:
        """Activate tenant account"""
        tenant = await self.get_by_id(tenant_id)
        if not tenant:
            return False
            
        tenant.is_active = True
        try:
            await self.db.commit()
            await self.db.refresh(tenant)
            return True
        except SQLAlchemyError:
            await self.db.rollback()
            raise

    async def deactivate_tenant(self, tenant_id: str) -> bool:
        """Deactivate tenant account"""
        tenant = await self.get_by_id(tenant_id)
        if not tenant:
            return False
            
        tenant.is_active = False
        try:
            await self.db.commit()
            await self.db.refresh(tenant)
            return True
        except SQLAlchemyError:
            await self.db.rollback()
            raise

    async def soft_delete_tenant(self, tenant_id: str) -> bool:
        """Soft delete tenant (mark as deleted)"""
        # Use raw query to find even soft-deleted tenants
        stmt = select(Tenant).where(Tenant.id == tenant_id)
        result = await self.db.execute(stmt)
        tenant = result.scalars().first()
        
        if not tenant:
            return False
            
        tenant.is_deleted = True
        tenant.is_active = False  # Also deactivate when deleting
        try:
            await self.db.commit()
            await self.db.refresh(tenant)
            return True
        except SQLAlchemyError:
            await self.db.rollback()
            raise

    # ========== VALIDATION OPERATIONS ==========
    
    async def is_domain_available(self, domain: str, exclude_tenant_id: Optional[str] = None) -> bool:
        """Check if a domain is available for use"""
        domain = domain.lower().strip()
        stmt = select(Tenant).where(
            Tenant.domain == domain,
            Tenant.is_deleted.is_(False)
        )
        
        if exclude_tenant_id:
            stmt = stmt.where(Tenant.id != exclude_tenant_id)
            
        result = await self.db.execute(stmt)
        existing = result.scalars().first()
        return existing is None

    async def validate_tenant_exists(self, tenant_id: str) -> bool:
        """Check if tenant exists and is active"""
        tenant = await self.get_by_id(tenant_id)
        return tenant is not None and tenant.is_active
