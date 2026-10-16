from __future__ import annotations
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from models.log_user import LogUser
import bcrypt

class LogUserRepository:
    def __init__(self, db: AsyncSession):
        """Initialize repository with async database session"""
        self.db = db

    # ========== READ OPERATIONS ==========
    
    async def get_by_email(self, email: str) -> Optional[LogUser]:
        """Get log_user by email (excludes soft-deleted users) - case insensitive"""
        stmt = select(LogUser).where(
            func.lower(LogUser.email) == email.lower(),
            LogUser.is_deleted.is_(False),
            LogUser.is_active.is_(True)
        )
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def get_by_id(self, user_id: int) -> Optional[LogUser]:
        """Get log_user by auto-increment ID (excludes soft-deleted users)"""
        stmt = select(LogUser).where(
            LogUser.id == user_id,
            LogUser.is_deleted.is_(False)
        )
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def get_by_user_id_email(self, email: str) -> Optional[LogUser]:
        """Get log_user by user_id field (which stores email) - case insensitive"""
        stmt = select(LogUser).where(
            func.lower(LogUser.user_id) == email.lower(),
            LogUser.is_deleted.is_(False)
        )
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def get_users_by_tenant_id(self, tenant_id: str) -> list[LogUser]:
        """Get all users associated with a specific tenant"""
        stmt = select(LogUser).where(
            LogUser.tenant_id == tenant_id,
            LogUser.is_deleted.is_(False),
            LogUser.is_active.is_(True)
        ).order_by(LogUser.created_at.desc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # ========== AUTHENTICATION OPERATIONS ==========
    
    async def authenticate_user(self, email: str, password: str) -> Optional[LogUser]:
        """
        Authenticate user by checking email and password
        
        Args:
            email: The user's email address
            password: The user's password (plain text)
            
        Returns:
            LogUser object if authentication succeeds, None otherwise
        """
        user = await self.get_by_email(email)
        if not user:
            return None
        
        # Verify password using bcrypt
        if not self.verify_password(password, user.password_hash):
            return None
            
        return user

    # ========== PASSWORD OPERATIONS ==========
    
    def verify_password(self, plain_password: str, hashed_password) -> bool:
        """
        Verify a plain password against a hashed password using bcrypt
        
        Args:
            plain_password: The plain text password to verify
            hashed_password: The stored bcrypt hash (str or bytes)
        Returns:
            True if password matches, False otherwise
        """
        try:
            # Reject empty or None stored hashes immediately
            if not hashed_password:
                return False
            # Support both bytes and str stored hashes
            if isinstance(hashed_password, bytes):
                hashed_bytes = hashed_password
            else:
                hashed_bytes = hashed_password.encode('utf-8')
            # Handle legacy plain-text stored passwords (str only)
            if isinstance(hashed_password, str) and not hashed_password.startswith(('$2b$', '$2a$', '$2y$')):
                # Legacy plain-text password - compare directly but log warning
                # TODO: Remove this after all passwords are migrated to bcrypt
                return plain_password == hashed_password
            # Proper bcrypt verification
            return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_bytes)
        except Exception:
            return False

    def hash_password(self, password: str) -> str:
        """
        Hash a password using bcrypt
        
        Args:
            password: Plain text password
            
        Returns:
            Bcrypt hashed password
        """
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    # ========== CREATE OPERATIONS ==========
    
    async def create_user(
        self,
        *,
        email: str,
        password: str,
        name: str,
        role: str = "user",
        is_admin: bool = False,
        client_id: Optional[str] = None,
        picture: Optional[str] = None,
        avatar: Optional[str] = None,
        preferences: Optional[str] = None,
        tenant_id: Optional[str] = None,
        district: Optional[str] = None,
        district_name: Optional[str] = None,
    ) -> LogUser:
        """
        Create a new log_user with hashed password
        
        Args:
            email: User's email address
            password: Plain text password (will be hashed)
            name: User's name
            role: User's role (default: "user")
            is_admin: Whether user is admin (default: False)
            client_id: Client ID (optional)
            picture: Profile picture URL (optional)
            avatar: Avatar URL (optional)
            preferences: User preferences JSON string (optional)
            tenant_id: Associated tenant UUID (optional)
            district: User's district/location (optional)
            district_name: User's district name (optional)
            
        Returns:
            Created LogUser object
            
        Raises:
            IntegrityError: If email already exists
        """
        # Hash the password before storing
        hashed_password = self.hash_password(password)
        
        # Normalize email to lowercase for consistent storage
        email_lower = email.lower()
        
        user = LogUser(
            user_id=email_lower,  # Set user_id to email (primary key)
            email=email_lower,
            password_hash=hashed_password,
            name=name,
            role=role,
            is_active=True,
            is_deleted=False,
            is_admin=is_admin,
            client_id=client_id,
            picture=picture,
            avatar=avatar,
            preferences=preferences,
            tenant_id=tenant_id,
            district=district,
            district_name=district_name,
        )
        
        try:
            self.db.add(user)
            await self.db.commit()
            await self.db.refresh(user)  # Load generated fields (timestamps)
            return user
        except IntegrityError:
            await self.db.rollback()
            # Handle unique constraint violations by checking email regardless of soft-delete or active flags
            stmt = select(LogUser).where(func.lower(LogUser.email) == email.lower())
            result = await self.db.execute(stmt)
            existing = result.scalars().first()
            if existing:
                raise ValueError(f"User with email {email} already exists")
            raise

    # ========== UPDATE OPERATIONS ==========
    
    async def update_password(self, email: str, new_password: str) -> Optional[LogUser]:
        """
        Update user's password
        
        Args:
            email: User's email
            new_password: New plain text password (will be hashed)
            
        Returns:
            Updated LogUser object or None if user not found
        """
        user = await self.get_by_email(email)
        if not user:
            return None

        # Hash the new password
        user.password_hash = self.hash_password(new_password)

        try:
            await self.db.commit()
            await self.db.refresh(user)
            return user
        except SQLAlchemyError:
            await self.db.rollback()
            raise

    async def update_profile(
        self,
        *,
        email: str,
        name: Optional[str] = None,
        picture: Optional[str] = None,
        avatar: Optional[str] = None,
        preferences: Optional[str] = None,
        role: Optional[str] = None,
        is_admin: Optional[bool] = None,
        client_id: Optional[str] = None,
        district: Optional[str] = None,
        district_name: Optional[str] = None,
    ) -> Optional[LogUser]:
        """Update user profile fields"""
        user = await self.get_by_email(email)
        if not user:
            return None

        # Update only provided fields
        if name is not None:
            user.name = name
        if picture is not None:
            user.picture = picture
        if avatar is not None:
            user.avatar = avatar
        if preferences is not None:
            user.preferences = preferences
        if role is not None:
            user.role = role
        if is_admin is not None:
            user.is_admin = is_admin
        if client_id is not None:
            user.client_id = client_id
        if district is not None:
            user.district = district
        if district_name is not None:
            user.district_name = district_name

        try:
            await self.db.commit()
            await self.db.refresh(user)  # Refresh to get updated timestamps
            return user
        except SQLAlchemyError:
            await self.db.rollback()
            raise

    # ========== STATUS OPERATIONS ==========
    
    async def deactivate_user(self, email: str) -> bool:
        """Deactivate user account"""
        user = await self.get_by_email(email)
        if not user:
            return False
            
        user.is_active = False
        try:
            await self.db.commit()
            await self.db.refresh(user)
            return True
        except SQLAlchemyError:
            await self.db.rollback()
            raise

    async def soft_delete_user(self, email: str) -> bool:
        """Soft delete user (mark as deleted)"""
        # Use raw query to find even soft-deleted users
        stmt = select(LogUser).where(func.lower(LogUser.email) == email.lower())
        result = await self.db.execute(stmt)
        user = result.scalars().first()
        
        if not user:
            return False
            
        user.is_deleted = True
        try:
            await self.db.commit()
            await self.db.refresh(user)
            return True
        except SQLAlchemyError:
            await self.db.rollback()
            raise
