from __future__ import annotations
from typing import Optional, List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from models.user import User

class UsersRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    # ---------- READ ----------
    async def get_by_email(self, email: str) -> Optional[User]:
        stmt = select(User).where(User.email == email, User.is_deleted.is_(False))
        result = await self.db.execute(stmt)
        return result.scalars().first()

    async def get_all_users(self) -> List[User]:
        stmt = select(User).where(User.is_deleted.is_(False)).order_by(User.id.asc())
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    # ---------- CREATE ----------
    async def create_user(
        self,
        *,
        email: str,
        name: str,
        picture: Optional[str] = None,
        role: str = "user",
        is_admin: bool = False,
        client_id: Optional[str] = None,
        avatar: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> User:
        user = User(
            email=email,
            password_hash="",     # SSO users: keep empty (column exists)
            name=name,
            role=role,
            is_active=True,
            is_deleted=False,
            is_admin=is_admin,
            avatar=avatar,
            picture=picture,
            preferences=preferences,
            client_id=client_id,
        )
        try:
            self.db.add(user)
            await self.db.commit()
            # ensure fields (id, timestamps, defaults) are loaded if expire_on_commit=True
            await self.db.refresh(user)
            return user
        except IntegrityError:
            await self.db.rollback()
            # if unique email constraint hit, fetch and return it (or re-raise as you prefer)
            existing = await self.get_by_email(email)
            if existing:
                return existing
            raise

    # ---------- UPDATE (profile) ----------
    async def update_profile(
        self,
        *,
        email: str,
        name: Optional[str] = None,
        picture: Optional[str] = None,
        avatar: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
        role: Optional[str] = None,
        is_admin: Optional[bool] = None,
        client_id: Optional[str] = None,
    ) -> Optional[User]:
        user = await self.get_by_email(email)
        if not user:
            return None

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

        try:
            await self.db.commit()       # no begin()
            await self.db.refresh(user)  # pull updated row (timestamps, etc.)
            return user
        except Exception:
            await self.db.rollback()
            raise

    # ---------- STATUS CHANGES ----------
    async def deactivate_user(self, email: str) -> bool:
        user = await self.get_by_email(email)
        if not user:
            return False
        user.is_active = False
        try:
            await self.db.commit()
            await self.db.refresh(user)
            return True
        except Exception:
            await self.db.rollback()
            raise

    async def soft_delete_user(self, email: str) -> bool:
        # We want to mark even if already soft-deleted filter would hide it
        stmt = select(User).where(User.email == email)
        result = await self.db.execute(stmt)
        user = result.scalars().first()
        if not user:
            return False
        user.is_deleted = True
        try:
            await self.db.commit()
            # no need to refresh for caller typically, but harmless
            await self.db.refresh(user)
            return True
        except Exception:
            await self.db.rollback()
            raise
