import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from DB.async_session_logs import AsyncSessionLogsDB
from DB.runtime import get_db as get_async_db
from api.tenants import get_tenant_auth_user

logger = logging.getLogger(__name__)

router = APIRouter()


class SavedAsset(BaseModel):
    image_url: str
    user_input: Optional[str] = None
    created_at: Optional[str] = None
    confidence: Optional[str] = None
    dimensions: Optional[str] = None
    method: Optional[str] = None
    agent_name: Optional[str] = None
    session_id: Optional[str] = None


class SavedAssetsResponse(BaseModel):
    assets: List[SavedAsset] = []


@router.get("/saved-assets", response_model=SavedAssetsResponse)
async def get_saved_assets(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID (UUID)"),
    db: AsyncSession = Depends(get_async_db),
    user: Dict[str, Any] = Depends(get_tenant_auth_user),
):
    """
    Get generated image assets for the authenticated user + agent. Max 50.
    """
    try:
        user_email = user.get('email') if isinstance(user, dict) else None
        if not user_email:
            raise HTTPException(status_code=400, detail="User email not found in token")

        if agent_id:
            try:
                uuid.UUID(agent_id)
            except (ValueError, TypeError):
                raise HTTPException(status_code=400, detail=f"Invalid agent_id format: {agent_id}")

        session_logs_db = AsyncSessionLogsDB()
        assets, error = await session_logs_db.get_saved_assets(
            db=db,
            user_email=user_email,
            agent_id=agent_id,
            limit=50,
        )

        if error:
            logger.error(f"Error fetching saved assets: {error}")
            raise HTTPException(status_code=500, detail=f"Error fetching assets: {error}")

        return SavedAssetsResponse(assets=[SavedAsset(**a) for a in (assets or [])])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_saved_assets: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching saved assets: {str(e)}")
