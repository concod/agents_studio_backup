import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from DB.async_projects import AsyncProjectsDB
from api.tenants import get_tenant_auth_user

logger = logging.getLogger(__name__)

router = APIRouter()
projects_db = AsyncProjectsDB()


# ── Pydantic models ──────────────────────────────────────────────

class CreateProjectRequest(BaseModel):
    project_name: str
    description: Optional[str] = None
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None


class UpdateProjectRequest(BaseModel):
    project_name: Optional[str] = None
    description: Optional[str] = None


class AddAssetRequest(BaseModel):
    asset_url: Optional[str] = None
    asset_name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[dict] = None


class AssetResponse(BaseModel):
    id: str
    project_id: str
    asset_url: Optional[str] = None
    asset_name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[dict] = None
    created_at: Optional[str] = None


class ProjectResponse(BaseModel):
    id: str
    user_email: str
    project_name: str
    description: Optional[str] = None
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    status: str
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    asset_count: int = 0


class ProjectDetailResponse(ProjectResponse):
    assets: List[AssetResponse] = []


class ProjectListResponse(BaseModel):
    projects: List[ProjectResponse] = []


# ── Helpers ──────────────────────────────────────────────────────

def _get_user_email(user: Dict[str, Any]) -> str:
    email = user.get("email") if isinstance(user, dict) else None
    if not email:
        raise HTTPException(status_code=400, detail="User email not found in token")
    return email


# ── Project endpoints ────────────────────────────────────────────

@router.post("/projects", response_model=ProjectResponse)
async def create_project(
    body: CreateProjectRequest,
    user: Dict[str, Any] = Depends(get_tenant_auth_user),
):
    """Create a new project."""
    user_email = _get_user_email(user)

    if body.agent_id:
        try:
            uuid.UUID(body.agent_id)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail=f"Invalid agent_id format: {body.agent_id}")

    project, error = await projects_db.create_project(
        user_email=user_email,
        project_name=body.project_name,
        description=body.description,
        agent_id=body.agent_id,
        agent_name=body.agent_name,
    )

    if error:
        logger.error(f"Error creating project: {error}")
        raise HTTPException(status_code=500, detail=f"Error creating project: {error}")

    return ProjectResponse(**project)


@router.get("/projects", response_model=ProjectListResponse)
async def list_projects(
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    user: Dict[str, Any] = Depends(get_tenant_auth_user),
):
    """List all active projects for the authenticated user."""
    user_email = _get_user_email(user)

    if agent_id:
        try:
            uuid.UUID(agent_id)
        except (ValueError, TypeError):
            raise HTTPException(status_code=400, detail=f"Invalid agent_id format: {agent_id}")

    projects, error = await projects_db.get_projects_by_user(
        user_email=user_email,
        agent_id=agent_id,
    )

    if error:
        logger.error(f"Error listing projects: {error}")
        raise HTTPException(status_code=500, detail=f"Error listing projects: {error}")

    return ProjectListResponse(
        projects=[ProjectResponse(**p) for p in (projects or [])]
    )


@router.get("/projects/{project_id}", response_model=ProjectDetailResponse)
async def get_project(
    project_id: str,
    user: Dict[str, Any] = Depends(get_tenant_auth_user),
):
    """Get a project with all its assets."""
    user_email = _get_user_email(user)

    project, error = await projects_db.get_project_by_id(
        project_id=project_id,
        user_email=user_email,
    )

    if error:
        if error == "Project not found":
            raise HTTPException(status_code=404, detail="Project not found")
        logger.error(f"Error getting project: {error}")
        raise HTTPException(status_code=500, detail=f"Error getting project: {error}")

    assets = project.pop("assets", [])
    return ProjectDetailResponse(
        **project,
        assets=[AssetResponse(**a) for a in assets],
    )


@router.put("/projects/{project_id}", response_model=dict)
async def update_project(
    project_id: str,
    body: UpdateProjectRequest,
    user: Dict[str, Any] = Depends(get_tenant_auth_user),
):
    """Update project name and/or description."""
    user_email = _get_user_email(user)

    success, error = await projects_db.update_project(
        project_id=project_id,
        user_email=user_email,
        project_name=body.project_name,
        description=body.description,
    )

    if error:
        if "not found" in error.lower():
            raise HTTPException(status_code=404, detail=error)
        logger.error(f"Error updating project: {error}")
        raise HTTPException(status_code=500, detail=f"Error updating project: {error}")

    return {"success": True}


@router.delete("/projects/{project_id}", response_model=dict)
async def delete_project(
    project_id: str,
    user: Dict[str, Any] = Depends(get_tenant_auth_user),
):
    """Soft delete (archive) a project."""
    user_email = _get_user_email(user)

    success, error = await projects_db.delete_project(
        project_id=project_id,
        user_email=user_email,
    )

    if error:
        if "not found" in error.lower():
            raise HTTPException(status_code=404, detail=error)
        logger.error(f"Error deleting project: {error}")
        raise HTTPException(status_code=500, detail=f"Error deleting project: {error}")

    return {"success": True}


# ── Asset endpoints ──────────────────────────────────────────────

@router.post("/projects/{project_id}/assets", response_model=AssetResponse)
async def add_asset(
    project_id: str,
    body: AddAssetRequest,
    user: Dict[str, Any] = Depends(get_tenant_auth_user),
):
    """Save an asset to a project."""
    user_email = _get_user_email(user)

    asset, error = await projects_db.add_asset_to_project(
        project_id=project_id,
        user_email=user_email,
        asset_url=body.asset_url,
        asset_name=body.asset_name,
        description=body.description,
        metadata=body.metadata,
    )

    if error:
        if "not found" in error.lower():
            raise HTTPException(status_code=404, detail=error)
        logger.error(f"Error adding asset: {error}")
        raise HTTPException(status_code=500, detail=f"Error adding asset: {error}")

    return AssetResponse(**asset)


@router.delete("/projects/{project_id}/assets/{asset_id}", response_model=dict)
async def delete_asset(
    project_id: str,
    asset_id: str,
    user: Dict[str, Any] = Depends(get_tenant_auth_user),
):
    """Remove an asset from a project."""
    user_email = _get_user_email(user)

    success, error = await projects_db.delete_asset(
        asset_id=asset_id,
        project_id=project_id,
        user_email=user_email,
    )

    if error:
        if "not found" in error.lower():
            raise HTTPException(status_code=404, detail=error)
        logger.error(f"Error deleting asset: {error}")
        raise HTTPException(status_code=500, detail=f"Error deleting asset: {error}")

    return {"success": True}
