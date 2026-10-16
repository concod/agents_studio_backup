"""
API routes for Store Hub application.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, Request, Query
from sqlalchemy.ext.asyncio import AsyncSession

from .dependencies import get_store_service
from ..application.store_service import StoreService
from ..infrastructure.utils.date_utils import get_date_range_defaults

logger = logging.getLogger(__name__)

# Create router with prefix
router = APIRouter(
    prefix="/api/agents/custom/store-hub",
    tags=["Store Hub"]
)


def create_routes(limiter, get_db, get_tenant_auth_user):
    """
    Create routes with dependencies from main.py.
    This allows us to inject the limiter and auth dependencies.
    """
    
    @router.get("/graph")
    @limiter.limit("60/minute")
    async def get_dashboard(
        request: Request,
        district_id: Optional[str] = Query(None, description="District ID filter"),
        store_id: Optional[str] = Query(None, description="Store ID filter"),
        start_date: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD)"),
        end_date: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD)"),
        db: AsyncSession = Depends(get_db),
        user: Dict[str, Any] = Depends(get_tenant_auth_user),
        store_service: StoreService = Depends(lambda: get_store_service(use_mock_data=False))
    ):
        """
        Get Store Hub dashboard with KPIs, priority stores, and performance charts.
        
        **Requires**: User authentication from log_users table (Bearer token from /api/sessions-login)
        
        Args:
            district_id: The district ID filter (optional)
            store_id: The store ID filter (optional)
            start_date: Start date filter in YYYY-MM-DD format (optional)
            end_date: End date filter in YYYY-MM-DD format (optional)
            user: Authenticated user from log_users table (automatically injected via dependency)
        
        Returns:
            Dashboard data including KPI cards, top priority stores, negative themes, and performance chart
        """
        logger.info(f"Dashboard request: district_id={district_id}, store_id={store_id}, start_date={start_date}, end_date={end_date}")
        
        return await store_service.get_dashboard(
            district_id=district_id,
            store_id=store_id,
            start_date=start_date,
            end_date=end_date
        )
    
    
    @router.get("/store/graph")
    @limiter.limit("60/minute")
    async def get_store_deep_dive(
        request: Request,
        district_id: Optional[str] = Query(None, description="District ID filter"),
        store_id: Optional[str] = Query(None, description="Store ID filter"),
        start_date: Optional[str] = Query(None, description="Start date filter (YYYY-MM-DD)"),
        end_date: Optional[str] = Query(None, description="End date filter (YYYY-MM-DD)"),
        db: AsyncSession = Depends(get_db),
        user: Dict[str, Any] = Depends(get_tenant_auth_user),
        store_service: StoreService = Depends(lambda: get_store_service(use_mock_data=False))
    ):
        """
        Get detailed store deep dive analysis with comprehensive metrics.
        
        **Requires**: User authentication from log_users table (Bearer token from /api/sessions-login)
        
        Args:
            district_id: The district ID filter (optional)
            store_id: The store ID filter (optional)
            start_date: Start date filter in YYYY-MM-DD format (optional)
            end_date: End date filter in YYYY-MM-DD format (optional)
            user: Authenticated user from log_users table (automatically injected via dependency)
        
        Returns:
            Store deep dive data including detailed KPIs, themes, and analysis
        """
        logger.info(f"Store deep dive request: district_id={district_id}, store_id={store_id}, start_date={start_date}, end_date={end_date}")
        
        return await store_service.get_store_deep_dive(
            district_id=district_id,
            store_id=store_id,
            start_date=start_date,
            end_date=end_date
        )
    
    
    @router.get("/all-stores")
    @limiter.limit("60/minute")
    async def get_all_stores(
        request: Request,
        district_id: Optional[str] = Query(None, description="District ID filter", example="Miller"),
        db: AsyncSession = Depends(get_db),
        user: Dict[str, Any] = Depends(get_tenant_auth_user),
        store_service: StoreService = Depends(lambda: get_store_service(use_mock_data=False))
    ):
        """
        Get all stores in a district with their ratings and metrics.
        
        **Requires**: User authentication from log_users table (Bearer token from /api/sessions-login)
        
        Args:
            district_id: The district ID filter (optional, defaults to "Miller")
            user: Authenticated user from log_users table (automatically injected via dependency)
        
        Returns:
            All stores data including ratings, tags, KPIs, and sentiment breakdown
        """
        logger.info(f"All stores request: district_id={district_id}")
        
        return await store_service.get_all_stores(district_id=district_id)
    
    
    @router.get("/date-ranges")
    @limiter.limit("60/minute")
    async def get_date_ranges(
        request: Request,
        user: Dict[str, Any] = Depends(get_tenant_auth_user)
    ):
        """
        Get default date ranges for the frontend date picker.
        
        **Requires**: User authentication from log_users table (Bearer token from /api/sessions-login)
        
        Returns:
            Date range defaults including current month, week, and quarter
        """
        logger.info("Date ranges request")
        return get_date_range_defaults()
    
    return router

