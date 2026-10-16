"""
Store service - Application layer.
Orchestrates business logic and use cases for Store Hub.
"""

import logging
from typing import Optional, Dict, Any

from ..domain.repositories import IStoreRepository
from ..infrastructure.utils.date_utils import get_date_range_defaults

logger = logging.getLogger(__name__)


class StoreService:
    """
    Store service for handling Store Hub business logic.
    
    This service implements the application layer, coordinating between
    the API layer and the domain layer.
    """
    
    def __init__(self, store_repository: IStoreRepository):
        """
        Initialize store service.
        
        Args:
            store_repository: Repository for accessing store data
        """
        self.store_repository = store_repository
    
    async def get_dashboard(
        self,
        district_id: Optional[str] = None,
        store_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get dashboard data for stores.
        
        Args:
            district_id: District ID filter
            store_id: Store ID filter
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            
        Returns:
            Dashboard response with KPIs, charts, store information, and date range defaults as dict
        """
        logger.info(f"Getting dashboard for district={district_id}, store={store_id}, start_date={start_date}, end_date={end_date}")
        
        data = await self.store_repository.get_dashboard_data(
            district_id=district_id,
            store_id=store_id,
            start_date=start_date,
            end_date=end_date
        )
        
        # Add date range defaults for frontend date picker
        if isinstance(data, dict):
            data["dateRangeDefaults"] = get_date_range_defaults()
        
        # Return raw dict without validation
        return data
    
    async def get_store_deep_dive(
        self,
        district_id: Optional[str] = None,
        store_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get detailed store deep dive data.
        
        Args:
            district_id: District ID filter
            store_id: Store ID filter
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            
        Returns:
            Store deep dive response with detailed metrics, analysis, and date range defaults as dict
        """
        logger.info(f"Getting store deep dive for district={district_id}, store={store_id}, start_date={start_date}, end_date={end_date}")
        
        data = await self.store_repository.get_store_deep_dive(
            district_id=district_id,
            store_id=store_id,
            start_date=start_date,
            end_date=end_date
        )
        
        # Add date range defaults for frontend date picker
        if isinstance(data, dict):
            data["dateRangeDefaults"] = get_date_range_defaults()
        
        # Return raw dict without validation
        return data
    
    async def get_all_stores(
        self,
        district_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get all stores in a district.
        
        Args:
            district_id: District ID filter
            
        Returns:
            All stores response with ratings, KPIs, and sentiment data as dict
        """
        logger.info(f"Getting all stores for district={district_id}")
        
        data = await self.store_repository.get_all_stores(
            district_id=district_id
        )
        
        # Return raw dict without validation
        return data

