"""
Implementation of store repository.
Coordinates between external API clients and mock data providers.
"""

from typing import Dict, Any, Optional
import logging

from .repositories import IStoreRepository
from ..infrastructure.clients.tools_universe_client import ToolsUniverseClient
from ..infrastructure.data.mock_data import MockDataProvider

logger = logging.getLogger(__name__)


class StoreRepository(IStoreRepository):
    """
    Implementation of store repository.
    
    This repository can use either:
    - External API (Tools Universe) for real data
    - Mock data provider for testing/development
    """
    
    def __init__(
        self,
        tools_universe_client: Optional[ToolsUniverseClient] = None,
        use_mock_data: bool = False
    ):
        """
        Initialize store repository.
        
        Args:
            tools_universe_client: Client for Tools Universe API (optional if using mock data)
            use_mock_data: If True, use mock data instead of external API
        """
        self.tools_universe_client = tools_universe_client
        self.use_mock_data = use_mock_data
        self.mock_provider = MockDataProvider()
    
    async def get_dashboard_data(
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
            Dashboard data
        """
        if self.use_mock_data:
            logger.info("Using mock data for dashboard")
            return self.mock_provider.get_dashboard_data()
        
        if not self.tools_universe_client:
            raise ValueError("Tools Universe client is required when not using mock data")
        
        logger.info(f"Fetching dashboard data from API for district={district_id}, store={store_id}, start_date={start_date}, end_date={end_date}")
        return await self.tools_universe_client.fetch_revv_analysis(
            district_id=district_id,
            store_id=store_id,
            start_date=start_date,
            end_date=end_date
        )
    
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
            Store deep dive data
        """
        if self.use_mock_data:
            logger.info("Using mock data for store deep dive")
            return self.mock_provider.get_dashboard_data()  # Same structure for now
        
        if not self.tools_universe_client:
            raise ValueError("Tools Universe client is required when not using mock data")
        
        logger.info(f"Fetching store deep dive from API for district={district_id}, store={store_id}, start_date={start_date}, end_date={end_date}")
        return await self.tools_universe_client.fetch_store_deep_dive(
            district_id=district_id,
            store_id=store_id,
            start_date=start_date,
            end_date=end_date
        )
    
    async def get_all_stores(
        self,
        district_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get all stores in a district.
        
        Args:
            district_id: District ID filter
            
        Returns:
            All stores data
        """
        if self.use_mock_data:
            logger.info(f"Using mock data for all stores in district={district_id}")
            return self.mock_provider.get_all_stores_data()
        
        if not self.tools_universe_client:
            raise ValueError("Tools Universe client is required when not using mock data")
        
        logger.info(f"Fetching district stores from API for district={district_id}")
        return await self.tools_universe_client.fetch_district_stores(
            district_id=district_id
        )

