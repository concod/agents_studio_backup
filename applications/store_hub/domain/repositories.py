"""
Repository interfaces for Store Hub domain.
Defines contracts for data access without implementation details.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class IStoreRepository(ABC):
    """Interface for store data repository."""
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass

