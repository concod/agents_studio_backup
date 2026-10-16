"""
Dependency injection for Store Hub API.
Provides instances of services and repositories.
"""

import os
from functools import lru_cache

from ..infrastructure.clients.tools_universe_client import ToolsUniverseClient
from ..domain.store_repository_impl import StoreRepository
from ..application.store_service import StoreService


@lru_cache()
def get_tools_universe_client() -> ToolsUniverseClient:
    """
    Get Tools Universe API client instance.
    
    Reads configuration from environment variables:
    - TOOLS_UNIVERSE_BASE_URL: Base URL for the API (default: http://localhost:8001)
    - TOOLS_UNIVERSE_API_KEY: API key for authentication
    
    Returns:
        Configured Tools Universe client
    """
    base_url = os.getenv("TOOLS_UNIVERSE_BASE_URL", "http://localhost:8001")
    api_key = os.getenv("TOOLS_UNIVERSE_API_KEY", "")
    
    return ToolsUniverseClient(
        base_url=base_url,
        api_key=api_key,
        timeout=120.0
    )


def get_store_repository(use_mock_data: bool = False) -> StoreRepository:
    """
    Get store repository instance.
    
    Args:
        use_mock_data: If True, use mock data instead of external API
        
    Returns:
        Store repository instance
    """
    if use_mock_data:
        return StoreRepository(use_mock_data=True)
    
    client = get_tools_universe_client()
    return StoreRepository(tools_universe_client=client, use_mock_data=False)


def get_store_service(use_mock_data: bool = False) -> StoreService:
    """
    Get store service instance.
    
    Args:
        use_mock_data: If True, use mock data instead of external API
        
    Returns:
        Store service instance
    """
    repository = get_store_repository(use_mock_data=use_mock_data)
    return StoreService(store_repository=repository)

