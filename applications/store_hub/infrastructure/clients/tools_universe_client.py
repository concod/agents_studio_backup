"""
Client for Tools Universe API.
Handles communication with external REVV analysis and store deep dive services.
"""

import logging
import json
from typing import Dict, Any, Optional
import httpx
from fastapi import HTTPException

logger = logging.getLogger(__name__)


class ToolsUniverseClient:
    """Client for Tools Universe API."""
    
    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0):
        """
        Initialize Tools Universe client.
        
        Args:
            base_url: Base URL for Tools Universe API
            api_key: API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = httpx.Timeout(timeout, connect=10.0)
    
    def _get_db_config(self) -> Dict[str, Any]:
        """Get database configuration for API requests."""
        # Import here to avoid circular dependency
        from main import get_bigquery_credentials
        
        return {
            "db_type": "bigquery",
            "project_id": "tools-universe-29042025",
            "database": "petsupplyplus",
            "credentials_json": get_bigquery_credentials()
        }
    
    async def fetch_revv_analysis(
        self,
        district_id: Optional[str] = None,
        store_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch REVV analysis data from Tools Universe API.
        
        Args:
            district_id: District ID filter
            store_id: Store ID filter
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            
        Returns:
            REVV analysis data
            
        Raises:
            HTTPException: If API call fails
        """
        request_data = {"db_config": self._get_db_config()}
        
        if store_id:
            request_data["store_id"] = store_id
        if district_id:
            request_data["district"] = district_id
        if start_date:
            request_data["start_date"] = start_date
        if end_date:
            request_data["end_date"] = end_date
        
        endpoint = f"{self.base_url}/internal/storehub/revv_analysis"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info(f"Calling REVV analysis API with district_id={district_id}, store_id={store_id}, start_date={start_date}, end_date={end_date}")
                
                # Same format as original: "Authorization": f"Bearer {TOOLS_UNIVERSE_API_KEY}"
                # Handle empty API key by using empty string (some APIs accept Bearer without token)
                api_key = self.api_key.strip() if self.api_key else ""
                
                response = await client.post(
                    endpoint,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}"
                    },
                    json=request_data
                )
                
                logger.info(f"REVV API response status: {response.status_code}")
                response.raise_for_status()
                
                data = response.json()
                logger.info("REVV API response received successfully")
                return data
                
        except httpx.TimeoutException as e:
            logger.error(f"Timeout calling REVV analysis API after {self.timeout.read}s: {str(e)}")
            raise HTTPException(
                status_code=504,
                detail=f"REVV analysis API request timed out after {self.timeout.read} seconds. The analysis may still be processing in the background."
            )
        except httpx.ConnectError as e:
            logger.error(f"Connection error calling REVV analysis API: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail="Could not connect to REVV analysis API. Please ensure the service is running."
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling REVV analysis API: Status {e.response.status_code}")
            logger.error(f"Request data: {request_data}")
            logger.error(f"Response text: {e.response.text[:1000] if hasattr(e.response, 'text') else 'N/A'}")
            
            # Try to parse error message from response
            error_detail = str(e)
            try:
                if hasattr(e.response, 'text') and e.response.text:
                    error_json = e.response.json()
                    if isinstance(error_json, dict) and 'detail' in error_json:
                        error_detail = error_json['detail']
                    else:
                        error_detail = e.response.text[:500]
            except:
                error_detail = e.response.text[:500] if hasattr(e.response, 'text') else str(e)
            
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"REVV analysis API returned error {e.response.status_code}: {error_detail}"
            )
        except Exception as e:
            logger.error(f"Unexpected error calling REVV analysis API: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch REVV analysis data: {str(e)}"
            )
    
    async def fetch_store_deep_dive(
        self,
        district_id: Optional[str] = None,
        store_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch store deep dive data from Tools Universe API.
        
        Args:
            district_id: District ID filter
            store_id: Store ID filter
            start_date: Start date filter (YYYY-MM-DD)
            end_date: End date filter (YYYY-MM-DD)
            
        Returns:
            Store deep dive data
            
        Raises:
            HTTPException: If API call fails
        """
        request_data = {"db_config": self._get_db_config()}
        
        if store_id:
            request_data["store_id"] = store_id
        if district_id:
            request_data["district"] = district_id
        if start_date:
            request_data["start_date"] = start_date
        if end_date:
            request_data["end_date"] = end_date
        
        endpoint = f"{self.base_url}/internal/storehub/store_deep_dive"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info(f"Calling store deep dive API with district_id={district_id}, store_id={store_id}, start_date={start_date}, end_date={end_date}")
                
                # Same format as original: "Authorization": f"Bearer {TOOLS_UNIVERSE_API_KEY}"
                # Handle empty API key by using empty string (some APIs accept Bearer without token)
                api_key = self.api_key.strip() if self.api_key else ""
                
                response = await client.post(
                    endpoint,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {api_key}"
                    },
                    json=request_data
                )
                
                logger.info(f"Store deep dive API response status: {response.status_code}")
                response.raise_for_status()
                
                data = response.json()
                logger.info("Store deep dive API response received successfully")
                return data
                
        except httpx.TimeoutException as e:
            logger.error(f"Timeout calling store deep dive API after {self.timeout.read}s: {str(e)}")
            raise HTTPException(
                status_code=504,
                detail=f"Store deep dive API request timed out after {self.timeout.read} seconds. The analysis may still be processing in the background."
            )
        except httpx.ConnectError as e:
            logger.error(f"Connection error calling store deep dive API: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail="Could not connect to store deep dive API. Please ensure the service is running."
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling store deep dive API: Status {e.response.status_code}")
            logger.error(f"Request data: {request_data}")
            logger.error(f"Response text: {e.response.text[:1000] if hasattr(e.response, 'text') else 'N/A'}")
            
            # Try to parse error message from response
            error_detail = str(e)
            try:
                if hasattr(e.response, 'text') and e.response.text:
                    error_json = e.response.json()
                    if isinstance(error_json, dict) and 'detail' in error_json:
                        error_detail = error_json['detail']
                    else:
                        error_detail = e.response.text[:500]
            except:
                error_detail = e.response.text[:500] if hasattr(e.response, 'text') else str(e)
            
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Store deep dive API returned error {e.response.status_code}: {error_detail}"
            )
        except Exception as e:
            logger.error(f"Unexpected error calling store deep dive API: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch store deep dive data: {str(e)}"
            )
    
    async def fetch_district_stores(
        self,
        district_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch district stores data from Tools Universe API.
        
        Args:
            district_id: District ID filter
            
        Returns:
            District stores data
            
        Raises:
            HTTPException: If API call fails
        """
        request_data = {"db_config": self._get_db_config()}
        
        if district_id:
            request_data["district"] = district_id
        
        endpoint = f"{self.base_url}/internal/storehub/district_stores"
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info(f"Calling district stores API with district_id={district_id}")
                
                # Handle empty API key by using empty string
                api_key = self.api_key.strip() if self.api_key else ""
                
                headers = {"Content-Type": "application/json"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                
                response = await client.post(
                    endpoint,
                    headers=headers,
                    json=request_data
                )
                
                logger.info(f"District stores API response status: {response.status_code}")
                response.raise_for_status()
                
                data = response.json()
                logger.info("District stores API response received successfully")
                return data
                
        except httpx.TimeoutException as e:
            logger.error(f"Timeout calling district stores API after {self.timeout.read}s: {str(e)}")
            raise HTTPException(
                status_code=504,
                detail=f"District stores API request timed out after {self.timeout.read} seconds. The request may still be processing in the background."
            )
        except httpx.ConnectError as e:
            logger.error(f"Connection error calling district stores API: {str(e)}")
            raise HTTPException(
                status_code=503,
                detail="Could not connect to district stores API. Please ensure the service is running."
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling district stores API: Status {e.response.status_code}")
            logger.error(f"Request data: {request_data}")
            logger.error(f"Response text: {e.response.text[:1000] if hasattr(e.response, 'text') else 'N/A'}")
            
            # Try to parse error message from response
            error_detail = str(e)
            try:
                if hasattr(e.response, 'text') and e.response.text:
                    error_json = e.response.json()
                    if isinstance(error_json, dict) and 'detail' in error_json:
                        error_detail = error_json['detail']
                    else:
                        error_detail = e.response.text[:500]
            except:
                error_detail = e.response.text[:500] if hasattr(e.response, 'text') else str(e)
            
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"District stores API returned error {e.response.status_code}: {error_detail}"
            )
        except Exception as e:
            logger.error(f"Unexpected error calling district stores API: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to fetch district stores data: {str(e)}"
            )

