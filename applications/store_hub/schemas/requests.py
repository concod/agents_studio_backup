"""
Request schemas for Store Hub API.
"""

from typing import Optional
from datetime import date
from pydantic import BaseModel, Field


class DashboardRequest(BaseModel):
    """Request model for dashboard data."""
    district_id: Optional[str] = Field(None, description="District ID filter")
    store_id: Optional[str] = Field(None, description="Store ID filter")
    start_date: Optional[str] = Field(None, description="Start date filter (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date filter (YYYY-MM-DD)")


class StoreRequest(BaseModel):
    """Request model for store deep dive data."""
    district_id: Optional[str] = Field(None, description="District ID filter")
    store_id: Optional[str] = Field(None, description="Store ID filter")
    start_date: Optional[str] = Field(None, description="Start date filter (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date filter (YYYY-MM-DD)")


class AllStoresRequest(BaseModel):
    """Request model for all stores list."""
    district_id: Optional[str] = Field(None, description="District ID filter", example="Miller")

