"""
Date utility functions for Store Hub application.
"""

from typing import Dict, Any


def get_date_range_defaults() -> Dict[str, Any]:
    """
    Get default date ranges for the frontend date picker.
    
    Returns:
        Dictionary containing start_date and end_date
    """
    return {
        "start_date": "2023-01-01",
        "end_date": "2025-10-31"
    }

