"""
Mock data provider for Store Hub application.
Contains hardcoded data for testing and development.
"""

from typing import Dict, Any


class MockDataProvider:
    """Provides mock data for Store Hub application."""
    
    @staticmethod
    def get_dashboard_data() -> Dict[str, Any]:
        """Get mock dashboard data."""
        return {
            "storeFilter": {
                "currentSelection": "All Stores",
                "options": ["All Stores", "PSP Downtown", "Store 1", "Store 2"]
            },
            "districtFilter": {
                "currentSelection": "Miller",
                "options": ["Miller"]
            },
            "kpiCards": [
                {
                    "id": "kpi_1",
                    "title": "Net Sales Comp (QTD)",
                    "icon": "sales",
                    "value": "$5.7M",
                    "trend": {
                        "value": 30,
                        "direction": "up",
                        "label": "Vs LY"
                    },
                    "sparklineData": [5, 10, 8, 15, 12, 18, 20]
                },
                {
                    "id": "kpi_2",
                    "title": "REVV Overall Score",
                    "icon": "score",
                    "value": "85/100",
                    "status": "Good",
                    "sparklineData": None
                },
                {
                    "id": "kpi_3",
                    "title": "NRR Audit Average",
                    "icon": "audit",
                    "value": "$5.7M",
                    "trend": {
                        "value": 22,
                        "direction": "up",
                        "label": "Vs LY"
                    },
                    "sparklineData": [15, 12, 18, 10, 14, 13, 16]
                },
                {
                    "id": "kpi_4",
                    "title": "Labor SPH (Sales Per Hour)",
                    "icon": "labor",
                    "value": "$5.7M",
                    "trend": {
                        "value": 30,
                        "direction": "up",
                        "label": "Vs LY"
                    },
                    "sparklineData": [8, 9, 12, 11, 14, 10, 15]
                }
            ],
            "topPriorityStores": {
                "title": "Top Priority Stores",
                "subtitle": "Action Required",
                "stores": [
                    {
                        "id": "store_A",
                        "name": "PSP Downtown",
                        "location": "Downtown, Downtown, U...",
                        "tags": ["Low Rev Store", "Inventory Issue", "Action Required"]
                    },
                    {
                        "id": "store_B",
                        "name": "PSP Downtown",
                        "location": "Downtown, Downtown, U...",
                        "tags": ["Low Rev Store", "Inventory Issue", "Action Required"]
                    }
                ]
            },
            "topNegativeThemes": {
                "title": "Top Negative Customer Theme",
                "subtitle": "Most frequent customer concern",
                "items": [
                    {
                        "id": "theme_001",
                        "title": "Out of Stock",
                        "percentage": 15,
                        "description": "15% of negative comments",
                        "tags": ["PSP Downtown", "PSP Downtown"]
                    },
                    {
                        "id": "theme_002",
                        "title": "Service Issues",
                        "percentage": 15,
                        "description": "15% of negative comments",
                        "tags": ["PSP Downtown"]
                    }
                ]
            },
            "performanceChart": {
                "title": "Performance Progress Chart",
                "subtitle": "Most frequent customer concern",
                "chart": {
                    "title": "Sales Comp",
                    "xAxisLabel": "Category",
                    "dataPoints": [
                        {"x": -800000, "y": 40000},
                        {"x": -700000, "y": -50000},
                        {"x": -500000, "y": 0},
                        {"x": -400000, "y": -80000},
                        {"x": -300000, "y": -30000},
                        {"x": -200000, "y": 40000},
                        {"x": 200000, "y": -20000},
                        {"x": 600000, "y": -10000},
                        {"x": 700000, "y": -60000},
                        {"x": 800000, "y": -40000}
                    ]
                }
            }
        }
    
    @staticmethod
    def get_all_stores_data() -> Dict[str, Any]:
        """Get mock all stores data."""
        return {
            "allStores": {
                "title": "All Stores in Miller District",
                "subtitle": "Ranked by average rating (ascending)",
                "stores": [
                    {
                        "id": "0049",
                        "name": "Canton - Cromer",
                        "location": {
                            "district": "Miller",
                            "label": "Canton, 44709-2908"
                        },
                        "average_rating": 71.43,
                        "tags": [
                            {
                                "themes": [
                                    {
                                        "type": "positive",
                                        "value": "Staff Interaction / Service Quality"
                                    }
                                ]
                            }
                        ],
                        "topIssue": None,
                        "kpiDeltas": {
                            "netSales": {
                                "value": None,
                                "unit": "percent",
                                "direction": None
                            },
                            "nrr": {
                                "value": None,
                                "unit": "points",
                                "direction": "flat"
                            },
                            "revv": {
                                "value": 71.43,
                                "unit": "points",
                                "direction": "down"
                            }
                        },
                        "sentimentBreakdown": {
                            "positive": 85.71,
                            "neutral": 0.0,
                            "negative": 14.29
                        }
                    }
                ]
            }
        }

