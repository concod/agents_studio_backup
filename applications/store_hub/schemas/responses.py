"""
Response schemas for Store Hub API.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field


class Trend(BaseModel):
    """Trend information for KPIs."""
    value: float
    direction: str = Field(..., description="up, down, or flat")
    label: str


class SparklineDataPoint(BaseModel):
    """Sparkline data point."""
    x: str  # Date string like "2025-06-01"
    y: float


class Metric(BaseModel):
    """Metric information."""
    label: str
    value: Optional[float]
    direction: Optional[str] = None


class KPICard(BaseModel):
    """KPI card data."""
    id: str
    title: str
    metricKey: Optional[str] = None
    icon: str
    value: Optional[Union[str, float]] = None  # Can be string, number, or None
    unit: Optional[str] = None
    metrics: Optional[List[Metric]] = None
    trend: Optional[Trend] = None
    status: Optional[str] = None
    sparklineData: Optional[List[Union[float, SparklineDataPoint]]] = None
    expandable: Optional[bool] = None
    detail: Optional[Dict[str, Any]] = None


class FilterOption(BaseModel):
    """Filter dropdown options."""
    currentSelection: str
    options: List[str]


class PriorityStore(BaseModel):
    """Priority store information."""
    id: str
    name: str
    location: str
    tags: List[str]


class TopPriorityStores(BaseModel):
    """Top priority stores section."""
    title: str
    subtitle: str
    stores: List[PriorityStore]


class NegativeTheme(BaseModel):
    """Negative customer theme."""
    id: str
    title: str
    percentage: float
    description: str
    tags: List[str]


class TopNegativeThemes(BaseModel):
    """Top negative themes section."""
    title: str
    subtitle: str
    items: List[NegativeTheme]


class DataPoint(BaseModel):
    """Chart data point."""
    x: float
    y: float
    # For performance chart, dataPoints can be arrays [x, y, label]
    # We'll use Union to handle both cases


class Chart(BaseModel):
    """Chart configuration."""
    title: str
    xAxisLabel: str
    yAxisLabel: Optional[str] = None
    dataPoints: List[Any]  # Can be DataPoint objects, arrays [x, y, label], or other formats


class PerformanceChart(BaseModel):
    """Performance chart section."""
    title: str
    subtitle: str
    chart: Chart


class DashboardResponse(BaseModel):
    """Response model for dashboard data."""
    districtFilter: FilterOption
    dataTimeDetail: Optional[Dict[str, Any]] = None
    overallSummary: Optional[Dict[str, Any]] = None
    kpiCards: List[KPICard]
    topPriorityStores: Optional[TopPriorityStores] = None
    topNegativeThemes: Optional[TopNegativeThemes] = None
    topPositiveThemes: Optional[Dict[str, Any]] = None
    topNeutralThemes: Optional[Dict[str, Any]] = None
    performanceChart: PerformanceChart
    allStores: Optional[Dict[str, Any]] = None


class StoreDeepDiveResponse(BaseModel):
    """Response model for store deep dive data."""
    districtFilter: Optional[FilterOption] = None
    district: Optional[str] = None  # Some APIs return district as a string
    dataTimeDetail: Optional[Dict[str, Any]] = None
    overallSummary: Optional[Dict[str, Any]] = None
    kpiCards: Optional[List[KPICard]] = None
    topPriorityStores: Optional[TopPriorityStores] = None
    topNegativeThemes: Optional[TopNegativeThemes] = None
    topPositiveThemes: Optional[Dict[str, Any]] = None
    topNeutralThemes: Optional[Dict[str, Any]] = None
    performanceChart: Optional[PerformanceChart] = None
    allStores: Optional[Dict[str, Any]] = None
    # Allow any additional fields from the API response
    model_config = {"extra": "allow"}


class StoreLocation(BaseModel):
    """Store location information."""
    district: str
    label: str


class Theme(BaseModel):
    """Theme information."""
    type: str = Field(..., description="positive, negative, or neutral")
    value: str


class TagGroup(BaseModel):
    """Tag group with themes."""
    themes: List[Theme]


class KPIDelta(BaseModel):
    """KPI delta information."""
    value: Optional[float]
    unit: str
    direction: Optional[str]


class KPIDeltas(BaseModel):
    """All KPI deltas for a store."""
    netSales: KPIDelta
    nrr: KPIDelta
    revv: KPIDelta


class SentimentBreakdown(BaseModel):
    """Sentiment breakdown percentages."""
    positive: float
    neutral: float
    negative: float


class Store(BaseModel):
    """Store information."""
    id: str
    name: str
    location: StoreLocation
    average_rating: float
    tags: List[TagGroup]
    topIssue: Optional[str]
    kpiDeltas: KPIDeltas
    sentimentBreakdown: SentimentBreakdown


class AllStoresData(BaseModel):
    """All stores data container."""
    title: str
    subtitle: str
    stores: List[Store]


class AllStoresResponse(BaseModel):
    """Response model for all stores list."""
    allStores: AllStoresData

