# Store Hub Application - Domain-Driven Design Architecture

## Overview

The Store Hub application is a bounded context within the agentic platform that provides store management, analytics, and performance tracking capabilities. It has been refactored from a monolithic structure in `main.py` to follow **Domain-Driven Design (DDD)** principles.

## Architecture

This application follows a clean, layered architecture based on DDD principles:

```
applications/store_hub/
├── api/                    # API Layer (Presentation)
│   ├── __init__.py
│   ├── routes.py          # FastAPI route definitions
│   └── dependencies.py    # Dependency injection setup
├── application/            # Application Layer (Use Cases)
│   ├── __init__.py
│   └── store_service.py   # Business logic orchestration
├── domain/                 # Domain Layer (Business Logic)
│   ├── __init__.py
│   ├── repositories.py    # Repository interfaces
│   └── store_repository_impl.py  # Repository implementation
├── infrastructure/         # Infrastructure Layer (External Dependencies)
│   ├── __init__.py
│   ├── clients/
│   │   ├── __init__.py
│   │   └── tools_universe_client.py  # External API client
│   └── data/
│       ├── __init__.py
│       └── mock_data.py   # Mock data provider
├── schemas/                # Data Transfer Objects (DTOs)
│   ├── __init__.py
│   ├── requests.py        # Request models
│   └── responses.py       # Response models
├── __init__.py
└── README.md              # This file
```

## Layer Responsibilities

### 1. API Layer (`api/`)

**Purpose**: Handle HTTP requests and responses

- **`routes.py`**: Defines FastAPI endpoints for Store Hub
  - `/api/agents/custom/store-hub/graph` - Dashboard data
  - `/api/agents/custom/store-hub/store/graph` - Store deep dive
  - `/api/agents/custom/store-hub/all-stores` - All stores list
  
- **`dependencies.py`**: Dependency injection configuration
  - Creates and configures service instances
  - Manages client initialization
  - Provides mock data toggle

**SOLID Principles Applied**:
- **Single Responsibility**: Routes only handle HTTP concerns
- **Dependency Inversion**: Depends on abstractions (service interfaces)

### 2. Application Layer (`application/`)

**Purpose**: Orchestrate business logic and use cases

- **`store_service.py`**: Application service that coordinates between API and domain layers
  - `get_dashboard()` - Fetch dashboard data
  - `get_store_deep_dive()` - Fetch detailed store analysis
  - `get_all_stores()` - Fetch all stores in a district

**SOLID Principles Applied**:
- **Single Responsibility**: Each method handles one use case
- **Open/Closed**: Easy to extend with new use cases without modifying existing code
- **Dependency Inversion**: Depends on repository interface, not implementation

### 3. Domain Layer (`domain/`)

**Purpose**: Core business logic and rules

- **`repositories.py`**: Defines `IStoreRepository` interface
  - Abstract methods for data access
  - Technology-agnostic contracts

- **`store_repository_impl.py`**: Implements `IStoreRepository`
  - Coordinates between external APIs and mock data
  - Handles data retrieval logic
  - Can switch between real and mock data

**SOLID Principles Applied**:
- **Interface Segregation**: Clean, focused repository interface
- **Liskov Substitution**: Any implementation of `IStoreRepository` can be substituted
- **Dependency Inversion**: Depends on abstractions (client interfaces)

### 4. Infrastructure Layer (`infrastructure/`)

**Purpose**: Handle external dependencies and data sources

- **`clients/tools_universe_client.py`**: External API client
  - Communicates with Tools Universe API
  - Handles HTTP requests, timeouts, and errors
  - Manages authentication

- **`data/mock_data.py`**: Mock data provider
  - Provides hardcoded data for testing/development
  - Enables development without external dependencies

**SOLID Principles Applied**:
- **Single Responsibility**: Each client handles one external system
- **Open/Closed**: Easy to add new clients without modifying existing ones

### 5. Schemas Layer (`schemas/`)

**Purpose**: Define data contracts (DTOs)

- **`requests.py`**: Pydantic models for API requests
- **`responses.py`**: Pydantic models for API responses

**Benefits**:
- Type safety
- Automatic validation
- Clear API contracts
- Self-documenting code

## SOLID Principles Summary

### Single Responsibility Principle (SRP)
✅ Each class/module has one reason to change:
- Routes handle HTTP
- Services orchestrate business logic
- Repositories manage data access
- Clients handle external APIs

### Open/Closed Principle (OCP)
✅ Open for extension, closed for modification:
- New use cases can be added without changing existing services
- New data sources can be added by implementing `IStoreRepository`
- New clients can be added without affecting existing code

### Liskov Substitution Principle (LSP)
✅ Implementations can be substituted:
- Any `IStoreRepository` implementation works with `StoreService`
- Mock and real repositories are interchangeable

### Interface Segregation Principle (ISP)
✅ Focused, minimal interfaces:
- `IStoreRepository` only defines necessary methods
- Clients don't depend on methods they don't use

### Dependency Inversion Principle (DIP)
✅ Depend on abstractions, not concretions:
- `StoreService` depends on `IStoreRepository` interface
- Routes depend on `StoreService` abstraction
- Easy to swap implementations for testing

## Configuration

### Environment Variables

```bash
# Tools Universe API Configuration
TOOLS_UNIVERSE_BASE_URL=https://tools-universe.devs.impact-agents.ai
TOOLS_UNIVERSE_API_KEY=your_api_key_here

# BigQuery Configuration (handled in main.py)
# Uses get_bigquery_credentials() function
```

### Mock Data Mode

To use mock data instead of external APIs:

```python
# In dependencies.py
store_service = get_store_service(use_mock_data=True)
```

## API Endpoints

### 1. Dashboard (`GET /api/agents/custom/store-hub/graph`)

**Query Parameters**:
- `district_id` (optional): Filter by district
- `store_id` (optional): Filter by store

**Response**: Dashboard data with KPIs, priority stores, themes, and charts

**Authentication**: Required (Bearer token)

### 2. Store Deep Dive (`GET /api/agents/custom/store-hub/store/graph`)

**Query Parameters**:
- `district_id` (optional): Filter by district
- `store_id` (optional): Filter by store

**Response**: Detailed store analysis with comprehensive metrics

**Authentication**: Required (Bearer token)

### 3. All Stores (`GET /api/agents/custom/store-hub/all-stores`)

**Query Parameters**:
- `district_id` (optional): Filter by district (default: "Miller")

**Response**: List of all stores with ratings, KPIs, and sentiment data

**Authentication**: Required (Bearer token)

**Note**: Currently returns hardcoded data for Canton - Cromer store (ID: 0049)

## Testing

### Unit Testing

```python
# Example: Test StoreService with mock repository
from applications.store_hub.application.store_service import StoreService
from applications.store_hub.domain.store_repository_impl import StoreRepository

# Create service with mock data
repository = StoreRepository(use_mock_data=True)
service = StoreService(repository)

# Test use cases
dashboard = await service.get_dashboard(district_id="Miller")
assert dashboard.districtFilter.currentSelection == "Miller"
```

### Integration Testing

```python
# Example: Test with real API client
from applications.store_hub.infrastructure.clients.tools_universe_client import ToolsUniverseClient
from applications.store_hub.domain.store_repository_impl import StoreRepository
from applications.store_hub.application.store_service import StoreService

# Create service with real client
client = ToolsUniverseClient(
    base_url="https://tools-universe.devs.impact-agents.ai",
    api_key="test_key"
)
repository = StoreRepository(tools_universe_client=client)
service = StoreService(repository)

# Test with real API
dashboard = await service.get_dashboard(district_id="Miller", store_id="0049")
```

## Migration from Old Code

The old monolithic routes in `main.py` (lines 3340-3852) have been:
1. ✅ Refactored into layered DDD architecture
2. ✅ Commented out with deprecation notice
3. ✅ Replaced with new router registration

### What Changed:

**Before** (in `main.py`):
```python
@app.get("/api/agents/custom/store-hub/graph")
async def get_store_hub_graph(...):
    # 200+ lines of mixed concerns
    # HTTP handling + business logic + API calls
```

**After** (DDD architecture):
```python
# API Layer
@router.get("/graph")
async def get_dashboard(...):
    return await store_service.get_dashboard(...)

# Application Layer
class StoreService:
    async def get_dashboard(...):
        data = await self.store_repository.get_dashboard_data(...)
        return DashboardResponse(**data)

# Domain Layer
class StoreRepository(IStoreRepository):
    async def get_dashboard_data(...):
        return await self.tools_universe_client.fetch_revv_analysis(...)

# Infrastructure Layer
class ToolsUniverseClient:
    async def fetch_revv_analysis(...):
        # HTTP client logic
```

## Benefits of This Architecture

### 1. **Maintainability**
- Clear separation of concerns
- Easy to locate and fix bugs
- Self-documenting structure

### 2. **Testability**
- Each layer can be tested independently
- Easy to mock dependencies
- Fast unit tests with mock data

### 3. **Scalability**
- Easy to add new features
- Can extract to microservice if needed
- Clear boundaries between components

### 4. **Flexibility**
- Swap implementations easily (mock vs real)
- Change external APIs without affecting business logic
- Add new data sources without breaking existing code

### 5. **Team Collaboration**
- Different teams can work on different layers
- Clear interfaces reduce merge conflicts
- New developers can understand structure quickly

## Future Enhancements

### Potential Improvements:

1. **Add Caching Layer**
   ```python
   # infrastructure/cache/redis_cache.py
   class RedisCacheRepository(IStoreRepository):
       async def get_dashboard_data(...):
           # Check cache first, then fallback to API
   ```

2. **Add Event Sourcing**
   ```python
   # domain/events.py
   class StoreAnalyzedEvent:
       store_id: str
       timestamp: datetime
   ```

3. **Add CQRS Pattern**
   ```python
   # application/commands/
   # application/queries/
   ```

4. **Extract to Microservice**
   - Already has clear boundaries
   - Can run independently
   - Easy to containerize

## Troubleshooting

### Common Issues:

1. **Import Errors**
   - Ensure `backend/` is in Python path
   - Check `__init__.py` files exist in all directories

2. **Circular Dependencies**
   - Use dependency injection
   - Import at function level if needed

3. **Mock Data Not Working**
   - Check `use_mock_data=True` in dependencies
   - Verify mock data structure matches schemas

4. **API Timeouts**
   - Check `TOOLS_UNIVERSE_BASE_URL` environment variable
   - Verify network connectivity
   - Increase timeout in client configuration

## Contributing

When adding new features:

1. **Add new use case**: Create method in `StoreService`
2. **Add new data source**: Implement `IStoreRepository` or create new repository interface
3. **Add new endpoint**: Add route in `routes.py` that calls service
4. **Add new external API**: Create new client in `infrastructure/clients/`
5. **Update schemas**: Add/modify request/response models in `schemas/`

Always follow the layered architecture and SOLID principles!

## Contact

For questions or issues with this application, please contact the platform team.

---

**Version**: 1.0.0  
**Last Updated**: January 2026  
**Architecture**: Domain-Driven Design (DDD)  
**Framework**: FastAPI + Pydantic

