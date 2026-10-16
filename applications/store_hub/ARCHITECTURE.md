# Store Hub - Domain-Driven Design Architecture

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLIENT (Frontend)                               │
│                    HTTP Requests with Bearer Token                       │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        API LAYER (Presentation)                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  routes.py                                                         │ │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │ │
│  │  │  get_dashboard   │  │ get_store_deep   │  │ get_all_stores  │ │ │
│  │  │  /graph          │  │ _dive            │  │ /all-stores     │ │ │
│  │  │                  │  │ /store/graph     │  │                 │ │ │
│  │  └────────┬─────────┘  └────────┬─────────┘  └────────┬────────┘ │ │
│  │           │                     │                      │          │ │
│  │           └─────────────────────┼──────────────────────┘          │ │
│  │                                 │                                 │ │
│  │  dependencies.py                │                                 │ │
│  │  ┌──────────────────────────────▼───────────────────────────────┐ │ │
│  │  │  get_store_service() - Dependency Injection                  │ │ │
│  │  └──────────────────────────────┬───────────────────────────────┘ │ │
│  └─────────────────────────────────┼─────────────────────────────────┘ │
└────────────────────────────────────┼───────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER (Use Cases)                         │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  store_service.py                                                  │ │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │ │
│  │  │  get_dashboard() │  │ get_store_deep   │  │ get_all_stores()│ │ │
│  │  │                  │  │ _dive()          │  │                 │ │ │
│  │  │  Orchestrates    │  │ Orchestrates     │  │ Orchestrates    │ │ │
│  │  │  business logic  │  │ business logic   │  │ business logic  │ │ │
│  │  └────────┬─────────┘  └────────┬─────────┘  └────────┬────────┘ │ │
│  │           │                     │                      │          │ │
│  │           └─────────────────────┼──────────────────────┘          │ │
│  │                                 │                                 │ │
│  │                                 ▼                                 │ │
│  │                    Depends on IStoreRepository                    │ │
│  └─────────────────────────────────┼─────────────────────────────────┘ │
└────────────────────────────────────┼───────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     DOMAIN LAYER (Business Logic)                        │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  repositories.py (Interface)                                       │ │
│  │  ┌──────────────────────────────────────────────────────────────┐ │ │
│  │  │  IStoreRepository (ABC)                                      │ │ │
│  │  │  - get_dashboard_data()                                      │ │ │
│  │  │  - get_store_deep_dive()                                     │ │ │
│  │  │  - get_all_stores()                                          │ │ │
│  │  └──────────────────────────────┬───────────────────────────────┘ │ │
│  │                                 │                                 │ │
│  │                                 ▼                                 │ │
│  │  store_repository_impl.py (Implementation)                        │ │
│  │  ┌──────────────────────────────────────────────────────────────┐ │ │
│  │  │  StoreRepository implements IStoreRepository                 │ │ │
│  │  │  ┌────────────────────┐  ┌──────────────────────────────┐   │ │ │
│  │  │  │ use_mock_data?     │  │ tools_universe_client        │   │ │ │
│  │  │  │ Yes: MockData      │  │ No: External API             │   │ │ │
│  │  │  └────────┬───────────┘  └────────┬─────────────────────┘   │ │ │
│  │  │           │                       │                          │ │ │
│  │  └───────────┼───────────────────────┼──────────────────────────┘ │ │
│  └──────────────┼───────────────────────┼────────────────────────────┘ │
└─────────────────┼───────────────────────┼──────────────────────────────┘
                  │                       │
                  ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                  INFRASTRUCTURE LAYER (External Dependencies)            │
│  ┌───────────────────────────────┐  ┌──────────────────────────────┐   │
│  │  data/mock_data.py            │  │  clients/                    │   │
│  │  ┌─────────────────────────┐  │  │  tools_universe_client.py    │   │
│  │  │  MockDataProvider       │  │  │  ┌────────────────────────┐  │   │
│  │  │  - get_dashboard_data() │  │  │  │  ToolsUniverseClient   │  │   │
│  │  │  - get_all_stores_data()│  │  │  │  - fetch_revv_analysis │  │   │
│  │  └─────────────────────────┘  │  │  │  - fetch_store_deep    │  │   │
│  │                                │  │  │    _dive               │  │   │
│  │  Returns hardcoded JSON       │  │  └──────────┬─────────────┘  │   │
│  └────────────────────────────────┘  └─────────────┼────────────────┘   │
└──────────────────────────────────────────────────┼──────────────────────┘
                                                   │
                                                   ▼
                                    ┌──────────────────────────────┐
                                    │  External API                │
                                    │  Tools Universe              │
                                    │  - /internal/revv_analysis   │
                                    │  - /internal/store_deep_dive │
                                    └──────────────────────────────┘
```

## Data Flow

### Example: Get Dashboard Request

```
1. Client Request
   ↓
   GET /api/agents/custom/store-hub/graph?district_id=Miller
   Headers: Authorization: Bearer <token>

2. API Layer (routes.py)
   ↓
   get_dashboard(request, district_id, store_id, db, user, store_service)
   - Validates authentication (user dependency)
   - Validates database connection (db dependency)
   - Injects StoreService (store_service dependency)

3. Application Layer (store_service.py)
   ↓
   store_service.get_dashboard(district_id, store_id)
   - Calls repository to get data
   - Validates data with Pydantic model
   - Returns DashboardResponse

4. Domain Layer (store_repository_impl.py)
   ↓
   store_repository.get_dashboard_data(district_id, store_id)
   - Checks if use_mock_data flag is set
   - If mock: calls MockDataProvider
   - If real: calls ToolsUniverseClient

5a. Infrastructure Layer - Mock Path
    ↓
    mock_provider.get_dashboard_data()
    - Returns hardcoded JSON
    - Fast, no external dependencies

5b. Infrastructure Layer - Real Path
    ↓
    tools_universe_client.fetch_revv_analysis(district_id, store_id)
    - Constructs request with BigQuery credentials
    - Makes HTTP POST to external API
    - Handles timeouts, errors, retries
    - Returns API response

6. Response Flow (back up the stack)
   ↓
   Repository → Service → Route → Client
   - Data is validated at each layer
   - Pydantic models ensure type safety
   - Errors are caught and transformed appropriately
```

## Dependency Injection Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  main.py                                                        │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  # Import Store Hub router                                │ │
│  │  from applications.store_hub.api.routes import (          │ │
│  │      router, create_routes                                │ │
│  │  )                                                         │ │
│  │                                                            │ │
│  │  # Create router with dependencies                        │ │
│  │  store_hub_router = create_routes(                        │ │
│  │      limiter=limiter,              # Rate limiting        │ │
│  │      get_db=get_db,                # Database session     │ │
│  │      get_tenant_auth_user=get_tenant_auth_user  # Auth   │ │
│  │  )                                                         │ │
│  │                                                            │ │
│  │  # Register router                                        │ │
│  │  app.include_router(store_hub_router, tags=["Store Hub"])│ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  applications/store_hub/api/routes.py                           │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  def create_routes(limiter, get_db, get_tenant_auth_user):│ │
│  │      # Routes are created with injected dependencies      │ │
│  │                                                            │ │
│  │      @router.get("/graph")                                │ │
│  │      @limiter.limit("60/minute")                          │ │
│  │      async def get_dashboard(                             │ │
│  │          request: Request,                                │ │
│  │          district_id: Optional[str],                      │ │
│  │          store_id: Optional[str],                         │ │
│  │          db: AsyncSession = Depends(get_db),              │ │
│  │          user: Dict = Depends(get_tenant_auth_user),      │ │
│  │          store_service = Depends(get_store_service)       │ │
│  │      ):                                                    │ │
│  │          return await store_service.get_dashboard(...)    │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  applications/store_hub/api/dependencies.py                     │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │  def get_store_service(use_mock_data=False):              │ │
│  │      repository = get_store_repository(use_mock_data)     │ │
│  │      return StoreService(repository)                      │ │
│  │                                                            │ │
│  │  def get_store_repository(use_mock_data=False):           │ │
│  │      if use_mock_data:                                    │ │
│  │          return StoreRepository(use_mock_data=True)       │ │
│  │      client = get_tools_universe_client()                 │ │
│  │      return StoreRepository(                              │ │
│  │          tools_universe_client=client                     │ │
│  │      )                                                     │ │
│  │                                                            │ │
│  │  def get_tools_universe_client():                         │ │
│  │      return ToolsUniverseClient(                          │ │
│  │          base_url=os.getenv("TOOLS_UNIVERSE_BASE_URL"),   │ │
│  │          api_key=os.getenv("TOOLS_UNIVERSE_API_KEY")      │ │
│  │      )                                                     │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## SOLID Principles in Action

### 1. Single Responsibility Principle (SRP)

```
routes.py           → HTTP handling only
store_service.py    → Business logic orchestration only
store_repository.py → Data access only
clients.py          → External API communication only
```

### 2. Open/Closed Principle (OCP)

```
# Adding new data source (OPEN for extension)
class DatabaseStoreRepository(IStoreRepository):
    async def get_dashboard_data(...):
        # Query database directly
        pass

# No need to modify existing code (CLOSED for modification)
```

### 3. Liskov Substitution Principle (LSP)

```
# Any IStoreRepository implementation works
service = StoreService(MockRepository())      # ✅ Works
service = StoreService(APIRepository())       # ✅ Works
service = StoreService(DatabaseRepository())  # ✅ Works
```

### 4. Interface Segregation Principle (ISP)

```
# Clean, focused interface
class IStoreRepository(ABC):
    @abstractmethod
    async def get_dashboard_data(...): pass
    
    @abstractmethod
    async def get_store_deep_dive(...): pass
    
    @abstractmethod
    async def get_all_stores(...): pass

# No unnecessary methods
```

### 5. Dependency Inversion Principle (DIP)

```
# High-level module (StoreService)
class StoreService:
    def __init__(self, store_repository: IStoreRepository):
        self.store_repository = store_repository  # Depends on abstraction

# Low-level module (StoreRepository)
class StoreRepository(IStoreRepository):
    # Implements abstraction
    pass

# Both depend on abstraction (IStoreRepository)
```

## Testing Strategy

### Unit Tests

```python
# Test Service Layer (with mock repository)
async def test_get_dashboard():
    mock_repo = MockStoreRepository()
    service = StoreService(mock_repo)
    result = await service.get_dashboard("Miller", "0049")
    assert result.districtFilter.currentSelection == "Miller"

# Test Repository Layer (with mock client)
async def test_repository_with_mock_data():
    repo = StoreRepository(use_mock_data=True)
    data = await repo.get_dashboard_data("Miller", "0049")
    assert "kpiCards" in data

# Test Client Layer (with mock HTTP)
async def test_client_timeout_handling():
    client = ToolsUniverseClient(base_url="http://test", api_key="test")
    with pytest.raises(HTTPException) as exc:
        await client.fetch_revv_analysis()
    assert exc.value.status_code == 504
```

### Integration Tests

```python
# Test full stack (API → Service → Repository → Client)
async def test_dashboard_endpoint():
    response = client.get(
        "/api/agents/custom/store-hub/graph",
        params={"district_id": "Miller"},
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert "kpiCards" in response.json()
```

## Configuration

### Environment Variables

```bash
# Required
TOOLS_UNIVERSE_BASE_URL=https://tools-universe.devs.impact-agents.ai
TOOLS_UNIVERSE_API_KEY=your_api_key

# Optional (for BigQuery)
# Handled by get_bigquery_credentials() in main.py
```

### Feature Flags

```python
# Use mock data (for development/testing)
store_service = get_store_service(use_mock_data=True)

# Use real API (for production)
store_service = get_store_service(use_mock_data=False)
```

## Error Handling

```
┌─────────────────────────────────────────────────────────────┐
│  Error Handling Flow                                        │
│                                                             │
│  Client Request                                             │
│       ↓                                                     │
│  API Layer (routes.py)                                      │
│       ↓                                                     │
│  Application Layer (store_service.py)                       │
│       ↓                                                     │
│  Domain Layer (store_repository.py)                         │
│       ↓                                                     │
│  Infrastructure Layer (tools_universe_client.py)            │
│       ↓                                                     │
│  External API                                               │
│       │                                                     │
│       ├─ TimeoutException → HTTPException(504)              │
│       ├─ ConnectError → HTTPException(503)                  │
│       ├─ HTTPStatusError → HTTPException(status_code)       │
│       └─ Exception → HTTPException(500)                     │
│                                                             │
│  Errors bubble up with proper HTTP status codes            │
└─────────────────────────────────────────────────────────────┘
```

## Performance Considerations

### Caching Strategy (Future Enhancement)

```
┌─────────────────────────────────────────────────────────────┐
│  Potential Caching Layer                                    │
│                                                             │
│  Request → Check Cache → Cache Hit? → Return Cached Data   │
│                ↓                                            │
│           Cache Miss                                        │
│                ↓                                            │
│           Fetch from API                                    │
│                ↓                                            │
│           Store in Cache                                    │
│                ↓                                            │
│           Return Data                                       │
└─────────────────────────────────────────────────────────────┘
```

### Timeout Configuration

```python
# Client timeout settings
ToolsUniverseClient(
    base_url="...",
    api_key="...",
    timeout=120.0  # 120 seconds for heavy operations
)
```

## Deployment

### Docker Container (Future)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy only Store Hub application
COPY applications/store_hub /app/applications/store_hub
COPY requirements.txt /app/

RUN pip install -r requirements.txt

# Run as microservice
CMD ["uvicorn", "applications.store_hub.api.routes:router", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes Deployment (Future)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: store-hub-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: store-hub
  template:
    metadata:
      labels:
        app: store-hub
    spec:
      containers:
      - name: store-hub
        image: store-hub:latest
        env:
        - name: TOOLS_UNIVERSE_BASE_URL
          valueFrom:
            configMapKeyRef:
              name: store-hub-config
              key: api_url
        - name: TOOLS_UNIVERSE_API_KEY
          valueFrom:
            secretKeyRef:
              name: store-hub-secrets
              key: api_key
```

## Conclusion

This architecture provides:
- ✅ Clean separation of concerns
- ✅ Easy to test and maintain
- ✅ SOLID principles throughout
- ✅ Ready for microservice extraction
- ✅ Professional, scalable codebase

---

**Architecture**: Domain-Driven Design (DDD)  
**Framework**: FastAPI + Pydantic  
**Status**: Production Ready ✅

