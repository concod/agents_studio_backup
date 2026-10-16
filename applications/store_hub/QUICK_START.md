# Store Hub - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Understanding the Structure

```
store_hub/
├── api/              → FastAPI routes (HTTP layer)
├── application/      → Business logic (Use cases)
├── domain/           → Core logic (Repositories)
├── infrastructure/   → External APIs & data
└── schemas/          → Request/Response models
```

### 2. Making Changes

#### Adding a New Endpoint

**Step 1**: Add route in `api/routes.py`

```python
@router.get("/new-endpoint")
@limiter.limit("60/minute")
async def get_new_data(
    request: Request,
    param: str = Query(None),
    db: AsyncSession = Depends(get_db),
    user: Dict[str, Any] = Depends(get_tenant_auth_user),
    store_service: StoreService = Depends(lambda: get_store_service())
):
    """Your endpoint documentation"""
    return await store_service.get_new_data(param)
```

**Step 2**: Add method in `application/store_service.py`

```python
async def get_new_data(self, param: Optional[str] = None) -> NewDataResponse:
    """Business logic for new data"""
    data = await self.store_repository.get_new_data(param)
    return NewDataResponse(**data)
```

**Step 3**: Add method in `domain/repositories.py` (interface)

```python
@abstractmethod
async def get_new_data(self, param: Optional[str] = None) -> Dict[str, Any]:
    """Get new data"""
    pass
```

**Step 4**: Implement in `domain/store_repository_impl.py`

```python
async def get_new_data(self, param: Optional[str] = None) -> Dict[str, Any]:
    """Implementation"""
    if self.use_mock_data:
        return self.mock_provider.get_new_data()
    return await self.tools_universe_client.fetch_new_data(param)
```

**Step 5**: Add schemas in `schemas/responses.py`

```python
class NewDataResponse(BaseModel):
    """Response model for new data"""
    data: List[str]
    count: int
```

Done! ✅

#### Adding Mock Data

Edit `infrastructure/data/mock_data.py`:

```python
@staticmethod
def get_new_data() -> Dict[str, Any]:
    """Get mock new data."""
    return {
        "data": ["item1", "item2"],
        "count": 2
    }
```

#### Adding External API Call

Edit `infrastructure/clients/tools_universe_client.py`:

```python
async def fetch_new_data(self, param: Optional[str] = None) -> Dict[str, Any]:
    """Fetch new data from external API"""
    endpoint = f"{self.base_url}/internal/new_endpoint"
    
    async with httpx.AsyncClient(timeout=self.timeout) as client:
        response = await client.post(
            endpoint,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            },
            json={"param": param}
        )
        response.raise_for_status()
        return response.json()
```

### 3. Testing Your Changes

#### Test with Mock Data

```python
# In dependencies.py, temporarily set:
store_service = get_store_service(use_mock_data=True)
```

#### Test with Real API

```python
# Set environment variables
export TOOLS_UNIVERSE_BASE_URL="https://tools-universe.devs.impact-agents.ai"
export TOOLS_UNIVERSE_API_KEY="your_key"

# In dependencies.py:
store_service = get_store_service(use_mock_data=False)
```

#### Test with cURL

```bash
# Get your auth token first
TOKEN="your_bearer_token"

# Test dashboard
curl -X GET "http://localhost:8000/api/agents/custom/store-hub/graph?district_id=Miller" \
  -H "Authorization: Bearer $TOKEN"

# Test store deep dive
curl -X GET "http://localhost:8000/api/agents/custom/store-hub/store/graph?district_id=Miller&store_id=0049" \
  -H "Authorization: Bearer $TOKEN"

# Test all stores
curl -X GET "http://localhost:8000/api/agents/custom/store-hub/all-stores?district_id=Miller" \
  -H "Authorization: Bearer $TOKEN"
```

### 4. Common Tasks

#### Switch Between Mock and Real Data

```python
# In api/dependencies.py

# For development (mock data)
def get_store_service() -> StoreService:
    return get_store_service(use_mock_data=True)

# For production (real API)
def get_store_service() -> StoreService:
    return get_store_service(use_mock_data=False)
```

#### Add Request Validation

```python
# In schemas/requests.py
class NewRequest(BaseModel):
    """Request with validation"""
    param: str = Field(..., min_length=1, max_length=100)
    optional_param: Optional[int] = Field(None, ge=0, le=1000)
```

#### Add Response Formatting

```python
# In schemas/responses.py
class NewResponse(BaseModel):
    """Response with formatting"""
    data: List[str]
    count: int = Field(..., ge=0)
    timestamp: datetime = Field(default_factory=datetime.now)
```

#### Handle Errors

```python
# In infrastructure/clients/tools_universe_client.py
try:
    response = await client.post(...)
    response.raise_for_status()
    return response.json()
except httpx.TimeoutException:
    raise HTTPException(status_code=504, detail="Request timed out")
except httpx.HTTPStatusError as e:
    raise HTTPException(status_code=e.response.status_code, detail=str(e))
```

### 5. Debugging

#### Enable Logging

```python
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# In your code
logger.debug(f"Processing request with param={param}")
logger.info(f"Fetched {len(data)} items")
logger.error(f"Failed to fetch data: {error}")
```

#### Check Router Registration

```python
# In main.py, after app setup
print(app.routes)  # Should show Store Hub routes
```

#### Verify Dependencies

```python
# Test dependency injection
from applications.store_hub.api.dependencies import get_store_service

service = get_store_service(use_mock_data=True)
result = await service.get_dashboard("Miller", "0049")
print(result)
```

### 6. Best Practices

#### ✅ DO

- Keep functions small (< 30 lines)
- Use type hints everywhere
- Add docstrings to all public methods
- Use Pydantic models for validation
- Handle errors at the appropriate layer
- Write tests for new features
- Use dependency injection
- Follow SOLID principles

#### ❌ DON'T

- Mix business logic in routes
- Make database calls from services
- Skip error handling
- Hardcode configuration values
- Bypass the layered architecture
- Modify interfaces without updating implementations
- Add circular dependencies

### 7. File Checklist

When adding a new feature, you may need to modify:

- [ ] `api/routes.py` - Add HTTP endpoint
- [ ] `api/dependencies.py` - Add dependency if needed
- [ ] `application/store_service.py` - Add business logic
- [ ] `domain/repositories.py` - Add interface method
- [ ] `domain/store_repository_impl.py` - Implement method
- [ ] `infrastructure/clients/tools_universe_client.py` - Add API call
- [ ] `infrastructure/data/mock_data.py` - Add mock data
- [ ] `schemas/requests.py` - Add request model
- [ ] `schemas/responses.py` - Add response model

### 8. Troubleshooting

#### Import Errors

```python
# Make sure you're in the backend directory
cd /path/to/backend

# Check Python path
import sys
print(sys.path)

# Should include backend directory
```

#### Circular Dependencies

```python
# Move imports inside functions
def get_store_service():
    from applications.store_hub.application.store_service import StoreService
    # ...
```

#### Type Errors

```python
# Use Optional for nullable fields
from typing import Optional

param: Optional[str] = None
```

#### Pydantic Validation Errors

```python
# Check your model matches the data
try:
    response = DashboardResponse(**data)
except ValidationError as e:
    print(e.json())  # Shows which fields failed
```

### 9. Useful Commands

```bash
# Check code style
cd backend
python -m pylint applications/store_hub/

# Format code
python -m black applications/store_hub/

# Type checking
python -m mypy applications/store_hub/

# Run tests
pytest applications/store_hub/tests/

# Start server
python main.py

# Check routes
python -c "from main import app; print([r.path for r in app.routes])"
```

### 10. Getting Help

#### Documentation

- `README.md` - Complete architecture guide
- `ARCHITECTURE.md` - Visual diagrams and flows
- `REFACTORING_COMPLETE.md` - Migration details

#### Code Examples

Look at existing endpoints:
- `get_dashboard()` - Simple use case
- `get_store_deep_dive()` - Complex use case
- `get_all_stores()` - Mock data use case

#### Common Patterns

```python
# Pattern 1: Simple data fetch
async def get_simple_data(self, param: str) -> SimpleResponse:
    data = await self.repository.get_data(param)
    return SimpleResponse(**data)

# Pattern 2: Data transformation
async def get_transformed_data(self, param: str) -> TransformedResponse:
    raw_data = await self.repository.get_data(param)
    transformed = self._transform(raw_data)
    return TransformedResponse(**transformed)

# Pattern 3: Multiple data sources
async def get_combined_data(self, param: str) -> CombinedResponse:
    data1 = await self.repository.get_data1(param)
    data2 = await self.repository.get_data2(param)
    combined = {**data1, **data2}
    return CombinedResponse(**combined)
```

## 🎯 Quick Reference

### Layer Responsibilities

| Layer | Responsibility | Example |
|-------|---------------|---------|
| API | HTTP handling | `routes.py` |
| Application | Business logic | `store_service.py` |
| Domain | Core logic | `repositories.py` |
| Infrastructure | External deps | `clients/`, `data/` |
| Schemas | Data contracts | `requests.py`, `responses.py` |

### Dependency Flow

```
Routes → Service → Repository → Client/Data Provider
```

### Error Handling

```
Client → Infrastructure → Domain → Application → API → User
         (catch)         (catch)   (catch)      (format)
```

## 📚 Next Steps

1. Read `README.md` for detailed architecture
2. Check `ARCHITECTURE.md` for visual diagrams
3. Review existing code in each layer
4. Try adding a simple endpoint
5. Write tests for your changes

Happy coding! 🚀

---

**Need Help?** Check the comprehensive documentation in `README.md` or `ARCHITECTURE.md`

