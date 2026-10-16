# Applications - Domain-Driven Design Architecture

## Overview

This directory contains **bounded contexts** (applications) for the agentic platform. Each subdirectory represents a separate domain/application following **Domain-Driven Design (DDD)** principles.

## Philosophy

Instead of having all API endpoints in a single monolithic `main.py` file, we organize code by **business domains**. Each application is:

- **Self-contained**: All code for a domain lives in one place
- **Independently testable**: Can be tested without other applications
- **Loosely coupled**: Minimal dependencies on other applications
- **Highly cohesive**: Related functionality grouped together
- **Microservice-ready**: Can be extracted to a separate service easily

## Current Applications

### 1. Store Hub (`store_hub/`)

**Domain**: Store management, analytics, and performance tracking

**APIs**:
- `GET /api/agents/custom/store-hub/graph` - Dashboard data
- `GET /api/agents/custom/store-hub/store/graph` - Store deep dive
- `GET /api/agents/custom/store-hub/all-stores` - All stores list

**Status**: ✅ Complete (Refactored from main.py)

**Documentation**:
- [`store_hub/README.md`](store_hub/README.md) - Complete guide
- [`store_hub/ARCHITECTURE.md`](store_hub/ARCHITECTURE.md) - Visual diagrams
- [`store_hub/QUICK_START.md`](store_hub/QUICK_START.md) - Quick reference

## Architecture Pattern

All applications follow the same layered DDD architecture:

```
application_name/
├── api/                    # API Layer (Presentation)
│   ├── __init__.py
│   ├── routes.py          # FastAPI routes
│   └── dependencies.py    # Dependency injection
├── application/            # Application Layer (Use Cases)
│   ├── __init__.py
│   └── *_service.py       # Business logic orchestration
├── domain/                 # Domain Layer (Business Logic)
│   ├── __init__.py
│   ├── repositories.py    # Repository interfaces
│   └── *_repository_impl.py  # Repository implementations
├── infrastructure/         # Infrastructure Layer (External Dependencies)
│   ├── __init__.py
│   ├── clients/           # External API clients
│   └── data/              # Data providers (mock, cache, etc.)
├── schemas/                # Data Transfer Objects (DTOs)
│   ├── __init__.py
│   ├── requests.py        # Request models
│   └── responses.py       # Response models
├── __init__.py
└── README.md              # Application documentation
```

## SOLID Principles

All applications adhere to SOLID principles:

### Single Responsibility Principle (SRP)
Each class/module has one reason to change:
- Routes handle HTTP
- Services orchestrate business logic
- Repositories manage data access
- Clients handle external APIs

### Open/Closed Principle (OCP)
Open for extension, closed for modification:
- New use cases can be added without changing existing services
- New data sources can be added by implementing repository interfaces

### Liskov Substitution Principle (LSP)
Implementations can be substituted:
- Any repository implementation works with the service
- Mock and real implementations are interchangeable

### Interface Segregation Principle (ISP)
Focused, minimal interfaces:
- Repository interfaces only define necessary methods
- Clients don't depend on methods they don't use

### Dependency Inversion Principle (DIP)
Depend on abstractions, not concretions:
- Services depend on repository interfaces
- Routes depend on service abstractions
- Easy to swap implementations for testing

## Adding a New Application

### Step 1: Create Directory Structure

```bash
cd backend/applications
mkdir -p new_app/{api,application,domain,infrastructure/{clients,data},schemas}
touch new_app/{api,application,domain,infrastructure,infrastructure/clients,infrastructure/data,schemas}/__init__.py
```

### Step 2: Define Schemas

```python
# new_app/schemas/requests.py
from pydantic import BaseModel

class NewRequest(BaseModel):
    param: str

# new_app/schemas/responses.py
class NewResponse(BaseModel):
    data: str
```

### Step 3: Create Repository Interface

```python
# new_app/domain/repositories.py
from abc import ABC, abstractmethod

class INewRepository(ABC):
    @abstractmethod
    async def get_data(self, param: str) -> dict:
        pass
```

### Step 4: Implement Repository

```python
# new_app/domain/new_repository_impl.py
class NewRepository(INewRepository):
    async def get_data(self, param: str) -> dict:
        # Implementation
        return {"data": param}
```

### Step 5: Create Service

```python
# new_app/application/new_service.py
class NewService:
    def __init__(self, repository: INewRepository):
        self.repository = repository
    
    async def get_data(self, param: str) -> NewResponse:
        data = await self.repository.get_data(param)
        return NewResponse(**data)
```

### Step 6: Create Routes

```python
# new_app/api/routes.py
from fastapi import APIRouter

router = APIRouter(prefix="/api/new-app", tags=["New App"])

def create_routes(limiter, get_db, get_tenant_auth_user):
    @router.get("/data")
    @limiter.limit("60/minute")
    async def get_data(request: Request, param: str):
        service = get_new_service()
        return await service.get_data(param)
    
    return router
```

### Step 7: Setup Dependencies

```python
# new_app/api/dependencies.py
def get_new_service() -> NewService:
    repository = NewRepository()
    return NewService(repository)
```

### Step 8: Register in main.py

```python
# backend/main.py
from applications.new_app.api.routes import router, create_routes

new_app_router = create_routes(limiter, get_db, get_tenant_auth_user)
app.include_router(new_app_router, tags=["New App"])
```

### Step 9: Document

Create `new_app/README.md` with:
- Overview
- Architecture
- API documentation
- Examples
- Testing guide

## Testing Applications

### Unit Tests

```python
# Test service with mock repository
async def test_service():
    mock_repo = MockRepository()
    service = NewService(mock_repo)
    result = await service.get_data("test")
    assert result.data == "test"
```

### Integration Tests

```python
# Test full stack
async def test_endpoint():
    response = client.get("/api/new-app/data?param=test")
    assert response.status_code == 200
```

## Benefits of This Architecture

### 1. Maintainability 📈
- Clear separation of concerns
- Easy to locate and fix bugs
- Self-documenting structure

### 2. Testability 🧪
- Each layer can be tested independently
- Easy to mock dependencies
- Fast unit tests

### 3. Scalability 📊
- Easy to add new features
- Can extract to microservice
- Clear boundaries

### 4. Team Collaboration 👥
- Different teams can work on different apps
- Clear interfaces reduce conflicts
- New developers onboard quickly

### 5. Code Reusability ♻️
- Shared patterns across applications
- Common infrastructure components
- Consistent architecture

## Migration Strategy

### From Monolithic main.py

1. **Identify Domain**: Group related endpoints
2. **Create Structure**: Set up DDD folders
3. **Extract Code**: Move code to appropriate layers
4. **Add Tests**: Ensure functionality preserved
5. **Integrate**: Register router in main.py
6. **Deprecate**: Comment out old code
7. **Document**: Add README and guides

### Example: Store Hub Migration

**Before**: 501 lines in main.py  
**After**: 17 well-organized files  
**Result**: 99.2% reduction in main.py, improved maintainability

See [`store_hub/REFACTORING_COMPLETE.md`](store_hub/REFACTORING_COMPLETE.md) for details.

## Best Practices

### DO ✅

- Follow the layered architecture
- Use dependency injection
- Write comprehensive tests
- Document your APIs
- Use type hints everywhere
- Handle errors appropriately
- Keep functions small (< 30 lines)
- Use Pydantic for validation

### DON'T ❌

- Mix concerns across layers
- Skip the repository pattern
- Hardcode configuration
- Bypass dependency injection
- Create circular dependencies
- Modify shared interfaces without coordination
- Add business logic in routes

## Common Patterns

### Pattern 1: Simple CRUD

```python
# Service
async def get_item(self, id: str) -> ItemResponse:
    data = await self.repository.get_item(id)
    return ItemResponse(**data)

async def create_item(self, request: CreateItemRequest) -> ItemResponse:
    data = await self.repository.create_item(request.dict())
    return ItemResponse(**data)
```

### Pattern 2: External API Integration

```python
# Client
async def fetch_external_data(self, param: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{self.base_url}/data?param={param}")
        response.raise_for_status()
        return response.json()
```

### Pattern 3: Mock Data for Development

```python
# Repository
def __init__(self, client=None, use_mock=False):
    self.client = client
    self.use_mock = use_mock
    self.mock_provider = MockDataProvider()

async def get_data(self, param: str) -> dict:
    if self.use_mock:
        return self.mock_provider.get_data()
    return await self.client.fetch_data(param)
```

## Troubleshooting

### Import Errors

```python
# Ensure backend/ is in Python path
import sys
sys.path.insert(0, '/path/to/backend')
```

### Circular Dependencies

```python
# Import inside functions
def get_service():
    from applications.app.service import Service
    return Service()
```

### Type Errors

```python
# Use Optional for nullable fields
from typing import Optional
param: Optional[str] = None
```

## Future Enhancements

### Potential Applications

1. **User Management** (`user_management/`)
   - User CRUD
   - Authentication
   - Authorization

2. **Analytics** (`analytics/`)
   - Metrics aggregation
   - Reporting
   - Dashboards

3. **Notifications** (`notifications/`)
   - Email notifications
   - Push notifications
   - SMS notifications

4. **Audit Logging** (`audit_logging/`)
   - Activity tracking
   - Compliance reporting
   - Security monitoring

### Shared Infrastructure

Consider creating `applications/shared/` for:
- Common clients
- Shared utilities
- Base classes
- Common schemas

## Documentation

Each application should have:

- `README.md` - Complete guide
- `ARCHITECTURE.md` - Visual diagrams (optional)
- `QUICK_START.md` - Quick reference (optional)
- Inline code documentation
- API documentation (OpenAPI/Swagger)

## Monitoring & Observability

### Logging

```python
import logging

logger = logging.getLogger(__name__)

# In your code
logger.info(f"Processing request: {param}")
logger.error(f"Failed to process: {error}")
```

### Metrics

```python
# Add metrics collection
from prometheus_client import Counter, Histogram

request_count = Counter('app_requests_total', 'Total requests')
request_duration = Histogram('app_request_duration_seconds', 'Request duration')
```

### Tracing

```python
# Add distributed tracing
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

with tracer.start_as_current_span("operation_name"):
    # Your code
    pass
```

## Contributing

When adding a new application:

1. Follow the standard DDD structure
2. Apply SOLID principles
3. Write comprehensive tests
4. Document thoroughly
5. Review with team
6. Update this README

## Resources

- [Domain-Driven Design](https://martinfowler.com/bliki/DomainDrivenDesign.html)
- [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- [SOLID Principles](https://en.wikipedia.org/wiki/SOLID)
- [FastAPI Best Practices](https://fastapi.tiangolo.com/tutorial/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

## Contact

For questions about this architecture or adding new applications, please contact the platform team.

---

**Architecture**: Domain-Driven Design (DDD)  
**Framework**: FastAPI + Pydantic  
**Status**: Active Development 🚀

