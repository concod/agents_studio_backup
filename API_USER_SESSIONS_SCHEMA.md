# API Schema: `/api/user-sessions`

## Endpoint Information

- **URL**: `/api/user-sessions`
- **Method**: `GET`
- **Authentication**: Required (Bearer Token)
- **Content-Type**: `application/json`

---

## Request Schema

### Headers

| Header | Type | Required | Description |
|--------|------|----------|-------------|
| `Authorization` | string | Yes | Bearer token from `/api/sessions-login` |
| `Content-Type` | string | No | `application/json` (optional) |

**Example:**
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Query Parameters

| Parameter | Type | Required | Default | Min | Max | Description |
|-----------|------|----------|---------|-----|-----|-------------|
| `limit` | integer | No | 100 | 1 | 1000 | Maximum number of sessions to return |
| `offset` | integer | No | 0 | 0 | - | Number of sessions to skip for pagination |

**Example:**
```
GET /api/user-sessions?limit=50&offset=0
```

---

## Response Schema

### Success Response (200 OK)

```json
{
  "user_email": "string",
  "total_sessions": "integer",
  "sessions": [
    {
      "session_id": "string (UUID)",
      "first_request": "string (ISO 8601 datetime)",
      "last_request": "string (ISO 8601 datetime)",
      "request_count": "integer",
      "avg_execution_time": "float",
      "agents_used": "string",
      "conversation": {
        "session_id": "string",
        "messages": [
          {
            "id": "integer",
            "timestamp": "string",
            "user_input": "string | null",
            "response": "string | null",
            "execution_time_seconds": "float | null",
            "feedback": "string | null",
            "ip_address": "string | null",
            "execution_id": "string | null"
          }
        ]
      } | null
    }
  ]
}
```

### Pydantic Models

#### `UserSessionsResponse`
```python
class UserSessionsResponse(BaseModel):
    user_email: str                    # Email of the authenticated user
    total_sessions: int                 # Total number of sessions returned
    sessions: List[UserSessionSummary]  # List of session summaries
```

#### `UserSessionSummary`
```python
class UserSessionSummary(BaseModel):
    session_id: str                     # UUID of the session
    first_request: str                  # ISO 8601 datetime of first request
    last_request: str                   # ISO 8601 datetime of last request
    request_count: int                  # Number of requests in this session
    avg_execution_time: float          # Average execution time in seconds
    agents_used: str                   # Comma-separated list of agent names
    conversation: Optional[Dict[str, Any]]  # Full conversation details (optional)
```

#### `Conversation` (nested in `conversation` field)
```python
{
    "session_id": "string",
    "messages": [
        {
            "id": int,
            "timestamp": str,
            "user_input": Optional[str],
            "response": Optional[str],
            "execution_time_seconds": Optional[float],
            "feedback": Optional[str],
            "ip_address": Optional[str],
            "execution_id": Optional[str]
        }
    ]
}
```

---

## Error Response Schemas

### 400 Bad Request - User Email Not Found

```json
{
  "detail": "User email not found in token"
}
```

**Schema:**
```python
{
    "detail": str  # Error message
}
```

### 401 Unauthorized - Missing/Invalid Token

```json
{
  "detail": "Could not validate credentials for tenant access"
}
```

**Schema:**
```python
{
    "detail": str  # Error message
}
```

### 500 Internal Server Error

```json
{
  "detail": "Error fetching sessions: <error_message>"
}
```

**Schema:**
```python
{
    "detail": str  # Error message with details
}
```

---

## Complete Example

### Request

```http
GET /api/user-sessions?limit=10&offset=0 HTTP/1.1
Host: localhost:8000
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ1c2VyQGV4YW1wbGUuY29tIiwiZXhwIjoxNjE2MjM5MDIyfQ.signature
Content-Type: application/json
```

### Success Response (200 OK)

```json
{
  "user_email": "user@example.com",
  "total_sessions": 2,
  "sessions": [
    {
      "session_id": "123e4567-e89b-12d3-a456-426614174000",
      "first_request": "2025-01-01T10:00:00Z",
      "last_request": "2025-01-01T11:30:00Z",
      "request_count": 5,
      "avg_execution_time": 2.5,
      "agents_used": "Chatbot Agent, Support Agent",
      "conversation": {
        "session_id": "123e4567-e89b-12d3-a456-426614174000",
        "messages": [
          {
            "id": 1,
            "timestamp": "2025-01-01T10:00:00Z",
            "user_input": "Hello",
            "response": "Hi! How can I help you?",
            "execution_time_seconds": 1.2,
            "feedback": null,
            "ip_address": "192.168.1.1",
            "execution_id": "exec-123"
          },
          {
            "id": 2,
            "timestamp": "2025-01-01T10:05:00Z",
            "user_input": "What's the weather?",
            "response": "I can help you check the weather.",
            "execution_time_seconds": 2.1,
            "feedback": null,
            "ip_address": "192.168.1.1",
            "execution_id": "exec-124"
          }
        ]
      }
    },
    {
      "session_id": "987e6543-e21b-43d2-b654-321987654321",
      "first_request": "2025-01-02T14:00:00Z",
      "last_request": "2025-01-02T14:15:00Z",
      "request_count": 3,
      "avg_execution_time": 1.8,
      "agents_used": "Support Agent",
      "conversation": null
    }
  ]
}
```

---

## TypeScript/JavaScript Types

```typescript
interface UserSessionsResponse {
  user_email: string;
  total_sessions: number;
  sessions: UserSessionSummary[];
}

interface UserSessionSummary {
  session_id: string;
  first_request: string;  // ISO 8601 datetime
  last_request: string;   // ISO 8601 datetime
  request_count: number;
  avg_execution_time: number;
  agents_used: string;
  conversation: Conversation | null;
}

interface Conversation {
  session_id: string;
  messages: Message[];
}

interface Message {
  id: number;
  timestamp: string;
  user_input: string | null;
  response: string | null;
  execution_time_seconds: number | null;
  feedback: string | null;
  ip_address: string | null;
  execution_id: string | null;
}
```

---

## Python Types

```python
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from datetime import datetime

class Message(BaseModel):
    id: int
    timestamp: str
    user_input: Optional[str] = None
    response: Optional[str] = None
    execution_time_seconds: Optional[float] = None
    feedback: Optional[str] = None
    ip_address: Optional[str] = None
    execution_id: Optional[str] = None

class Conversation(BaseModel):
    session_id: str
    messages: List[Message]

class UserSessionSummary(BaseModel):
    session_id: str
    first_request: str
    last_request: str
    request_count: int
    avg_execution_time: float
    agents_used: str
    conversation: Optional[Dict[str, Any]] = None

class UserSessionsResponse(BaseModel):
    user_email: str
    total_sessions: int
    sessions: List[UserSessionSummary]
```

---

## Validation Rules

### Query Parameters

- `limit`: 
  - Must be an integer
  - Minimum: 1
  - Maximum: 1000
  - Default: 100

- `offset`: 
  - Must be an integer
  - Minimum: 0
  - Default: 0

### Response Fields

- `user_email`: 
  - Must be a valid email format
  - Extracted from JWT token

- `session_id`: 
  - Must be a valid UUID format

- `first_request` / `last_request`: 
  - Must be ISO 8601 datetime format
  - Example: `2025-01-01T10:00:00Z`

- `request_count`: 
  - Must be a positive integer

- `avg_execution_time`: 
  - Must be a non-negative float
  - Represents seconds

---

## Authentication Flow

1. **Login** to get access token:
   ```
   POST /api/sessions-login
   Body: { "email": "user@example.com", "password": "password" }
   Response: { "access_token": "..." }
   ```

2. **Use token** in Authorization header:
   ```
   GET /api/user-sessions
   Header: Authorization: Bearer <access_token>
   ```

---

## Notes

- The endpoint extracts `user_email` from the JWT token in the Authorization header
- Sessions are matched by extracting `user_email` from JWT tokens stored in `request_headers` JSONB field
- The `conversation` field may be `null` if conversation details are not available
- Pagination is supported via `limit` and `offset` parameters
- The endpoint uses tenant authentication (`get_tenant_auth_user`) which validates against the `log_users` table

