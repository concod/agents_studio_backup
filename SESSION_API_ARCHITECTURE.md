# Session API Architecture Documentation

## Overview

The session system in this application provides a way to track and maintain conversation history across multiple requests. Sessions are identified by unique UUIDs (`session_id`) and are used for:
- **Conversation History Management**: Storing chat history in cache
- **Request Logging**: Tracking all interactions in the database
- **User Tracking**: Identifying users across multiple requests
- **Analytics**: Analyzing user interactions and agent performance

---

## Session Architecture

### 1. Session Creation Flow

Sessions are **implicitly created** - there's no explicit "create session" endpoint. Sessions are created automatically when:

1. **First Request**: When a user makes their first request without an existing `session_id` cookie
2. **Session ID Generation**: A new UUID is generated: `str(uuid.uuid4())`
3. **Cookie Setting**: The `session_id` is set as an HTTP-only cookie in the response
4. **Database Logging**: Session records are created when requests are logged to the database

#### Code Flow:

```python
# Step 1: Generate or retrieve session_id
session_id = request.cookies.get("session_id") or str(uuid.uuid4())

# Step 2: Set cookie (in run_agent function)
if response is not None:
    response.set_cookie(key="session_id", value=session_id, httponly=True)

# Step 3: Log to database (after agent execution)
await session_logs_db.log_session(
    session_id=session_id,
    timestamp=timestamp_utc,
    # ... other fields
)
```

**Key Files:**
- `backend/main.py` (lines 5403, 4410, 5356)
- `backend/DB/async_session_logs.py` (log_session method)

---

## 2. Session API Endpoints

### 2.1 Session Creation Endpoints (Implicit)

These endpoints automatically create sessions if one doesn't exist:

#### `/api/agent/execute` (POST)
- **Purpose**: Execute a single agent
- **Session Handling**: 
  - Retrieves `session_id` from cookies or generates new one
  - Sets cookie in response
  - Logs session to database
- **Location**: `backend/main.py:5376`

#### `/api/agent/langgraph/execute` (POST)
- **Purpose**: Execute LangGraph agent (Text-to-SQL)
- **Session Handling**: Same as `/api/agent/execute`
- **Location**: `backend/main.py:5461`

#### `/api/agent/execute/stream` (POST)
- **Purpose**: Stream agent execution results
- **Session Handling**: Same as `/api/agent/execute`
- **Location**: `backend/main.py:5655`

#### `/api/agent/infer` (POST)
- **Purpose**: Agent inference endpoint
- **Session Handling**: Same pattern
- **Location**: `backend/main.py:5106`

#### `/api/multi-agent/execute` (POST)
- **Purpose**: Execute multi-agent workflows
- **Session Handling**: Same pattern
- **Location**: `backend/main.py:6701`

### 2.2 Session Management Endpoints

#### `/reset-session` (GET)
- **Purpose**: Reset/clear current session
- **Behavior**: 
  - Deletes `session_id` cookie
  - Returns new `session_id` for fresh session
- **Location**: `backend/main.py:3269`
- **Rate Limit**: 10/minute

```python
@app.get("/reset-session")
async def reset_session(request: Request, response: Response):
    session_id = request.cookies.get("session_id") or request.headers.get("X-Session-ID")
    new_session_id = str(uuid.uuid4())
    if session_id:
        delete_session_cookie(response=response, cookie_name="session_id")
        logger.info(f"Cleared session history for session_id: {session_id}")
    return {"message": "Session reset successfully", "new_session_id": new_session_id}
```

### 2.3 Session Logging & Retrieval Endpoints

#### `/api/fetch_session_logs` (GET)
- **Purpose**: Fetch session logs for a specific agent
- **Authentication**: Required (Bearer token)
- **Query Parameters**:
  - `agent_id` (optional): Filter by agent ID
  - `query_all` (bool, default: true): Include all related data
- **Response**: Nested structure with sessions and messages
- **Location**: `backend/api/session_logs.py:145`

#### `/api/filtered-logs` (GET)
- **Purpose**: Get filtered session logs with advanced filtering
- **Authentication**: Required (Bearer token)
- **Query Parameters**:
  - `session_id` (optional): Filter by session ID
  - `from_date` (optional): Start date filter
  - `to_date` (optional): End date filter
  - `agent_id` (optional): Filter by agent ID
- **Location**: `backend/api/session_logs.py:98`

#### `/api/sessions-logs` (GET)
- **Purpose**: Get all sessions with summary data for a specific date
- **Query Parameters**:
  - `date` (optional): Date in YYYY-MM-DD format
- **Response**: List of sessions with statistics
- **Location**: `backend/main.py:7486`

#### `/api/sessions-logs/dates` (GET)
- **Purpose**: Get list of dates for which logs are available
- **Response**: Array of date strings
- **Location**: `backend/main.py:7447`

#### `/api/sessions-logs/stats` (GET)
- **Purpose**: Get session statistics for a specific date
- **Query Parameters**:
  - `date` (optional): Date in YYYY-MM-DD format
- **Response**: Statistics (total requests, sessions, execution times, etc.)
- **Location**: `backend/main.py:7588`

#### `/api/sessions-logs/{session_id}/conversation` (GET)
- **Purpose**: Get conversation data for a specific session
- **Path Parameters**:
  - `session_id`: The session ID to retrieve
- **Query Parameters**:
  - `date` (optional): Date filter
- **Response**: Conversation messages and metadata
- **Location**: `backend/main.py:7656`

#### `/api/session/execution-feedback` (POST)
- **Purpose**: Submit feedback for a specific execution
- **Request Body**:
  ```json
  {
    "execution_id": "uuid-timestamp",
    "session_id": "uuid",
    "feedback": "user feedback text"
  }
  ```
- **Location**: `backend/main.py:4226`

---

## 3. Session Storage Architecture

### 3.1 Database Storage

**Table**: `session_logs` (PostgreSQL)

**Schema** (from `backend/models/session_log.py`):
```python
class SessionLog(Base):
    __tablename__ = "session_logs"
    
    id = Column(Integer, primary_key=True)
    session_id = Column(UUID, nullable=False)  # Primary identifier
    timestamp = Column(DateTime(timezone=True))
    ip_address = Column(INET)
    user_agent = Column(Text)
    request_headers = Column(JSONB)
    user_input = Column(Text)
    response = Column(JSONB)
    execution_time_seconds = Column(Float)
    agent_id = Column(UUID)
    agent_name = Column(String(100))
    endpoint = Column(String(255))
    method = Column(String(10))
    execution_id = Column(String(120))
    feedback = Column(Text)
    created_at = Column(DateTime(timezone=True))
```

**Database Access Layer**:
- `backend/DB/async_session_logs.py` - Async database operations
- `backend/services/session_log_service.py` - Service layer for session logs

### 3.2 Cache Storage (Conversation History)

**Cache Key Format**: `history:{agentId}:{session_id}`

**Purpose**: Store conversation history for agents with memory enabled

**Example**:
```python
cache_key = f"history:{agentId}:{session_id}"
history_data = cache.get(cache_key)
history = json.loads(history_data) if history_data else []
```

**Location**: `backend/main.py:4482`

**Cache Backend**: 
- Redis (if available)
- DiskCache (fallback)

### 3.3 File Storage (Legacy/Backup)

**Location**: `request_logs/session_logs_{date}.json`

**Format**: JSON file with structure:
```json
{
  "session_id_1": [
    {
      "timestamp": "2025-01-01T10:00:00Z",
      "user_input": "...",
      "response": {...},
      ...
    }
  ],
  "session_id_2": [...]
}
```

**Location**: `backend/main.py:5338`

---

## 4. Session Usage Patterns

### 4.1 Conversation History Management

Sessions are used to maintain conversation context:

```python
# Retrieve history
cache_key = f"history:{agentId}:{session_id}"
history_data = cache.get(cache_key)
history = json.loads(history_data) if history_data else []

# Update history after agent execution
history.append({
    "role": "user",
    "content": userInput
})
history.append({
    "role": "assistant", 
    "content": agent_response
})

# Store back to cache
cache.setex(cache_key, 3600, json.dumps(history))  # 1 hour TTL
```

### 4.2 Request Logging

Every request is logged with session context:

```python
await session_logs_db.log_session(
    session_id=session_id,
    timestamp=timestamp_utc,
    ip_address=request.client.host,
    user_agent=request.headers.get("user-agent"),
    user_input=userInput,
    response=response_data,
    execution_time_seconds=execution_time,
    agent_id=agentId,
    agent_name=agent_name,
    execution_id=execution_id
)
```

### 4.3 Session Identification

Sessions are identified via:
1. **HTTP Cookies**: `session_id` cookie (primary method)
2. **Request Headers**: `X-Session-ID` header (fallback)
3. **Request Body**: Some endpoints accept `session_id` in JSON body

**Retrieval Pattern**:
```python
session_id = (
    request.cookies.get("session_id") or 
    request.headers.get("X-Session-ID") or
    json_data.get("session_id") or
    str(uuid.uuid4())  # Generate new if none exists
)
```

---

## 5. Session Lifecycle

### 5.1 Creation
1. User makes first request without `session_id` cookie
2. System generates UUID: `str(uuid.uuid4())`
3. Cookie is set in response: `response.set_cookie(key="session_id", value=session_id, httponly=True)`
4. First log entry is created in database

### 5.2 Continuation
1. Subsequent requests include `session_id` cookie
2. System retrieves existing session_id from cookie
3. New log entries are added to same session
4. Conversation history is retrieved from cache using session_id

### 5.3 Reset
1. User calls `/reset-session` endpoint
2. Cookie is deleted
3. New session_id is generated
4. Old session data remains in database (for analytics) but is no longer active

### 5.4 Expiration
- **Cache**: Conversation history expires after 1 hour (3600 seconds)
- **Database**: Session logs are permanent (no automatic deletion)
- **Cookies**: Browser-managed (typically session-based or expires with browser session)

---

## 6. Session Security & Privacy

### 6.1 Cookie Security
- **HttpOnly**: Prevents JavaScript access (XSS protection)
- **SameSite**: Not explicitly set (defaults to browser behavior)
- **Secure**: Not explicitly set (should be enabled in production with HTTPS)

### 6.2 Session Isolation
- Each `session_id` is a unique UUID (very low collision probability)
- Sessions are isolated by `session_id` in database queries
- Cache keys include both `agentId` and `session_id` for isolation

### 6.3 Data Privacy
- Session logs include IP addresses and user agents
- Request headers are stored (may contain sensitive data)
- No automatic PII scrubbing (should be added for production)

---

## 7. Integration Points

### 7.1 Agent Execution
- All agent execution endpoints use sessions
- Session ID is passed to `run_agent()` function
- Used for conversation history retrieval

### 7.2 Multi-Agent Workflows
- Multi-agent executor uses sessions for workflow tracking
- Cache key: `history:{multi_agent_id}:{session_id or user_email}`

### 7.3 Analytics & Reporting
- Session logs are queried for analytics
- Used to generate reports on agent usage
- Tracks user engagement metrics

### 7.4 Feedback System
- Feedback is associated with `session_id` and `execution_id`
- Stored in `session_logs` table

---

## 8. Key Components

### 8.1 Database Layer
- **`AsyncSessionLogsDB`** (`backend/DB/async_session_logs.py`)
  - `log_session()`: Insert session log
  - `get_session_conversation()`: Retrieve conversation
  - `get_sessions_by_date()`: Get sessions for date
  - `update_feedback()`: Update feedback

### 8.2 Service Layer
- **`SessionLogService`** (`backend/services/session_log_service.py`)
  - `get_session_logs()`: Query logs with filters
  - `query_session_logs()`: Query logs for agent

### 8.3 API Layer
- **`session_logs_router`** (`backend/api/session_logs.py`)
  - FastAPI router for session log endpoints
  - Includes authentication middleware

### 8.4 Models
- **`SessionLog`** (`backend/models/session_log.py`)
  - SQLAlchemy model for database table

---

## 9. Best Practices & Recommendations

### 9.1 Current Implementation
✅ **Good Practices**:
- UUID-based session IDs (no collisions)
- HttpOnly cookies (XSS protection)
- Async database operations
- Comprehensive logging

⚠️ **Areas for Improvement**:
- Add `Secure` flag to cookies (HTTPS only)
- Add `SameSite` cookie attribute
- Implement session expiration/cleanup
- Add PII scrubbing for logs
- Add rate limiting per session
- Implement session validation middleware

### 9.2 Usage Guidelines

1. **Always check for existing session_id** before generating new one
2. **Set cookie in response** when creating new session
3. **Log all requests** to database for analytics
4. **Use cache for conversation history** (not database)
5. **Include session_id in responses** for client-side tracking

---

## 10. Example Flows

### 10.1 First Request Flow
```
1. Client → POST /api/agent/execute (no cookie)
2. Server generates: session_id = uuid4()
3. Server executes agent
4. Server sets cookie: session_id
5. Server logs to database
6. Server returns response with session_id
```

### 10.2 Subsequent Request Flow
```
1. Client → POST /api/agent/execute (with session_id cookie)
2. Server retrieves: session_id from cookie
3. Server retrieves history from cache: history:{agentId}:{session_id}
4. Server executes agent with history context
5. Server updates cache with new conversation
6. Server logs to database
7. Server returns response
```

### 10.3 Session Reset Flow
```
1. Client → GET /reset-session (with session_id cookie)
2. Server deletes cookie
3. Server generates: new_session_id = uuid4()
4. Server returns: {"new_session_id": "..."}
5. Client stores new_session_id (localStorage or sets cookie)
```

---

## 11. API Endpoint Summary

| Endpoint | Method | Purpose | Auth Required |
|----------|--------|---------|---------------|
| `/api/agent/execute` | POST | Execute agent (creates session) | API Key |
| `/api/agent/langgraph/execute` | POST | Execute LangGraph agent | API Key |
| `/api/agent/execute/stream` | POST | Stream agent execution | API Key |
| `/api/multi-agent/execute` | POST | Execute multi-agent | API Key |
| `/reset-session` | GET | Reset session | No |
| `/api/fetch_session_logs` | GET | Fetch logs by agent | Yes |
| `/api/filtered-logs` | GET | Filtered logs query | Yes |
| `/api/sessions-logs` | GET | Get sessions by date | No |
| `/api/sessions-logs/dates` | GET | Get available dates | No |
| `/api/sessions-logs/stats` | GET | Get statistics | No |
| `/api/sessions-logs/{session_id}/conversation` | GET | Get conversation | No |
| `/api/session/execution-feedback` | POST | Submit feedback | Yes |

---

## 12. Related Files

### Core Files
- `backend/main.py` - Main API endpoints and session creation logic
- `backend/DB/async_session_logs.py` - Database operations
- `backend/services/session_log_service.py` - Service layer
- `backend/api/session_logs.py` - API router
- `backend/models/session_log.py` - Database model

### Supporting Files
- `backend/database.py` - Database connection
- `backend/DB/runtime.py` - Async database runtime
- `backend/utils.py` - Utility functions

---

## Conclusion

The session system is designed to be **stateless at the API level** but **stateful for user experience**. Sessions are automatically created, managed via cookies, and used extensively for:
- Conversation history management
- Request logging and analytics
- User tracking across requests
- Feedback collection

The architecture supports both authenticated and unauthenticated users, with sessions being the primary mechanism for maintaining context across multiple interactions.

