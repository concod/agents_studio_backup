# User Sessions API Documentation

## Overview

This document describes the new API endpoint for fetching all sessions related to a logged-in user.

## Endpoint

### `GET /api/user-sessions`

Get all sessions for the currently authenticated user (uses simple login/register, not OAuth).

**Authentication**: Required (Bearer token from `/api/sessions-login`)

**Query Parameters**:
- `limit` (int, optional): Maximum number of sessions to return. Default: 100, Max: 1000
- `offset` (int, optional): Number of sessions to skip for pagination. Default: 0

**Request Headers**:
```
Authorization: Bearer <access_token>
```

**Note**: This endpoint uses the same authentication as store-hub APIs - simple email/password login via `/api/sessions-login`, not Google OAuth.

**Response**:
```json
{
  "user_email": "user@example.com",
  "total_sessions": 5,
  "sessions": [
    {
      "session_id": "123e4567-e89b-12d3-a456-426614174000",
      "first_request": "2025-01-01T10:00:00Z",
      "last_request": "2025-01-01T11:30:00Z",
      "request_count": 10,
      "avg_execution_time": 2.5,
      "agents_used": "Agent 1, Agent 2",
      "conversation": {
        "session_id": "...",
        "messages": [...]
      }
    }
  ]
}
```

## Implementation Details

### How Sessions are Linked to Users

Currently, sessions are linked to users by extracting `user_email` from JWT tokens stored in the `request_headers` JSONB field of the `session_logs` table.

**Process**:
1. Query all session logs
2. Extract `Authorization` header from `request_headers`
3. Decode JWT token to get `user_email` (from `sub` or `email` claim)
4. Match sessions where `user_email` matches the authenticated user
5. Group by `session_id` and return session summaries

### Database Method

**Location**: `backend/DB/async_session_logs.py`

**Method**: `get_sessions_by_user_email(user_email: str, limit: int, offset: int)`

This method:
- Queries session logs from database
- Extracts user_email from JWT tokens in request_headers
- Groups sessions by session_id
- Returns session summaries with conversation details

### Performance Considerations

**Current Implementation**:
- Queries all recent logs and filters in Python
- Extracts JWT tokens from request_headers JSONB
- Suitable for small to medium datasets

**Recommended Improvement**:
For better performance, add a `user_email` column to the `session_logs` table:

```sql
ALTER TABLE session_logs 
ADD COLUMN user_email VARCHAR(255);

CREATE INDEX idx_session_logs_user_email ON session_logs(user_email);
```

Then use the `get_sessions_by_user_email_v2()` method which queries directly by `user_email` column.

## Usage Example

### Step 1: Login to get access token
```bash
curl -X POST "http://localhost:8000/api/sessions-login" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "your_password"
  }'
```

### Step 2: Use the access_token to get user sessions
```bash
curl -X GET "http://localhost:8000/api/user-sessions?limit=50&offset=0" \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN"
```

### Python
```python
import requests

# Step 1: Login
login_response = requests.post(
    "http://localhost:8000/api/sessions-login",
    json={
        "email": "user@example.com",
        "password": "your_password"
    }
)
access_token = login_response.json()["access_token"]

# Step 2: Get user sessions
headers = {
    "Authorization": f"Bearer {access_token}"
}

response = requests.get(
    "http://localhost:8000/api/user-sessions",
    headers=headers,
    params={
        "limit": 50,
        "offset": 0
    }
)

sessions = response.json()
print(f"Found {sessions['total_sessions']} sessions for {sessions['user_email']}")
for session in sessions['sessions']:
    print(f"Session {session['session_id']}: {session['request_count']} requests")
```

### JavaScript
```javascript
// Step 1: Login
const loginResponse = await fetch('/api/sessions-login', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    email: 'user@example.com',
    password: 'your_password'
  })
});

const loginData = await loginResponse.json();
const accessToken = loginData.access_token;

// Step 2: Get user sessions
const response = await fetch('/api/user-sessions?limit=50&offset=0', {
  headers: {
    'Authorization': `Bearer ${accessToken}`
  }
});

const data = await response.json();
console.log(`Found ${data.total_sessions} sessions for ${data.user_email}`);
data.sessions.forEach(session => {
  console.log(`Session ${session.session_id}: ${session.request_count} requests`);
});
```

## Error Handling

**401 Unauthorized**: Missing or invalid token
```json
{
  "detail": "Could not validate credentials for tenant access"
}
```

**400 Bad Request**: User email not found in token
```json
{
  "detail": "User email not found in token"
}
```

**400 Bad Request**: User email not found in token
```json
{
  "detail": "User email not found in token"
}
```

**500 Internal Server Error**: Database or processing error
```json
{
  "detail": "Error fetching sessions: <error_message>"
}
```

## Future Enhancements

1. **Add user_email column**: For better query performance
2. **Caching**: Cache user sessions for frequently accessed users
3. **Filtering**: Add filters for date range, agent_id, etc.
4. **Sorting**: Add sorting options (by date, request count, etc.)
5. **Export**: Add ability to export user sessions

## Related Files

- `backend/api/session_logs.py` - API endpoint definition
- `backend/DB/async_session_logs.py` - Database methods
- `backend/services/auth_service.py` - Authentication service
- `backend/models/session_log.py` - Session log model

## Migration Script (Optional)

To add `user_email` column for better performance:

```sql
-- Add user_email column
ALTER TABLE session_logs 
ADD COLUMN user_email VARCHAR(255);

-- Create index for faster queries
CREATE INDEX idx_session_logs_user_email ON session_logs(user_email);

-- Backfill existing data (optional)
-- This would require extracting user_email from request_headers
-- and updating existing records
```

Then update `log_session()` calls to include `user_email` parameter.

