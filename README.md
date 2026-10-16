# IAgent Studio

IAgent Studio is a web application for creating, managing, and deploying AI agents, built with FastAPI and a dynamic vanilla JavaScript frontend.

## Features

*   **Agent Management:** Create, configure, edit, duplicate, and delete AI agents.
*   **Tool Integration:** Define and manage custom tools (using OpenAPI schemas) or utilize built-in integrations.
*   **Agent Configuration:** Specify LLM provider, model, API keys, role, backstory, instructions, and select tools for each agent.
*   **Marketplace:** Discover and import pre-built agents (requires `/api/marketplace/agents` endpoint).
*   **User Management:** Placeholder for managing users and roles.
*   **Settings:** Placeholder for managing account and subscription details.
*   **Dynamic Frontend:** Single Page Application (SPA) experience using vanilla JavaScript for page loading and UI updates.
*   **Notifications:** In-app notification system.

## Technology Stack

*   **Backend:** Python, FastAPI
*   **Frontend:** HTML, CSS, Vanilla JavaScript
*   **Data Storage:** JSON files (for agents, tools, notifications)

## Project Structure

```
/
├── static/
│   ├── css/             # CSS stylesheets (main, marketplace, users, settings, etc.)
│   ├── data/            # JSON data files (e.g., notifications.json)
│   ├── images/          # Icons and illustrations
│   ├── js/              # JavaScript files (main.js, marketplace.js)
│   ├── pages/           # HTML fragments for different pages (agents.html, tools.html, etc.)
│   └── index.html       # Main HTML shell
├── tool_schemas/        # Directory for storing tool OpenAPI schemas (JSON)
├── agents.json          # Agent data store
├── tools.json           # Built-in tool data store
├── custom_tools.json    # Custom tool data store
├── main.py              # FastAPI application entry point
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## Setup and Running

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: You might need to create `requirements.txt` if it doesn't exist. Based on `main.py`, it should at least contain `fastapi`, `uvicorn[standard]`, `pydantic`)*

4.  **Run the FastAPI application:**
    ```bash
    uvicorn main:app --reload
    ```
    The `--reload` flag enables auto-reloading when code changes, useful for development.

5.  **Access the application:**
    Open your web browser and navigate to `http://localhost:8002` (or the address provided by uvicorn).

## How to Contribute

(Optional: Add guidelines for contributing if this is an open project).

---
*This README was generated based on the project structure and features observed.* 


# Session Logs

# Session Logs Management System

A FastAPI application for managing and querying session logs with PostgreSQL integration using SQLAlchemy ORM.

## 🚀 Setup and Running the Server

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd session-logs-system
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure the database**
   - Create a `.env` file in the project root
   - Add your PostgreSQL connection string:
     ```
     DATABASE_URL=postgresql://username:password@localhost:5432/dbname
     ```
   - Set other environment variables if needed (JWT_SECRET, etc.)

5. **Run database migrations**
   ```bash
   alembic upgrade head
   ```

6. **Start the development server**
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   The API will be available at `http://localhost:8000`
   
   Interactive API documentation will be available at:
   - Swagger UI: `http://localhost:8000/docs`
   - ReDoc: `http://localhost:8000/redoc`

## 🔐 Authentication

All API endpoints (except login) require authentication using JWT Bearer tokens.

## 📚 API Endpoints

### 1. Login

**Endpoint:** `POST /api/sessions-login`

Authenticate a user and retrieve an access token.

**Request Body:**
```json
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "user_name": "John Doe",
  "user_email": "user@example.com",
  "user_picture": "https://example.com/avatar.jpg"
}
```

### 2. Fetch Session Logs

**Endpoint:** `GET /api/fetch_session_logs`

Fetch session logs for a specific agent.

**Query Parameters:**
- `agent_id` (string, optional): Filter by agent ID
- `query_all` (boolean, default: true): Whether to include all related data

**Headers:**
```
Authorization: Bearer <access_token>
```

**Response:**
```json
{
  "agent_id": "550e8400-e29b-41d4-a716-446655440000",
  "agent_name": "Agent 1",
  "sessions": [
    {
      "session_id": "123e4567-e89b-12d3-a456-426614174000",
      "start_time": "2025-06-01T10:00:00",
      "end_time": "2025-06-01T11:30:00",
      "status": "completed",
      "messages": [
        {
          "message_id": "msg_001",
          "timestamp": "2025-06-01T10:05:23",
          "content": "Session started",
          "type": "system"
        }
      ]
    }
  ]
}
```

### 3. Filtered Logs

**Endpoint:** `GET /api/filtered-logs`

Fetch logs with advanced filtering options.

**Query Parameters:**
- `session_id` (string, optional): Filter by session ID
- `from_date` (datetime, optional): Start date for filtering
- `to_date` (datetime, optional): End date for filtering
- `agent_id` (string, optional): Filter by agent ID

**Headers:**
```
Authorization: Bearer <access_token>
```

**Example Request:**
```
GET /api/filtered-logs?from_date=2025-06-01T00:00:00&to_date=2025-06-30T23:59:59&agent_id=550e8400-e29b-41d4-a716-446655440000
```

**Response:**
```json
[
  {
    "session_id": "123e4567-e89b-12d3-a456-426614174000",
    "agent_id": "550e8400-e29b-41d4-a716-446655440000",
    "start_time": "2025-06-01T10:00:00",
    "end_time": "2025-06-01T11:30:00",
    "status": "completed",
    "created_at": "2025-06-01T10:00:00"
  }
]
```

## 🛠 Development

- **Code Formatting:**
  ```bash
  black .
  isort .
  ```

- **Linting:**
  ```bash
  flake8 .
  ```

- **Running Tests:**
  ```bash
  # Add test command once available
  ```

## 📝 Notes

- All timestamps are in UTC
- JWT tokens expire after 30 minutes
- Ensure proper error handling in client applications
