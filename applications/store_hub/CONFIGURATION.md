# Store Hub Configuration Guide

## Environment Variables

The Store Hub application requires the following environment variables to be set in your `.env` file:

### Required Configuration

```bash
# Tools Universe API Base URL
# Default: http://localhost:8001
TOOLS_UNIVERSE_BASE_URL=http://localhost:8001

# Tools Universe API Key
# Get this from your Tools Universe dashboard
TOOLS_UNIVERSE_API_KEY=your_api_key_here
```

## Configuration Options

### Local Development (Recommended)

For local development, use localhost:

```bash
# In your .env file
TOOLS_UNIVERSE_BASE_URL=http://localhost:8001
TOOLS_UNIVERSE_API_KEY=your_local_api_key
```

**Prerequisites**:
- Tools Universe API must be running locally on port 8001
- Start the Tools Universe service before running the Store Hub APIs

### Production/Remote API

If you need to connect to a remote Tools Universe instance:

```bash
# In your .env file
TOOLS_UNIVERSE_BASE_URL=https://tools-universe.devs.impact-agents.ai
TOOLS_UNIVERSE_API_KEY=your_production_api_key
```

## How to Set Up

### Step 1: Create/Update `.env` File

```bash
cd /Users/prachipathrayan/work-repositories/agentic-studio/backend

# Create .env if it doesn't exist
touch .env

# Add the configuration
echo "TOOLS_UNIVERSE_BASE_URL=http://localhost:8001" >> .env
echo "TOOLS_UNIVERSE_API_KEY=your_api_key_here" >> .env
```

### Step 2: Start Tools Universe API (Local)

```bash
# In a separate terminal, start your Tools Universe API
cd /path/to/tools-universe
python main.py  # or however you start it

# It should be running on http://localhost:8001
```

### Step 3: Verify Configuration

```bash
# Check if Tools Universe API is accessible
curl http://localhost:8001/health  # or appropriate health check endpoint
```

### Step 4: Restart Your Application

```bash
# The application will automatically reload and pick up the new environment variables
# If not, restart manually:
cd /Users/prachipathrayan/work-repositories/agentic-studio/backend
python main.py
```

## Testing the Configuration

### Test with cURL

```bash
# Get your auth token first
TOKEN="your_bearer_token"

# Test dashboard endpoint
curl -X GET "http://localhost:8000/api/agents/custom/store-hub/graph?district_id=Miller" \
  -H "Authorization: Bearer $TOKEN"

# Should return data from your local Tools Universe API
```

### Expected Behavior

**With Correct Configuration**:
- ✅ APIs connect to `http://localhost:8001`
- ✅ Data is fetched from your local Tools Universe instance
- ✅ No connection errors

**With Incorrect Configuration**:
- ❌ Connection errors (503)
- ❌ Timeout errors (504)
- ❌ APIs fall back to mock data (if configured)

## Troubleshooting

### Issue: Connection Refused

**Error**: `Connection error calling REVV analysis API: All connection attempts failed`

**Solutions**:
1. Verify Tools Universe API is running:
   ```bash
   curl http://localhost:8001/health
   ```

2. Check the port number:
   ```bash
   # Make sure TOOLS_UNIVERSE_BASE_URL uses the correct port
   # Default is 8001, but yours might be different
   ```

3. Check firewall/network:
   ```bash
   # Ensure localhost connections are allowed
   ```

### Issue: Wrong API Key

**Error**: `401 Unauthorized` or `403 Forbidden`

**Solution**:
```bash
# Update your API key in .env
TOOLS_UNIVERSE_API_KEY=correct_api_key_here
```

### Issue: Wrong Endpoint

**Error**: `404 Not Found`

**Solution**:
- Verify your Tools Universe API has the endpoints:
  - `POST /internal/revv_analysis`
  - `POST /internal/store_deep_dive`

## Using Mock Data (Development)

If you don't have Tools Universe API running locally, you can use mock data:

### Option 1: Environment Variable

```bash
# In your .env file
USE_MOCK_DATA=true
```

Then update `dependencies.py`:
```python
def get_store_service(use_mock_data: bool = None) -> StoreService:
    if use_mock_data is None:
        use_mock_data = os.getenv("USE_MOCK_DATA", "false").lower() == "true"
    repository = get_store_repository(use_mock_data=use_mock_data)
    return StoreService(store_repository=repository)
```

### Option 2: Modify Routes Temporarily

In `api/routes.py`, change:
```python
store_service: StoreService = Depends(lambda: get_store_service(use_mock_data=True))
```

## Default Values

If environment variables are not set, the following defaults are used:

| Variable | Default Value | Description |
|----------|--------------|-------------|
| `TOOLS_UNIVERSE_BASE_URL` | `http://localhost:8001` | Base URL for Tools Universe API |
| `TOOLS_UNIVERSE_API_KEY` | `""` (empty) | API key for authentication |

## API Endpoints

The Store Hub application will call these endpoints on your Tools Universe API:

### 1. REVV Analysis
- **Endpoint**: `POST /internal/revv_analysis`
- **Used by**: `/api/agents/custom/store-hub/graph`
- **Purpose**: Dashboard data with KPIs and analytics

### 2. Store Deep Dive
- **Endpoint**: `POST /internal/store_deep_dive`
- **Used by**: `/api/agents/custom/store-hub/store/graph`
- **Purpose**: Detailed store analysis

### Request Format

Both endpoints expect:
```json
{
  "db_config": {
    "db_type": "bigquery",
    "project_id": "tools-universe-29042025",
    "database": "petsupplyplus",
    "credentials_json": "..."
  },
  "district": "Miller",
  "store_id": "0049"
}
```

## Security Notes

⚠️ **Important**:
- Never commit your `.env` file to version control
- Keep your API keys secure
- Use different API keys for development and production
- Rotate API keys regularly

## Support

For issues with:
- **Store Hub APIs**: Check this documentation
- **Tools Universe API**: Refer to [Tools Universe Documentation](https://tools-universe.devs.impact-agents.ai)
- **Configuration**: See `applications/store_hub/README.md`

---

**Last Updated**: January 14, 2026  
**Configuration Version**: 1.0.0

