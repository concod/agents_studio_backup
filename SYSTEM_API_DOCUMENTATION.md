# System Management API Documentation

This document describes the enhanced system management API endpoints that provide comprehensive monitoring and cleanup capabilities for the Agentic Studio backend.

## Overview

The system API provides endpoints to monitor system resources, manage temporary files, and perform maintenance tasks. These endpoints help prevent disk space issues and monitor system health.

## Endpoints

### 1. GET `/api/system/memory` - Comprehensive System Statistics

**Enhanced endpoint that provides detailed system information including memory, disk, CPU, and temporary folder statistics.**

#### Response Example:
```json
{
  "memory": {
    "process_usage_mb": 245.6,
    "process_percent": 1.2,
    "rss_mb": 245.6,
    "vms_mb": 2048.3,
    "active_executors": 0,
    "request_count": 42
  },
  "system_memory": {
    "total_gb": 16.0,
    "available_gb": 8.5,
    "used_gb": 7.5,
    "percent_used": 46.9,
    "free_gb": 8.5
  },
  "disk": {
    "root": {
      "total_gb": 500.0,
      "used_gb": 250.0,
      "free_gb": 250.0,
      "percent_used": 50.0
    },
    "tmp": {
      "total_gb": 500.0,
      "used_gb": 250.0,
      "free_gb": 250.0,
      "percent_used": 50.0
    }
  },
  "cpu": {
    "percent_used": 15.2,
    "core_count": 8,
    "load_average": [1.2, 1.5, 1.8]
  },
  "temp_folders": {
    "counts": {
      "cs_folders": 25,
      "cm_folders": 5,
      "cc_folders": 5,
      "crew_folders": 0,
      "crewai_folders": 2,
      "total_temp_folders": 37
    },
    "cleanup_stats": {
      "scanned": 37,
      "deletable": 30,
      "preserved": 0,
      "total_size_mb": 45.2,
      "patterns_found": {
        "/tmp/cs_*": 25,
        "/tmp/cm_*": 5,
        "/tmp/cc_*": 5
      }
    }
  },
  "gc_stats": {
    "generation_0": 123,
    "generation_1": 45,
    "generation_2": 12,
    "total_collections": 180
  },
  "timestamp": "2025-09-23T18:30:00.000Z",
  "uptime_seconds": 3600.5
}
```

#### Usage:
```bash
curl http://localhost:8000/api/system/memory
```

---

### 2. GET `/api/system/temp-stats` - Temporary Folder Statistics

**Get detailed statistics about temporary folders without running cleanup.**

#### Response Example:
```json
{
  "folder_counts": {
    "cs_folders": 25,
    "cm_folders": 5,
    "cc_folders": 5,
    "crew_folders": 0,
    "crewai_folders": 2,
    "total_temp_folders": 37
  },
  "folder_sizes_mb": {
    "cs_folders": 35.2,
    "cm_folders": 5.1,
    "cc_folders": 4.9,
    "crewai_folders": 0.0
  },
  "cleanup_preview": {
    "deletable_count": 30,
    "deletable_size_mb": 45.2,
    "preserved_count": 0,
    "patterns_found": {
      "/tmp/cs_*": 25,
      "/tmp/cm_*": 5,
      "/tmp/cc_*": 5
    }
  },
  "recommendations": {
    "should_cleanup": false,
    "cleanup_urgency": "low"
  },
  "timestamp": "2025-09-23T18:30:00.000Z"
}
```

#### Usage:
```bash
curl http://localhost:8000/api/system/temp-stats
```

---

### 3. POST `/api/system/cleanup-temp` - Clean Up Temporary Folders

**Clean up old temporary folders created by agent execution.**

#### Parameters:
- `max_age_hours` (int, optional): Delete folders older than this many hours (default: 24)
- `dry_run` (bool, optional): Preview cleanup without actually deleting (default: false)

#### Response Example:
```json
{
  "success": true,
  "dry_run": false,
  "max_age_hours": 24,
  "stats": {
    "scanned": 37,
    "deleted": 30,
    "preserved": 0,
    "errors": 0,
    "total_size_mb": 45.2,
    "patterns_found": {
      "/tmp/cs_*": 25,
      "/tmp/cm_*": 5,
      "/tmp/cc_*": 5
    }
  },
  "message": "Deleted 30 items, freed 45.2MB",
  "timestamp": "2025-09-23T18:30:00.000Z"
}
```

#### Usage:
```bash
# Preview cleanup (dry run)
curl -X POST 'http://localhost:8000/api/system/cleanup-temp?dry_run=true&max_age_hours=24'

# Actually clean up folders older than 24 hours
curl -X POST 'http://localhost:8000/api/system/cleanup-temp?max_age_hours=24'

# Aggressive cleanup - folders older than 6 hours
curl -X POST 'http://localhost:8000/api/system/cleanup-temp?max_age_hours=6'
```

---

### 4. POST `/api/system/gc` - Force Garbage Collection

**Force Python garbage collection and return memory statistics.**

#### Response Example:
```json
{
  "initial_memory_mb": 250.5,
  "final_memory_mb": 235.2,
  "memory_freed_mb": 15.3,
  "objects_collected": 1250,
  "active_executors": 0
}
```

#### Usage:
```bash
curl -X POST http://localhost:8000/api/system/gc
```

---

## Temporary Folder Types

The system creates different types of temporary folders during agent execution:

| Pattern | Description | Safe to Delete |
|---------|-------------|----------------|
| `/tmp/cs_*` | Storage directories for each execution | ✅ Yes (after execution) |
| `/tmp/cm_*` | Memory directories for each execution | ✅ Yes (after execution) |
| `/tmp/cc_*` | Cache directories for each execution | ✅ Yes (after execution) |
| `/tmp/crew*` | General CrewAI temporary files | ✅ Yes (after execution) |
| `/tmp/embed_cache*` | Embedding cache (reusable) | ❌ No (preserve for performance) |
| `/tmp/vector_store*` | Vector store (reusable) | ❌ No (preserve for performance) |
| `/tmp/crewai_db*` | Database storage | ❌ No (may contain important data) |
| `/tmp/crewai_storage_kb_*` | Knowledge base storage | ❌ No (preserve for reuse) |

## Monitoring and Alerts

### Cleanup Recommendations

The API provides automatic recommendations based on temporary folder size:

- **Low urgency**: < 25MB of temporary files
- **Medium urgency**: 25-100MB of temporary files  
- **High urgency**: > 100MB of temporary files

### Automated Cleanup

You can set up automated cleanup using cron jobs:

```bash
# Clean up folders older than 24 hours every hour
0 * * * * curl -X POST 'http://localhost:8000/api/system/cleanup-temp?max_age_hours=24' > /dev/null 2>&1

# Clean up folders older than 6 hours every 30 minutes (more aggressive)
*/30 * * * * curl -X POST 'http://localhost:8000/api/system/cleanup-temp?max_age_hours=6' > /dev/null 2>&1
```

## Rate Limiting

All endpoints have rate limiting to prevent abuse:

- `/api/system/memory`: 100 requests/minute
- `/api/system/temp-stats`: 60 requests/minute
- `/api/system/cleanup-temp`: 10 requests/minute
- `/api/system/gc`: 5 requests/minute

## Error Handling

All endpoints return consistent error responses:

```json
{
  "error": "Error description",
  "success": false
}
```

Common error scenarios:
- System resource access denied
- Temporary folder access permissions
- Disk space issues
- Service unavailable

## Security Considerations

- These endpoints provide system-level information and should be protected
- Consider implementing authentication for production environments
- Monitor usage to detect potential abuse
- Regular cleanup prevents disk space exhaustion attacks

## Testing

Use the included test script to verify functionality:

```bash
python3 test_system_api.py
```

This will test all endpoints and provide a comprehensive system health report.

## Integration Examples

### Dashboard Integration

```javascript
// Fetch system stats for dashboard
async function getSystemStats() {
    const response = await fetch('/api/system/memory');
    const stats = await response.json();
    
    // Update dashboard with memory, disk, and temp folder info
    updateDashboard(stats);
}

// Check if cleanup is needed
async function checkCleanupNeeded() {
    const response = await fetch('/api/system/temp-stats');
    const stats = await response.json();
    
    if (stats.recommendations.should_cleanup) {
        showCleanupAlert(stats.recommendations.cleanup_urgency);
    }
}
```

### Monitoring Script

```bash
#!/bin/bash
# System health check script

STATS=$(curl -s http://localhost:8000/api/system/memory)
DISK_USAGE=$(echo $STATS | jq -r '.disk.root.percent_used')
TEMP_SIZE=$(echo $STATS | jq -r '.temp_folders.cleanup_stats.total_size_mb')

if (( $(echo "$DISK_USAGE > 80" | bc -l) )); then
    echo "WARNING: Disk usage is ${DISK_USAGE}%"
fi

if (( $(echo "$TEMP_SIZE > 100" | bc -l) )); then
    echo "WARNING: Temporary files using ${TEMP_SIZE}MB"
    # Trigger cleanup
    curl -X POST 'http://localhost:8000/api/system/cleanup-temp?max_age_hours=12'
fi
```

---

## Changelog

### v1.1.0 (Current)
- Enhanced `/api/system/memory` with comprehensive system statistics
- Added `/api/system/temp-stats` for detailed temporary folder analysis
- Added `/api/system/cleanup-temp` for automated cleanup
- Added cleanup recommendations and urgency levels
- Improved error handling and logging

### v1.0.0 (Previous)
- Basic `/api/system/memory` endpoint with process memory stats
- Basic garbage collection endpoint
