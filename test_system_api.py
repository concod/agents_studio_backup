#!/usr/bin/env python3
"""
Test script for the new system API endpoints.
This tests the enhanced /api/system/memory endpoint and the new cleanup endpoints.
"""

import requests
import json
import sys
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_endpoint(endpoint, method="GET", params=None, data=None):
    """Test an API endpoint and return the response"""
    try:
        url = f"{BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, params=params, timeout=30)
        elif method == "POST":
            response = requests.post(url, params=params, json=data, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        print(f"\n{'='*60}")
        print(f"Testing {method} {endpoint}")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success!")
            return data
        else:
            print(f"❌ Failed: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection failed - is the server running on {BASE_URL}?")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def format_size(size_mb):
    """Format size in MB to human readable"""
    if size_mb < 1:
        return f"{size_mb * 1024:.1f} KB"
    elif size_mb > 1024:
        return f"{size_mb / 1024:.1f} GB"
    else:
        return f"{size_mb:.1f} MB"

def main():
    print("🚀 Testing Enhanced System API Endpoints")
    print(f"Base URL: {BASE_URL}")
    print(f"Test Time: {datetime.now()}")
    
    # Test 1: Enhanced system memory endpoint
    print("\n📊 Testing enhanced system statistics...")
    system_stats = test_endpoint("/api/system/memory")
    
    if system_stats:
        print("\n📈 System Statistics Summary:")
        
        # Memory info
        if "memory" in system_stats:
            mem = system_stats["memory"]
            print(f"  Process Memory: {format_size(mem.get('process_usage_mb', 0))}")
            print(f"  Active Executors: {mem.get('active_executors', 0)}")
            print(f"  Request Count: {mem.get('request_count', 0)}")
        
        # System memory
        if "system_memory" in system_stats:
            sys_mem = system_stats["system_memory"]
            print(f"  System Memory: {format_size(sys_mem.get('used_gb', 0) * 1024)} / {format_size(sys_mem.get('total_gb', 0) * 1024)} ({sys_mem.get('percent_used', 0):.1f}%)")
        
        # Disk usage
        if "disk" in system_stats:
            disk = system_stats["disk"]
            if "root" in disk:
                root = disk["root"]
                print(f"  Root Disk: {format_size(root.get('used_gb', 0) * 1024)} / {format_size(root.get('total_gb', 0) * 1024)} ({root.get('percent_used', 0):.1f}%)")
        
        # CPU
        if "cpu" in system_stats:
            cpu = system_stats["cpu"]
            print(f"  CPU Usage: {cpu.get('percent_used', 0):.1f}% ({cpu.get('core_count', 0)} cores)")
        
        # Temp folders
        if "temp_folders" in system_stats:
            temp = system_stats["temp_folders"]
            counts = temp.get("counts", {})
            cleanup = temp.get("cleanup_stats", {})
            print(f"  Temp Folders: {counts.get('total_temp_folders', 0)} total")
            print(f"  Deletable: {cleanup.get('deletable', 0)} items ({format_size(cleanup.get('total_size_mb', 0))})")
    
    # Test 2: Temp folder statistics
    print("\n🗂️ Testing temp folder statistics...")
    temp_stats = test_endpoint("/api/system/temp-stats")
    
    if temp_stats:
        print("\n📁 Temporary Folder Details:")
        
        counts = temp_stats.get("folder_counts", {})
        sizes = temp_stats.get("folder_sizes_mb", {})
        
        for folder_type, count in counts.items():
            if count > 0:
                size = sizes.get(folder_type, 0)
                print(f"  {folder_type}: {count} folders ({format_size(size)})")
        
        cleanup = temp_stats.get("cleanup_preview", {})
        if cleanup:
            print(f"\n🧹 Cleanup Preview:")
            print(f"  Deletable: {cleanup.get('deletable_count', 0)} items")
            print(f"  Size to free: {format_size(cleanup.get('deletable_size_mb', 0))}")
        
        recommendations = temp_stats.get("recommendations", {})
        if recommendations:
            print(f"\n💡 Recommendations:")
            print(f"  Should cleanup: {recommendations.get('should_cleanup', False)}")
            print(f"  Urgency: {recommendations.get('cleanup_urgency', 'low')}")
    
    # Test 3: Cleanup dry run
    print("\n🔍 Testing cleanup dry run...")
    cleanup_preview = test_endpoint("/api/system/cleanup-temp", "POST", params={"dry_run": True, "max_age_hours": 24})
    
    if cleanup_preview:
        print(f"\n🧹 Cleanup Preview Results:")
        print(f"  Success: {cleanup_preview.get('success', False)}")
        print(f"  Message: {cleanup_preview.get('message', 'N/A')}")
        
        stats = cleanup_preview.get("stats", {})
        if stats:
            print(f"  Would scan: {stats.get('scanned', 0)} items")
            print(f"  Would delete: {stats.get('deleted', 0)} items")
            print(f"  Would free: {format_size(stats.get('total_size_mb', 0))}")
    
    # Test 4: Garbage collection
    print("\n🗑️ Testing garbage collection...")
    gc_stats = test_endpoint("/api/system/gc", "POST")
    
    if gc_stats:
        print(f"\n♻️ Garbage Collection Results:")
        print(f"  Initial Memory: {format_size(gc_stats.get('initial_memory_mb', 0))}")
        print(f"  Final Memory: {format_size(gc_stats.get('final_memory_mb', 0))}")
        print(f"  Memory Freed: {format_size(gc_stats.get('memory_freed_mb', 0))}")
        print(f"  Objects Collected: {gc_stats.get('objects_collected', 0)}")
    
    print(f"\n{'='*60}")
    print("✅ API Testing Complete!")
    print("\n📋 Available Endpoints:")
    print("  GET  /api/system/memory      - Comprehensive system statistics")
    print("  GET  /api/system/temp-stats  - Temporary folder statistics")
    print("  POST /api/system/cleanup-temp - Clean up temporary folders")
    print("  POST /api/system/gc          - Force garbage collection")
    print("\n💡 Usage Examples:")
    print("  # Get system stats")
    print("  curl http://localhost:8000/api/system/memory")
    print("  ")
    print("  # Preview cleanup")
    print("  curl -X POST 'http://localhost:8000/api/system/cleanup-temp?dry_run=true&max_age_hours=24'")
    print("  ")
    print("  # Actually clean up")
    print("  curl -X POST 'http://localhost:8000/api/system/cleanup-temp?max_age_hours=24'")

if __name__ == "__main__":
    main()
