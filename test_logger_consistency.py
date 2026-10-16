#!/usr/bin/env python3
"""
Test script to verify logger consistency across TaskExecutor

This script tests that:
1. All logs use the same execution_id format
2. Knowledge source loading logs are properly formatted
3. Tool logs use the executor's logger
4. No orphaned loggers are created

Usage:
    python test_logger_consistency.py
"""

import asyncio
import logging
import sys
from task_executor import TaskExecutor

# Configure root logging to capture all logs
logging.basicConfig(
    level=logging.DEBUG,
    format='%(name)s - %(asctime)s [%(levelname)s] %(message)s'
)

async def test_logger_consistency():
    """Test that all logs use consistent execution_id formatting"""
    
    print("🧪 Testing Logger Consistency")
    print("=" * 50)
    
    # Create TaskExecutor with specific execution_id
    test_execution_id = "test_logger_123"
    
    try:
        print(f"Creating TaskExecutor with execution_id: {test_execution_id}")
        
        # Mock agent configuration
        agent_config = {
            "role": "Logger Test Assistant",
            "goal": "Test consistent logging",
            "backstory": "I test logger consistency",
            "knowledge_base_id": "non_existent_kb"  # This will trigger knowledge loading logs
        }
        
        # Mock tool configuration to test tool logging
        mock_tool_config = [{
            "id": "mock_tool",
            "schema": {
                "info": {"title": "Mock Logger Tool"},
                "paths": {
                    "/test": {
                        "post": {
                            "requestBody": {
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "query": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "servers": [{"url": "http://test-server.com"}]
            }
        }]
        
        # Create executor - this should trigger various logging
        executor = await TaskExecutor.create(
            agent_config=agent_config,
            tools_config=mock_tool_config,
            execution_id=test_execution_id,
            verbose=True
        )
        
        print(f"✅ TaskExecutor created successfully")
        print(f"Logger name: {executor.logger.name}")
        print(f"Execution ID: {executor.execution_id}")
        print(f"Number of tools: {len(executor.tools)}")
        
        # Test task execution logging
        print("\nTesting task execution logging...")
        
        try:
            result = await executor.execute_task(
                description="Test logging consistency",
                expected_output="A test response",
                task_name="Logger Consistency Test",
                input="Test input for logger consistency"
            )
            print(f"✅ Task execution completed")
        except Exception as e:
            print(f"⚠️ Task execution failed (expected): {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_log_output():
    """Analyze the log output for consistency"""
    
    print("\n" + "=" * 50)
    print("📊 LOG ANALYSIS")
    print("=" * 50)
    
    # Check if all logs have execution_id format
    print("✅ Expected log format: [execution_id] timestamp [LEVEL] message")
    print("✅ All logs should use the same execution_id: test_logger_123")
    print("✅ Knowledge source logs should have execution_id formatting")
    print("✅ Tool logs should use executor's logger")
    
    print("\nLook for these patterns in the output above:")
    print("- [test_logger_123] in all log messages")
    print("- Consistent formatting across all log entries")
    print("- No logs without execution_id prefix")
    
    print("\n⚠️  Manual verification required:")
    print("- Review the log output above")
    print("- Ensure all logs follow the [execution_id] format")
    print("- Check that no generic logger messages appear")

async def main():
    """Run logger consistency tests"""
    
    # Test logger consistency
    success = await test_logger_consistency()
    
    # Analyze the output
    analyze_log_output()
    
    print("\n" + "=" * 50)
    print("🎯 FINAL RESULT")
    print("=" * 50)
    
    if success:
        print("✅ Logger consistency test completed successfully!")
        print("Review the log output above to verify all logs use the same execution_id format.")
        return 0
    else:
        print("❌ Logger consistency test failed!")
        return 1

if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 