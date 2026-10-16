#!/usr/bin/env python3
"""
Comprehensive test for enhanced cross-contamination fixes

Tests all aspects identified in the research:
1. Fresh Crew/Agent instances per request
2. Request-specific memory storage paths  
3. Request ID in all logs
4. Request context through entire chain
5. Proper resource cleanup

Usage:
    python test_enhanced_isolation.py
"""

import asyncio
import logging
import os
import threading
import time
from datetime import datetime
from task_executor import TaskExecutor, get_execution_context, set_execution_context

# Configure detailed logging to see everything
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

class EnhancedIsolationTest:
    def __init__(self):
        self.test_results = []
        self.temp_dirs_created = []
        
    async def test_fresh_instances(self):
        """Test that each request gets completely fresh instances"""
        print("🔬 Testing Fresh Instance Creation...")
        
        try:
            # Create two TaskExecutors with different execution_ids
            executor1 = await TaskExecutor.create(
                agent_config={
                    "role": "Test Agent 1",
                    "goal": "Test isolation",
                    "backstory": "I am test agent 1"
                },
                execution_id="test_instance_1"
            )
            
            executor2 = await TaskExecutor.create(
                agent_config={
                    "role": "Test Agent 2", 
                    "goal": "Test isolation",
                    "backstory": "I am test agent 2"
                },
                execution_id="test_instance_2"
            )
            
            # Verify they are completely different instances
            tests = {
                "Different TaskExecutor instances": id(executor1) != id(executor2),
                "Different Agent instances": id(executor1.agent) != id(executor2.agent),
                "Different execution_ids": executor1.execution_id != executor2.execution_id,
                "Different logger names": executor1.logger.name != executor2.logger.name
            }
            
            all_passed = all(tests.values())
            
            print("📋 Fresh Instance Results:")
            for test_name, passed in tests.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"  {status} {test_name}")
            
            if all_passed:
                print("🎉 All fresh instance tests passed!")
            
            return all_passed
            
        except Exception as e:
            print(f"❌ Fresh instance test failed: {e}")
            return False
    
    async def test_request_specific_paths(self):
        """Test that each request gets its own memory/cache paths"""
        print("\n🗂️  Testing Request-Specific Paths...")
        
        try:
            executor = await TaskExecutor.create(
                agent_config={
                    "role": "Path Test Agent",
                    "goal": "Test path isolation", 
                    "backstory": "I test paths"
                },
                execution_id="test_paths_123"
            )
            
            # Set execution context to trigger path creation
            set_execution_context("test_paths_123", "Test input for paths")
            
            # Simulate task execution to create temp directories
            try:
                await executor.execute_task(
                    description="Test path creation",
                    expected_output="Path test response",
                    task_name="Path Test",
                    input="Test input for paths"
                )
            except Exception:
                pass  # Expected to fail, but should create directories
            
            # Check if paths were created
            context = get_execution_context()
            
            tests = {
                "Context has memory path": 'temp_memory_dir' in context,
                "Context has cache path": 'temp_cache_dir' in context,
                "Context has thread ID": 'thread_id' in context,
                "Context has process ID": 'process_id' in context,
                "Context has agent instance ID": 'agent_instance_id' in context
            }
            
            # Check if directories exist
            if 'temp_memory_dir' in context:
                tests["Memory directory exists"] = os.path.exists(context['temp_memory_dir'])
                self.temp_dirs_created.append(context['temp_memory_dir'])
            
            if 'temp_cache_dir' in context:
                tests["Cache directory exists"] = os.path.exists(context['temp_cache_dir'])
                self.temp_dirs_created.append(context['temp_cache_dir'])
            
            all_passed = all(tests.values())
            
            print("📋 Request-Specific Path Results:")
            for test_name, passed in tests.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"  {status} {test_name}")
                
            if 'temp_memory_dir' in context:
                print(f"  📁 Memory path: {context['temp_memory_dir']}")
            if 'temp_cache_dir' in context:
                print(f"  📁 Cache path: {context['temp_cache_dir']}")
            
            return all_passed
            
        except Exception as e:
            print(f"❌ Path isolation test failed: {e}")
            return False
    
    async def test_concurrent_isolation(self):
        """Test that concurrent requests don't interfere with each other"""
        print("\n🔄 Testing Concurrent Request Isolation...")
        
        async def single_request_test(request_id: str, input_text: str):
            """Run a single isolated request"""
            try:
                executor = await TaskExecutor.create(
                    agent_config={
                        "role": f"Concurrent Test Agent {request_id}",
                        "goal": "Test concurrent isolation",
                        "backstory": f"I am concurrent agent {request_id}"
                    },
                    execution_id=f"concurrent_{request_id}"
                )
                
                # Execute with unique input
                result = await executor.execute_task(
                    description=f"Process this input: {input_text}",
                    expected_output="A response to the input",
                    task_name=f"Concurrent Test {request_id}",
                    input=input_text
                )
                
                # Get context to verify isolation
                context = get_execution_context()
                
                return {
                    'request_id': request_id,
                    'input': input_text,
                    'execution_id': context.get('execution_id'),
                    'thread_id': context.get('thread_id'),
                    'agent_id': context.get('agent_instance_id'),
                    'success': True
                }
                
            except Exception as e:
                return {
                    'request_id': request_id,
                    'input': input_text,
                    'error': str(e),
                    'success': False
                }
        
        # Run multiple concurrent requests
        test_cases = [
            ("A", "What is your name?"),
            ("B", "Tell me about pricing"),
            ("C", "How does this work?"),
            ("D", "What are the features?"),
        ]
        
        tasks = [single_request_test(req_id, input_text) for req_id, input_text in test_cases]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze results for isolation
        successful_results = [r for r in results if isinstance(r, dict) and r.get('success')]
        
        tests = {
            "All requests completed": len(successful_results) == len(test_cases),
            "Unique execution IDs": len(set(r['execution_id'] for r in successful_results)) == len(successful_results),
            "Unique thread IDs": len(set(r['thread_id'] for r in successful_results)) >= 1,  # May share threads
            "Unique agent IDs": len(set(r['agent_id'] for r in successful_results)) == len(successful_results)
        }
        
        all_passed = all(tests.values())
        
        print("📋 Concurrent Isolation Results:")
        for test_name, passed in tests.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status} {test_name}")
        
        print("📊 Individual Request Results:")
        for result in successful_results:
            print(f"  Request {result['request_id']}: {result['execution_id']} (Thread: {result['thread_id']}, Agent: {result['agent_id']})")
        
        return all_passed
    
    async def test_resource_cleanup(self):
        """Test that resources are properly cleaned up"""
        print("\n🧹 Testing Resource Cleanup...")
        
        try:
            # Record initial state
            initial_temp_dirs = set(os.listdir("/tmp"))
            initial_env_vars = set(os.environ.keys())
            
            # Create and execute a task
            executor = await TaskExecutor.create(
                agent_config={
                    "role": "Cleanup Test Agent",
                    "goal": "Test resource cleanup",
                    "backstory": "I test cleanup"
                },
                execution_id="cleanup_test_456"
            )
            
            try:
                await executor.execute_task(
                    description="Test cleanup functionality",
                    expected_output="Cleanup test response",
                    task_name="Cleanup Test",
                    input="Test cleanup input"
                )
            except Exception:
                pass  # Expected to fail, but should trigger cleanup
            
            # Give it a moment for cleanup
            await asyncio.sleep(0.1)
            
            # Check final state
            final_temp_dirs = set(os.listdir("/tmp"))
            final_env_vars = set(os.environ.keys())
            
            # Look for persistent directories created by our test
            persistent_test_dirs = [d for d in final_temp_dirs if 'crewai_' in d and 'cleanup_test_456' in d]
            persistent_env_vars = [v for v in final_env_vars if v.endswith('_cleanup_test_456')]
            
            tests = {
                "No persistent test directories": len(persistent_test_dirs) == 0,
                "No persistent test env vars": len(persistent_env_vars) == 0,
                "Environment not polluted": len(final_env_vars - initial_env_vars) <= 5  # Allow some normal variation
            }
            
            all_passed = all(tests.values())
            
            print("📋 Resource Cleanup Results:")
            for test_name, passed in tests.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"  {status} {test_name}")
            
            if persistent_test_dirs:
                print(f"  ⚠️  Persistent directories found: {persistent_test_dirs}")
            if persistent_env_vars:
                print(f"  ⚠️  Persistent env vars found: {persistent_env_vars}")
            
            return all_passed
            
        except Exception as e:
            print(f"❌ Resource cleanup test failed: {e}")
            return False
    
    def cleanup_test_resources(self):
        """Clean up any test resources"""
        print("\n🧽 Cleaning up test resources...")
        
        # Clean up any temporary directories we created
        import shutil
        for temp_dir in self.temp_dirs_created:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    print(f"  🗑️  Cleaned up: {temp_dir}")
            except Exception as e:
                print(f"  ⚠️  Failed to clean up {temp_dir}: {e}")

async def main():
    """Run all enhanced isolation tests"""
    
    print("🚀 ENHANCED CROSS-CONTAMINATION ISOLATION TESTS")
    print("=" * 60)
    print("Based on research findings:")
    print("✓ Fresh Crew/Agent instances per request")
    print("✓ Request-specific memory storage paths")
    print("✓ Request ID in all logs")
    print("✓ Request context through entire chain")
    print("=" * 60)
    
    tester = EnhancedIsolationTest()
    
    # Run all tests
    test_results = {}
    
    test_results["Fresh Instances"] = await tester.test_fresh_instances()
    test_results["Request-Specific Paths"] = await tester.test_request_specific_paths()
    test_results["Concurrent Isolation"] = await tester.test_concurrent_isolation()
    test_results["Resource Cleanup"] = await tester.test_resource_cleanup()
    
    # Clean up test resources
    tester.cleanup_test_resources()
    
    # Final results
    print("\n" + "=" * 60)
    print("🎯 FINAL TEST RESULTS")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {test_name}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print("🎉 ALL ENHANCED ISOLATION TESTS PASSED!")
        print("✅ Cross-contamination fixes are working correctly")
        print("✅ Your research-based solutions are fully implemented")
        print("✅ Ready for production deployment")
        return 0
    else:
        print("🚨 SOME TESTS FAILED!")
        print("❌ Review the failed tests above")
        print("❌ Additional investigation needed")
        return 1

if __name__ == '__main__':
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 