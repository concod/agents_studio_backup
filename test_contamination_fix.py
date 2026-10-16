#!/usr/bin/env python3
"""
Test script to verify cross-contamination fixes

This script tests the fixes by:
1. Creating multiple TaskExecutor instances with different inputs
2. Running them concurrently
3. Verifying that each gets the correct response
4. Checking logs for proper isolation

Usage:
    python test_contamination_fix.py
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from task_executor import TaskExecutor, set_execution_context, get_execution_context

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class ContaminationTest:
    def __init__(self):
        self.test_results = []
        
    async def create_test_executor(self, execution_id: str, test_input: str):
        """Create a TaskExecutor for testing"""
        
        # Mock agent configuration
        agent_config = {
            "role": "Test Assistant",
            "goal": "Help with testing",
            "backstory": "I am a test assistant designed to validate input isolation",
        }
        
        executor = await TaskExecutor.create(
            agent_config=agent_config,
            tools_config=[],  # No tools for simple testing
            log_file=f"logs/test_execution_{execution_id}.log",
            user_email="test@example.com",
            execution_id=execution_id
        )
        
        return executor
    
    async def run_single_test(self, test_id: str, user_input: str):
        """Run a single test case"""
        try:
            logger.info(f"Starting test {test_id} with input: '{user_input}'")
            
            # Create executor with unique execution_id
            execution_id = f"test_{test_id}_{int(time.time())}"
            executor = await self.create_test_executor(execution_id, user_input)
            
            # Set execution context
            set_execution_context(execution_id, user_input)
            
            # Execute task
            result = await executor.execute_task(
                description=f"Respond to the following user input: {user_input}. Simply acknowledge what they asked about.",
                expected_output="A simple acknowledgment of what the user asked about",
                task_name=f"Test Task {test_id}",
                input=user_input
            )
            
            # Verify context integrity
            context = get_execution_context()
            context_intact = context.get('user_input') == user_input
            
            test_result = {
                'test_id': test_id,
                'execution_id': execution_id,
                'input': user_input,
                'result': result,
                'context_intact': context_intact,
                'success': bool(result and context_intact),
                'timestamp': datetime.now().isoformat()
            }
            
            self.test_results.append(test_result)
            logger.info(f"Test {test_id} completed. Context intact: {context_intact}")
            
            return test_result
            
        except Exception as e:
            logger.error(f"Test {test_id} failed: {e}")
            error_result = {
                'test_id': test_id,
                'execution_id': execution_id if 'execution_id' in locals() else 'unknown',
                'input': user_input,
                'result': f'ERROR: {str(e)}',
                'context_intact': False,
                'success': False,
                'timestamp': datetime.now().isoformat()
            }
            self.test_results.append(error_result)
            return error_result
    
    async def run_concurrent_tests(self):
        """Run multiple tests concurrently to check for cross-contamination"""
        
        test_cases = [
            ("A", "What is your name?"),
            ("B", "Tell me about pricing"),
            ("C", "How does this work?"),
            ("D", "What are the features?"),
            ("E", "Do you have customer testimonials?"),
            ("F", "What is PlanSmart?"),
            ("G", "How can I contact support?"),
            ("H", "What are your business hours?"),
        ]
        
        logger.info(f"Running {len(test_cases)} concurrent tests...")
        
        # Run all tests concurrently
        tasks = [self.run_single_test(test_id, user_input) for test_id, user_input in test_cases]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    def analyze_results(self):
        """Analyze test results for contamination"""
        
        logger.info("Analyzing test results...")
        
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r['success'])
        context_intact_count = sum(1 for r in self.test_results if r['context_intact'])
        
        contamination_detected = False
        contamination_issues = []
        
        # Check for cross-contamination signs
        for result in self.test_results:
            if not result['context_intact']:
                contamination_detected = True
                contamination_issues.append(f"Test {result['test_id']}: Context not intact")
            
            # Check if result contains input from other tests
            for other_result in self.test_results:
                if (result['test_id'] != other_result['test_id'] and 
                    other_result['input'].lower() in result['result'].lower()):
                    contamination_detected = True
                    contamination_issues.append(
                        f"Test {result['test_id']} result contains input from Test {other_result['test_id']}"
                    )
        
        analysis = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'context_intact_count': context_intact_count,
            'success_rate': (successful_tests / total_tests) * 100 if total_tests > 0 else 0,
            'context_integrity_rate': (context_intact_count / total_tests) * 100 if total_tests > 0 else 0,
            'contamination_detected': contamination_detected,
            'contamination_issues': contamination_issues,
            'test_results': self.test_results
        }
        
        return analysis
    
    def print_analysis(self, analysis):
        """Print analysis results"""
        
        print("\n" + "="*60)
        print("CROSS-CONTAMINATION TEST RESULTS")
        print("="*60)
        
        print(f"Total Tests: {analysis['total_tests']}")
        print(f"Successful Tests: {analysis['successful_tests']}")
        print(f"Success Rate: {analysis['success_rate']:.1f}%")
        print(f"Context Integrity Rate: {analysis['context_integrity_rate']:.1f}%")
        
        if analysis['contamination_detected']:
            print("\n🚨 CONTAMINATION DETECTED:")
            for issue in analysis['contamination_issues']:
                print(f"  - {issue}")
        else:
            print("\n✅ NO CONTAMINATION DETECTED")
        
        print("\nIndividual Test Results:")
        print("-" * 60)
        for result in analysis['test_results']:
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            print(f"Test {result['test_id']}: {status}")
            print(f"  Input: {result['input']}")
            print(f"  Context Intact: {result['context_intact']}")
            if not result['success']:
                print(f"  Result: {result['result'][:100]}...")
            print()
        
        print("="*60)
        
        # Final verdict
        if analysis['contamination_detected']:
            print("🚨 VERDICT: FIXES MAY NOT BE WORKING - CONTAMINATION DETECTED")
            print("Review the issues above and check the logs for more details.")
        else:
            print("✅ VERDICT: FIXES APPEAR TO BE WORKING - NO CONTAMINATION DETECTED")
            print("Continue monitoring in production for ongoing validation.")
        
        print("="*60)

async def main():
    """Main test function"""
    
    print("Starting Cross-Contamination Fix Tests...")
    print("This will test concurrent execution isolation.")
    
    # Create test instance
    tester = ContaminationTest()
    
    # Run concurrent tests
    await tester.run_concurrent_tests()
    
    # Analyze results
    analysis = tester.analyze_results()
    
    # Print analysis
    tester.print_analysis(analysis)
    
    # Save results
    with open('contamination_test_results.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nDetailed results saved to contamination_test_results.json")
    
    # Return exit code based on results
    return 0 if not analysis['contamination_detected'] else 1

if __name__ == '__main__':
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 