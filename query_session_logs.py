import os
import uuid
import psycopg2
import psycopg2.extras
import asyncio
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union, Awaitable, Callable, TypeVar
from functools import wraps

# Import the async get_db_connection from main
from main import get_db_connection

# Type variable for async functions
T = TypeVar('T')

def run_async(coro: Callable[..., Awaitable[T]]) -> Callable[..., T]:
    """Decorator to run async functions in a sync context"""
    @wraps(coro)
    def wrapper(*args, **kwargs) -> T:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro(*args, **kwargs))
    return wrapper


def query_session_logs(agent_id_str: str, query_all: bool = False) -> List[Dict[str, Any]]:
    """
    Query session logs for a specific agent with optional full query
    This is a synchronous wrapper around the async implementation.

    Args:
        agent_id_str: UUID string of the agent
        query_all: If True, query all columns, otherwise query selected columns

    Returns:
        List of dictionaries containing query results
    """
    async def _query_async() -> List[Dict[str, Any]]:
        """Async implementation of the query"""
        from DB import DatabasePool
        
        # Validate agent_id
        try:
            agent_id = str(uuid.UUID(agent_id_str))
        except (ValueError, TypeError) as e:
            print(f"Invalid agent_id: {str(e)}")
            return []
            
        conn = None
        cursor = None
        db_pool = DatabasePool.get_instance()
        
        try:
            # Get a database connection
            conn = await get_db_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
            if query_all:
                # Query all columns
                query = """
                    SELECT 
                        id, session_id, timestamp, ip_address, user_agent, 
                        request_headers, user_input, response, execution_time_seconds,
                        agent_id, agent_name, endpoint, method, created_at, 
                        feedback, execution_id
                    FROM session_logs
                    WHERE agent_id = %s
                    ORDER BY timestamp DESC
                """
                cursor.execute(query, (agent_id,))
                results = cursor.fetchall()
                
                # Print all columns results
                print("\nAll Columns Results:")
                print("-" * 80)
                for result in results:
                    print(f"\nSession ID: {result['session_id']}")
                    print(f"Timestamp: {result['timestamp']}")
                    print(f"Endpoint: {result['endpoint']}")
                    print(f"Method: {result['method']}")
                    print(f"User Input: {result['user_input']}")
                    print(f"Response: {result['response']}")
                    print("-" * 50)
                    
                return [dict(row) for row in results]
                
            else:
                # Query selected columns (grouped by session)
                query = """
                    SELECT 
                        session_id,
                        MIN(timestamp) as first_timestamp,
                        MAX(timestamp) as last_timestamp,
                        COUNT(id) as interaction_count
                    FROM session_logs
                    WHERE agent_id = %s
                    GROUP BY session_id
                    ORDER BY MIN(timestamp) DESC
                """
                cursor.execute(query, (agent_id,))
                results = cursor.fetchall()
                
                # Print the results
                print("\nSession Logs Grouped by Session ID:")
                print("-" * 80)
                print(f"Total sessions found: {len(results)}")
                print("-" * 80)

                for result in results:
                    print(f"\nSession ID: {result['session_id']}")
                    print(f"First Interaction: {result['first_timestamp']}")
                    print(f"Last Interaction: {result['last_timestamp']}")
                    print(f"Total Interactions: {result['interaction_count']}")
                    print("-" * 50)
                    
                return [dict(row) for row in results]
                
        except Exception as e:
            print(f"Error querying session logs: {str(e)}")
            return []
            
        finally:
            # Clean up resources
            if cursor:
                cursor.close()
            # The connection is managed by the DatabasePool, so we don't close it here
    
    # Run the async function and return the result
    return asyncio.get_event_loop().run_until_complete(_query_async())


if __name__ == "__main__":
    # Load environment variables
    agent_name = os.getenv("AGENT_NAME")
    agent_id_str = os.getenv("AGENT_ID")

    print("\nQuerying selected columns (default behavior):")
    query_session_logs(agent_id_str)

    print("\nQuerying all columns:")
    all_columns_results = query_session_logs(agent_id_str, query_all=True)

    # Print first few results of all columns query
    print("\nFirst few results of all columns query:")
    for result in all_columns_results[:3]:  # Show first 3 results
        print(f"\nSession ID: {result.session_id}")
        print(f"Timestamp: {result.timestamp}")
        print(f"Endpoint: {result.endpoint}")
        print(f"Method: {result.method}")
        print(f"User Input: {result.user_input}")
        print("-" * 50)
