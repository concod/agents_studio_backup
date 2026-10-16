from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, cast

import psycopg2
import psycopg2.extras


def row_to_dict(row: Any) -> Dict[str, Any]:
    """Convert a database row to a dictionary.
    
    Args:
        row: The database row to convert
        
    Returns:
        Dictionary representation of the row
    """
    if row is None:
        return {}
    if isinstance(row, dict):
        return dict(row)
    if hasattr(row, '_asdict'):  # For SQLAlchemy-like objects
        return cast(Dict[str, Any], row._asdict())
    if hasattr(row, '_fields'):  # For namedtuple-like objects
        return {key: getattr(row, key) for key in row._fields}
    return dict(row)


class SessionLogService:
    """Service for handling session log operations."""
    
    def __init__(self, db_connection: psycopg2.extensions.connection):
        """Initialize the service with a database connection.
        
        Args:
            db_connection: Active psycopg2 database connection
        """
        self.conn = db_connection
        self.cursor = None

    def format_timestamp(self, timestamp_str: str) -> str:
        """
        Format a timestamp string to a more readable format.

        Args:
            timestamp_str: The input timestamp string in ISO format

        Returns:
            Formatted timestamp string (e.g., "09th June, 2025, 15:08:39")
        """
        try:
            # Handle different timestamp formats
            if 'T' in timestamp_str:
                dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            else:
                # Try parsing with timezone
                try:
                    dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f %z")
                except ValueError:
                    dt = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S %z")
            
            # Format the date with proper ordinal suffix
            day = dt.day
            if 11 <= day <= 13:
                suffix = 'th'
            else:
                suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(day % 10, 'th')
                
            return dt.strftime(f'%d{suffix} %B, %Y, %H:%M:%S')
            
        except (ValueError, TypeError) as e:
            # If parsing fails, return the original string
            print(f"Error formatting timestamp {timestamp_str}: {str(e)}")
            return timestamp_str

    def get_session_logs(
        self,
        session_id: Optional[str] = None,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
        agent_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get session logs based on the provided filters.

        Args:
            session_id: Filter by session ID
            from_date: Filter logs after this date
            to_date: Filter logs before this date
            agent_id: Filter by agent ID

        Returns:
            List of session log dictionaries
        """
        cursor = None
        try:
            cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            query = """
                SELECT 
                    id,
                    session_id,
                    timestamp,
                    user_input,
                    CASE 
                        WHEN response->'content'->>'text' IS NOT NULL THEN response->'content'->>'text'
                        ELSE response::text
                    END as response,
                    execution_time_seconds,
                    feedback,
                    ip_address,
                    execution_id,
                    agent_id,
                    created_at
                FROM session_logs
                WHERE 1=1
            """
            params: List[Any] = []
            
            if session_id:
                query += " AND session_id = %s"
                params.append(session_id)
                
            if from_date:
                query += " AND timestamp >= %s"
                params.append(from_date)
                
            if to_date:
                # Add 1 day to include the entire to_date day
                next_day = to_date.replace(hour=23, minute=59, second=59) + timedelta(seconds=1)
                query += " AND timestamp < %s"
                params.append(next_day)
                
            if agent_id:
                query += " AND agent_id = %s"
                params.append(agent_id)
            
            # Order by timestamp in descending order (newest first)
            query += " ORDER BY timestamp DESC"
            
            cursor.execute(query, params)
            logs = cursor.fetchall()
            return [row_to_dict(log) for log in logs]
            
        except Exception as e:
            print(f"Error getting session logs: {str(e)}")
            return []
        finally:
            if cursor:
                cursor.close()

    def close(self) -> None:
        """Close the database cursor."""
        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()

    def query_session_logs(self, agent_id: str, query_all: bool = False) -> List:
        """
        Query session logs for a specific agent with optional full query

        Args:
            agent_id (str): UUID string of the agent
            query_all (bool): If True, query all columns, otherwise query selected columns

        Returns:
            List: List of dictionaries containing query results
        """
        cursor = None
        try:
            cursor = self.conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            
            if query_all:
                # Query all columns
                query = """
                    SELECT *
                    FROM session_logs
                    WHERE agent_id = %s
                    ORDER BY timestamp DESC
                """
                cursor.execute(query, (agent_id,))
            else:
                # Query selected columns
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
            return [dict(row) for row in results]

        except Exception as e:
            print(f"Error executing query: {str(e)}")
            return []
        finally:
            if cursor:
                cursor.close()
