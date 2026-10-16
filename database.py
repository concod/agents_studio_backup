from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from psycopg2 import pool, sql, errors
from config import settings
import os


# REMOVE OR REIMPORT FROM MAIN
import logging
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

db_pool, db_connection = None, None


# Database connection configuration
DB_CONFIG = {
    "host": settings.DB_HOST,
    "port": settings.DB_PORT,
    "database": settings.DB_NAME,
    "user": settings.DB_USER,
    "password": settings.DB_PASSWORD
}


# Database connection function with retry logic (synchronous version)
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def get_db_connection_sync():
    """Get a synchronous database connection from the pool"""
    try:
        global db_pool
        if db_pool is None:
            db_pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                **DB_CONFIG
            )
        return db_pool.getconn()
    except Exception as e:
        logger.error(f"Error getting database connection: {str(e)}")
        raise

def release_db_connection(conn):
    """Release a database connection back to the pool"""
    try:
        if conn is not None:
            db_pool.putconn(conn)
    except Exception as e:
        logger.error(f"Error releasing database connection: {str(e)}")


async def get_db():
    """Dependency that provides a database connection for each request"""
    conn = None
    try:
        conn = get_db_connection_sync()
        yield conn
    finally:
        if conn is not None:
            release_db_connection(conn)