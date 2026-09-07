from psycopg_pool import ConnectionPool

from dotenv import load_dotenv
import os

load_dotenv()

def create_connection_pool() -> ConnectionPool:
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError("DATABASE_URL is not set")
    
    return ConnectionPool(
        conninfo=database_url,
        min_size=1,
        max_size=5,
        open=True
    )