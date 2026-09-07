from checkpoint.database import create_connection_pool

def test_postgres_connection():
    pool = create_connection_pool()

    try:
        with pool.connection() as conn:
            result = conn.execute("SELECT 1").fetchone()

        assert result == (1,)

    finally:
        pool.close()