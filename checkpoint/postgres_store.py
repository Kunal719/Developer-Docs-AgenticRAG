from psycopg_pool import ConnectionPool
from checkpoint.models import RAGCheckpoint

class PostgresCheckpointStore:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def save(self, run_id: str, checkpoint: RAGCheckpoint) -> None:
        checkpoint_data = checkpoint.model_dump_json()

        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO rag_checkpoints (run_id, checkpoint)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (run_id) 
                DO UPDATE SET 
                    checkpoint = EXCLUDED.checkpoint,
                    updated_at = NOW();
                """,
                (run_id, checkpoint_data)
            )

    def load(self, run_id: str) -> RAGCheckpoint:
        with self._pool.connection() as conn:
            row = conn.execute(
                """
                SELECT checkpoint
                FROM rag_checkpoints
                WHERE run_id = %s
                """,
                (run_id,)
            ).fetchone()

        if row is None:
            raise KeyError(f"No checkpoint found for run_id: {run_id}")

        return RAGCheckpoint.model_validate(row[0])

    def delete(self, run_id: str) -> None:
        with self._pool.connection() as conn:
            result = conn.execute(
                """
                DELETE FROM rag_checkpoints
                WHERE run_id = %s
                """,
                (run_id,)
            )

        if result.rowcount == 0:
            raise KeyError(f"No checkpoint found for run_id: {run_id}")

        