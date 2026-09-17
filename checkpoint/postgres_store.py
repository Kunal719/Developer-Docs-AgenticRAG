from psycopg_pool import ConnectionPool
from checkpoint.models import RAGCheckpoint, RunStatus

class PostgresCheckpointStore:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    def save(self, run_id: str, checkpoint: RAGCheckpoint) -> None:
        checkpoint_data = checkpoint.model_dump_json()

        with self._pool.connection() as conn:
            conn.execute(
                """
                INSERT INTO rag_checkpoints (run_id, checkpoint, status, last_heartbeat_at)
                VALUES (%s, %s::jsonb, %s, NOW())
                ON CONFLICT (run_id) 
                DO UPDATE SET 
                    checkpoint = EXCLUDED.checkpoint,
                    updated_at = NOW();
                """,
                (run_id, checkpoint_data, RunStatus.RUNNING.value)
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


    def update_status(self, run_id: str, status: RunStatus, error: str | None = None) -> None:
        with self._pool.connection() as conn:
            result = conn.execute(
                """
                UPDATE rag_checkpoints
                SET
                    status = %s,
                    error = %s,
                    updated_at = NOW(),
                    completed_at = CASE WHEN %s in (%s, %s) THEN NOW() ELSE completed_at END
                WHERE run_id = %s
                """,
                (
                    status.value,
                    error,
                    status.value,
                    RunStatus.COMPLETED.value,
                    RunStatus.FAILED.value,
                    run_id
                )
            )

        if result.rowcount == 0:
            raise KeyError(f"No checkpoint found for run_id: {run_id}")

    def get_status(self, run_id: str) -> RunStatus:
        with self._pool.connection() as conn:
            row = conn.execute(
                """
                SELECT status from rag_checkpoints
                WHERE run_id = %s
                """,
                (run_id,)
            ).fetchone()

        if row is None:
            raise KeyError(f"No checkpoint found for run_id: {run_id}")

        return RunStatus(row[0])

    def list_runs(self) -> list[dict]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    run_id,
                    status,
                    updated_at,
                    completed_at,
                    error
                FROM rag_checkpoints
                ORDER BY updated_at DESC
                """
            ).fetchall()

        return [
            {
                "run_id": row[0],
                "status": RunStatus(row[1]),
                "updated_at": row[2],
                "completed_at": row[3],
                "error": row[4],
            }
            for row in rows
        ]

    def heartbeat(self, run_id: str) -> None:
        with self._pool.connection() as conn:
            result = conn.execute(
                """
                UPDATE rag_checkpoints
                SET
                    last_heartbeat_at = NOW()
                WHERE run_id = %s
                """,
                (run_id,)
            )

        if result.rowcount == 0:
            raise KeyError(f"No checkpoint found for run_id: {run_id}")

    # def delete(self, run_id: str) -> None:
    #     with self._pool.connection() as conn:
    #         result = conn.execute(
    #             """
    #             DELETE FROM rag_checkpoints
    #             WHERE run_id = %s
    #             """,
    #             (run_id,)
    #         )

    #     if result.rowcount == 0:
    #         raise KeyError(f"No checkpoint found for run_id: {run_id}")

        