from psycopg_pool import ConnectionPool
from checkpoint.models import RAGCheckpoint, RunStatus

class PostgresCheckpointStore:
    def __init__(self, pool: ConnectionPool):
        self._pool = pool

    @property
    def pool(self) -> ConnectionPool:
        return self._pool

    def create_run(self, run_id: str, user_id: str, checkpoint: RAGCheckpoint) -> None:
        checkpoint_data = checkpoint.model_dump_json()

        with self._pool.connection() as conn:

            # Serialize the run creation for the specific user
            conn.execute(
                """
                SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))
                """,
                (user_id,)
            )

            result = conn.execute(
                """
                SELECT COUNT(*) FROM rag_checkpoints 
                WHERE user_id = %s
                AND status IN (%s, %s, %s)
                AND created_at > NOW() - INTERVAL '24 hours'
                """,
                (user_id, RunStatus.RUNNING.value, RunStatus.INTERRUPTED.value, RunStatus.COMPLETED.value)
            )

            active_run_count = result.fetchone()[0]

            if active_run_count >= 5:
                raise ValueError("You have reached the maximum of 5 questions within the last 24 hours")

            conn.execute(
                """
                INSERT INTO rag_checkpoints (run_id, user_id, checkpoint, status, created_at, last_heartbeat_at)
                VALUES (%s, %s, %s::jsonb, %s, NOW(), NOW())
                """,
                (run_id, user_id, checkpoint_data, RunStatus.RUNNING.value)
            ) 
    
    def save(self, run_id: str, user_id: str, checkpoint: RAGCheckpoint) -> None:
        checkpoint_data = checkpoint.model_dump_json()

        with self._pool.connection() as conn:
            result = conn.execute(
                """
                UPDATE rag_checkpoints
                SET
                    checkpoint = %s::jsonb,
                    updated_at = NOW()
                WHERE
                    run_id = %s
                    AND user_id = %s
                """,
                (checkpoint_data, run_id, user_id)
            )

        if result.rowcount == 0:
            raise KeyError(f"No checkpoint found for run_id: {run_id}")

    def load(self, run_id: str, user_id: str) -> RAGCheckpoint:
        with self._pool.connection() as conn:
            row = conn.execute(
                """
                SELECT checkpoint
                FROM rag_checkpoints
                WHERE run_id = %s
                AND user_id = %s
                """,
                (run_id, user_id)
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
                    completed_at = CASE WHEN %s in (%s, %s, %s) THEN NOW() ELSE completed_at END
                WHERE run_id = %s AND status = %s
                """,
                (
                    status.value,
                    error,
                    status.value,
                    RunStatus.COMPLETED.value,
                    RunStatus.FAILED.value,
                    RunStatus.INTERRUPTED.value,
                    run_id,
                    RunStatus.RUNNING.value
                )
            )

        if result.rowcount == 0:
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
            
            raise ValueError(f"Cannot transition run {run_id} from '{row[0]}' to '{status.value}")

    def get_status(self, run_id: str, user_id: str) -> RunStatus:

        with self._pool.connection() as conn:
            row = conn.execute(
                """
                SELECT status from rag_checkpoints
                WHERE run_id = %s
                AND user_id = %s
                """,
                (run_id, user_id)
            ).fetchone()

        print("GET STATUS ROW:", row)

        if row is None:
            raise KeyError(f"No checkpoint found for run_id: {run_id}")

        return RunStatus(row[0])

    def list_runs(self, user_id: str) -> list[dict]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    run_id,
                    checkpoint ->> 'original_question',
                    status, 
                    updated_at,
                    completed_at,
                    error
                FROM rag_checkpoints
                WHERE user_id = %s
                ORDER BY updated_at DESC
                """,
                (user_id,)
            ).fetchall()

        return [
            {
                "run_id": row[0],
                "question": row[1],
                "status": RunStatus(row[2]),
                "updated_at": row[3],
                "completed_at": row[4],
                "error": row[5],
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


    def interrupt_stale_runs(self, stale_after_seconds: int) -> list[str]:
        with self._pool.connection() as conn:
            result = conn.execute(
                """
                UPDATE rag_checkpoints
                SET
                    status = %s,
                    error = %s,
                    updated_at = NOW(),
                    completed_at = NOW()
                WHERE
                    status = %s
                    AND last_heartbeat_at < NOW() - (%s * INTERVAL '1 second')
                RETURNING run_id
                """,
                (
                    RunStatus.INTERRUPTED.value,
                    "Run heartbeat timed out/stale",
                    RunStatus.RUNNING.value,
                    stale_after_seconds
                ),
            ).fetchall()

        return [row[0] for row in result]

    def resume_run(self, run_id: str, user_id: str) -> None:
        with self._pool.connection() as conn:
            result = conn.execute(
                """
                UPDATE rag_checkpoints
                SET
                    status = %s,
                    error = NULL,
                    updated_at = NOW(),
                    last_heartbeat_at = NOW(),
                    completed_at = NULL
                WHERE
                    run_id = %s
                    AND status = %s
                    AND user_id = %s
                """,
                (
                    RunStatus.RUNNING.value,
                    run_id,
                    RunStatus.INTERRUPTED.value,
                    user_id
                ),
            )

        if result.rowcount == 0:
            with self._pool.connection() as conn:
                row = conn.execute(
                    """
                    SELECT status
                    FROM rag_checkpoints
                    WHERE run_id = %s
                    AND user_id = %s
                    """,
                    (run_id, user_id),
                ).fetchone()

            if row is None:
                raise KeyError(f"No checkpoint found for run_id: {run_id}")

            raise ValueError(f"Cannot resume run {run_id} from status '{row[0]}'")

        