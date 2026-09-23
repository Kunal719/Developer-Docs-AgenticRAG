import time

from fastapi.testclient import TestClient

from app.api.main import app
from checkpoint.database import create_connection_pool
from checkpoint.models import RAGCheckpoint, RunStatus
from checkpoint.postgres_store import PostgresCheckpointStore
from models.query_input import QueryInput
from models.rag_state import RAGStage


def make_checkpoint():
    return RAGCheckpoint(
        query_input=QueryInput(
            question="How does LangGraph work?"
        ),
        stage=RAGStage.ROUTED,
        generation_round=0,
    )


def test_stale_run_monitor_marks_stale_run_interrupted():
    pool = create_connection_pool()
    store = PostgresCheckpointStore(pool)

    run_id = "test-stale-monitor-integration"

    try:
        store.save(run_id, make_checkpoint())

        # Simulate a worker that stopped sending heartbeats.
        with pool.connection() as conn:
            conn.execute(
                """
                UPDATE rag_checkpoints
                SET last_heartbeat_at = NOW() - INTERVAL '60 seconds'
                WHERE run_id = %s
                """,
                (run_id,),
            )

        assert store.get_status(run_id) == RunStatus.RUNNING

        # Starting TestClient starts the real FastAPI lifespan,
        # which starts the stale-run monitor.
        with TestClient(app):
            deadline = time.time() + 25

            while time.time() < deadline:
                status = store.get_status(run_id)

                if status == RunStatus.INTERRUPTED:
                    break

                time.sleep(0.5)

            else:
                raise AssertionError(
                    "Stale run was not marked INTERRUPTED "
                    "by the background monitor"
                )

        assert store.get_status(run_id) == RunStatus.INTERRUPTED

    finally:
        with pool.connection() as conn:
            conn.execute(
                """
                DELETE FROM rag_checkpoints
                WHERE run_id = %s
                """,
                (run_id,),
            )

        pool.close()