import pytest

from checkpoint.database import create_connection_pool
from checkpoint.models import RAGCheckpoint, RunStatus
from checkpoint.postgres_store import PostgresCheckpointStore

from models.query_input import QueryInput
from models.rag_state import RAGStage


def test_resume_run_transitions_interrupted_to_running():
    pool = create_connection_pool()
    store = PostgresCheckpointStore(pool)

    run_id = "test-resume-run"

    checkpoint = RAGCheckpoint(
        query_input=QueryInput(
            question="How does LangGraph work?"
        ),
        stage=RAGStage.ROUTED,
        generation_round=0,
    )

    try:
        store.save(run_id, checkpoint)

        # save() creates the run as RUNNING.
        assert store.get_status(run_id) == RunStatus.RUNNING

        # Simulate an interrupted run.
        with pool.connection() as conn:
            conn.execute(
                """
                UPDATE rag_checkpoints
                SET status = %s
                WHERE run_id = %s
                """,
                (
                    RunStatus.INTERRUPTED.value,
                    run_id,
                ),
            )

        assert store.get_status(run_id) == RunStatus.INTERRUPTED

        # Act
        store.resume_run(run_id)

        # Assert
        assert store.get_status(run_id) == RunStatus.RUNNING

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


def test_resume_run_raises_key_error_for_missing_run():
    pool = create_connection_pool()
    store = PostgresCheckpointStore(pool)

    run_id = "does-not-exist"

    try:
        with pytest.raises(KeyError):
            store.resume_run(run_id)

    finally:
        pool.close()


def test_resume_run_raises_value_error_for_invalid_status():
    pool = create_connection_pool()
    store = PostgresCheckpointStore(pool)

    run_id = "test-resume-invalid-status"

    checkpoint = RAGCheckpoint(
        query_input=QueryInput(
            question="How does LangGraph work?"
        ),
        stage=RAGStage.ROUTED,
        generation_round=0,
    )

    try:
        store.save(run_id, checkpoint)

        # save() leaves the run in RUNNING state.
        assert store.get_status(run_id) == RunStatus.RUNNING

        # resume_run() should only accept INTERRUPTED runs.
        with pytest.raises(ValueError):
            store.resume_run(run_id)

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