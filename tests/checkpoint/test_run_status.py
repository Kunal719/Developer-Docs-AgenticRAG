import time

import pytest

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


def test_running_can_be_marked_completed():
    pool = create_connection_pool()
    store = PostgresCheckpointStore(pool)

    run_id = "test-status-completed"

    try:
        store.save(run_id, make_checkpoint())

        store.update_status(
            run_id,
            RunStatus.COMPLETED,
        )

        assert store.get_status(run_id) == RunStatus.COMPLETED

    finally:
        with pool.connection() as conn:
            conn.execute(
                "DELETE FROM rag_checkpoints WHERE run_id = %s",
                (run_id,),
            )

        pool.close()


def test_running_can_be_marked_failed():
    pool = create_connection_pool()
    store = PostgresCheckpointStore(pool)

    run_id = "test-status-failed"

    try:
        store.save(run_id, make_checkpoint())

        store.update_status(
            run_id,
            RunStatus.FAILED,
            "Something went wrong",
        )

        assert store.get_status(run_id) == RunStatus.FAILED

    finally:
        with pool.connection() as conn:
            conn.execute(
                "DELETE FROM rag_checkpoints WHERE run_id = %s",
                (run_id,),
            )

        pool.close()


def test_terminal_run_cannot_be_marked_interrupted():
    pool = create_connection_pool()
    store = PostgresCheckpointStore(pool)

    run_id = "test-terminal-transition"

    try:
        store.save(run_id, make_checkpoint())

        store.update_status(
            run_id,
            RunStatus.COMPLETED,
        )

        with pytest.raises(ValueError):
            store.update_status(
                run_id,
                RunStatus.INTERRUPTED,
            )

        assert store.get_status(run_id) == RunStatus.COMPLETED

    finally:
        with pool.connection() as conn:
            conn.execute(
                "DELETE FROM rag_checkpoints WHERE run_id = %s",
                (run_id,),
            )

        pool.close()


def test_stale_running_run_is_interrupted():
    pool = create_connection_pool()
    store = PostgresCheckpointStore(pool)

    run_id = "test-stale-run"

    try:
        store.save(run_id, make_checkpoint())

        # Make the heartbeat old enough to be considered stale.
        with pool.connection() as conn:
            conn.execute(
                """
                UPDATE rag_checkpoints
                SET last_heartbeat_at = NOW() - INTERVAL '60 seconds'
                WHERE run_id = %s
                """,
                (run_id,),
            )

        interrupted = store.interrupt_stale_runs(
            stale_after_seconds=30
        )

        assert run_id in interrupted
        assert store.get_status(run_id) == RunStatus.INTERRUPTED

    finally:
        with pool.connection() as conn:
            conn.execute(
                "DELETE FROM rag_checkpoints WHERE run_id = %s",
                (run_id,),
            )

        pool.close()


def test_fresh_running_run_is_not_interrupted():
    pool = create_connection_pool()
    store = PostgresCheckpointStore(pool)

    run_id = "test-fresh-run"

    try:
        store.save(run_id, make_checkpoint())

        interrupted = store.interrupt_stale_runs(
            stale_after_seconds=30
        )

        assert run_id not in interrupted
        assert store.get_status(run_id) == RunStatus.RUNNING

    finally:
        with pool.connection() as conn:
            conn.execute(
                "DELETE FROM rag_checkpoints WHERE run_id = %s",
                (run_id,),
            )

        pool.close()


def test_completed_run_is_not_interrupted_even_with_old_heartbeat():
    pool = create_connection_pool()
    store = PostgresCheckpointStore(pool)

    run_id = "test-completed-stale-run"

    try:
        store.save(run_id, make_checkpoint())

        store.update_status(
            run_id,
            RunStatus.COMPLETED,
        )

        with pool.connection() as conn:
            conn.execute(
                """
                UPDATE rag_checkpoints
                SET last_heartbeat_at = NOW() - INTERVAL '60 seconds'
                WHERE run_id = %s
                """,
                (run_id,),
            )

        interrupted = store.interrupt_stale_runs(
            stale_after_seconds=30
        )

        assert run_id not in interrupted
        assert store.get_status(run_id) == RunStatus.COMPLETED

    finally:
        with pool.connection() as conn:
            conn.execute(
                "DELETE FROM rag_checkpoints WHERE run_id = %s",
                (run_id,),
            )

        pool.close()