import pytest

from checkpoint.database import create_connection_pool
from checkpoint.models import RAGCheckpoint
from checkpoint.postgres_store import PostgresCheckpointStore
from models.query_input import QueryInput
from models.rag_state import RAGStage

import time
from unittest.mock import Mock

import threading

from pipeline.agentic_loop import AgenticRAG


def test_heartbeat_updates_last_heartbeat_at():
    pool = create_connection_pool()
    store = PostgresCheckpointStore(pool)

    run_id = "test-heartbeat-run"

    checkpoint = RAGCheckpoint(
        query_input=QueryInput(
            question="How does LangGraph work?"
        ),
        stage=RAGStage.ROUTED,
        generation_round=0,
    )

    try:
        store.save(run_id, checkpoint)

        store.heartbeat(run_id)

        with pool.connection() as conn:
            row = conn.execute(
                """
                SELECT last_heartbeat_at
                FROM rag_checkpoints
                WHERE run_id = %s
                """,
                (run_id,),
            ).fetchone()

        assert row is not None
        assert row[0] is not None

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


def test_heartbeat_missing_checkpoint():
    pool = create_connection_pool()
    store = PostgresCheckpointStore(pool)

    try:
        with pytest.raises(KeyError):
            store.heartbeat("does-not-exist")

    finally:
        pool.close()

def test_save_initializes_last_heartbeat_at():
    pool = create_connection_pool()
    store = PostgresCheckpointStore(pool)
    run_id = "test-save-heartbeat-run"

    checkpoint = RAGCheckpoint(
        query_input=QueryInput(question="How does LangGraph work?"),
        stage=RAGStage.ROUTED,
        generation_round=0,
    )

    try:
        store.save(run_id, checkpoint)

        with pool.connection() as conn:
            row = conn.execute(
                """
                SELECT last_heartbeat_at
                FROM rag_checkpoints
                WHERE run_id = %s
                """,
                (run_id,),
            ).fetchone()

        assert row is not None
        assert row[0] is not None

    finally:
        with pool.connection() as conn:
            conn.execute(
                "DELETE FROM rag_checkpoints WHERE run_id = %s",
                (run_id,),
            )
        pool.close()


def test_agentic_rag_heartbeat_lifecycle():
    checkpoint_store = Mock()

    agentic_rag = AgenticRAG.__new__(AgenticRAG)
    agentic_rag._checkpoint_store = checkpoint_store

    run_id = "test-heartbeat-lifecycle"

    stop_event = threading.Event()

    heartbeat_thread = threading.Thread(
        target=agentic_rag._heartbeat_loop,
        args=(run_id, stop_event),
        daemon=True,
    )

    heartbeat_thread.start()

    # Give the heartbeat loop enough time to execute at least once.
    time.sleep(6)

    stop_event.set()
    heartbeat_thread.join(timeout=1)

    checkpoint_store.heartbeat.assert_called()

    assert not heartbeat_thread.is_alive()