from checkpoint.database import create_connection_pool
from checkpoint.models import RAGCheckpoint
from checkpoint.postgres_store import PostgresCheckpointStore

from models.query_input import QueryInput
from models.rag_state import RAGStage

import pytest


def test_save_checkpoint():
    pool = create_connection_pool()
    store = PostgresCheckpointStore(pool)

    checkpoint = RAGCheckpoint(
        query_input={
            "question": "How does LangGraph work?"
        },
        stage="draft_generated",
        generation_round=1,
    )

    run_id = "test-run-123"

    try:
        store.save(run_id, checkpoint)

        with pool.connection() as conn:
            row = conn.execute(
                """
                SELECT run_id, checkpoint
                FROM rag_checkpoints
                WHERE run_id = %s
                """,
                (run_id,),
            ).fetchone()

        assert row is not None
        assert row[0] == run_id

        saved_checkpoint = RAGCheckpoint.model_validate(row[1])

        assert saved_checkpoint == checkpoint

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

def test_load_checkpoint():
    pool = create_connection_pool()
    store = PostgresCheckpointStore(pool)

    checkpoint = RAGCheckpoint(
        query_input={
            "question": "How does LangGraph work?"
        },
        stage="draft_generated",
        generation_round=1,
    )

    run_id = "test-load-123"

    try:
        store.save(run_id, checkpoint)

        loaded_checkpoint = store.load(run_id)

        assert loaded_checkpoint == checkpoint

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

def test_delete_checkpoint():
    pool = create_connection_pool()
    store = PostgresCheckpointStore(pool)

    run_id = "test-delete-run"

    checkpoint = RAGCheckpoint(
        query_input={
            "question": "How does LangGraph work?"
        },
        stage="draft_generated",
        generation_round=1,
    )

    try:
        store.save(run_id, checkpoint)

        store.delete(run_id)

        with pool.connection() as conn:
            row = conn.execute(
                """
                SELECT checkpoint
                FROM rag_checkpoints
                WHERE run_id = %s
                """,
                (run_id,),
            ).fetchone()

        assert row is None

    finally:
        pool.close()

def test_delete_missing_checkpoint():
    pool = create_connection_pool()
    store = PostgresCheckpointStore(pool)

    try:
        with pytest.raises(KeyError):
            store.delete("does-not-exist")

    finally:
        pool.close()


def test_save_overwrites_existing_checkpoint():
    pool = create_connection_pool()
    store = PostgresCheckpointStore(pool)

    run_id = "test-upsert-run"

    checkpoint_1 = RAGCheckpoint(
        query_input=QueryInput(question="First question"),
        stage=RAGStage.ROUTED,
        generation_round=0,
    )

    checkpoint_2 = RAGCheckpoint(
        query_input=QueryInput(question="Second question"),
        stage=RAGStage.DRAFT_GENERATED,
        generation_round=1,
    )

    try:
        store.save(run_id, checkpoint_1)
        store.save(run_id, checkpoint_2)

        loaded_checkpoint = store.load(run_id)

        assert loaded_checkpoint == checkpoint_2

    finally:
        store.delete(run_id)
        pool.close()