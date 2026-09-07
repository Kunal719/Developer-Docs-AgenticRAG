import pytest

from checkpoint.models import RAGCheckpoint
from checkpoint.store import InMemoryCheckpointStore
from models.query_input import QueryInput
from models.rag_state import RAGStage


@pytest.fixture
def checkpoint():
    return RAGCheckpoint(
        query_input=QueryInput(
            question="How do I resume an interrupted graph?"
        ),
        stage=RAGStage.DRAFT_GENERATED,
        generation_round=1,
    )


@pytest.fixture
def store():
    return InMemoryCheckpointStore()


def test_save_and_load(store, checkpoint):
    run_id = "run-123"

    store.save(run_id, checkpoint)

    loaded = store.load(run_id)

    assert loaded == checkpoint


def test_save_overwrites_existing_checkpoint(store, checkpoint):
    run_id = "run-123"

    store.save(run_id, checkpoint)

    updated_checkpoint = checkpoint.model_copy(
        update={
            "generation_round": 2,
            "stage": RAGStage.FINAL_EVIDENCE_READY,
        }
    )

    store.save(run_id, updated_checkpoint)

    loaded = store.load(run_id)

    assert loaded == updated_checkpoint


def test_delete(store, checkpoint):
    run_id = "run-123"

    store.save(run_id, checkpoint)
    store.delete(run_id)

    with pytest.raises(KeyError):
        store.load(run_id)


def test_load_missing_run_id_raises_key_error(store):
    with pytest.raises(KeyError):
        store.load("does-not-exist")


def test_delete_missing_run_id_raises_key_error(store):
    with pytest.raises(KeyError):
        store.delete("does-not-exist")