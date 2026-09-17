from unittest.mock import MagicMock
from uuid import UUID

from models.query_input import QueryInput
from app.services.rag_service import RAGService


def test_run_normalizes_input_before_agentic_rag():
    service = RAGService()

    normalized_input = QueryInput(
        question="Why is my graph not resuming after a restart?",
        observed_behavior="The graph starts from the beginning after restart.",
        expected_behavior="The graph should resume from the checkpoint.",
    )

    normalizer = MagicMock()
    normalizer.normalize.return_value = normalized_input

    agentic_rag = MagicMock()
    agentic_rag.run.return_value = MagicMock()

    service._input_normalizer = normalizer
    service._agentic_rag = agentic_rag

    result = service.run(
        "why my graph is not resuming after restart?"
    )

    normalizer.normalize.assert_called_once_with(
        "why my graph is not resuming after restart?"
    )

    agentic_rag.run.assert_called_once()

    call_kwargs = agentic_rag.run.call_args.kwargs

    assert call_kwargs["query_input"] == normalized_input
    assert "run_id" in call_kwargs

    assert result.run_id == call_kwargs["run_id"]

from unittest.mock import MagicMock

import pytest

from checkpoint.models import RunStatus
from models.query_input import QueryInput
from app.services.rag_service import RAGService


def test_run_marks_failed_when_agentic_rag_fails_after_checkpoint():
    service = RAGService()

    normalized_input = QueryInput(
        question="How do I create a checkpoint?"
    )

    normalizer = MagicMock()
    normalizer.normalize.return_value = normalized_input

    agentic_rag = MagicMock()
    agentic_rag.run.side_effect = RuntimeError("Generation failed")

    checkpoint_store = MagicMock()

    service._input_normalizer = normalizer
    service._agentic_rag = agentic_rag
    service._checkpoint_store = checkpoint_store

    with pytest.raises(RuntimeError, match="Generation failed"):
        service.run("How do I create a checkpoint?")

    checkpoint_store.update_status.assert_called_once()

    args = checkpoint_store.update_status.call_args.args

    assert args[0] == agentic_rag.run.call_args.kwargs["run_id"]
    assert args[1] == RunStatus.FAILED
    assert args[2] == "Generation failed"

def test_resume_uses_checkpoint_query_input_without_normalizing():
    service = RAGService()

    run_id = "test-run-123"

    checkpoint = MagicMock()
    checkpoint.query_input = QueryInput(
        question="How do I create a checkpoint?"
    )

    checkpoint_store = MagicMock()
    checkpoint_store.load.return_value = checkpoint

    normalizer = MagicMock()

    agentic_rag = MagicMock()
    agentic_rag.run.return_value = MagicMock()

    service._checkpoint_store = checkpoint_store
    service._input_normalizer = normalizer
    service._agentic_rag = agentic_rag

    result = service.resume(run_id)

    checkpoint_store.load.assert_called_once_with(run_id)

    agentic_rag.run.assert_called_once_with(
        query_input=checkpoint.query_input,
        run_id=run_id,
        resume=True,
    )

    normalizer.normalize.assert_not_called()

    assert result.run_id == run_id