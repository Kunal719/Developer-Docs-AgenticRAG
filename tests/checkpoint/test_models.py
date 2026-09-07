from models.rag_state import RAGStage
from models.query_input import QueryInput
from checkpoint.models import RAGCheckpoint


def test_rag_checkpoint_creation():
    query_input = QueryInput(
        question="How do I resume an interrupted graph?"
    )

    checkpoint = RAGCheckpoint(
        query_input=query_input,
        stage=RAGStage.DRAFT_GENERATED,
        generation_round=1,
        reranker_query="resume interrupted graph checkpoint",
    )

    assert checkpoint.schema_version == 1
    assert checkpoint.stage == RAGStage.DRAFT_GENERATED
    assert checkpoint.generation_round == 1
    assert checkpoint.reranker_query == "resume interrupted graph checkpoint"


def test_rag_checkpoint_serializes_to_json():
    checkpoint = RAGCheckpoint(
        query_input=QueryInput(
            question="How do I resume an interrupted graph?"
        ),
        stage=RAGStage.DRAFT_GENERATED,
        generation_round=1,
    )

    data = checkpoint.model_dump(mode="json")

    assert isinstance(data, dict)
    assert data["schema_version"] == 1
    assert data["stage"] == "draft_generated"
    assert data["generation_round"] == 1
    assert data["query_input"]["question"] == (
        "How do I resume an interrupted graph?"
    )


def test_query_input_round_trip_through_pydantic_checkpoint():
    query_input = QueryInput(
        question="How do I resume a LangGraph interrupt?",
        code_snippet="graph.invoke(...)",
        error="GraphInterrupt",
        logs="execution interrupted",
        observed_behavior="Execution pauses",
        expected_behavior="Execution should resume from checkpoint",
    )

    checkpoint = RAGCheckpoint(
        query_input=query_input,
        stage=RAGStage.INITIALIZED,
        generation_round=0,
    )

    data = checkpoint.model_dump(mode="json")

    assert isinstance(data, dict)
    assert data["query_input"]["question"] == query_input.question
    assert data["query_input"]["code_snippet"] == query_input.code_snippet
    assert data["query_input"]["error"] == query_input.error

    restored = RAGCheckpoint.model_validate(data)

    assert restored.query_input == query_input
    assert restored.query_input.question == query_input.question
    assert restored.query_input.code_snippet == query_input.code_snippet
    assert restored.query_input.error == query_input.error


def test_rag_checkpoint_full_round_trip():
    checkpoint = RAGCheckpoint(
        query_input=QueryInput(
            question="How do I resume an interrupted graph?",
            code_snippet="graph.invoke(...)",
        ),
        stage=RAGStage.FINAL_EVIDENCE_READY,
        generation_round=1,
        reranker_query="resume interrupted graph checkpoint",
    )

    data = checkpoint.model_dump(mode="json")

    restored = RAGCheckpoint.model_validate(data)

    assert restored == checkpoint