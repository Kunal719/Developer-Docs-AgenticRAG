from langchain_core.documents import Document

from models.rag_state import RAGState, RAGStage
from models.query_input import QueryInput
from models.evidence import EvidenceItem

from checkpoint.serialization import create_checkpoint, restore_state
from checkpoint.store import InMemoryCheckpointStore


class FakeEvidenceResolver:
    def __init__(self, documents: dict[tuple[str, str], Document]):
        self._documents = documents

    def __call__(self, corpus: str, chunk_id: str) -> Document:
        return self._documents[(corpus, chunk_id)]


def test_checkpoint_recovery_end_to_end():
    document = Document(
        page_content="LangGraph interrupt documentation",
        metadata={
            "corpus": "documentation",
            "chunk_id": "doc-123",
        },
    )

    evidence = EvidenceItem(
        document=document,
        source="initial",
        query="How do LangGraph interrupts work?",
    )

    original_state = RAGState(
        query_input=QueryInput(
            question="How do LangGraph interrupts work?",
            code_snippet="graph.invoke(...)",
            error="GraphInterrupt",
        ),
        stage=RAGStage.DRAFT_GENERATED,
        generation_round=1,
        reranker_query="LangGraph interrupt resume checkpoint",
        initial_evidence=[evidence],
    )

    # Create checkpoint from runtime state
    checkpoint = create_checkpoint(original_state)

    # Persist checkpoint
    store = InMemoryCheckpointStore()
    run_id = "run-123"

    store.save(run_id, checkpoint)

    # Load checkpoint
    loaded_checkpoint = store.load(run_id)

    # Resolver reconstructs Documents from corpus + chunk_id
    resolver = FakeEvidenceResolver(
        {
            ("documentation", "doc-123"): document,
        }
    )

    # Restore runtime state
    restored_state = restore_state(
        loaded_checkpoint,
        resolver,
    )

    # Verify recovered state
    assert restored_state.query_input == original_state.query_input
    assert restored_state.stage == original_state.stage
    assert restored_state.generation_round == original_state.generation_round
    assert restored_state.reranker_query == original_state.reranker_query

    assert len(restored_state.initial_evidence) == 1

    restored_evidence = restored_state.initial_evidence[0]

    assert restored_evidence.document == document
    assert restored_evidence.source == evidence.source
    assert restored_evidence.query == evidence.query
    assert restored_evidence.gap == evidence.gap