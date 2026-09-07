from langchain_core.documents import Document

from models.rag_state import RAGStage, RAGState
from models.query_input import QueryInput
from models.evidence import EvidenceItem

from checkpoint.models import RAGCheckpoint, EvidenceItemCheckpoint
from checkpoint.serialization import (
    create_checkpoint,
    restore_state,
)


class FakeEvidenceResolver:
    def __init__(self, documents: dict[tuple[str, str], Document]):
        self._documents = documents

    def __call__(self, corpus: str, chunk_id: str) -> Document:
        return self._documents[(corpus, chunk_id)]


def test_evidence_items_to_checkpoint():
    document = Document(
        page_content="Documentation content",
        metadata={
            "corpus": "documentation",
            "chunk_id": "doc-123",
        },
    )

    evidence = EvidenceItem(
        document=document,
        source="initial",
        query="How do interrupts work?",
    )

    state = RAGState(
        query_input=QueryInput(
            question="How do interrupts work?"
        ),
        initial_evidence=[evidence],
    )

    checkpoint = create_checkpoint(state)

    assert len(checkpoint.initial_evidence) == 1

    evidence_checkpoint = checkpoint.initial_evidence[0]

    assert isinstance(evidence_checkpoint, EvidenceItemCheckpoint)
    assert evidence_checkpoint.corpus == "documentation"
    assert evidence_checkpoint.chunk_id == "doc-123"
    assert evidence_checkpoint.source == "initial"
    assert evidence_checkpoint.query == "How do interrupts work?"


def test_checkpoint_restores_evidence_with_resolver():
    document = Document(
        page_content="Documentation content",
        metadata={
            "corpus": "documentation",
            "chunk_id": "doc-123",
        },
    )

    resolver = FakeEvidenceResolver(
        {
            ("documentation", "doc-123"): document,
        }
    )

    checkpoint = RAGCheckpoint(
        query_input=QueryInput(
            question="How do interrupts work?"
        ),
        stage=RAGStage.DRAFT_GENERATED,
        generation_round=1,
        initial_evidence=[
            EvidenceItemCheckpoint(
                corpus="documentation",
                chunk_id="doc-123",
                source="initial",
                query="How do interrupts work?",
            )
        ],
    )

    state = restore_state(checkpoint, resolver)

    assert len(state.initial_evidence) == 1

    evidence = state.initial_evidence[0]

    assert evidence.document == document
    assert evidence.source == "initial"
    assert evidence.query == "How do interrupts work?"


def test_rag_state_checkpoint_round_trip():
    document = Document(
        page_content="Implementation content",
        metadata={
            "corpus": "implementation",
            "chunk_id": "code-456",
        },
    )

    evidence = EvidenceItem(
        document=document,
        source="gap_rerank",
        query="resume interrupted graph",
    )

    original_state = RAGState(
        query_input=QueryInput(
            question="How do I resume an interrupted graph?",
            code_snippet="graph.invoke(...)",
        ),
        stage=RAGStage.FINAL_EVIDENCE_READY,
        generation_round=1,
        reranker_query="resume interrupted graph",
        final_evidence=[evidence],
    )

    checkpoint = create_checkpoint(original_state)

    resolver = FakeEvidenceResolver(
        {
            ("implementation", "code-456"): document,
        }
    )

    restored_state = restore_state(checkpoint, resolver)

    assert restored_state.query_input == original_state.query_input
    assert restored_state.stage == original_state.stage
    assert restored_state.generation_round == original_state.generation_round
    assert restored_state.reranker_query == original_state.reranker_query
    assert restored_state.final_evidence == original_state.final_evidence
    assert restored_state.generated_answer == original_state.generated_answer
    assert restored_state.gaps == original_state.gaps