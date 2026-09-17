import pytest
from unittest.mock import MagicMock

from langchain_core.documents import Document

from pipeline.agentic_loop import AgenticRAG

from models.query_input import QueryInput
from models.generation import GeneratedAnswer, RetrievalGap
from models.rag_state import RAGState, RAGStage
from models.evidence import EvidenceItem

from checkpoint.database import create_connection_pool
from checkpoint.postgres_store import PostgresCheckpointStore
from checkpoint.serialization import create_checkpoint


@pytest.fixture
def postgres_checkpoint_store():
    pool = create_connection_pool()
    store = PostgresCheckpointStore(pool)

    yield store

    pool.close()


def test_postgres_checkpoint_save_and_load(postgres_checkpoint_store):
    run_id = "test-postgres-save-load"

    document = Document(
        page_content="PostgreSQL checkpoint evidence",
        metadata={
            "corpus": "documentation",
            "chunk_id": "postgres-test-chunk",
        },
    )

    evidence = EvidenceItem(
        document=document,
        source="initial_rerank",
        query="Test PostgreSQL checkpointing",
    )

    state = RAGState(
        query_input=QueryInput(
            question="Test PostgreSQL checkpointing"
        ),
        stage=RAGStage.INITIAL_EVIDENCE_READY,
        generation_round=0,
        reranker_query="Test PostgreSQL checkpointing",
        initial_evidence=[evidence],
        evidence_pool=[evidence],
    )

    checkpoint = create_checkpoint(state)

    postgres_checkpoint_store.save(run_id, checkpoint)

    loaded = postgres_checkpoint_store.load(run_id)

    assert loaded == checkpoint

    postgres_checkpoint_store.delete(run_id)


def test_postgres_checkpoint_upsert(postgres_checkpoint_store):
    run_id = "test-postgres-upsert"

    first_state = RAGState(
        query_input=QueryInput(
            question="Original question"
        ),
        stage=RAGStage.ROUTED,
    )

    second_state = RAGState(
        query_input=QueryInput(
            question="Updated question"
        ),
        stage=RAGStage.INITIAL_EVIDENCE_READY,
    )

    postgres_checkpoint_store.save(
        run_id,
        create_checkpoint(first_state),
    )

    postgres_checkpoint_store.save(
        run_id,
        create_checkpoint(second_state),
    )

    loaded = postgres_checkpoint_store.load(run_id)

    assert loaded.query_input.question == "Updated question"
    assert loaded.stage == RAGStage.INITIAL_EVIDENCE_READY

    postgres_checkpoint_store.delete(run_id)


def test_agentic_rag_recovers_from_postgres_checkpoint(postgres_checkpoint_store):
    run_id = "test-postgres-agentic-recovery"

    document = Document(
        page_content="Evidence restored from PostgreSQL",
        metadata={
            "corpus": "documentation",
            "chunk_id": "postgres-recovery-chunk",
        },
    )

    evidence = EvidenceItem(
        document=document,
        source="gap_rerank",
        query="Find missing information",
    )

    gap = RetrievalGap(
        missing_topic="Missing information",
        suggested_query="Find missing information",
        priority="important",
        reason="Required for a complete answer",
    )

    state = RAGState(
        query_input=QueryInput(
            question="Test PostgreSQL recovery"
        ),
        stage=RAGStage.FINAL_EVIDENCE_READY,
        generation_round=0,
        reranker_query="Test PostgreSQL recovery",
        evidence_pool=[evidence],
        final_evidence=[evidence],
        generated_answer=GeneratedAnswer(
            answer="Initial draft",
            gaps=[gap],
            is_complete=False,
        ),
        gaps=[gap],
    )

    postgres_checkpoint_store.save(
        run_id,
        create_checkpoint(state),
    )

    evidence_resolver = MagicMock(return_value=document)

    agentic_rag = AgenticRAG(
        query_router=MagicMock(),
        generation_llm=MagicMock(),
        doc_hybrid_retriever=MagicMock(),
        code_hybrid_retriever=MagicMock(),
        multi_corpus_retriever=MagicMock(),
        multi_query_strategy=MagicMock(),
        reranker=MagicMock(),
        code_vector_store=MagicMock(),
        evidence_selector=MagicMock(),
        checkpoint_store=postgres_checkpoint_store,
        evidence_resolver=evidence_resolver,
    )

    complete_answer = GeneratedAnswer(
        answer="Final complete answer",
        gaps=[],
        is_complete=True,
    )

    agentic_rag._generate_gap_round = MagicMock(
        side_effect=lambda state, observer: (
            setattr(state, "generated_answer", complete_answer),
            setattr(state, "generation_round", 1),
        )
    )

    result = agentic_rag.run(
        query_input=QueryInput(
            question="Test PostgreSQL recovery"
        ),
        run_id=run_id,
    )

    assert result == complete_answer

    # Recovery must not rerun routing, retrieval, or reranking.
    agentic_rag._query_router.route.assert_not_called()
    agentic_rag._reranker.rerank.assert_not_called()

    agentic_rag._generate_gap_round.assert_called_once()

    evidence_resolver.assert_any_call(
        "documentation",
        "postgres-recovery-chunk",
    )

    # Successful completion removes the persistent checkpoint.
    with pytest.raises(KeyError):
        postgres_checkpoint_store.load(run_id)

def test_postgres_checkpoint_remains_after_generation_crash(postgres_checkpoint_store):
    run_id = "test-postgres-generation-crash"

    state = RAGState(
        query_input=QueryInput(
            question="Test crash recovery"
        ),
        stage=RAGStage.FINAL_EVIDENCE_READY,
        generation_round=0,
        reranker_query="Test crash recovery",
        generated_answer=GeneratedAnswer(
            answer="Initial draft",
            gaps=[
                RetrievalGap(
                    missing_topic="Missing information",
                    suggested_query="Find missing information",
                    priority="important",
                    reason="Required for a complete answer",
                )
            ],
            is_complete=False,
        ),
    )

    checkpoint = create_checkpoint(state)

    postgres_checkpoint_store.save(
        run_id,
        checkpoint,
    )

    agentic_rag = AgenticRAG(
        query_router=MagicMock(),
        generation_llm=MagicMock(),
        doc_hybrid_retriever=MagicMock(),
        code_hybrid_retriever=MagicMock(),
        multi_corpus_retriever=MagicMock(),
        multi_query_strategy=MagicMock(),
        reranker=MagicMock(),
        code_vector_store=MagicMock(),
        evidence_selector=MagicMock(),
        checkpoint_store=postgres_checkpoint_store,
        evidence_resolver=MagicMock(),
    )

    # Simulate generation failing before the new checkpoint
    # can be persisted.
    agentic_rag._generate_gap_round = MagicMock(
        side_effect=RuntimeError("Simulated generation crash")
    )

    with pytest.raises(RuntimeError, match="Simulated generation crash"):
        agentic_rag.run(
            query_input=QueryInput(
                question="Test crash recovery"
            ),
            run_id=run_id,
        )

    # The previously persisted checkpoint must still exist.
    loaded = postgres_checkpoint_store.load(run_id)

    assert loaded.stage == RAGStage.FINAL_EVIDENCE_READY
    assert loaded.generation_round == 0
    assert loaded.generated_answer.answer == "Initial draft"
    assert loaded.generated_answer.is_complete is False

    # Cleanup
    postgres_checkpoint_store.delete(run_id)