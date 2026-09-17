import pytest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from checkpoint.store import InMemoryCheckpointStore
from models.query_input import QueryInput
from models.generation import GeneratedAnswer, RetrievalGap
from models.retrieval_plan import RetrievalPlan
from models.rag_state import RAGStage, RAGState
from models.evidence import EvidenceItem

from checkpoint.serialization import create_checkpoint

from pipeline.agentic_loop import AgenticRAG


def test_checkpoint_remains_after_final_generation_failure():
    # ---------------------------------------------------------
    # Arrange
    # ---------------------------------------------------------

    checkpoint_store = InMemoryCheckpointStore()

    query_router = MagicMock()

    query_router.route.return_value = RetrievalPlan(
        retrieval_mode="direct",
        reasoning="Test retrieval plan",
    )

    generation_llm = MagicMock()

    doc_hybrid_retriever = MagicMock()
    code_hybrid_retriever = MagicMock()
    multi_corpus_retriever = MagicMock()
    multi_query_strategy = MagicMock()
    reranker = MagicMock()
    code_vector_store = MagicMock()
    evidence_selector = MagicMock()
    evidence_resolver = MagicMock()

    document = Document(
        page_content="Test documentation",
        metadata={
            "corpus": "documentation",
            "chunk_id": "test-chunk-1",
        },
    )

    reranker.rerank.return_value = [document]

    gap = RetrievalGap(
        missing_topic="Missing information",
        suggested_query="Find missing information",
        priority="important",
        reason="Required for a complete answer",
    )

    initial_answer = GeneratedAnswer(
        answer="Initial draft",
        gaps=[gap],
        is_complete=False,
    )

    agentic_rag = AgenticRAG(
    query_router=query_router,
    generation_llm=generation_llm,
    doc_hybrid_retriever=doc_hybrid_retriever,
    code_hybrid_retriever=code_hybrid_retriever,
    multi_corpus_retriever=multi_corpus_retriever,
    multi_query_strategy=multi_query_strategy,
    reranker=reranker,
    code_vector_store=code_vector_store,
    evidence_selector=evidence_selector,
    checkpoint_store=checkpoint_store,
    evidence_resolver=evidence_resolver,
    )

    gap_evidence = EvidenceItem(
        document=document,
        source="gap_rerank",
        query=gap.suggested_query,
        gap=gap,
    )

    evidence_selector.select_evidence.return_value = [
        gap_evidence
    ]

    # First generation succeeds.
    # Second generation fails.
    agentic_rag._generate = MagicMock(
        side_effect=[
            initial_answer,
            RuntimeError("Final generation failed"),
        ]
    )

    with patch(
        "pipeline.agentic_loop.retrieve_candidates",
        return_value=[document],
    ), patch.object(
        agentic_rag,
        "_process_gap",
        return_value=([gap_evidence], MagicMock()),
    ):

        # ---------------------------------------------------------
        # Act
        # ---------------------------------------------------------

        with pytest.raises(RuntimeError, match="Final generation failed"):
            agentic_rag.run(
                query_input=QueryInput(
                    question="Test question"
                )
            )

    # ---------------------------------------------------------
    # Assert
    # ---------------------------------------------------------

    # A checkpoint must remain because the run did not complete.
    assert len(checkpoint_store._checkpoints) == 1

    run_id = next(iter(checkpoint_store._checkpoints))

    checkpoint = checkpoint_store.load(run_id)

    assert checkpoint.stage == RAGStage.FINAL_EVIDENCE_READY
    assert checkpoint.query_input.question == "Test question"
    assert checkpoint.reranker_query is not None

    assert len(checkpoint.initial_evidence) == 1
    assert checkpoint.initial_evidence[0].chunk_id == "test-chunk-1"

    assert len(checkpoint.evidence_pool) == 2
    assert len(checkpoint.final_evidence) == 1

    assert checkpoint.generated_answer == initial_answer
    assert checkpoint.gaps == [gap]

def test_checkpoint_deleted_after_successful_completion():
    # ---------------------------------------------------------
    # Arrange
    # ---------------------------------------------------------

    checkpoint_store = InMemoryCheckpointStore()

    query_router = MagicMock()

    query_router.route.return_value = RetrievalPlan(
        retrieval_mode="direct",
        reasoning="Test retrieval plan",
    )

    generation_llm = MagicMock()

    doc_hybrid_retriever = MagicMock()
    code_hybrid_retriever = MagicMock()
    multi_corpus_retriever = MagicMock()
    multi_query_strategy = MagicMock()
    reranker = MagicMock()
    code_vector_store = MagicMock()
    evidence_selector = MagicMock()
    evidence_resolver = MagicMock()

    document = Document(
        page_content="Test documentation",
        metadata={
            "corpus": "documentation",
            "chunk_id": "test-chunk-1",
        },
    )

    reranker.rerank.return_value = [document]

    complete_answer = GeneratedAnswer(
        answer="Complete answer",
        gaps=[],
        is_complete=True,
    )

    agentic_rag = AgenticRAG(
    query_router=query_router,
    generation_llm=generation_llm,
    doc_hybrid_retriever=doc_hybrid_retriever,
    code_hybrid_retriever=code_hybrid_retriever,
    multi_corpus_retriever=multi_corpus_retriever,
    multi_query_strategy=multi_query_strategy,
    reranker=reranker,
    code_vector_store=code_vector_store,
    evidence_selector=evidence_selector,
    checkpoint_store=checkpoint_store,
    evidence_resolver=evidence_resolver,
    )

    # Initial generation immediately produces a complete answer.
    agentic_rag._generate = MagicMock(
        return_value=complete_answer
    )

    with patch(
        "pipeline.agentic_loop.retrieve_candidates",
        return_value=[document],
    ):

        # ---------------------------------------------------------
        # Act
        # ---------------------------------------------------------

        result = agentic_rag.run(
            query_input=QueryInput(
                question="Test question"
            )
        )

    # ---------------------------------------------------------
    # Assert
    # ---------------------------------------------------------

    assert result == complete_answer

    # The run completed successfully, so its checkpoint
    # should have been deleted.
    assert len(checkpoint_store._checkpoints) == 0

def test_resume_from_final_evidence_ready_with_complete_answer():
    # ---------------------------------------------------------
    # Arrange
    # ---------------------------------------------------------

    checkpoint_store = InMemoryCheckpointStore()

    query_router = MagicMock()
    generation_llm = MagicMock()
    doc_hybrid_retriever = MagicMock()
    code_hybrid_retriever = MagicMock()
    multi_corpus_retriever = MagicMock()
    multi_query_strategy = MagicMock()
    reranker = MagicMock()
    code_vector_store = MagicMock()
    evidence_selector = MagicMock()
    evidence_resolver = MagicMock()

    document = Document(
        page_content="Test documentation",
        metadata={
            "corpus": "documentation",
            "chunk_id": "test-chunk-1",
        },
    )

    evidence = EvidenceItem(
        document=document,
        source="gap_rerank",
        query="Find missing information",
        gap=None,
    )

    draft_answer = GeneratedAnswer(
        answer="Draft answer",
        gaps=[
            RetrievalGap(
                missing_topic="Missing information",
                suggested_query="Find missing information",
                priority="important",
                reason="Required for a complete answer",
            )
        ],
        is_complete=False,
    )

    final_answer = GeneratedAnswer(
        answer="Final answer",
        gaps=[],
        is_complete=True,
    )

    # Resolver reconstructs the document stored in the checkpoint.
    evidence_resolver.return_value = document

    # Build the checkpoint representing a crash after
    # FINAL_EVIDENCE_READY.
    agentic_rag = AgenticRAG(
        query_router=query_router,
        generation_llm=generation_llm,
        doc_hybrid_retriever=doc_hybrid_retriever,
        code_hybrid_retriever=code_hybrid_retriever,
        multi_corpus_retriever=multi_corpus_retriever,
        multi_query_strategy=multi_query_strategy,
        reranker=reranker,
        code_vector_store=code_vector_store,
        evidence_selector=evidence_selector,
        checkpoint_store=checkpoint_store,
        evidence_resolver=evidence_resolver,
    )

    agentic_rag._generate_gap_round = MagicMock(
        side_effect=lambda state, observer: (
            setattr(state, "generated_answer", final_answer),
            setattr(state, "generation_round", 1),
        )
    )

    # Save a checkpoint as if the process had crashed
    # immediately after FINAL_EVIDENCE_READY.
    checkpoint = agentic_rag._checkpoint_store

    from checkpoint.serialization import create_checkpoint
    from models.rag_state import RAGState

    state = RAGState(
        query_input=QueryInput(
            question="Test question"
        ),
        stage=RAGStage.FINAL_EVIDENCE_READY,
        generation_round=0,
        reranker_query="Test question",
        final_evidence=[evidence],
        generated_answer=draft_answer,
        gaps=draft_answer.gaps,
    )

    checkpoint.save(
        "test-run",
        create_checkpoint(state),
    )

    # ---------------------------------------------------------
    # Act
    # ---------------------------------------------------------

    result = agentic_rag.run(
        query_input=QueryInput(question="Test question"),
        run_id="test-run",
    )

    # ---------------------------------------------------------
    # Assert
    # ---------------------------------------------------------

    assert result == final_answer

    agentic_rag._generate_gap_round.assert_called_once()

    # The run completed, so the checkpoint must be deleted.
    assert len(checkpoint_store._checkpoints) == 0

def test_resume_from_gap_generation_completed_continues_gap_loop():
    # ---------------------------------------------------------
    # Arrange
    # ---------------------------------------------------------

    checkpoint_store = InMemoryCheckpointStore()

    query_router = MagicMock()
    generation_llm = MagicMock()
    doc_hybrid_retriever = MagicMock()
    code_hybrid_retriever = MagicMock()
    multi_corpus_retriever = MagicMock()
    multi_query_strategy = MagicMock()
    reranker = MagicMock()
    code_vector_store = MagicMock()
    evidence_selector = MagicMock()
    evidence_resolver = MagicMock()

    document = Document(
        page_content="Test documentation",
        metadata={
            "corpus": "documentation",
            "chunk_id": "test-chunk-1",
        },
    )

    next_gap = RetrievalGap(
        missing_topic="Another missing topic",
        suggested_query="Find next missing information",
        priority="important",
        reason="Still required for a complete answer",
    )

    # This is the answer produced by the generation that
    # already completed before the crash.
    generated_answer = GeneratedAnswer(
        answer="Answer after previous gap round",
        gaps=[next_gap],
        is_complete=False,
    )

    state = RAGState(
        query_input=QueryInput(
            question="Test question"
        ),
        stage=RAGStage.GAP_GENERATION_COMPLETED,
        generation_round=1,
        reranker_query="Test question",
        generated_answer=generated_answer,
    )

    checkpoint_store.save(
        "test-run",
        create_checkpoint(state),
    )

    agentic_rag = AgenticRAG(
        query_router=query_router,
        generation_llm=generation_llm,
        doc_hybrid_retriever=doc_hybrid_retriever,
        code_hybrid_retriever=code_hybrid_retriever,
        multi_corpus_retriever=multi_corpus_retriever,
        multi_query_strategy=multi_query_strategy,
        reranker=reranker,
        code_vector_store=code_vector_store,
        evidence_selector=evidence_selector,
        checkpoint_store=checkpoint_store,
        evidence_resolver=evidence_resolver,
    )

    next_gap_evidence = EvidenceItem(
        document=document,
        source="gap_rerank",
        query=next_gap.suggested_query,
        gap=next_gap,
    )

    evidence_selector.select_evidence.return_value = [
        next_gap_evidence
    ]

    # The NEXT generation produces a complete answer.
    complete_answer = GeneratedAnswer(
        answer="Final complete answer",
        gaps=[],
        is_complete=True,
    )

    agentic_rag._generate_gap_round = MagicMock(
        side_effect=lambda state, observer: (
            setattr(state, "generated_answer", complete_answer),
            setattr(state, "generation_round", 2),
        )
    )

    with patch.object(
        agentic_rag,
        "_process_gap",
        return_value=([next_gap_evidence], MagicMock()),
    ):

        # ---------------------------------------------------------
        # Act
        # ---------------------------------------------------------

        result = agentic_rag.run(
            query_input=QueryInput(
                question="Test question"
            ),
            run_id="test-run",
        )

    # ---------------------------------------------------------
    # Assert
    # ---------------------------------------------------------

    assert result == complete_answer

    # We resumed AFTER a generation had already completed.
    # Therefore the resumed run must not repeat that generation.
    #
    # It should process the gap from the persisted generated
    # answer and then perform exactly one NEW generation.
    agentic_rag._generate_gap_round.assert_called_once()

    # The run completed successfully, so the checkpoint
    # should have been deleted.
    assert len(checkpoint_store._checkpoints) == 0

def test_resume_from_routed_continues_initial_retrieval():
    # ---------------------------------------------------------
    # Arrange
    # ---------------------------------------------------------

    checkpoint_store = InMemoryCheckpointStore()

    query_router = MagicMock()
    generation_llm = MagicMock()
    doc_hybrid_retriever = MagicMock()
    code_hybrid_retriever = MagicMock()
    multi_corpus_retriever = MagicMock()
    multi_query_strategy = MagicMock()
    reranker = MagicMock()
    code_vector_store = MagicMock()
    evidence_selector = MagicMock()
    evidence_resolver = MagicMock()

    retrieval_plan = RetrievalPlan(
        retrieval_mode="direct",
        reasoning="Persisted retrieval plan",
    )

    document = Document(
        page_content="Retrieved documentation",
        metadata={
            "corpus": "documentation",
            "chunk_id": "test-chunk-1",
        },
    )

    reranker.rerank.return_value = [document]

    state = RAGState(
        query_input=QueryInput(
            question="Test question"
        ),
        stage=RAGStage.ROUTED,
        retrieval_plan=retrieval_plan,
    )

    checkpoint_store.save(
        "test-run",
        create_checkpoint(state),
    )

    evidence_resolver.return_value = document

    agentic_rag = AgenticRAG(
        query_router=query_router,
        generation_llm=generation_llm,
        doc_hybrid_retriever=doc_hybrid_retriever,
        code_hybrid_retriever=code_hybrid_retriever,
        multi_corpus_retriever=multi_corpus_retriever,
        multi_query_strategy=multi_query_strategy,
        reranker=reranker,
        code_vector_store=code_vector_store,
        evidence_selector=evidence_selector,
        checkpoint_store=checkpoint_store,
        evidence_resolver=evidence_resolver,
    )

    # ---------------------------------------------------------
    # Act
    # ---------------------------------------------------------

    with patch(
        "pipeline.agentic_loop.retrieve_candidates",
        return_value=[document],
    ):

        # We only want to test the recovery through retrieval.
        # Stop execution after INITIAL_EVIDENCE_READY.
        with patch.object(
            agentic_rag,
            "_generate",
            side_effect=RuntimeError("Stop after retrieval"),
        ):

            with pytest.raises(
                RuntimeError,
                match="Stop after retrieval",
            ):
                agentic_rag.run(
                    query_input=QueryInput(
                        question="Test question"
                    ),
                    run_id="test-run",
                )

    # ---------------------------------------------------------
    # Assert
    # ---------------------------------------------------------

    # Routing must NOT happen again.
    query_router.route.assert_not_called()

    # Retrieval should happen exactly once during recovery.
    reranker.rerank.assert_called_once()

    # The persisted retrieval plan must have been used.
    retrieved_checkpoint = checkpoint_store.load("test-run")

    assert (
        retrieved_checkpoint.retrieval_plan
        == retrieval_plan
    )

    # Retrieval completed before generation failed, so the
    # checkpoint should now be INITIAL_EVIDENCE_READY.
    assert (
        retrieved_checkpoint.stage
        == RAGStage.INITIAL_EVIDENCE_READY
    )

    assert len(retrieved_checkpoint.initial_evidence) == 1
    assert (
        retrieved_checkpoint.initial_evidence[0].chunk_id
        == "test-chunk-1"
    )


def test_resume_from_initial_evidence_ready_continues_initial_generation():
    # ---------------------------------------------------------
    # Arrange
    # ---------------------------------------------------------

    checkpoint_store = InMemoryCheckpointStore()

    query_router = MagicMock()
    generation_llm = MagicMock()
    doc_hybrid_retriever = MagicMock()
    code_hybrid_retriever = MagicMock()
    multi_corpus_retriever = MagicMock()
    multi_query_strategy = MagicMock()
    reranker = MagicMock()
    code_vector_store = MagicMock()
    evidence_selector = MagicMock()
    evidence_resolver = MagicMock()

    document = Document(
        page_content="Persisted initial evidence",
        metadata={
            "corpus": "documentation",
            "chunk_id": "test-chunk-1",
        },
    )

    initial_evidence = EvidenceItem(
        document=document,
        source="initial",
        query="Test question",
        gap=None,
    )

    draft_answer = GeneratedAnswer(
        answer="Initial complete answer",
        gaps=[],
        is_complete=True,
    )

    state = RAGState(
        query_input=QueryInput(
            question="Test question"
        ),
        stage=RAGStage.INITIAL_EVIDENCE_READY,
        generation_round=0,
        reranker_query="Test question",
        initial_evidence=[initial_evidence],
        evidence_pool=[initial_evidence],
    )

    checkpoint_store.save(
        "test-run",
        create_checkpoint(state),
    )

    evidence_resolver.return_value = document

    agentic_rag = AgenticRAG(
        query_router=query_router,
        generation_llm=generation_llm,
        doc_hybrid_retriever=doc_hybrid_retriever,
        code_hybrid_retriever=code_hybrid_retriever,
        multi_corpus_retriever=multi_corpus_retriever,
        multi_query_strategy=multi_query_strategy,
        reranker=reranker,
        code_vector_store=code_vector_store,
        evidence_selector=evidence_selector,
        checkpoint_store=checkpoint_store,
        evidence_resolver=evidence_resolver,
    )

    agentic_rag._generate = MagicMock(
        return_value=draft_answer
    )

    # ---------------------------------------------------------
    # Act
    # ---------------------------------------------------------

    result = agentic_rag.run(
        query_input=QueryInput(
            question="Test question"
        ),
        run_id="test-run",
    )

    # ---------------------------------------------------------
    # Assert
    # ---------------------------------------------------------

    assert result == draft_answer

    # Initial generation should happen exactly once.
    agentic_rag._generate.assert_called_once()

    # Routing must NOT happen again.
    query_router.route.assert_not_called()

    # Retrieval/reranking must NOT happen again.
    reranker.rerank.assert_not_called()

    # Both initial_evidence and evidence_pool are restored
    # from the checkpoint, so the resolver is called once
    # for each collection.
    assert evidence_resolver.call_count == 2

    evidence_resolver.assert_any_call(
        "documentation",
        "test-chunk-1",
    )

    # The run completed successfully, so the checkpoint
    # should have been deleted.
    assert len(checkpoint_store._checkpoints) == 0

def test_resume_from_gaps_processed_continues_final_evidence_selection():
    # ---------------------------------------------------------
    # Arrange
    # ---------------------------------------------------------

    checkpoint_store = InMemoryCheckpointStore()

    query_router = MagicMock()
    generation_llm = MagicMock()
    doc_hybrid_retriever = MagicMock()
    code_hybrid_retriever = MagicMock()
    multi_corpus_retriever = MagicMock()
    multi_query_strategy = MagicMock()
    reranker = MagicMock()
    code_vector_store = MagicMock()
    evidence_selector = MagicMock()
    evidence_resolver = MagicMock()

    document = Document(
        page_content="Retrieved gap evidence",
        metadata={
            "corpus": "documentation",
            "chunk_id": "test-chunk-1",
        },
    )

    gap = RetrievalGap(
        missing_topic="Missing information",
        suggested_query="Find missing information",
        priority="important",
        reason="Required for a complete answer",
    )

    gap_evidence = EvidenceItem(
        document=document,
        source="gap_rerank",
        query=gap.suggested_query,
        gap=gap,
    )

    generated_answer = GeneratedAnswer(
        answer="Initial draft",
        gaps=[gap],
        is_complete=False,
    )

    state = RAGState(
        query_input=QueryInput(
            question="Test question"
        ),
        stage=RAGStage.GAPS_PROCESSED,
        generation_round=0,
        reranker_query="Test question",
        evidence_pool=[gap_evidence],
        generated_answer=generated_answer,
        gaps=[gap],
    )

    checkpoint_store.save(
        "test-run",
        create_checkpoint(state),
    )

    evidence_resolver.return_value = document

    evidence_selector.select_evidence.return_value = [
        gap_evidence
    ]

    agentic_rag = AgenticRAG(
        query_router=query_router,
        generation_llm=generation_llm,
        doc_hybrid_retriever=doc_hybrid_retriever,
        code_hybrid_retriever=code_hybrid_retriever,
        multi_corpus_retriever=multi_corpus_retriever,
        multi_query_strategy=multi_query_strategy,
        reranker=reranker,
        code_vector_store=code_vector_store,
        evidence_selector=evidence_selector,
        checkpoint_store=checkpoint_store,
        evidence_resolver=evidence_resolver,
    )

    # The next generation completes the answer.
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

    # ---------------------------------------------------------
    # Act
    # ---------------------------------------------------------

    result = agentic_rag.run(
        query_input=QueryInput(
            question="Test question"
        ),
        run_id="test-run",
    )

    # ---------------------------------------------------------
    # Assert
    # ---------------------------------------------------------

    assert result == complete_answer

    # Routing must NOT happen during recovery.
    query_router.route.assert_not_called()

    # Retrieval/reranking must NOT happen during recovery.
    reranker.rerank.assert_not_called()

    # Final evidence selection must happen exactly once.
    evidence_selector.select_evidence.assert_called_once()

    # The next generation must happen exactly once.
    agentic_rag._generate_gap_round.assert_called_once()

    # The run completed successfully, so the checkpoint
    # should have been deleted.
    assert len(checkpoint_store._checkpoints) == 0


def test_restore_state_fails_when_checkpoint_evidence_cannot_be_resolved():
    # ---------------------------------------------------------
    # Arrange
    # ---------------------------------------------------------

    checkpoint_store = InMemoryCheckpointStore()
    evidence_resolver = MagicMock()

    document = Document(
        page_content="Retrieved evidence",
        metadata={
            "corpus": "documentation",
            "chunk_id": "missing-chunk",
        },
    )

    evidence_item = EvidenceItem(
        document=document,
        source="initial_rerank",
        query="Test question",
    )

    state = RAGState(
        query_input=QueryInput(
            question="Test question"
        ),
        stage=RAGStage.INITIAL_EVIDENCE_READY,
        generation_round=0,
        reranker_query="Test question",
        initial_evidence=[evidence_item],
        evidence_pool=[evidence_item],
    )

    checkpoint_store.save(
        "test-run",
        create_checkpoint(state),
    )

    evidence_resolver.side_effect = LookupError(
        "Could not find document with chunk_id: missing-chunk"
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
        checkpoint_store=checkpoint_store,
        evidence_resolver=evidence_resolver,
    )

    # ---------------------------------------------------------
    # Act / Assert
    # ---------------------------------------------------------

    with pytest.raises(
        LookupError,
        match="Could not find document with chunk_id: missing-chunk",
    ):
        agentic_rag.run(
            query_input=QueryInput(
                question="Test question"
            ),
            run_id="test-run",
        )

    # Recovery must fail rather than silently retrieving
    # replacement evidence.
    evidence_resolver.assert_called_once_with(
        "documentation",
        "missing-chunk",
    )