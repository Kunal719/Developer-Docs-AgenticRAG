from dotenv import load_dotenv

from ingestion.embedding_model import get_embedding_model
from ingestion.bm25_index import load_bm25_index
from ingestion.vector_store import build_vector_store

from models.query_input import QueryInput
from models.generation import GeneratedAnswer, RetrievalGap
from models.retrieval_plan import RetrievalPlan

from retrieval.retrieval_orchestrator import retrieve_candidates
from retrieval.dense_retriever import DenseRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.multi_corpus_retriever import MultiCorpusRetriever
from retrieval.multi_query_retrieval_strategy import MultiQueryRetrievalStrategy

from reranking.cross_encoder import CrossEncoderReranker

from app.llm import get_llm
from pipeline.agentic_loop import AgenticRAG

from config import (
    DENSE_TOP_K,
    DOC_HYBRID_TOP_K,
    CODE_HYBRID_TOP_K,
    RRF_K,
    RERANKER_MODEL,
    RERANK_TOP_K,
    MODEL_NAME,
    MODEL_PROVIDER,
    MODEL_TEMP,
    DOC_PERSISTENT_DIRECTORY,
    CODE_PERSISTENT_DIRECTORY,
    DOC_COLLECTION_NAME,
    CODE_COLLECTION_NAME,
    DOC_BM25_PATH,
    CODE_BM25_PATH,
    MULTI_QUERY_TOP_K
)


def build_test_agent():
    load_dotenv()

    # -------------------------
    # Embeddings
    # -------------------------
    embedding_model = get_embedding_model()

    # -------------------------
    # Vector stores
    # -------------------------
    doc_vector_store = build_vector_store(
        embedding_model=embedding_model,
        persistent_dir=DOC_PERSISTENT_DIRECTORY,
        collection_name=DOC_COLLECTION_NAME,
        rebuild=False,
    )

    code_vector_store = build_vector_store(
        embedding_model=embedding_model,
        persistent_dir=CODE_PERSISTENT_DIRECTORY,
        collection_name=CODE_COLLECTION_NAME,
        rebuild=False,
    )

    # -------------------------
    # BM25 indexes
    # -------------------------
    doc_bm25_index = load_bm25_index(DOC_BM25_PATH)
    code_bm25_index = load_bm25_index(CODE_BM25_PATH)

    # -------------------------
    # Documentation retriever
    # -------------------------
    doc_dense = DenseRetriever(
        vector_store=doc_vector_store,
        top_k=DENSE_TOP_K,
    )

    doc_bm25 = BM25Retriever(
        bm25_index=doc_bm25_index,
    )

    doc_hybrid = HybridRetriever(
        bm25_retriever=doc_bm25,
        dense_retriever=doc_dense,
        top_k=DOC_HYBRID_TOP_K,
        rrf_k=RRF_K,
    )

    # -------------------------
    # Implementation retriever
    # -------------------------
    code_dense = DenseRetriever(
        vector_store=code_vector_store,
        top_k=DENSE_TOP_K,
    )

    code_bm25 = BM25Retriever(
        bm25_index=code_bm25_index,
    )

    code_hybrid = HybridRetriever(
        bm25_retriever=code_bm25,
        dense_retriever=code_dense,
        top_k=CODE_HYBRID_TOP_K,
        rrf_k=RRF_K,
    )

    # -------------------------
    # Multi-corpus retriever
    # -------------------------
    multi_corpus = MultiCorpusRetriever(
        retrievers=[
            doc_hybrid,
            code_hybrid,
        ]
    )

    # -------------------------
    # Multi-query strategy
    # -------------------------
    multi_query_strategy = MultiQueryRetrievalStrategy(
        top_k=MULTI_QUERY_TOP_K,
        rrf_k=RRF_K,
    )

    # -------------------------
    # Reranker
    # -------------------------
    reranker = CrossEncoderReranker(
        rerank_model=RERANKER_MODEL,
        top_k=RERANK_TOP_K,
    )

    # -------------------------
    # LLMs
    # -------------------------
    router_llm = get_llm(
        model_name=MODEL_NAME,
        model_provider=MODEL_PROVIDER,
        temperature=MODEL_TEMP,
    )

    generation_llm = get_llm(
        model_name=MODEL_NAME,
        model_provider=MODEL_PROVIDER,
        temperature=MODEL_TEMP,
    )

    # -------------------------
    # Agent
    # -------------------------
    query_input = QueryInput(
        question="How does interrupt work internally?",
    )

    return AgenticRAG(
        query_input=query_input,
        reranker=reranker,
        router_llm=router_llm,
        generation_llm=generation_llm,
        doc_hybrid_retriever=doc_hybrid,
        code_hybrid_retriever=code_hybrid,
        multi_corpus_retriever=multi_corpus,
        multi_query_strategy=multi_query_strategy,
        code_vector_store=code_vector_store,
    )


def test_agentic_first_round():

    agent = build_test_agent()

    result = agent.run()

    print("\n================ GENERATED ANSWER ================")
    print(result.answer)

    print("\nIs complete:", result.is_complete)

    print("\nGaps:")
    for gap in result.gaps:
        print("-", gap.missing_topic)
        print("  Query:", gap.suggested_query)
        print("  Filters:", gap.suggested_filters)

    # -------------------------
    # Structural assertions
    # -------------------------

    assert isinstance(result, GeneratedAnswer)

    assert isinstance(result.answer, str)
    assert result.answer.strip()

    assert isinstance(result.is_complete, bool)

    assert isinstance(result.gaps, list)

    for gap in result.gaps:
        assert isinstance(gap, RetrievalGap)
        assert gap.missing_topic
        assert gap.suggested_query


def test_gap_retrieval():
    agent = build_test_agent()

    # First round: obtain a real GeneratedAnswer and its gaps
    result = agent.run()

    assert result.gaps, "Expected at least one retrieval gap"

    gap = result.gaps[0]

    print("\n================ GAP ================")
    print("Missing topic:", gap.missing_topic)
    print("Suggested query:", gap.suggested_query)
    print("Suggested filters:", gap.suggested_filters)

    # Build a direct retrieval plan from the LLM-generated gap
    gap_plan = RetrievalPlan(
        retrieval_mode="direct",
        reasoning="Gap retrieval test",
        metadata_filters=gap.suggested_filters,
    )

    # Retrieve using the actual suggested_query
    gap_candidates = retrieve_candidates(
        query_input=agent._query_input,
        plan=gap_plan,
        doc_hybrid_retriever=agent._doc_hybrid_retriever,
        code_hybrid_retriever=agent._code_hybrid_retriever,
        multi_corpus_retriever=agent._multi_corpus_retriever,
        multi_query_strategy=agent._multi_query_strategy,
        code_vector_store=agent._code_vector_store,
        query_override=gap.suggested_query,
    )

    print("\n================ GAP RESULTS ================")
    print("Retrieved:", len(gap_candidates))

    for i, doc in enumerate(gap_candidates[:10], start=1):
        print(
            f"{i}. "
            f"corpus={doc.metadata.get('corpus')} | "
            f"source={doc.metadata.get('source')} | "
            f"module={doc.metadata.get('module_name')} | "
            f"class={doc.metadata.get('class_name')} | "
            f"function={doc.metadata.get('function_name')}"
        )

    assert gap_candidates


def test_agentic_gap_round():
    agent = build_test_agent()

    # Initial round + gap retrieval + second generation
    result = agent.run()

    print("\n================ FINAL ANSWER ================")
    print(result.answer)

    print("\nIs complete:", result.is_complete)

    print("\nRemaining gaps:", len(result.gaps))
    for gap in result.gaps:
        print("-", gap.missing_topic)
        print("  Query:", gap.suggested_query)

    # Structural checks
    assert isinstance(result, GeneratedAnswer)
    assert result.answer.strip()
    assert isinstance(result.is_complete, bool)
    assert isinstance(result.gaps, list)