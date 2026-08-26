from dotenv import load_dotenv

from ingestion.embedding_model import get_embedding_model
from ingestion.bm25_index import load_bm25_index
from ingestion.vector_store import build_vector_store

from models.query_input import QueryInput
from models.retrieval_plan import MetadataFilters, RetrievalPlan

from retrieval.dense_retriever import DenseRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.multi_corpus_retriever import MultiCorpusRetriever
from retrieval.multi_query_retrieval_strategy import MultiQueryRetrievalStrategy

from retrieval.retrieval_orchestrator import retrieve_candidates

from config import (
    DENSE_TOP_K,
    DOC_HYBRID_TOP_K,
    CODE_HYBRID_TOP_K,
    RRF_K,
    DOC_PERSISTENT_DIRECTORY,
    CODE_PERSISTENT_DIRECTORY,
    DOC_COLLECTION_NAME,
    CODE_COLLECTION_NAME,
    DOC_BM25_PATH,
    CODE_BM25_PATH,
)


def build_test_resources():
    load_dotenv()

    embedding_model = get_embedding_model()

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

    doc_bm25_index = load_bm25_index(DOC_BM25_PATH)
    code_bm25_index = load_bm25_index(CODE_BM25_PATH)

    doc_dense_retriever = DenseRetriever(
        vector_store=doc_vector_store,
        top_k=DENSE_TOP_K,
    )

    doc_bm25_retriever = BM25Retriever(
        bm25_index=doc_bm25_index,
    )

    doc_hybrid_retriever = HybridRetriever(
        bm25_retriever=doc_bm25_retriever,
        dense_retriever=doc_dense_retriever,
        top_k=DOC_HYBRID_TOP_K,
        rrf_k=RRF_K,
    )

    code_dense_retriever = DenseRetriever(
        vector_store=code_vector_store,
        top_k=DENSE_TOP_K,
    )

    code_bm25_retriever = BM25Retriever(
        bm25_index=code_bm25_index,
    )

    code_hybrid_retriever = HybridRetriever(
        bm25_retriever=code_bm25_retriever,
        dense_retriever=code_dense_retriever,
        top_k=CODE_HYBRID_TOP_K,
        rrf_k=RRF_K,
    )

    multi_corpus_retriever = MultiCorpusRetriever(
        retrievers=[
            doc_hybrid_retriever,
            code_hybrid_retriever,
        ]
    )

    multi_query_strategy = MultiQueryRetrievalStrategy(
        top_k=40,
        rrf_k=RRF_K,
    )

    return (
        doc_hybrid_retriever,
        code_hybrid_retriever,
        multi_corpus_retriever,
        multi_query_strategy,
        code_vector_store,
    )


def make_query_input():
    return QueryInput(
        question="How does interrupt work internally?",
    )


def test_both_corpora():
    (
        doc_hybrid,
        code_hybrid,
        multi_corpus,
        multi_query,
        code_vector_store,
    ) = build_test_resources()

    plan = RetrievalPlan(
        retrieval_mode="direct",
        reasoning="Test both corpora",
        metadata_filters=MetadataFilters(),
    )

    results = retrieve_candidates(
        query_input=make_query_input(),
        plan=plan,
        doc_hybrid_retriever=doc_hybrid,
        code_hybrid_retriever=code_hybrid,
        multi_corpus_retriever=multi_corpus,
        multi_query_strategy=multi_query,
        code_vector_store=code_vector_store,
    )

    assert results
    corpora = {doc.metadata.get("corpus") for doc in results}

    print("\nBoth corpora:", corpora)

    assert "documentation" in corpora
    assert "implementation" in corpora


def test_documentation_only():
    (
        doc_hybrid,
        code_hybrid,
        multi_corpus,
        multi_query,
        code_vector_store,
    ) = build_test_resources()

    plan = RetrievalPlan(
        retrieval_mode="direct",
        reasoning="Test documentation corpus",
        metadata_filters=MetadataFilters(
            corpus="documentation",
        ),
    )

    results = retrieve_candidates(
        query_input=make_query_input(),
        plan=plan,
        doc_hybrid_retriever=doc_hybrid,
        code_hybrid_retriever=code_hybrid,
        multi_corpus_retriever=multi_corpus,
        multi_query_strategy=multi_query,
        code_vector_store=code_vector_store,
    )

    assert results
    corpora = {doc.metadata.get("corpus") for doc in results}

    print("\nDocumentation only:", corpora)

    assert corpora == {"documentation"}


def test_implementation_only():
    (
        doc_hybrid,
        code_hybrid,
        multi_corpus,
        multi_query,
        code_vector_store,
    ) = build_test_resources()

    plan = RetrievalPlan(
        retrieval_mode="direct",
        reasoning="Test implementation corpus",
        metadata_filters=MetadataFilters(
            corpus="implementation",
        ),
    )

    results = retrieve_candidates(
        query_input=make_query_input(),
        plan=plan,
        doc_hybrid_retriever=doc_hybrid,
        code_hybrid_retriever=code_hybrid,
        multi_corpus_retriever=multi_corpus,
        multi_query_strategy=multi_query,
        code_vector_store=code_vector_store,
    )

    assert results
    corpora = {doc.metadata.get("corpus") for doc in results}

    print("\nImplementation only:", corpora)

    assert corpora == {"implementation"}


def test_exact_entity_retrieval():
    (
        doc_hybrid,
        code_hybrid,
        multi_corpus,
        multi_query,
        code_vector_store,
    ) = build_test_resources()

    plan = RetrievalPlan(
        retrieval_mode="direct",
        reasoning="Test exact implementation entity lookup",
        metadata_filters=MetadataFilters(
            corpus="implementation",
            function_names=["put_writes"],
        ),
    )

    results = retrieve_candidates(
        query_input=make_query_input(),
        plan=plan,
        doc_hybrid_retriever=doc_hybrid,
        code_hybrid_retriever=code_hybrid,
        multi_corpus_retriever=multi_corpus,
        multi_query_strategy=multi_query,
        code_vector_store=code_vector_store,
    )

    assert results

    print("\nExact entity retrieval:")
    for doc in results[:5]:
        print(
            doc.metadata.get("source"),
            doc.metadata.get("function_name"),
        )

    # Every result should be implementation code because the
    # normal retrieval is explicitly restricted to that corpus.
    assert all(
        doc.metadata.get("corpus") == "implementation"
        for doc in results
    )