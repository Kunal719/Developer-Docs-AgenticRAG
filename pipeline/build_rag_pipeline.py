from app.llm import get_llm
from langchain_core.language_models.chat_models import BaseChatModel
from retrieval.rag_chain import build_rag_chain
from retrieval.base import BaseRetriever
from retrieval.dense_retriever import DenseRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.multi_corpus_retriever import MultiCorpusRetriever
from reranking.cross_encoder import CrossEncoderReranker
from langchain_chroma import Chroma
from langchain_core.runnables import Runnable



# Configuration
from config import (
    DENSE_TOP_K,
    DOC_HYBRID_TOP_K,
    CODE_HYBRID_TOP_K,
    RRF_K,
    MODEL_NAME,
    MODEL_PROVIDER,
    MODEL_TEMP,
    RERANKER_MODEL,
    RERANK_TOP_K
)

def build_rag_pipeline(
        doc_vector_store: Chroma, 
        doc_bm25_index: BM25Retriever,
        code_vector_store: Chroma,
        code_bm25_index: BM25Retriever
        ) -> tuple[
        Runnable,
        HybridRetriever,
        HybridRetriever,
        MultiCorpusRetriever,
        Chroma,
        CrossEncoderReranker,
        BaseChatModel,
        ]:
    """
    Build the runtime RAG pipeline.

    Initializes the retrievers, reranker, language model,
    and constructs the RAG chain.

    Returns:
        The RAG chain, hybrid retriever, and reranker.
    """
    
    print("\nBuilding retrievers...")
    doc_dense_retriever = DenseRetriever(vector_store=doc_vector_store, top_k=DENSE_TOP_K)
    doc_bm25_retriever = BM25Retriever(bm25_index=doc_bm25_index)
    doc_hybrid_retriever = HybridRetriever(bm25_retriever=doc_bm25_retriever, dense_retriever=doc_dense_retriever, top_k=DOC_HYBRID_TOP_K, rrf_k=RRF_K)

    code_dense_retriever = DenseRetriever(vector_store=code_vector_store, top_k=DENSE_TOP_K)
    code_bm25_retriever = BM25Retriever(bm25_index=code_bm25_index)
    code_hybrid_retriever = HybridRetriever(bm25_retriever=code_bm25_retriever, dense_retriever=code_dense_retriever, top_k=CODE_HYBRID_TOP_K, rrf_k=RRF_K)

    multi_corpus_retriever = MultiCorpusRetriever(retrievers=[doc_hybrid_retriever, code_hybrid_retriever])
    
    print("Retrievers ready")

    print("\nInitializing LLM...")
    llm = get_llm(
        model_name=MODEL_NAME,
        model_provider=MODEL_PROVIDER,
        temperature=MODEL_TEMP
    )
    print("LLM ready")

    print("\nInitializing reranker...")
    cross_encoder_reranker = CrossEncoderReranker(
        rerank_model=RERANKER_MODEL,
        top_k=RERANK_TOP_K
    )
    print("Reranker ready")

    print("\nBuilding RAG pipeline...")
    rag_chain = build_rag_chain(
        retriever=multi_corpus_retriever,
        reranker=cross_encoder_reranker,
        llm=llm,
    )
    print("RAG pipeline ready")

    return (
    rag_chain,
    doc_hybrid_retriever,
    code_hybrid_retriever,
    multi_corpus_retriever,
    code_vector_store,
    cross_encoder_reranker,
    llm,
    )