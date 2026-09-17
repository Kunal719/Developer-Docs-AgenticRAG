from dotenv import load_dotenv
from pathlib import Path
import torch

from pipeline.initialization import initialize_resources
from pipeline.prepare_retrieval_resources import prepare_retrieval_resources

from pipeline.agentic_loop import AgenticRAG

from ingestion.embedding_model import get_embedding_model
from ingestion.markdown_parser import load_documents, load_selected_documents
from ingestion.metadata_extractor import extract_metadata
from ingestion.code_chunker import chunk_python_documents
from ingestion.chunker import chunk_documents
from ingestion.python_loader import (
    load_python_documents,
    load_selected_python_documents,
)
from ingestion.code_metadata_extractor import extract_code_metadata

from retrieval.dense_retriever import DenseRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.multi_corpus_retriever import MultiCorpusRetriever
from retrieval.multi_query_retrieval_strategy import MultiQueryRetrievalStrategy

from reranking.onnx_reranker import ONNXReranker
from reranking.cross_encoder import CrossEncoderReranker

from router.query_router import QueryRouter

from evaluation.agentic_benchmark_runner import AgenticBenchmarkRunner
from evaluation.result_writer import ResultWriter

from checkpoint.database import create_connection_pool
from checkpoint.postgres_store import PostgresCheckpointStore
from checkpoint.resolver import ChromaEvidenceResolver

from evidence.rerank_selector import RerankEvidenceSelector

from app.llm import get_llm

from config import (
    GITHUB_DOC_REPO_URL,
    GITHUB_CODE_REPO_URL,
    LOCAL_DOC_REPO_PATH,
    LOCAL_CODE_REPO_PATH,
    DOCUMENTATION_SPARSE_PATHS,
    IMPLEMENTATION_SPARSE_PATHS,
    DOC_PERSISTENT_DIRECTORY,
    CODE_PERSISTENT_DIRECTORY,
    DOC_COLLECTION_NAME,
    CODE_COLLECTION_NAME,
    DOCUMENTATION_PATH,
    IMPLEMENTATION_PATH,
    DOC_BM25_PATH,
    CODE_BM25_PATH,
    DOC_INDEX_PATH,
    CODE_INDEX_PATH,
    MODEL_NAME,
    MODEL_PROVIDER,
    MODEL_TEMP,
    JUDGE_MODEL_PROVIDER,
    JUDGE_MODEL_NAME,
    JUDGE_TEMP,
    RRF_K,
    DENSE_TOP_K,
    MULTI_QUERY_TOP_K,
    DOC_HYBRID_TOP_K,
    CODE_HYBRID_TOP_K,
    RERANKER_MODEL,
    RERANK_TOP_K
)


def main() -> None:
    """
    Run the Agentic RAG evaluation benchmark.
    """

    load_dotenv()

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)

    print("PyTorch threads:", torch.get_num_threads())
    print("PyTorch interop threads:", torch.get_num_interop_threads())

    print("=" * 60)
    print("Agentic Developer Documentation Assistant Evaluation")
    print("=" * 60)

    try:
        # ---------------------------------------------------------
        # Embedding model
        # ---------------------------------------------------------
        print("\nInitializing embedding model...")
        embedding_model = get_embedding_model()
        print("Embedding model ready\n")
        print(
            "[AFTER EMBEDDING] torch threads=",
            torch.get_num_threads(),
            "interop=",
            torch.get_num_interop_threads(),
        )

        # ---------------------------------------------------------
        # Documentation corpus
        # ---------------------------------------------------------
        print("=" * 50)
        print("Initializing Documentation Corpus...")
        print("=" * 50)

        doc_repository_status, doc_vector_store = initialize_resources(
            GITHUB_DOC_REPO_URL,
            LOCAL_DOC_REPO_PATH,
            DOCUMENTATION_SPARSE_PATHS,
            DOC_PERSISTENT_DIRECTORY,
            DOC_COLLECTION_NAME,
            embedding_model,
        )

        doc_vector_store, doc_bm25_index = prepare_retrieval_resources(
            repository_status=doc_repository_status,
            source_path=DOCUMENTATION_PATH,
            glob_pattern="**/*.mdx",
            vector_store=doc_vector_store,
            bm25_path=DOC_BM25_PATH,
            index_path=DOC_INDEX_PATH,
            loader=load_documents,
            selected_loader=load_selected_documents,
            metadata_extractor=extract_metadata,
            chunker=chunk_documents,
        )

        print("Documentation corpus ready\n")

        # ---------------------------------------------------------
        # Implementation corpus
        # ---------------------------------------------------------
        print("=" * 50)
        print("Initializing Implementation Corpus...")
        print("=" * 50)

        code_repository_status, code_vector_store = initialize_resources(
            GITHUB_CODE_REPO_URL,
            LOCAL_CODE_REPO_PATH,
            IMPLEMENTATION_SPARSE_PATHS,
            CODE_PERSISTENT_DIRECTORY,
            CODE_COLLECTION_NAME,
            embedding_model,
        )

        code_vector_store, code_bm25_index = prepare_retrieval_resources(
            repository_status=code_repository_status,
            source_path=IMPLEMENTATION_PATH,
            glob_pattern="**/*.py",
            vector_store=code_vector_store,
            bm25_path=CODE_BM25_PATH,
            index_path=CODE_INDEX_PATH,
            loader=load_python_documents,
            selected_loader=load_selected_python_documents,
            metadata_extractor=extract_code_metadata,
            chunker=chunk_python_documents,
        )

        print("Implementation corpus ready\n")

        # ---------------------------------------------------------
        # Runtime retrievers
        # ---------------------------------------------------------
        print("=" * 50)
        print("Building Agentic RAG Resources...")
        print("=" * 50)

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
            top_k=MULTI_QUERY_TOP_K,
            rrf_k=RRF_K,
        )

        # reranker = CrossEncoderReranker(
        #     rerank_model=RERANKER_MODEL,
        #     top_k=RERANK_TOP_K,
        # )

        checkpoint_pool = create_connection_pool()
        checkpoint_store = PostgresCheckpointStore(checkpoint_pool)

        evidence_resolver = ChromaEvidenceResolver(
            doc_store=doc_vector_store,
            code_store=code_vector_store,
        )

        reranker = CrossEncoderReranker(
            rerank_model="cross-encoder/ettin-reranker-32m-v1",
            top_k=RERANK_TOP_K,
            batch_size=16,
        )

        evidence_selector = RerankEvidenceSelector(reranker)


        # reranker = ONNXReranker(
        #     model_path="reranking/onnx/bge-reranker-base-o3",
        #     top_k=RERANK_TOP_K,
        #     batch_size=16,
        #     intra_op_num_threads=4,
        # )

        print("Retrieval resources ready\n")

        # ---------------------------------------------------------
        # LLMs
        # ---------------------------------------------------------
        print("Initializing router/generation LLM...")

        router_llm = get_llm(
            model_name=MODEL_NAME,
            model_provider=MODEL_PROVIDER,
            temperature=MODEL_TEMP,
            max_tokens=3000
        )

        generation_llm = get_llm(
            model_name=MODEL_NAME,
            model_provider=MODEL_PROVIDER,
            temperature=MODEL_TEMP,
        )

        print("Router/generation LLMs ready\n")

        query_router = QueryRouter(router_llm)

        # ---------------------------------------------------------
        # Judge
        # ---------------------------------------------------------
        print("Initializing judge LLM...")

        judge_llm = get_llm(
            model_name=JUDGE_MODEL_NAME,
            model_provider=JUDGE_MODEL_PROVIDER,
            max_tokens=8000,
            temperature=JUDGE_TEMP,
        )

        print("Judge LLM ready\n")

        # ---------------------------------------------------------
        # Benchmark
        # ---------------------------------------------------------
        print("Running Agentic benchmark...\n")

        benchmark_runner = AgenticBenchmarkRunner(
            questions_path=Path(
                "evaluation/questions/testing.json"
            ),
            router_llm=router_llm,
            generation_llm=generation_llm,
            judge_llm=judge_llm,
            doc_hybrid_retriever=doc_hybrid_retriever,
            code_hybrid_retriever=code_hybrid_retriever,
            multi_corpus_retriever=multi_corpus_retriever,
            multi_query_strategy=multi_query_strategy,
            code_vector_store=code_vector_store,
            reranker=reranker,
            evidence_selector=evidence_selector,
            checkpoint_store=checkpoint_store,
            evidence_resolver=evidence_resolver,
        )

        results = benchmark_runner.run()

        # ---------------------------------------------------------
        # Save results
        # ---------------------------------------------------------
        output_path = ResultWriter.save(results)

        print("\nBenchmark completed successfully.")
        print(f"\nResults saved to: {output_path}")

    except Exception as e:
        print("\nBenchmark execution failed:")
        print(e)
        return


if __name__ == "__main__":
    main()