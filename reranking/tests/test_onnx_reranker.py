from pathlib import Path
from time import perf_counter
from statistics import mean, median
import gc

from dotenv import load_dotenv

from pipeline.initialization import initialize_resources
from pipeline.prepare_retrieval_resources import prepare_retrieval_resources

from ingestion.embedding_model import get_embedding_model
from ingestion.markdown_parser import load_documents, load_selected_documents
from ingestion.metadata_extractor import extract_metadata
from ingestion.chunker import chunk_documents

from ingestion.python_loader import (
    load_python_documents,
    load_selected_python_documents,
)
from ingestion.code_metadata_extractor import extract_code_metadata
from ingestion.code_chunker import chunk_python_documents

from retrieval.dense_retriever import DenseRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.multi_corpus_retriever import MultiCorpusRetriever

from reranking.onnx_reranker import ONNXReranker

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
    DENSE_TOP_K,
    DOC_HYBRID_TOP_K,
    CODE_HYBRID_TOP_K,
    RRF_K,
)


MODEL_PATH = "reranking/onnx/bge-reranker-base"

QUERY = (
    "How does interrupt() resume execution after a "
    "Command(resume=...) is provided?"
)

# Candidate counts to measure.
DOCUMENT_COUNTS = [13, 21, 28, 38, 40]

# Thread counts to compare.
THREAD_COUNTS = [1, 2, 4, 8]

# Batch sizes to compare.
BATCH_SIZES = [4, 8, 16, 32]

# Number of measured repetitions per configuration.
RUNS = 3

WARMUP_RUNS = 1


def build_real_document_pool():
    """
    Initialize the same documentation/code resources used by the benchmark,
    then retrieve a large real-document pool once.

    Retrieval time is NOT included in reranker timings.
    """

    print("=" * 70)
    print("Initializing retrieval resources")
    print("=" * 70)

    embedding_model = get_embedding_model()

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

    multi_corpus = MultiCorpusRetriever(
        retrievers=[doc_hybrid, code_hybrid]
    )

    print("\nRetrieving real candidate pool...")

    # Use the existing production retrieval stack to obtain real documents.
    # Requesting a large hybrid pool here is only setup; this timing is not
    # included in reranker measurements.
    documents = multi_corpus.retrieve(
        query=QUERY,
        observer=None,
        filters=None,
    )

    # If the normal hybrid settings only produce 28 documents, supplement
    # with direct Chroma similarity search so the benchmark can test 38/40.
    if len(documents) < max(DOCUMENT_COUNTS):
        print(
            f"Hybrid retrieval produced {len(documents)} documents; "
            "supplementing from Chroma."
        )

        documents.extend(
            doc_vector_store.similarity_search(
                QUERY,
                k=max(DOCUMENT_COUNTS),
            )
        )

        documents.extend(
            code_vector_store.similarity_search(
                QUERY,
                k=max(DOCUMENT_COUNTS),
            )
        )

    # Preserve first occurrence by chunk_id.
    seen = set()
    unique_documents = []

    for document in documents:
        chunk_id = document.metadata.get("chunk_id")

        if chunk_id not in seen:
            seen.add(chunk_id)
            unique_documents.append(document)

    if len(unique_documents) < max(DOCUMENT_COUNTS):
        raise RuntimeError(
            f"Could only obtain {len(unique_documents)} unique documents; "
            f"need at least {max(DOCUMENT_COUNTS)}."
        )

    print(f"Prepared {len(unique_documents)} unique real documents.")

    return unique_documents


def benchmark_reranker(
    documents,
    *,
    threads,
    batch_size,
    document_count,
):
    reranker = ONNXReranker(
        model_path=MODEL_PATH,
        top_k=13,
        batch_size=batch_size,
        intra_op_num_threads=threads,
    )

    test_documents = documents[:document_count]

    # Warmup.
    for _ in range(WARMUP_RUNS):
        reranker.rerank(
            query=QUERY,
            documents=test_documents,
            top_k=13,
        )

    timings = []

    for _ in range(RUNS):
        start = perf_counter()

        reranker.rerank(
            query=QUERY,
            documents=test_documents,
            top_k=13,
        )

        timings.append(perf_counter() - start)

    del reranker
    gc.collect()

    return timings


def print_results(title, results):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print(
        f"{'Docs':>6} "
        f"{'Threads':>8} "
        f"{'Batch':>7} "
        f"{'Avg(s)':>10} "
        f"{'Median(s)':>12} "
        f"{'Min(s)':>10} "
        f"{'Max(s)':>10}"
    )

    print("-" * 70)

    for result in results:
        print(
            f"{result['docs']:>6} "
            f"{result['threads']:>8} "
            f"{result['batch']:>7} "
            f"{result['avg']:>10.3f} "
            f"{result['median']:>12.3f} "
            f"{result['min']:>10.3f} "
            f"{result['max']:>10.3f}"
        )


def main():
    load_dotenv()

    documents = build_real_document_pool()

    # -------------------------------------------------------------
    # Experiment 1: candidate count
    # Keep current production settings fixed:
    #   threads = 8
    #   batch_size = 16
    # -------------------------------------------------------------
    count_results = []

    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Candidate count")
    print("Threads=8, batch_size=16")
    print("=" * 70)

    for count in DOCUMENT_COUNTS:
        print(f"\nTesting {count} documents...")

        timings = benchmark_reranker(
            documents,
            threads=8,
            batch_size=16,
            document_count=count,
        )

        count_results.append(
            {
                "docs": count,
                "threads": 8,
                "batch": 16,
                "avg": mean(timings),
                "median": median(timings),
                "min": min(timings),
                "max": max(timings),
            }
        )

    print_results("Candidate Count Results", count_results)

    # -------------------------------------------------------------
    # Experiment 2: ONNX intra-op threads
    # Use 38 documents because this is close to your observed
    # initial candidate pool.
    # -------------------------------------------------------------
    thread_results = []

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: ONNX intra-op threads")
    print("Documents=38, batch_size=16")
    print("=" * 70)

    for threads in THREAD_COUNTS:
        print(f"\nTesting {threads} ONNX threads...")

        timings = benchmark_reranker(
            documents,
            threads=threads,
            batch_size=16,
            document_count=38,
        )

        thread_results.append(
            {
                "docs": 38,
                "threads": threads,
                "batch": 16,
                "avg": mean(timings),
                "median": median(timings),
                "min": min(timings),
                "max": max(timings),
            }
        )

    print_results("Thread Count Results", thread_results)

    # -------------------------------------------------------------
    # Experiment 3: batch size
    # Use 38 documents and the best/current thread setting.
    # Keep 8 here initially; use Experiment 2 to decide whether
    # this should be changed in the next run.
    # -------------------------------------------------------------
    batch_results = []

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Batch size")
    print("Documents=38, threads=8")
    print("=" * 70)

    for batch_size in BATCH_SIZES:
        print(f"\nTesting batch size {batch_size}...")

        timings = benchmark_reranker(
            documents,
            threads=8,
            batch_size=batch_size,
            document_count=38,
        )

        batch_results.append(
            {
                "docs": 38,
                "threads": 8,
                "batch": batch_size,
                "avg": mean(timings),
                "median": median(timings),
                "min": min(timings),
                "max": max(timings),
            }
        )

    print_results("Batch Size Results", batch_results)

    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
