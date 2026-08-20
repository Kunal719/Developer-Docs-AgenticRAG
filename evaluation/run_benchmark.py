from dotenv import load_dotenv
from pathlib import Path

from pipeline.initialization import initialize_resources
from pipeline.build_rag_pipeline import build_rag_pipeline
from pipeline.prepare_retrieval_resources import prepare_retrieval_resources

from ingestion.embedding_model import get_embedding_model
from ingestion.markdown_parser import load_documents, load_selected_documents
from ingestion.metadata_extractor import extract_metadata
from ingestion.code_chunker import chunk_python_documents
from ingestion.chunker import chunk_documents
from ingestion.python_loader import load_python_documents, load_selected_python_documents
from ingestion.code_metadata_extractor import extract_code_metadata

from retrieval.multi_query_retrieval_strategy import MultiQueryRetrievalStrategy

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
    JUDGE_MODEL_PROVIDER,
    JUDGE_MODEL_NAME,
    JUDGE_TEMP,
    RRF_K,
    MULTI_QUERY_TOP_K
)

from evaluation.benchmark_runner import BenchmarkRunner
from evaluation.result_writer import ResultWriter

from app.llm import get_llm


def main() -> None:
    """
    Run the evaluation benchmark against the developer documentation assistant.
    """

    load_dotenv()

    print("=" * 60)
    print("Developer Documentation Assistant Evaluation")
    print("=" * 60)
    print()

    try:
        print("\nInitializing embedding model...")
        embedding_model = get_embedding_model()
        print("Embedding model ready\n")

        # Initialize documentation repository and vector_store
        print("=" * 50)
        print("Initializing Documentation Corpus...")
        print("=" * 50)
        doc_repository_status, doc_vector_store = initialize_resources(
            GITHUB_DOC_REPO_URL, 
            LOCAL_DOC_REPO_PATH, 
            DOCUMENTATION_SPARSE_PATHS, 
            DOC_PERSISTENT_DIRECTORY, 
            DOC_COLLECTION_NAME, 
            embedding_model)

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
            chunker=chunk_documents
        )
        print("Documentation corpus ready\n")


        # Initialize implementation repository and vector_store
        print("=" * 50)
        print("Initializing Implementation Corpus...")
        print("=" * 50)
        code_repository_status, code_vector_store = initialize_resources(
            GITHUB_CODE_REPO_URL, 
            LOCAL_CODE_REPO_PATH, 
            IMPLEMENTATION_SPARSE_PATHS, 
            CODE_PERSISTENT_DIRECTORY, 
            CODE_COLLECTION_NAME, 
            embedding_model)

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
            chunker=chunk_python_documents
        )
        print("Implementation corpus ready\n")
        
        _, multi_corpus_retriever, cross_encoder_reranker, llm = build_rag_pipeline(
            doc_vector_store=doc_vector_store, 
            doc_bm25_index=doc_bm25_index,
            code_vector_store=code_vector_store,
            code_bm25_index=code_bm25_index
        )

        multi_query_retriever = MultiQueryRetrievalStrategy(
            retriever=multi_corpus_retriever,
            top_k=MULTI_QUERY_TOP_K,
            rrf_k=RRF_K
        )

        # Get judge LLM
        judge_llm = get_llm(model_name=JUDGE_MODEL_NAME, model_provider=JUDGE_MODEL_PROVIDER, max_tokens=5000, temperature=JUDGE_TEMP)

        print("\nRunning benchmark...")
        benchmark_runner = BenchmarkRunner(
            retriever=multi_corpus_retriever,
            multi_query_retriever=None,
            llm=llm,
            judge_llm=judge_llm,
            questions_path=Path("evaluation/questions/testing.json"),
            reranker=cross_encoder_reranker,
            retrieval_only=True
        )

        results = benchmark_runner.run()

        output_path = ResultWriter.save(results)

        print("\nBenchmark completed successfully.")
        print(f"\nResults saved to: {output_path}")

    except Exception as e:
        print(f"\nBenchmark execution failed:\n{e}")
        return


if __name__ == "__main__":
    main()