from dotenv import load_dotenv

from ingestion.embedding_model import get_embedding_model
from pipeline.initialization import initialize_resources
from pipeline.build_rag_pipeline import build_rag_pipeline
from pipeline.prepare_retrieval_resources import prepare_retrieval_resources

from ingestion.markdown_parser import load_documents, load_selected_documents
from ingestion.metadata_extractor import extract_metadata

from ingestion.code_chunker import chunk_python_documents
from ingestion.chunker import chunk_documents

from ingestion.python_loader import load_python_documents, load_selected_python_documents
from ingestion.code_metadata_extractor import extract_code_metadata

from retrieval.rag_chain import invoke_chain, stream_chain

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
    CODE_INDEX_PATH
)


def main() -> None:
    """
    Run the developer docs assistant
    """

    load_dotenv()

    print("=" * 60)
    print("Developer Documentation Assistant")
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
        
        rag_chain, _, _ , _= build_rag_pipeline(
            doc_vector_store=doc_vector_store, 
            doc_bm25_index=doc_bm25_index,
            code_vector_store=code_vector_store,
            code_bm25_index=code_bm25_index
        )

    except Exception as e:
        print(f"\nFailed to initialize the application:\n{e}")
        return

    print("\n" + "=" * 60)
    print("Ask questions about LangGraph.")
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:
        question = input("\nYou > ").strip()

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print("\nAssistant > Goodbye!")
            break

        try:
            # retrieved_docs = hybrid_retriever.retrieve(question)

            # print("\n" + "=" * 80)
            # print("Retrieved Documents")
            # print("=" * 80)
            # for i, doc in enumerate(retrieved_docs, start=1):
            #     print(f"{i}. {doc.metadata['source']}")
            #     # print(doc.page_content[:300])

            # for i, doc in enumerate(retrieved_docs, start=1):
            #     print(f"\nDocument {i}")
            #     print(f"Source     : {doc.metadata.get('source')}")
            #     print(f"Title      : {doc.metadata.get('title')}")
            #     print(f"Description: {doc.metadata.get('description')}")
            #     print(f"Category   : {doc.metadata.get('category')}")
            #     print(f"Has Code   : {doc.metadata.get('has_code')}")
            #     print(f"Chunk Length: {len(doc.page_content)}")
            #     print("\nContent Preview:")
            #     print(doc.page_content[:300])   # First 300 characters
            #     print("\n" + "-" * 80)
            
            # response = rag_chain.invoke(question)
            # print(f"\nAssistant > {response}")

            # response = invoke_chain(rag_chain, question, observer=None)
            # print(f"\nAssistant > {response}")

            print("\nAssistant > ", end="", flush=True)

            chain_input = {
                "retrieval": question,
                "reranker_query": question,
            }
            for token in stream_chain(
                rag_chain,
                chain_input
            ):
                print(token, end="", flush=True)

            print()

        except Exception as e:
            print(f"\nAssistant > Failed to process your request.\nError: {e}")


if __name__ == "__main__":
    main()