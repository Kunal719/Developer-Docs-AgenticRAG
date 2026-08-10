from ingestion.github_loader import RepositoryStatus
from langchain_chroma import Chroma
from langchain_core.documents import Document
from pipeline.prepare_chunks import prepare_chunks
from ingestion.bm25_index import build_bm25_index, save_bm25_index, load_bm25_index
from ingestion.document_index import update_index_metadata, detect_changes, ChangeSet
from ingestion.vector_store import delete_documents
from ingestion.chunker import chunk_documents, assign_chunk_id
from langchain_community.retrievers import BM25Retriever


from config import BM25_TOP_K, CHUNK_SIZE, CHUNK_OVERLAP
from pathlib import Path
from collections.abc import Callable

def prepare_retrieval_resources(
        repository_status: RepositoryStatus,
        source_path: Path,
        glob_pattern: str, 
        vector_store: Chroma,
        bm25_path: Path,
        index_path: Path,
        loader: Callable[[Path], list[Document]],
        selected_loader: Callable[[set[str], Path], list[Document]],
        metadata_extractor: Callable[[list[Document], Path], list[Document]]) -> tuple[Chroma, BM25Retriever]:
    """
    Prepare retrieval resources for the application.

    Depending on the repository status, this function updates the
    vector store, BM25 index, and metadata index before returning
    the ready-to-use retrieval resources.

    Returns:
        The vector store and BM25 index.

    """
    
    if repository_status is RepositoryStatus.CLONED:
        chunks = prepare_chunks(source_path, loader, metadata_extractor)

        # Create bm25 index
        print("\nCreating bm25 index...")
        bm25_index = build_bm25_index(chunks, top_k=BM25_TOP_K)
        save_bm25_index(bm25_index, bm25_path)
        print("bm25 index created")

        # Add documents to vector store
        print("\nAdding documents to vector store...")
        vector_store.add_documents(chunks)
        print("Documents added to vector store")

        # Creating the metadata index
        change_set = ChangeSet(
                        new={chunk.metadata["source"] for chunk in chunks},
                        updated=set(),
                        deleted=set(),
                        has_changes=False)
        print("\nUpdating metadata index...")
        update_index_metadata(
            index_path=index_path,
            change_set=change_set,
            source_path=source_path,
        )
        print("Metadata index updated")
            
    elif repository_status is RepositoryStatus.UPDATED:

        print("\nDetecting source changes...")
        change_set = detect_changes(source_path, index_path, glob_pattern)

        if change_set.has_changes:
            print("Detected source changes: "f"{len(change_set.new)} new, "f"{len(change_set.updated)} updated, "f"{len(change_set.deleted)} deleted.")
            # Check for deleted documents first, then new and updated documents
            if change_set.deleted or change_set.updated:
                print("\nDeleting old documents from vector store")
                sources_to_delete = change_set.deleted | change_set.updated
                delete_documents(vector_store, sources_to_delete)
                print("Old documents deleted from vector store")
        
            if change_set.new or change_set.updated:
                print("\nLoading and extracting metadata of new or updated documents...")
                new_and_updated_docs = change_set.new | change_set.updated
                documents = selected_loader(new_and_updated_docs, source_path)
                documents = metadata_extractor(documents, source_path)
                print(f"Loaded and extracted metadata of {len(documents)} new or updated documents")

                print("\nChunking documents and adding them to vector store...")
                print(f"Documents: {len(documents)}")

                chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                assign_chunk_id(chunks)
                print(f"Created {len(chunks)} chunks")
                vector_store.add_documents(chunks)
                print("New or updated documents added to vector store")

            # Update the metadata index
            print("\nUpdating metadata index...")
            update_index_metadata(
                index_path=index_path,
                change_set=change_set,
                source_path=source_path,
            )
            print("Metadata index updated")

            # Create BM25 index again
            print("\nCreating bm25 index...")
            chunks = prepare_chunks(source_path, loader, metadata_extractor)

            bm25_index = build_bm25_index(chunks, top_k=BM25_TOP_K)
            save_bm25_index(bm25_index, bm25_path)
            print("bm25 index created")
        
        else:
            print("No changes in source detected")
            print("\nLoading bm25 index...")
            bm25_index = load_bm25_index(bm25_path)
            print("bm25 index loaded")

    else:
        print("Repository already up to date. Skipping indexing")
        print("\nLoading bm25 index...")
        bm25_index = load_bm25_index(bm25_path)
        print("bm25 index loaded")

    return vector_store, bm25_index