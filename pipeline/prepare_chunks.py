from langchain_core.documents import Document
from ingestion.chunker import chunk_documents, assign_chunk_id

from collections.abc import Callable
from pathlib import Path

from config import CHUNK_SIZE, CHUNK_OVERLAP

def prepare_chunks(
        source_path: Path,
        loader: Callable[[Path], list[Document]],
        metadata_extractor: Callable[[list[Document], Path], list[Document]],
        chunker: Callable[[list[Document]], list[Document]]
        ) -> list[Document]:
    """
    Load documents, enrich their metadata, and prepare chunked documents for indexing

    Args:
        loader: Function to load documents from a directory
        metadata_extractor: Function to enrich documents with metadata
        source_path: Path to the documentation directory
        chunker: Function used to convert loaded documents into retrieval chunks.
    
    Returns:
        List of chunked documents
    """
    print("\nLoading documentation...")
    documents = loader(source_path)
    print(f"Loaded {len(documents)} documents")

    print("\nExtracting metadata...")
    documents = metadata_extractor(documents, source_path)
    print("Metadata extracted")

    print("\nChunking documents...")
    chunks = chunker(documents)
    assign_chunk_id(chunks)
    print(f"Created {len(chunks)} chunks")

    return chunks