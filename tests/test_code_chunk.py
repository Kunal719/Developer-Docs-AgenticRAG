from pathlib import Path

from ingestion.code_chunker import chunk_python_documents
from ingestion.code_metadata_extractor import extract_code_metadata
from ingestion.python_loader import load_python_documents


documents = load_python_documents(Path("tests"))
documents = extract_code_metadata(documents, Path("tests"))

chunks = chunk_python_documents(documents)

for chunk in chunks:
    if chunk.metadata.get("source") == "jsonplus.py":
        print("=" * 80)
        print(chunk.metadata)
        print(chunk.page_content)