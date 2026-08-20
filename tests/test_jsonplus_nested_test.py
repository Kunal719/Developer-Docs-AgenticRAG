from pathlib import Path

from ingestion.python_loader import load_python_documents
from ingestion.code_metadata_extractor import extract_code_metadata

from tests.code_chunker_nested_test import (
    experimental_chunk_python_documents
)


CODE_PATH = Path("data/raw/langgraph/libs")

TARGET_SOURCE = "checkpoint/langgraph/checkpoint/serde/jsonplus.py"


def main() -> None:
    print(f"Code path: {CODE_PATH.resolve()}")

    documents = load_python_documents(CODE_PATH)

    documents = extract_code_metadata(
        documents,
        CODE_PATH,
    )

    # Keep only jsonplus.py
    documents = [
        doc
        for doc in documents
        if doc.metadata["source"] == TARGET_SOURCE
    ]

    if not documents:
        print(f"Could not find: {TARGET_SOURCE}")
        return

    chunks = experimental_chunk_python_documents(documents)

    print("=" * 100)
    print(f"Source: {TARGET_SOURCE}")
    print(f"Chunks: {len(chunks)}")
    print("=" * 100)

    for chunk in chunks:
        print("\n" + "=" * 100)

        print(chunk.metadata)

        print("-" * 100)

        print(chunk.page_content)


if __name__ == "__main__":
    main()