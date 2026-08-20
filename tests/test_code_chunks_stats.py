from collections import Counter
from pathlib import Path
from statistics import mean, median

import tiktoken
from langchain_core.documents import Document

from tests.test_code_chunk import chunk_python_documents
from ingestion.python_loader import load_python_documents
from ingestion.code_metadata_extractor import extract_code_metadata


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CODE_PATH = Path("data/raw/langgraph/libs")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def percentile(values: list[int], percentile_value: float) -> float:
    """
    Calculate a percentile using linear interpolation.
    """
    if not values:
        return 0.0

    sorted_values = sorted(values)

    position = (len(sorted_values) - 1) * percentile_value
    lower = int(position)
    upper = min(lower + 1, len(sorted_values))

    if lower == upper:
        return float(sorted_values[lower])

    weight = position - lower

    return (
        sorted_values[lower]
        + weight * (sorted_values[upper] - sorted_values[lower])
    )


def print_distribution(
    name: str,
    values: list[int],
) -> None:
    """
    Print descriptive statistics for a distribution.
    """
    if not values:
        print(f"\n{name}: no values")
        return

    print(f"\n{name}")
    print("-" * 60)
    print(f"Mean:       {mean(values):,.2f}")
    print(f"Median:     {median(values):,.2f}")
    print(f"P90:        {percentile(values, 0.90):,.2f}")
    print(f"P95:        {percentile(values, 0.95):,.2f}")
    print(f"P99:        {percentile(values, 0.99):,.2f}")
    print(f"Maximum:    {max(values):,}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Code path: {CODE_PATH.resolve()}")

    documents = load_python_documents(CODE_PATH)

    documents = extract_code_metadata(documents, CODE_PATH)

    print(f"Python files: {len(documents):,}")

    chunks = chunk_python_documents(documents)

    print(f"Code chunks:  {len(chunks):,}")

    # -----------------------------------------------------------------------
    # Chunk type statistics
    # -----------------------------------------------------------------------

    chunk_types = Counter(
        chunk.metadata.get("chunk_type")
        for chunk in chunks
    )

    print("\nChunk types")
    print("-" * 60)

    for chunk_type, count in sorted(chunk_types.items()):
        print(f"{chunk_type:<20} {count:>8,}")

    # -----------------------------------------------------------------------
    # Character lengths
    # -----------------------------------------------------------------------

    character_lengths = [
        len(chunk.page_content)
        for chunk in chunks
    ]

    print_distribution(
        "Character length",
        character_lengths,
    )

    # -----------------------------------------------------------------------
    # Token lengths
    # -----------------------------------------------------------------------

    encoding = tiktoken.get_encoding("cl100k_base")

    token_lengths = [
        len(encoding.encode(chunk.page_content))
        for chunk in chunks
    ]

    print_distribution(
        "Token length",
        token_lengths,
    )

    # -----------------------------------------------------------------------
    # Largest chunks
    # -----------------------------------------------------------------------

    largest_chunks = sorted(
        chunks,
        key=lambda chunk: len(chunk.page_content),
        reverse=True,
    )[:10]

    print("\nTop 10 largest chunks")
    print("-" * 100)

    for rank, chunk in enumerate(largest_chunks, start=1):
        print(
            f"{rank:>2}. "
            f"{chunk.metadata.get('qualified_name', '<unknown>'):<60} "
            f"{len(chunk.page_content):>8,} chars | "
            f"{len(encoding.encode(chunk.page_content)):>7,} tokens | "
            f"{chunk.metadata.get('source')}"
        )

    # -----------------------------------------------------------------------
    # Duplicate chunk IDs
    # -----------------------------------------------------------------------

    # We intentionally calculate the same IDs your production pipeline uses.
    import hashlib

    chunk_ids = []

    for chunk in chunks:
        hash_input = (
            f"{chunk.metadata['source']}::{chunk.metadata.get('qualified_name', '')}::{chunk.page_content}"
        )

        chunk_id = hashlib.sha256(
            hash_input.encode("utf-8")
        ).hexdigest()

        chunk_ids.append(chunk_id)

    id_counts = Counter(chunk_ids)

    duplicate_ids = {
        chunk_id: count
        for chunk_id, count in id_counts.items()
        if count > 1
    }

    print("\nChunk ID integrity")
    print("-" * 60)
    print(f"Unique chunk IDs:     {len(id_counts):,}")
    print(f"Duplicate chunk IDs:  {len(duplicate_ids):,}")

    print("\nDuplicate chunk details")
    print("-" * 100)

    for duplicate_id in duplicate_ids:
        matching_chunks = [
            chunk
            for chunk, chunk_id in zip(chunks, chunk_ids)
            if chunk_id == duplicate_id
        ]

        for chunk in matching_chunks:
            print(
                f"source:          {chunk.metadata.get('source')}\n"
                f"chunk_type:      {chunk.metadata.get('chunk_type')}\n"
                f"class_name:     {chunk.metadata.get('class_name')}\n"
                f"function_name:   {chunk.metadata.get('function_name')}\n"
                f"qualified_name:  {chunk.metadata.get('qualified_name')}\n"
                f"chunk_id:        {duplicate_id}\n"
            )
            print(chunk.page_content)
            print("-" * 100)

    # -----------------------------------------------------------------------
    # Metadata integrity
    # -----------------------------------------------------------------------

    required_metadata = [
        "source",
        "corpus",
        "chunk_type",
        "module_name",
        "class_name",
        "function_name",
        "qualified_name",
    ]

    print("\nMetadata integrity")
    print("-" * 60)

    for field in required_metadata:
        missing = sum(
            field not in chunk.metadata
            for chunk in chunks
        )

        print(
            f"{field:<20} "
            f"{'OK' if missing == 0 else f'{missing:,} missing'}"
        )

    empty_chunks = sum(
        not chunk.page_content.strip()
        for chunk in chunks
    )

    print(f"{'Empty chunks':<20} {empty_chunks:,}")

    # -----------------------------------------------------------------------
    # Files with most chunks
    # -----------------------------------------------------------------------

    chunks_per_file = Counter(
        chunk.metadata["source"]
        for chunk in chunks
    )

    print("\nTop 10 files by number of chunks")
    print("-" * 80)

    for rank, (source, count) in enumerate(
        chunks_per_file.most_common(10),
        start=1,
    ):
        print(
            f"{rank:>2}. "
            f"{count:>5,} chunks  "
            f"{source}"
        )


if __name__ == "__main__":
    main()