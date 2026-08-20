from collections import Counter

import tiktoken


from ingestion.python_loader import load_python_documents
from ingestion.code_metadata_extractor import extract_code_metadata

from tests.code_chunker_nested_experiment import (
    experimental_chunk_python_documents,
)

from pathlib import Path


CODE_PATH = Path("data/raw/langgraph/libs")


def get_token_length(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def percentile(values: list[int], p: float) -> float:
    values = sorted(values)

    if not values:
        return 0

    index = (len(values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)

    weight = index - lower

    return (
        values[lower]
        + (values[upper] - values[lower]) * weight
    )


def main() -> None:
    print(f"Code path: {CODE_PATH.resolve()}")
    print()

    documents = load_python_documents(CODE_PATH)

    documents = extract_code_metadata(
        documents,
        CODE_PATH,
    )

    chunks = experimental_chunk_python_documents(documents)

    char_lengths = [
        len(chunk.page_content)
        for chunk in chunks
    ]

    token_lengths = [
        get_token_length(chunk.page_content)
        for chunk in chunks
    ]

    print("=" * 80)
    print("EXPERIMENTAL RECURSIVE NESTED-FUNCTION CHUNKING")
    print("=" * 80)

    print()
    print(f"Python files: {len(documents):,}")
    print(f"Code chunks:  {len(chunks):,}")

    print()
    print("Chunk types")
    print("-" * 60)

    chunk_types = Counter(
        chunk.metadata["chunk_type"]
        for chunk in chunks
    )

    for chunk_type, count in chunk_types.most_common():
        print(f"{chunk_type:<25} {count:,}")

    print()
    print("Character length")
    print("-" * 60)

    print(f"Mean:       {sum(char_lengths) / len(char_lengths):,.2f}")
    print(f"Median:     {percentile(char_lengths, 0.50):,.2f}")
    print(f"P90:        {percentile(char_lengths, 0.90):,.2f}")
    print(f"P95:        {percentile(char_lengths, 0.95):,.2f}")
    print(f"P99:        {percentile(char_lengths, 0.99):,.2f}")
    print(f"Maximum:    {max(char_lengths):,}")

    print()
    print("Token length")
    print("-" * 60)

    print(f"Mean:       {sum(token_lengths) / len(token_lengths):,.2f}")
    print(f"Median:     {percentile(token_lengths, 0.50):,.2f}")
    print(f"P90:        {percentile(token_lengths, 0.90):,.2f}")
    print(f"P95:        {percentile(token_lengths, 0.95):,.2f}")
    print(f"P99:        {percentile(token_lengths, 0.99):,.2f}")
    print(f"Maximum:    {max(token_lengths):,}")

    print()
    print("Top 20 largest chunks")
    print("-" * 100)

    largest = sorted(
        zip(chunks, char_lengths, token_lengths),
        key=lambda item: item[1],
        reverse=True,
    )[:20]

    for index, (chunk, char_length, token_length) in enumerate(
        largest,
        start=1,
    ):
        print(
            f"{index:2}. "
            f"{chunk.metadata['qualified_name']:<60} "
            f"{char_length:>8,} chars | "
            f"{token_length:>6,} tokens | "
            f"{chunk.metadata['source']}"
        )


if __name__ == "__main__":
    main()