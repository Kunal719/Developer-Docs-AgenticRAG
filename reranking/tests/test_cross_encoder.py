"""
Smoke/correctness test for the unified ONNX CrossEncoderReranker.

Tests these CPU variants:

  BGE FP32:
    reranking/onnx/bge-reranker-base

  BGE INT8:
    reranking/onnx/bge-reranker-base-int8

  BGE O3:
    reranking/onnx/bge-reranker-base-o3

  Ettin O2:
    cross-encoder/ettin-reranker-32m-v1 / onnx/model_O2.onnx

  Ettin O3:
    cross-encoder/ettin-reranker-32m-v1 / onnx/model_O3.onnx

This is intentionally a smoke/correctness test, not a quality benchmark.

Run from the project root:

    python test_cross_encoder_reranker.py

Optional:
    python test_cross_encoder_reranker.py --batch-size 1
    python test_cross_encoder_reranker.py --batch-size 16

The test checks:
  1. model/tokenizer loading
  2. ONNX input compatibility
  3. pair tokenization
  4. batched inference
  5. finite scores
  6. one score per document
  7. deterministic ranking for the same inputs
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Make sure imports work when this file is run from the project root
# or directly from another working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from reranking.cross_encoder_onnx_reranker import CrossEncoderReranker


# ---------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------

QUERY = (
    "How does interrupt() resume execution after a Command(resume=...) "
    "is provided in LangGraph?"
)

DOCUMENTS = [
    Document(
        page_content=(
            "LangGraph interrupts pause graph execution and return control "
            "to the caller. Execution can later continue when a Command "
            "containing resume data is supplied to the graph invocation."
        ),
        metadata={
            "chunk_id": "test-1",
            "source": "langgraph_interrupts.md",
            "corpus": "test",
        },
    ),
    Document(
        page_content=(
            "Python dictionaries store key-value pairs. A dictionary can "
            "be updated by assigning a new value to an existing key."
        ),
        metadata={
            "chunk_id": "test-2",
            "source": "python.md",
            "corpus": "test",
        },
    ),
    Document(
        page_content=(
            "A LangGraph checkpoint stores graph state so execution can "
            "be recovered. Checkpoints are associated with execution "
            "configuration and checkpoint namespaces."
        ),
        metadata={
            "chunk_id": "test-3",
            "source": "langgraph_checkpointing.md",
            "corpus": "test",
        },
    ),
    Document(
        page_content=(
            "Command can be used to combine state updates with control "
            "flow. The resume value is used when continuing execution "
            "after an interrupt."
        ),
        metadata={
            "chunk_id": "test-4",
            "source": "langgraph_command.md",
            "corpus": "test",
        },
    ),
    Document(
        page_content=(
            "ONNX Runtime executes machine-learning models using an "
            "InferenceSession. CPUExecutionProvider runs the model on CPU."
        ),
        metadata={
            "chunk_id": "test-5",
            "source": "onnx_runtime.md",
            "corpus": "test",
        },
    ),
]


# ---------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------

MODELS = [
    {
        "name": "BGE FP32",
        "model": "reranking/onnx/bge-reranker-base",
        "onnx_file": None,
    },
    {
        "name": "BGE INT8",
        "model": "reranking/onnx/bge-reranker-base-int8",
        "onnx_file": None,
    },
    {
        "name": "BGE O3",
        "model": "reranking/onnx/bge-reranker-base-o3",
        "onnx_file": None,
    },
    {
        "name": "Ettin O2",
        "model": "cross-encoder/ettin-reranker-32m-v1",
        "onnx_file": "onnx/model_O2.onnx",
    },
    {
        "name": "Ettin O3",
        "model": "cross-encoder/ettin-reranker-32m-v1",
        "onnx_file": "onnx/model_O3.onnx",
    },
]


def check_scores(scores: list[float], expected_count: int) -> None:
    """Validate the raw score output."""
    assert len(scores) == expected_count, (
        f"Expected {expected_count} scores, got {len(scores)}"
    )

    assert all(
        isinstance(score, (float, int))
        for score in scores
    ), f"Non-numeric score found: {scores}"

    assert all(
        math.isfinite(float(score))
        for score in scores
    ), f"Non-finite score found: {scores}"


def run_one(
    config: dict,
    batch_size: int,
    num_threads: int,
) -> bool:
    name = config["name"]

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    try:
        model = CrossEncoderReranker(
            rerank_model=config["model"],
            top_k=len(DOCUMENTS),
            batch_size=batch_size,
            max_length=384,
            onnx_file=config["onnx_file"],
            num_threads=num_threads,
        )

        # -------------------------------------------------------------
        # Direct score test
        # -------------------------------------------------------------

        pairs = [
            (QUERY, doc.page_content)
            for doc in DOCUMENTS
        ]

        print("\n[TEST] Running direct _score()...")
        scores = model._score(pairs)

        check_scores(
            scores,
            expected_count=len(DOCUMENTS),
        )

        print("[PASS] Score count:", len(scores))
        print("[PASS] All scores are finite.")

        # -------------------------------------------------------------
        # Rerank test
        # -------------------------------------------------------------

        print("\n[TEST] Running rerank()...")

        ranked = model.rerank(
            query=QUERY,
            documents=DOCUMENTS,
            top_k=len(DOCUMENTS),
        )

        assert len(ranked) == len(DOCUMENTS), (
            f"Expected {len(DOCUMENTS)} ranked docs, "
            f"got {len(ranked)}"
        )

        print("[PASS] Correct number of ranked documents.")

        # -------------------------------------------------------------
        # Verify ranking is consistent with the returned score order.
        # -------------------------------------------------------------

        score_by_id = {
            doc.metadata["chunk_id"]: score
            for doc, score in zip(DOCUMENTS, scores)
        }

        ranked_scores = [
            score_by_id[doc.metadata["chunk_id"]]
            for doc in ranked
        ]

        assert ranked_scores == sorted(
            ranked_scores,
            reverse=True,
        ), "rerank() output is not sorted by descending score."

        print("[PASS] Ranking is sorted by descending score.")

        # -------------------------------------------------------------
        # Print ranking
        # -------------------------------------------------------------

        print("\nRanking:")
        for rank, (doc, score) in enumerate(
            zip(ranked, ranked_scores),
            start=1,
        ):
            print(
                f"  {rank}. "
                f"{doc.metadata['chunk_id']:>6} "
                f"score={float(score): .6f} "
                f"source={doc.metadata['source']}"
            )

        # The first result should normally be one of the highly relevant
        # interrupt/resume documents. We deliberately do not hard-assert
        # this because this script is intended to validate the inference
        # pipeline, not establish model quality.
        top_id = ranked[0].metadata["chunk_id"]

        if top_id in {"test-1", "test-4"}:
            print(
                "[INFO] Top result is one of the expected "
                "interrupt/resume documents."
            )
        else:
            print(
                "[WARNING] Top result was not test-1/test-4. "
                "Inference still passed; inspect ranking manually."
            )

        return True

    except Exception as exc:
        print(
            f"\n[FAIL] {name}: "
            f"{type(exc).__name__}: {exc}"
        )
        return False


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="ONNX inference batch size (default: 4)",
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="ONNX Runtime intra-op CPU threads (default: 4)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "bge-fp32",
            "bge-int8",
            "bge-o3",
            "ettin-o2",
            "ettin-o3",
        ],
        help="Run only one model variant.",
    )

    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")

    if args.threads < 1:
        parser.error("--threads must be >= 1")

    selected = MODELS

    if args.model:
        mapping = {
            "bge-fp32": "BGE FP32",
            "bge-int8": "BGE INT8",
            "bge-o3": "BGE O3",
            "ettin-o2": "Ettin O2",
            "ettin-o3": "Ettin O3",
        }

        selected = [
            model
            for model in MODELS
            if model["name"] == mapping[args.model]
        ]

    print("=" * 70)
    print("Cross-Encoder ONNX Smoke Test")
    print("=" * 70)
    print(f"Query: {QUERY}")
    print(f"Documents: {len(DOCUMENTS)}")
    print(f"Batch size: {args.batch_size}")
    print(f"CPU threads: {args.threads}")

    results = {}

    for config in selected:
        results[config["name"]] = run_one(
            config=config,
            batch_size=args.batch_size,
            num_threads=args.threads,
        )

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    all_passed = True

    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{name:<15} {status}")

        if not passed:
            all_passed = False

    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
