from sentence_transformers import CrossEncoder
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from time import perf_counter
import numpy as np


MODEL_NAME = "BAAI/bge-reranker-base"
ONNX_MODEL_PATH = "reranking/onnx/bge-reranker-base"

QUERY = "How does LangGraph checkpoint recovery work?"

DOCUMENTS = [
    "LangGraph checkpoints persist graph state and allow execution to resume after failures.",
    "LangGraph supports streaming updates from graph nodes.",
    "A checkpointer saves state associated with a thread ID.",
    "Python dictionaries can be serialized to JSON.",
    "Durable execution allows workflows to resume from persisted state.",
    "LangGraph tasks can encapsulate side effects and their results can be replayed.",
    "Checkpointing does not provide exactly-once execution for external side effects.",
    "External API calls should be designed to be idempotent.",
    "A failed task may be executed again during recovery.",
    "LangGraph associates checkpoints with a thread identifier.",
    "Retry policies can cause failed work to execute again.",
    "The Functional API provides task-based durable execution.",
    "A node may be re-executed from the beginning after interruption.",
    "Synchronous durability persists checkpoints before continuing.",
    "Asynchronous checkpoint persistence can have a crash window.",
    "The graph API allows stateful workflows to be compiled with a checkpointer.",
    "External systems may need their own deduplication mechanisms.",
    "A stable operation identifier can be used for idempotency.",
    "Checkpoint recovery restores persisted workflow state.",
    "Uncheckpointed external side effects may be repeated."
]


def benchmark(name, fn, warmup_runs=2, benchmark_runs=5):
    # Warmup
    for _ in range(warmup_runs):
        fn()

    times = []

    for _ in range(benchmark_runs):
        start = perf_counter()
        fn()
        end = perf_counter()

        times.append(end - start)

    return {
        "name": name,
        "mean": np.mean(times),
        "median": np.median(times),
        "min": np.min(times),
        "max": np.max(times),
    }


def main():

    print("Loading PyTorch model...")
    pytorch_model = CrossEncoder(MODEL_NAME)

    print("Loading ONNX model...")
    tokenizer = AutoTokenizer.from_pretrained(ONNX_MODEL_PATH)

    onnx_model = ORTModelForSequenceClassification.from_pretrained(
        ONNX_MODEL_PATH,
        file_name="model.onnx",
    )

    print("\nModels loaded.\n")

    candidate_counts = [8, 10, 13, 20]

    for count in candidate_counts:

        documents = DOCUMENTS[:count]
        pairs = [(QUERY, document) for document in documents]

        print("=" * 60)
        print(f"Candidates: {count}")
        print("=" * 60)

        # ---------------------------------
        # PyTorch
        # ---------------------------------

        def run_pytorch():
            return pytorch_model.predict(
                pairs,
                batch_size=16,
            )

        pytorch_result = benchmark(
            "PyTorch",
            run_pytorch,
        )

        # ---------------------------------
        # ONNX
        # ---------------------------------

        def run_onnx():

            inputs = tokenizer(
                [QUERY] * len(documents),
                documents,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )

            outputs = onnx_model(**inputs)

            return outputs.logits.squeeze(-1).detach().numpy()

        onnx_result = benchmark(
            "ONNX",
            run_onnx,
        )

        # ---------------------------------
        # Compare
        # ---------------------------------

        pytorch_scores = run_pytorch()
        onnx_scores = run_onnx()

        pytorch_ranking = np.argsort(-pytorch_scores)
        onnx_ranking = np.argsort(-onnx_scores)

        print(
            f"PyTorch mean:   {pytorch_result['mean']:.4f}s"
        )
        print(
            f"PyTorch median: {pytorch_result['median']:.4f}s"
        )

        print(
            f"ONNX mean:      {onnx_result['mean']:.4f}s"
        )
        print(
            f"ONNX median:    {onnx_result['median']:.4f}s"
        )

        speedup = pytorch_result["mean"] / onnx_result["mean"]

        print(
            f"Speedup:        {speedup:.2f}x"
        )

        print(
            "Ranking identical:",
            np.array_equal(
                pytorch_ranking,
                onnx_ranking,
            ),
        )

        max_score_difference = np.max(
            np.abs(
                pytorch_scores - onnx_scores
            )
        )

        print(
            f"Max score difference: "
            f"{max_score_difference:.6f}"
        )

        print()


if __name__ == "__main__":
    main()