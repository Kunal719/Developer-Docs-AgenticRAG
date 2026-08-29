from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


MODEL_PATH = Path(
    "reranking/onnx/bge-reranker-base/model.onnx"
)

TOKENIZER_PATH = Path(
    "reranking/onnx/bge-reranker-base"
)

QUERY = "How does LangGraph checkpoint recovery work?"

DOCUMENTS = [
    "LangGraph checkpoints persist graph state and allow execution to resume after failures.",
    "LangGraph supports streaming updates from graph nodes.",
    "A checkpointer saves state associated with a thread ID.",
    "Python dictionaries can be serialized to JSON.",
    "Durable execution allows workflows to resume from persisted state.",
    "A failed task may execute again during recovery.",
    "External side effects should be idempotent.",
    "LangGraph tasks can encapsulate side effects and their results can be replayed.",
    "Checkpointing does not provide exactly-once execution for external side effects.",
    "A stable idempotency key can prevent duplicate operations.",
    "Retry policies can cause failed work to execute again.",
    "The Functional API provides task-based durable execution.",
    "A node may be re-executed from the beginning after interruption.",
    "Synchronous durability persists checkpoints before continuing.",
    "Asynchronous checkpoint persistence can have a crash window.",
    "External systems may need their own deduplication mechanisms.",
    "A stable operation identifier can be used for idempotency.",
    "Checkpoint recovery restores persisted workflow state.",
    "Uncheckpointed external side effects may be repeated.",
    "Graph execution can resume from persisted checkpoints.",
]


THREADS = 8

OPTIMIZATION_LEVELS = {
    "DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    "BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
    "EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
    "ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
}


def create_session(optimization_level):
    options = ort.SessionOptions()

    # Keep threading fixed so we are only testing
    # graph optimization.
    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = 1

    options.graph_optimization_level = optimization_level

    return ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def prepare_inputs(tokenizer, documents):
    inputs = tokenizer(
        [QUERY] * len(documents),
        documents,
        padding=True,
        truncation=True,
        return_tensors="np",
    )

    return {
        key: value
        for key, value in inputs.items()
    }


def predict(session, inputs):
    outputs = session.run(
        None,
        inputs,
    )

    return outputs[0].squeeze(-1)


def benchmark(
    session,
    inputs,
    warmup_runs=3,
    benchmark_runs=7,
):
    # Warmup
    for _ in range(warmup_runs):
        predict(session, inputs)

    times = []

    for _ in range(benchmark_runs):
        start = perf_counter()

        predict(session, inputs)

        end = perf_counter()

        times.append(end - start)

    return {
        "mean": np.mean(times),
        "median": np.median(times),
        "min": np.min(times),
    }


def main():

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH
    )

    candidate_counts = [8, 10, 13, 20]

    for candidate_count in candidate_counts:

        documents = DOCUMENTS[:candidate_count]

        print("\n" + "=" * 70)
        print(f"Candidates: {candidate_count}")
        print(f"ONNX Runtime threads: {THREADS}")
        print("=" * 70)

        inputs = prepare_inputs(
            tokenizer,
            documents,
        )

        baseline_scores = None
        baseline_ranking = None

        for name, optimization_level in OPTIMIZATION_LEVELS.items():

            print(
                f"\nTesting optimization: {name}"
            )

            session = create_session(
                optimization_level
            )

            result = benchmark(
                session,
                inputs,
            )

            scores = predict(
                session,
                inputs,
            )

            ranking = np.argsort(-scores)

            if baseline_scores is None:
                baseline_scores = scores
                baseline_ranking = ranking

            ranking_identical = np.array_equal(
                baseline_ranking,
                ranking,
            )

            correlation = np.corrcoef(
                baseline_scores,
                scores,
            )[0, 1]

            print(
                f"Mean:              {result['mean']:.4f}s"
            )

            print(
                f"Median:            {result['median']:.4f}s"
            )

            print(
                f"Min:               {result['min']:.4f}s"
            )

            print(
                f"Ranking identical: {ranking_identical}"
            )

            print(
                f"Score correlation:  {correlation:.4f}"
            )

            print(
                "Ranking:",
                (ranking + 1).tolist(),
            )

            del session


if __name__ == "__main__":
    main()