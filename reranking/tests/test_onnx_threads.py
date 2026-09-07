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


def create_session(num_threads: int):
    options = ort.SessionOptions()

    # Number of intra-op threads used by ONNX Runtime
    options.intra_op_num_threads = num_threads

    # Keep inter-op parallelism simple for this single-model workload.
    options.inter_op_num_threads = 1

    # Let ONNX Runtime optimize the graph normally.
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

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


def benchmark(session, inputs, warmup_runs=3, benchmark_runs=7):
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

    # Your machine has 4 physical cores / 8 logical processors.
    thread_counts = [1, 2, 4, 8]

    for candidate_count in candidate_counts:

        documents = DOCUMENTS[:candidate_count]

        print("\n" + "=" * 60)
        print(f"Candidates: {candidate_count}")
        print("=" * 60)

        inputs = prepare_inputs(
            tokenizer,
            documents,
        )

        for num_threads in thread_counts:

            print(
                f"\nTesting ONNX Runtime threads: "
                f"{num_threads}"
            )

            session = create_session(
                num_threads
            )

            result = benchmark(
                session,
                inputs,
            )

            print(
                f"Mean:   {result['mean']:.4f}s"
            )

            print(
                f"Median: {result['median']:.4f}s"
            )

            print(
                f"Min:    {result['min']:.4f}s"
            )

            # Release the session before testing
            # the next configuration.
            del session


if __name__ == "__main__":
    main()