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
    "Durable execution allows interrupted workflows to continue from saved state.",
    "A thread ID identifies the persisted execution state.",
    "Checkpoints provide recovery points during graph execution.",
    "External API operations should tolerate retries.",
    "The checkpointer stores state for a graph thread.",
    "A workflow can recover after a process failure.",
    "Side effects should be isolated from replay-sensitive graph logic.",
    "Recovery can cause unfinished work to execute again.",
    "Persisted state is used when resuming a workflow.",
    "Synchronous durability waits for checkpoint persistence.",
    "Asynchronous durability persists checkpoints in the background.",
    "Exit durability saves state when graph execution finishes.",
    "External operations can use deduplication keys.",
]


THREADS = 8

# These are now meaningful because we have 32 candidates.
BATCH_SIZES = [8, 16, 32]


def create_session():
    options = ort.SessionOptions()

    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = 1

    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    return ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def prepare_batches(tokenizer, documents, batch_size):
    batches = []

    for start in range(0, len(documents), batch_size):
        batch_documents = documents[
            start:start + batch_size
        ]

        inputs = tokenizer(
            [QUERY] * len(batch_documents),
            batch_documents,
            padding=True,
            truncation=True,
            return_tensors="np",
        )

        batches.append(
            {
                key: value
                for key, value in inputs.items()
            }
        )

    return batches


def predict(session, batches):
    all_scores = []

    for inputs in batches:
        outputs = session.run(
            None,
            inputs,
        )

        scores = outputs[0].squeeze(-1)

        all_scores.extend(
            np.atleast_1d(scores).tolist()
        )

    return np.asarray(all_scores)


def benchmark(
    session,
    batches,
    warmup_runs=3,
    benchmark_runs=7,
):
    for _ in range(warmup_runs):
        predict(session, batches)

    times = []

    for _ in range(benchmark_runs):
        start = perf_counter()

        predict(session, batches)

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

    print("Creating ONNX Runtime session...")

    session = create_session()

    candidate_count = len(DOCUMENTS)

    print("\n" + "=" * 70)
    print(f"Candidates: {candidate_count}")
    print(f"ONNX Runtime threads: {THREADS}")
    print("Graph optimization: ALL")
    print("=" * 70)

    baseline_scores = None
    baseline_ranking = None

    for batch_size in BATCH_SIZES:

        batches = prepare_batches(
            tokenizer,
            DOCUMENTS,
            batch_size,
        )

        print(
            f"\nBatch size: {batch_size}"
        )

        print(
            f"Number of inference calls: {len(batches)}"
        )

        result = benchmark(
            session,
            batches,
        )

        scores = predict(
            session,
            batches,
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
            "Top-10 ranking:",
            (ranking[:10] + 1).tolist(),
        )

    del session


if __name__ == "__main__":
    main()