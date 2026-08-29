from pathlib import Path
from time import perf_counter
import shutil

import numpy as np
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType


ONNX_MODEL_PATH = Path("reranking/onnx/bge-reranker-base")
INT8_MODEL_PATH = Path("reranking/onnx/bge-reranker-base-int8")

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


def create_int8_model():
    source_model = ONNX_MODEL_PATH / "model.onnx"
    INT8_MODEL_PATH.mkdir(parents=True, exist_ok=True)

    output_model = INT8_MODEL_PATH / "model.onnx"

    if output_model.exists():
        print("INT8 model already exists. Skipping quantization.")
        return

    print("Creating INT8 model...")

    quantize_dynamic(
        model_input=str(source_model),
        model_output=str(output_model),
        weight_type=QuantType.QInt8,
    )

    # Copy tokenizer/config files required by Transformers/Optimum
    for file_name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]:
        source_file = ONNX_MODEL_PATH / file_name

        if source_file.exists():
            shutil.copy2(
                source_file,
                INT8_MODEL_PATH / file_name,
            )

    print(f"INT8 model saved to: {output_model}")


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = ORTModelForSequenceClassification.from_pretrained(
        model_path,
        file_name="model.onnx",
    )

    return tokenizer, model


def predict(tokenizer, model, query, documents):
    inputs = tokenizer(
        [query] * len(documents),
        documents,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    outputs = model(**inputs)

    return outputs.logits.squeeze(-1).detach().numpy()


def benchmark(fn, warmup_runs=2, benchmark_runs=5):
    for _ in range(warmup_runs):
        fn()

    times = []

    for _ in range(benchmark_runs):
        start = perf_counter()
        fn()
        end = perf_counter()

        times.append(end - start)

    return {
        "mean": np.mean(times),
        "median": np.median(times),
    }


def main():

    create_int8_model()

    print("\nLoading FP32 ONNX model...")
    fp32_tokenizer, fp32_model = load_model(
        ONNX_MODEL_PATH
    )

    print("Loading INT8 ONNX model...")
    int8_tokenizer, int8_model = load_model(
        INT8_MODEL_PATH
    )

    print("\nModels loaded.\n")

    candidate_counts = [8, 10, 13, 20]

    for count in candidate_counts:

        documents = DOCUMENTS[:count]

        print("=" * 60)
        print(f"Candidates: {count}")
        print("=" * 60)

        # -------------------------
        # FP32 ONNX
        # -------------------------

        def run_fp32():
            return predict(
                fp32_tokenizer,
                fp32_model,
                QUERY,
                documents,
            )

        fp32_result = benchmark(run_fp32)

        # -------------------------
        # INT8 ONNX
        # -------------------------

        def run_int8():
            return predict(
                int8_tokenizer,
                int8_model,
                QUERY,
                documents,
            )

        int8_result = benchmark(run_int8)

        # Get scores for comparison
        fp32_scores = run_fp32()
        int8_scores = run_int8()

        fp32_ranking = np.argsort(-fp32_scores)
        int8_ranking = np.argsort(-int8_scores)

        top5_fp32 = set(fp32_ranking[:5])
        top5_int8 = set(int8_ranking[:5])

        speedup = (
            fp32_result["mean"]
            / int8_result["mean"]
        )

        score_correlation = np.corrcoef(
            fp32_scores,
            int8_scores,
        )[0, 1]

        print(
            f"FP32 mean:        "
            f"{fp32_result['mean']:.4f}s"
        )

        print(
            f"INT8 mean:        "
            f"{int8_result['mean']:.4f}s"
        )

        print(
            f"INT8 speedup:     "
            f"{speedup:.2f}x"
        )

        print(
            "Top-1 identical:  ",
            fp32_ranking[0] == int8_ranking[0],
        )

        print(
            "Top-5 identical:  ",
            top5_fp32 == top5_int8,
        )

        print(
            "Full ranking:      ",
            np.array_equal(
                fp32_ranking,
                int8_ranking,
            ),
        )

        print(
            f"Score correlation: "
            f"{score_correlation:.4f}"
        )

        print(
            "FP32 ranking:",
            (fp32_ranking + 1).tolist(),
        )

        print(
            "INT8 ranking:",
            (int8_ranking + 1).tolist(),
        )

        print()


if __name__ == "__main__":
    main()