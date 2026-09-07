from sentence_transformers import CrossEncoder
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

import numpy as np


MODEL_NAME = "BAAI/bge-reranker-base"
ONNX_MODEL_PATH = "reranking/onnx/bge-reranker-base"


TEST_CASES = [
    {
        "query": "How does LangGraph checkpoint recovery work?",
        "documents": [
            "LangGraph checkpoints persist graph state and allow execution to resume after failures.",
            "LangGraph supports streaming updates from graph nodes.",
            "A checkpointer saves state associated with a thread ID.",
            "Python dictionaries can be serialized to JSON.",
            "Durable execution allows workflows to resume from persisted state.",
            "A failed task may execute again during recovery.",
            "External side effects should be idempotent.",
        ],
    },
    {
        "query": "How does interrupt resume execution in LangGraph?",
        "documents": [
            "Calling interrupt pauses graph execution and waits for external input.",
            "A Command with resume provides the value needed to continue execution.",
            "The node containing interrupt is restarted when execution resumes.",
            "Checkpoints store graph state for durable execution.",
            "BM25 is a sparse retrieval algorithm.",
            "Python functions can accept keyword arguments.",
            "LangGraph supports conditional edges.",
        ],
    },
    {
        "query": "How does LangGraph Send handle parallel workers?",
        "documents": [
            "Send enables dynamically generated parallel work in LangGraph.",
            "Multiple workers can process items independently and fan their results back into the graph.",
            "Reducers combine updates produced by parallel nodes.",
            "operator.add can be used to accumulate values.",
            "Checkpointers persist graph state between executions.",
            "Dense retrieval uses vector similarity.",
            "Cross encoders score query-document pairs.",
        ],
    },
    {
        "query": "How should external side effects be handled with durable execution?",
        "documents": [
            "External side effects should be placed inside tasks and designed to be idempotent.",
            "Durable execution can replay unfinished work after a failure.",
            "Checkpointing does not provide exactly-once execution for external systems.",
            "A stable idempotency key can prevent duplicate operations.",
            "BM25 retrieves documents using lexical matching.",
            "A graph can contain multiple nodes.",
            "Vector stores support similarity search.",
        ],
    },
]


def get_onnx_scores(tokenizer, model, query, documents):
    inputs = tokenizer(
        [query] * len(documents),
        documents,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )

    outputs = model(**inputs)

    return outputs.logits.squeeze(-1).detach().numpy()


def ranking(scores):
    return np.argsort(-scores)


def top_k(scores, k):
    return ranking(scores)[:k]


def main():
    print("Loading PyTorch model...")
    pytorch_model = CrossEncoder(
        MODEL_NAME,
        max_length=512,
    )

    print("Loading ONNX model...")
    tokenizer = AutoTokenizer.from_pretrained(ONNX_MODEL_PATH)

    onnx_model = ORTModelForSequenceClassification.from_pretrained(
        ONNX_MODEL_PATH,
        file_name="model.onnx",
    )

    total_cases = len(TEST_CASES)

    top1_matches = 0
    top5_matches = 0
    ranking_matches = 0

    correlations = []

    print("\n================ EQUIVALENCE TEST ================\n")

    for case_number, case in enumerate(TEST_CASES, start=1):
        query = case["query"]
        documents = case["documents"]

        pairs = [(query, document) for document in documents]

        pytorch_scores = np.asarray(
            pytorch_model.predict(
                pairs,
                batch_size=16,
            )
        )

        onnx_scores = np.asarray(
            get_onnx_scores(
                tokenizer,
                onnx_model,
                query,
                documents,
            )
        )

        pytorch_ranking = ranking(pytorch_scores)
        onnx_ranking = ranking(onnx_scores)

        pytorch_top1 = pytorch_ranking[0]
        onnx_top1 = onnx_ranking[0]

        pytorch_top5 = set(top_k(pytorch_scores, 5))
        onnx_top5 = set(top_k(onnx_scores, 5))

        top1_match = pytorch_top1 == onnx_top1
        top5_match = pytorch_top5 == onnx_top5
        full_ranking_match = np.array_equal(
            pytorch_ranking,
            onnx_ranking,
        )

        correlation = np.corrcoef(
            pytorch_scores,
            onnx_scores,
        )[0, 1]

        correlations.append(correlation)

        top1_matches += top1_match
        top5_matches += top5_match
        ranking_matches += full_ranking_match

        print(f"Test case {case_number}")
        print(f"Query: {query}")

        print(
            "PyTorch ranking:",
            (pytorch_ranking + 1).tolist(),
        )

        print(
            "ONNX ranking:   ",
            (onnx_ranking + 1).tolist(),
        )

        print(f"Top-1 match:     {top1_match}")
        print(f"Top-5 match:     {top5_match}")
        print(f"Full ranking:    {full_ranking_match}")
        print(f"Correlation:     {correlation:.4f}")
        print()

    print("================ SUMMARY ================\n")

    print(
        f"Top-1 agreement: "
        f"{top1_matches}/{total_cases} "
        f"({top1_matches / total_cases:.1%})"
    )

    print(
        f"Top-5 agreement: "
        f"{top5_matches}/{total_cases} "
        f"({top5_matches / total_cases:.1%})"
    )

    print(
        f"Full ranking agreement: "
        f"{ranking_matches}/{total_cases} "
        f"({ranking_matches / total_cases:.1%})"
    )

    print(
        f"Mean score correlation: "
        f"{np.mean(correlations):.4f}"
    )


if __name__ == "__main__":
    main()