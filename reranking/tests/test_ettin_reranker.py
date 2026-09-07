import time
import torch
from sentence_transformers import CrossEncoder


MODEL = "cross-encoder/ettin-reranker-32m-v1"


def main():
    print("=" * 60)
    print("Ettin CPU Reranker Diagnostic")
    print("=" * 60)

    print("\nPyTorch configuration:")
    print("torch version:", torch.__version__)
    print("torch threads:", torch.get_num_threads())
    print("torch interop threads:", torch.get_num_interop_threads())

    print("\nLoading model...")

    load_start = time.perf_counter()

    model = CrossEncoder(
        MODEL,
        device="cpu",
    )

    load_end = time.perf_counter()

    print(
        f"Model loaded in: "
        f"{load_end - load_start:.3f}s"
    )

    print("Model device:", model.device)

    query = (
        "How does interrupt() resume execution after "
        "a Command(resume=...) is provided?"
    )

    document = (
        "When a graph is interrupted, execution can be resumed "
        "by invoking the graph again with a Command containing "
        "the resume value. The checkpoint stores the execution "
        "state required for resumption."
    )

    print("\n" + "=" * 60)
    print("TEST 1: Single pair")
    print("=" * 60)

    start = time.perf_counter()

    score = model.predict(
        [(query, document)],
        batch_size=1,
        show_progress_bar=False,
    )

    end = time.perf_counter()

    print("Score:", score)
    print(f"Time: {end - start:.3f}s")

    print("\n" + "=" * 60)
    print("TEST 2: 13 documents")
    print("=" * 60)

    documents_13 = [
        document
        for _ in range(13)
    ]

    pairs_13 = [
        (query, doc)
        for doc in documents_13
    ]

    start = time.perf_counter()

    scores_13 = model.predict(
        pairs_13,
        batch_size=16,
        show_progress_bar=True,
    )

    end = time.perf_counter()

    print(f"\n13 documents time: {end - start:.3f}s")
    print("Scores:", scores_13)

    print("\n" + "=" * 60)
    print("TEST 3: 32 documents")
    print("=" * 60)

    documents_32 = [
        document
        for _ in range(32)
    ]

    pairs_32 = [
        (query, doc)
        for doc in documents_32
    ]

    start = time.perf_counter()

    scores_32 = model.predict(
        pairs_32,
        batch_size=16,
        show_progress_bar=True,
    )

    end = time.perf_counter()

    print(f"\n32 documents time: {end - start:.3f}s")
    print("Number of scores:", len(scores_32))

    print("\n" + "=" * 60)
    print("TEST 4: Thread comparison")
    print("=" * 60)

    for threads in [1, 2, 4, 8]:

        torch.set_num_threads(threads)

        start = time.perf_counter()

        model.predict(
            pairs_13,
            batch_size=16,
            show_progress_bar=False,
        )

        end = time.perf_counter()

        print(
            f"threads={threads}: "
            f"{end - start:.3f}s"
        )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()