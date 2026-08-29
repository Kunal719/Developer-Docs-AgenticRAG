from concurrent.futures import ThreadPoolExecutor
from time import sleep, perf_counter


def dense():
    print("Dense started")
    sleep(2)
    print("Dense finished")
    return "dense result"


def bm25():
    print("BM25 started")
    sleep(2)
    print("BM25 finished")
    return "bm25 result"


def main():
    start = perf_counter()

    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(dense)
        bm25_future = executor.submit(bm25)

        dense_result = dense_future.result()
        bm25_result = bm25_future.result()

    end = perf_counter()

    print(dense_result)
    print(bm25_result)
    print(f"Total time: {end - start:.2f}s")


if __name__ == "__main__":
    main()