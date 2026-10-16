# # from reranking.base_reranker import BaseReranker
# # from langchain_core.documents import Document
# # from sentence_transformers import CrossEncoder
# # from evaluation.models import RetrievedDocument
# # from evaluation.observer import PipelineObserver
# # from time import perf_counter

# # class CrossEncoderReranker(BaseReranker):
# #     def __init__(self, rerank_model: str, top_k: int):
# #         self._rerank_model = CrossEncoder(rerank_model)
# #         self._top_k = top_k

# #     def rerank(self, query: str, documents: list[Document], observer: PipelineObserver | None = None, top_k: int | None = None) -> list[Document]:
# #         """
# #         Given a query and a list of documents, rerank the documents based on the query using cross-encoder reranker
# #         """
# #         if not documents:
# #             return []

# #         final_top_k = top_k if top_k is not None else self._top_k

# #         pairs = [(query, doc.page_content) for doc in documents]

# #         start_time = perf_counter()
# #         scores = self._rerank_model.predict(
# #             pairs,
# #             batch_size=16
# #         )

# #         scored_docs = list(zip(documents, scores))

# #         # Sort by score (descending)
# #         scored_docs.sort(key=lambda x: x[1], reverse=True)

# #         final_docs = [doc for doc, _ in scored_docs[:final_top_k]]
# #         final_scores = [score for _, score in scored_docs[:final_top_k]]
# #         end_time = perf_counter()

# #         if observer is not None:
# #             observer_docs: list[RetrievedDocument] = []

# #             for i, (doc, score) in enumerate(zip(final_docs, final_scores)):
# #                 observer_docs.append(RetrievedDocument(chunk_id=doc.metadata.get("chunk_id"), source=doc.metadata.get("source"), document_type=doc.metadata.get("corpus"), reranker_score=float(score), rank=i+1))

# #             observer.record_stage("Reranker", (end_time - start_time), final_top_k, observer_docs)

# #         # for doc in final_docs:
# #             # print(f"Corpus: {doc.metadata['corpus']} \n Source: {doc.metadata['source']}")
# #         return final_docs

# from reranking.base_reranker import BaseReranker

# from langchain_core.documents import Document

# from sentence_transformers import CrossEncoder

# from evaluation.models import RetrievedDocument
# from evaluation.observer import PipelineObserver

# from time import perf_counter
# import threading
# import torch

# class CrossEncoderReranker(BaseReranker):

#     def __init__(
#         self,
#         rerank_model: str,
#         top_k: int,
#         batch_size: int = 16,
#     ):
#         print(
#             f"[RERANKER INIT] Loading model: {rerank_model}"
#         )

#         init_start = perf_counter()

#         self._rerank_model = CrossEncoder(
#             rerank_model,
#             max_length=256,
#             backend="onnx",
#             device="cpu",
#         )

#         self._top_k = top_k
#         self._batch_size = batch_size

#         init_end = perf_counter()

#         print(
#             f"[RERANKER INIT] Model loaded in "
#             f"{init_end - init_start:.3f}s"
#         )

#         print(
#             f"[RERANKER INIT] device={self._rerank_model.device}"
#         )

#     def rerank(
#         self,
#         query: str,
#         documents: list[Document],
#         observer: PipelineObserver | None = None,
#         top_k: int | None = None,
#     ) -> list[Document]:

#         if not documents:
#             return []

#         final_top_k = (
#             top_k
#             if top_k is not None
#             else self._top_k
#         )

#         print(
#             f"\n[RERANK ENTER] "
#             f"thread={threading.current_thread().name} "
#             f"docs={len(documents)}"
#         )

#         print("[RERANK] Creating document/query pairs...")

#         pairs_start = perf_counter()

#         pairs = [
#             (query, doc.page_content)
#             for doc in documents
#         ]

#         pairs_end = perf_counter()

#         print(
#             f"[RERANK] Pairs created in "
#             f"{pairs_end - pairs_start:.3f}s"
#         )

#         print(
#             f"[RERANK] Starting prediction "
#             f"batch_size={self._batch_size}..."
#         )

#         start_time = perf_counter()

#         print("[RERANK] Starting prediction...")
#         print(
#             "[TORCH STATE]",
#             "threads=", torch.get_num_threads(),
#             "interop=", torch.get_num_interop_threads(),
#         )

#         scores = self._rerank_model.predict(
#             pairs,
#             batch_size=self._batch_size,
#             show_progress_bar=True,
#         )

#         end_time = perf_counter()

#         print(
#             f"[RERANK] Prediction finished "
#             f"thread={threading.current_thread().name} "
#             f"docs={len(documents)} "
#             f"duration={end_time - start_time:.3f}s"
#         )

#         scored_docs = list(
#             zip(documents, scores)
#         )

#         print("[RERANK] Sorting scores...")

#         scored_docs.sort(
#             key=lambda x: x[1],
#             reverse=True,
#         )

#         final_docs = [
#             doc
#             for doc, _ in scored_docs[:final_top_k]
#         ]

#         final_scores = [
#             score
#             for _, score in scored_docs[:final_top_k]
#         ]

#         total_end_time = perf_counter()

#         print(
#             f"[RERANK EXIT] "
#             f"thread={threading.current_thread().name} "
#             f"docs={len(documents)} "
#             f"total={total_end_time - start_time:.3f}s "
#             f"returned={len(final_docs)}"
#         )

#         if observer is not None:

#             observer_docs: list[RetrievedDocument] = []

#             for i, (
#                 doc,
#                 score,
#             ) in enumerate(
#                 zip(
#                     final_docs,
#                     final_scores,
#                 )
#             ):
#                 observer_docs.append(
#                     RetrievedDocument(
#                         chunk_id=doc.metadata.get(
#                             "chunk_id"
#                         ),
#                         source=doc.metadata.get(
#                             "source"
#                         ),
#                         document_type=doc.metadata.get(
#                             "corpus"
#                         ),
#                         reranker_score=float(score),
#                         rank=i + 1,
#                     )
#                 )

#             observer.record_stage(
#                 "Reranker",
#                 end_time - start_time,
#                 final_top_k,
#                 observer_docs,
#             )

#         return final_docs


#### LATEST
# from reranking.base_reranker import BaseReranker

# from langchain_core.documents import Document

# import numpy as np
# import onnxruntime as ort
# from transformers import AutoTokenizer
# from huggingface_hub import hf_hub_download

# from evaluation.models import RetrievedDocument
# from evaluation.observer import PipelineObserver

# from time import perf_counter
# import threading


# class CrossEncoderReranker(BaseReranker):

#     def __init__(
#         self,
#         rerank_model: str,
#         top_k: int,
#         batch_size: int = 16,
#         max_length: int = 750,
#         onnx_file: str = "onnx/model_O2.onnx",  # or "onnx/model_O4.onnx"
#         num_threads: int = 4,
#     ):
#         print(f"[RERANKER INIT] Loading model: {rerank_model} ({onnx_file})")

#         init_start = perf_counter()

#         self._tokenizer = AutoTokenizer.from_pretrained(rerank_model)
#         onnx_path = hf_hub_download(rerank_model, onnx_file)

#         sess_options = ort.SessionOptions()
#         sess_options.intra_op_num_threads = num_threads
#         self._session = ort.InferenceSession(
#             onnx_path,
#             sess_options,
#             providers=["CPUExecutionProvider"],
#         )
#         self._input_names = {i.name for i in self._session.get_inputs()}

#         self._top_k = top_k
#         self._batch_size = batch_size
#         self._max_length = max_length

#         init_end = perf_counter()

#         print(f"[RERANKER INIT] Model loaded in {init_end - init_start:.3f}s")
#         print(f"[RERANKER INIT] onnx_file={onnx_file} threads={num_threads}")

#     def _score(self, pairs: list[tuple[str, str]]) -> list[float]:
#         scores: list[float] = []

#         for i in range(0, len(pairs), self._batch_size):
#             batch = pairs[i : i + self._batch_size]

#             enc = self._tokenizer(
#                 [q for q, _ in batch],
#                 [d for _, d in batch],
#                 padding=True,
#                 truncation=True,
#                 max_length=self._max_length,
#                 return_tensors="np",
#             )

#             ort_inputs = {
#                 k: v for k, v in enc.items() if k in self._input_names
#             }

#             logits = self._session.run(None, ort_inputs)[0]
#             if logits.ndim > 1 and logits.shape[-1] == 1:
#                 logits = logits.squeeze(-1)
#             scores.extend(logits.reshape(-1).tolist())

#         return scores

#     def rerank(
#         self,
#         query: str,
#         documents: list[Document],
#         observer: PipelineObserver | None = None,
#         top_k: int | None = None,
#     ) -> list[Document]:

#         if not documents:
#             return []

#         final_top_k = top_k if top_k is not None else self._top_k

#         print(
#             f"\n[RERANK ENTER] "
#             f"thread={threading.current_thread().name} "
#             f"docs={len(documents)}"
#         )

#         print("[RERANK] Creating document/query pairs...")

#         pairs_start = perf_counter()
#         pairs = [(query, doc.page_content) for doc in documents]
#         pairs_end = perf_counter()

#         print(f"[RERANK] Pairs created in {pairs_end - pairs_start:.3f}s")
#         print(f"[RERANK] Starting prediction batch_size={self._batch_size}...")

#         start_time = perf_counter()

#         scores = self._score(pairs)

#         end_time = perf_counter()

#         print(
#             f"[RERANK] Prediction finished "
#             f"thread={threading.current_thread().name} "
#             f"docs={len(documents)} "
#             f"duration={end_time - start_time:.3f}s"
#         )

#         scored_docs = list(zip(documents, scores))

#         print("[RERANK] Sorting scores...")
#         scored_docs.sort(key=lambda x: x[1], reverse=True)

#         final_docs = [doc for doc, _ in scored_docs[:final_top_k]]
#         final_scores = [score for _, score in scored_docs[:final_top_k]]

#         total_end_time = perf_counter()

#         print(
#             f"[RERANK EXIT] "
#             f"thread={threading.current_thread().name} "
#             f"docs={len(documents)} "
#             f"total={total_end_time - start_time:.3f}s "
#             f"returned={len(final_docs)}"
#         )

#         if observer is not None:
#             observer_docs: list[RetrievedDocument] = []

#             for i, (doc, score) in enumerate(zip(final_docs, final_scores)):
#                 observer_docs.append(
#                     RetrievedDocument(
#                         chunk_id=doc.metadata.get("chunk_id"),
#                         source=doc.metadata.get("source"),
#                         document_type=doc.metadata.get("corpus"),
#                         reranker_score=float(score),
#                         rank=i + 1,
#                     )
#                 )

#             observer.record_stage(
#                 "Reranker",
#                 end_time - start_time,
#                 final_top_k,
#                 observer_docs,
#             )

#         return final_docs


import os
import threading
from time import perf_counter

import onnxruntime as ort
from transformers import AutoTokenizer

from langchain_core.documents import Document

from reranking.base_reranker import BaseReranker
from evaluation.models import RetrievedDocument
from evaluation.observer import PipelineObserver


class CrossEncoderReranker(BaseReranker):

    def __init__(
        self,
        rerank_model: str,
        top_k: int,
        batch_size: int = 16,
        max_length: int = 750,
        onnx_file: str = "model.onnx",
        num_threads: int = 4,
    ):
        print(
            f"[RERANKER INIT] "
            f"Loading model: {rerank_model} ({onnx_file})"
        )

        init_start = perf_counter()

        # ---------------------------------------------------------
        # Load tokenizer + ONNX model
        #
        # Supports:
        #   1. Local model directory
        #   2. Hugging Face model ID
        # ---------------------------------------------------------

        if os.path.isdir(rerank_model):
            # Local model
            self._tokenizer = AutoTokenizer.from_pretrained(
                rerank_model
            )

            onnx_path = os.path.join(
                rerank_model,
                onnx_file,
            )

            if not os.path.isfile(onnx_path):
                raise FileNotFoundError(
                    f"ONNX model not found: {onnx_path}"
                )

            print(
                f"[RERANKER INIT] "
                f"Using local ONNX model: {onnx_path}"
            )

        else:
            # Hugging Face model
            self._tokenizer = AutoTokenizer.from_pretrained(
                rerank_model
            )

            from huggingface_hub import hf_hub_download

            onnx_path = hf_hub_download(
                rerank_model,
                onnx_file,
            )

            print(
                f"[RERANKER INIT] "
                f"Using Hugging Face ONNX model: {onnx_path}"
            )

        # ---------------------------------------------------------
        # ONNX Runtime
        # ---------------------------------------------------------

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads

        self._session = ort.InferenceSession(
            onnx_path,
            sess_options,
            providers=["CPUExecutionProvider"],
        )

        self._input_names = {
            i.name for i in self._session.get_inputs()
        }

        self._top_k = top_k
        self._batch_size = batch_size
        self._max_length = max_length

        init_end = perf_counter()

        print(
            f"[RERANKER INIT] "
            f"Model loaded in {init_end - init_start:.3f}s"
        )

        print(
            f"[RERANKER INIT] "
            f"onnx_file={onnx_file} "
            f"threads={num_threads}"
        )

    def _score(
        self,
        pairs: list[tuple[str, str]],
    ) -> list[float]:

        scores: list[float] = []

        for i in range(0, len(pairs), self._batch_size):
            batch = pairs[i : i + self._batch_size]

            enc = self._tokenizer(
                [q for q, _ in batch],
                [d for _, d in batch],
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="np",
            )

            ort_inputs = {
                k: v
                for k, v in enc.items()
                if k in self._input_names
            }

            logits = self._session.run(
                None,
                ort_inputs,
            )[0]

            if logits.ndim > 1 and logits.shape[-1] == 1:
                logits = logits.squeeze(-1)

            scores.extend(
                logits.reshape(-1).tolist()
            )

        return scores

    def rerank(
        self,
        query: str,
        documents: list[Document],
        observer: PipelineObserver | None = None,
        top_k: int | None = None,
    ) -> list[Document]:

        if not documents:
            return []

        final_top_k = (
            top_k
            if top_k is not None
            else self._top_k
        )

        print(
            f"\n[RERANK ENTER] "
            f"thread={threading.current_thread().name} "
            f"docs={len(documents)}"
        )

        print(
            "[RERANK] Creating document/query pairs..."
        )

        pairs_start = perf_counter()

        pairs = [
            (query, doc.page_content)
            for doc in documents
        ]

        pairs_end = perf_counter()

        print(
            f"[RERANK] Pairs created in "
            f"{pairs_end - pairs_start:.3f}s"
        )

        print(
            f"[RERANK] Starting prediction "
            f"batch_size={self._batch_size}..."
        )

        start_time = perf_counter()

        scores = self._score(pairs)

        end_time = perf_counter()

        print(
            f"[RERANK] Prediction finished "
            f"thread={threading.current_thread().name} "
            f"docs={len(documents)} "
            f"duration={end_time - start_time:.3f}s"
        )

        scored_docs = list(
            zip(documents, scores)
        )

        print("[RERANK] Sorting scores...")

        scored_docs.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        final_docs = [
            doc
            for doc, _ in scored_docs[:final_top_k]
        ]

        final_scores = [
            score
            for _, score in scored_docs[:final_top_k]
        ]

        total_end_time = perf_counter()

        print(
            f"[RERANK EXIT] "
            f"thread={threading.current_thread().name} "
            f"docs={len(documents)} "
            f"total={total_end_time - start_time:.3f}s "
            f"returned={len(final_docs)}"
        )

        if observer is not None:
            observer_docs: list[RetrievedDocument] = []

            for i, (doc, score) in enumerate(
                zip(final_docs, final_scores)
            ):
                observer_docs.append(
                    RetrievedDocument(
                        chunk_id=doc.metadata.get("chunk_id"),
                        source=doc.metadata.get("source"),
                        document_type=doc.metadata.get("corpus"),
                        reranker_score=float(score),
                        rank=i + 1,
                    )
                )

            observer.record_stage(
                "Reranker",
                end_time - start_time,
                final_top_k,
                observer_docs,
            )

        return final_docs