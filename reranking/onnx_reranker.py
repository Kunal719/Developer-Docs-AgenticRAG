from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from reranking.base_reranker import BaseReranker
from langchain_core.documents import Document

from evaluation.models import RetrievedDocument
from evaluation.observer import PipelineObserver


class ONNXReranker(BaseReranker):

    def __init__(
        self,
        model_path: str,
        top_k: int,
        batch_size: int = 16,
        intra_op_num_threads: int = 8,
    ):
        self._model_path = Path(model_path)
        self._top_k = top_k
        self._batch_size = batch_size

        # Load tokenizer ONCE.
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_path
        )

        # Configure ONNX Runtime.
        session_options = ort.SessionOptions()

        session_options.intra_op_num_threads = (
            intra_op_num_threads
        )

        session_options.inter_op_num_threads = 1

        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Load ONNX model ONCE.
        self._session = ort.InferenceSession(
            str(self._model_path / "model.onnx"),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )

    def _predict(
        self,
        query: str,
        documents: list[Document],
    ) -> np.ndarray:

        all_scores = []

        # Process candidates in batches.
        for start in range(
            0,
            len(documents),
            self._batch_size,
        ):
            batch_documents = documents[
                start:start + self._batch_size
            ]

            inputs = self._tokenizer(
                [query] * len(batch_documents),
                [
                    document.page_content
                    for document in batch_documents
                ],
                padding=True,
                truncation=True,
                return_tensors="np",
            )

            outputs = self._session.run(
                None,
                {
                    key: value
                    for key, value in inputs.items()
                },
            )

            scores = outputs[0].squeeze(-1)

            all_scores.extend(
                np.atleast_1d(scores).tolist()
            )

        return np.asarray(all_scores)

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

        start_time = perf_counter()

        scores = self._predict(
            query=query,
            documents=documents,
        )

        scored_docs = list(
            zip(documents, scores)
        )

        # Highest score first.
        scored_docs.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        final_docs = [
            document
            for document, _ in scored_docs[:final_top_k]
        ]

        final_scores = [
            float(score)
            for _, score in scored_docs[:final_top_k]
        ]

        end_time = perf_counter()

        if observer is not None:

            observer_docs: list[
                RetrievedDocument
            ] = []

            for i, (
                document,
                score,
            ) in enumerate(
                zip(
                    final_docs,
                    final_scores,
                )
            ):
                observer_docs.append(
                    RetrievedDocument(
                        chunk_id=document.metadata.get(
                            "chunk_id"
                        ),
                        source=document.metadata.get(
                            "source"
                        ),
                        document_type=document.metadata.get(
                            "corpus"
                        ),
                        reranker_score=score,
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