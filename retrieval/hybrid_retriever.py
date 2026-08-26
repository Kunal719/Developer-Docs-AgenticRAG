from retrieval.base import BaseRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.dense_retriever import DenseRetriever

from models.retrieval_plan import MetadataFilters

from evaluation.observer import PipelineObserver
from evaluation.models import RetrievedDocument
from langchain_core.documents import Document
from time import perf_counter

class HybridRetriever(BaseRetriever):
    def __init__(self, bm25_retriever: BM25Retriever, dense_retriever: DenseRetriever, top_k: int, rrf_k: int):
        self._bm25_index = bm25_retriever
        self._dense_retriever = dense_retriever
        self._top_k = top_k
        self._rrf_k = rrf_k

    def _rrf(self, results_list: list[list[Document]]) -> list[Document]:
        scores = {}
        doc_map = {}

        for retrieved_docs in results_list:
            for rank, doc in enumerate(retrieved_docs):
                doc_id = doc.metadata["chunk_id"]
                doc_map[doc_id] = doc

                if doc_id not in scores:
                    scores[doc_id] = 0

                scores[doc_id] += 1 / (rank + self._rrf_k + 1)

        ranked_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        fused_docs = [doc_map[doc_id] for doc_id, _ in ranked_scores[:self._top_k]]
        fused_scores = [scores[doc_id] for doc_id, _ in ranked_scores[:self._top_k]]

        return fused_docs, fused_scores

    def retrieve(self, query: str, observer: PipelineObserver | None = None, filters: MetadataFilters = None) -> list[Document]:
        """
        Retrieve the most relevant documents using Dense Retriever and BM25 Retriever

        Args:
            query: Query string
            observer: The pipeline observer which records the metrics

        Returns:
            List of retrieved documents
        """

        dense_results = self._dense_retriever.retrieve(query, observer, filters)
        bm25_results = self._bm25_index.retrieve(query, observer, filters)

        start_time = perf_counter()
        fused_results, fused_scores = self._rrf([dense_results, bm25_results])
        end_time = perf_counter()

        if observer is not None:
            observer_docs: list[RetrievedDocument] = []

            for i, (doc, score) in enumerate(zip(fused_results, fused_scores)):
                observer_docs.append(RetrievedDocument(chunk_id=doc.metadata.get("chunk_id"), source=doc.metadata.get("source"), document_type=doc.metadata.get("corpus"), retriever_score=score, rank=i+1))

            observer.record_stage("RRF Fusion", (end_time - start_time), self._top_k, observer_docs)

        return fused_results

