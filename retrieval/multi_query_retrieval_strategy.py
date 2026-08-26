from retrieval.base import BaseRetriever

from models.retrieval_plan import MetadataFilters

from evaluation.observer import PipelineObserver
from evaluation.models import RetrievedDocument

from langchain_core.documents import Document

from time import perf_counter

class MultiQueryRetrievalStrategy():

    def __init__(self, top_k: int, rrf_k: int):
        self._top_k = top_k
        self._rrf_k = rrf_k

    def _rrf(self, results_list: list[list[Document]]) -> tuple[list[Document], list[float]]:

        scores = {}
        doc_map= {}

        for retrieved_docs in results_list:
            for rank, doc in enumerate(retrieved_docs):

                doc_id = doc.metadata["chunk_id"]
                doc_map[doc_id] = doc

                if doc_id not in scores:
                    scores[doc_id] = 0.0

                scores[doc_id] += 1 / (rank + self._rrf_k + 1)

        ranked_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        fused_docs = [doc_map[doc_id] for doc_id, _ in ranked_scores[:self._top_k]]
        fused_scores = [scores[doc_id] for doc_id, _ in ranked_scores[:self._top_k]]

        return fused_docs, fused_scores

    def retrieve(self, queries: list[str], retriever: BaseRetriever, filters: MetadataFilters | None = None, observer: PipelineObserver | None = None) -> list[Document]:

        query_results = []

        start_time = perf_counter()
        for query in queries:
            # We will not add internal query retrieval stages in observer
            documents = retriever.retrieve(query, observer=None, filters=filters)

            query_results.append(documents)

        fused_documents, fused_scores = self._rrf(query_results)
        end_time = perf_counter()

        if observer is not None:
            observed_docs = [
                RetrievedDocument(
                    chunk_id=doc.metadata.get("chunk_id"),
                    source=doc.metadata.get("source"),
                    document_type=doc.metadata.get("corpus"),
                    retriever_score=score,
                    rank=i + 1,
                )
                for i, (doc, score) in enumerate(zip(fused_documents, fused_scores))
            ]

            observer.record_stage("Multi Query Fusion", (end_time - start_time), self._top_k, observed_docs)

        return fused_documents