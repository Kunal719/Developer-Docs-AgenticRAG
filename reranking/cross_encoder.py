from reranking.base_reranker import BaseReranker
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from evaluation.models import RetrievedDocument
from evaluation.observer import PipelineObserver
from time import perf_counter

class CrossEncoderReranker(BaseReranker):
    def __init__(self, rerank_model: str, top_k: int):
        self._rerank_model = CrossEncoder(rerank_model)
        self._top_k = top_k

    def rerank(self, query: str, documents: list[Document], observer: PipelineObserver | None = None, top_k: int | None = None) -> list[Document]:
        """
        Given a query and a list of documents, rerank the documents based on the query using cross-encoder reranker
        """
        if not documents:
            return []

        final_top_k = top_k if top_k is not None else self._top_k

        pairs = [(query, doc.page_content) for doc in documents]

        start_time = perf_counter()
        scores = self._rerank_model.predict(pairs)

        scored_docs = list(zip(documents, scores))

        # Sort by score (descending)
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        final_docs = [doc for doc, _ in scored_docs[:final_top_k]]
        final_scores = [score for _, score in scored_docs[:final_top_k]]
        end_time = perf_counter()

        if observer is not None:
            observer_docs: list[RetrievedDocument] = []

            for i, (doc, score) in enumerate(zip(final_docs, final_scores)):
                observer_docs.append(RetrievedDocument(chunk_id=doc.metadata.get("chunk_id"), source=doc.metadata.get("source"), document_type=doc.metadata.get("corpus"), reranker_score=float(score), rank=i+1))

            observer.record_stage("Reranker", (end_time - start_time), final_top_k, observer_docs)

        # for doc in final_docs:
            # print(f"Corpus: {doc.metadata['corpus']} \n Source: {doc.metadata['source']}")
        return final_docs