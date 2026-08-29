from langchain_core.documents import Document
from langchain_chroma import Chroma

from retrieval.base import BaseRetriever
from retrieval.filters import build_chroma_where

from evaluation.models import RetrievedDocument
from evaluation.observer import PipelineObserver

from models.retrieval_plan import MetadataFilters

from time import perf_counter

class DenseRetriever(BaseRetriever):
    def __init__(self, vector_store: Chroma, top_k: int):
        self._vector_store = vector_store
        self._top_k = top_k

    def retrieve(self, query: str, observer: PipelineObserver | None = None, filters: MetadataFilters = None, gap_top_k: int | None = None) -> list[Document]:
        """
        Retrieve the most relevant documents using Dense Retriever

        Args:
            query: Query string
            observer: The pipeline observer which records the metrics

        Returns:
            List of retrieved documents
        """

        where_clauses = build_chroma_where(filters)

        search_kwargs = {"k": self._top_k}

        if where_clauses is not None:
            search_kwargs["filter"] = where_clauses

        retriever = self._vector_store.as_retriever(
            search_type = "similarity",
            search_kwargs = search_kwargs,
        )

        start_time = perf_counter()
        documents = retriever.invoke(query)
        end_time = perf_counter()

        if observer is not None:
            observer_docs: list[RetrievedDocument] = []

            for doc in documents:
                observer_docs.append(RetrievedDocument(chunk_id=doc.metadata.get("chunk_id"), source=doc.metadata.get("source"),document_type=doc.metadata.get("corpus"), retriever_score=None))

            observer.record_stage("Dense Retriever", (end_time - start_time), self._top_k, observer_docs)

        return documents   

         
