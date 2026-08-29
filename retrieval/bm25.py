from langchain_community.retrievers import BM25Retriever as LangChainBM25
from langchain_core.documents import Document

from retrieval.base import BaseRetriever
from retrieval.filters import matches_filter

from evaluation.models import RetrievedDocument
from evaluation.observer import PipelineObserver

from models.retrieval_plan import MetadataFilters

from time import perf_counter

class BM25Retriever(BaseRetriever):
    def __init__(self, bm25_index: LangChainBM25):
        self._bm25_index = bm25_index

    def retrieve(self, query: str, observer : PipelineObserver | None = None, filters: MetadataFilters = None, gap_top_k: int | None = None) -> list[Document]:
        """
        Retrieve the most relevant documents using BM25
        
        Args:
            query: Query string
            observer: The pipeline observer which records the metrics
        
        Returns:
            List of retrieved documents
        """ 

        start_time = perf_counter()
        documents = self._bm25_index.invoke(query)
        end_time = perf_counter()

        filtered_docs = []

        if filters is not None:
            for doc in documents:
                if matches_filter(doc, filters):
                    filtered_docs.append(doc)

            documents = filtered_docs

        if observer is not None:
            observer_docs: list[RetrievedDocument] = []

            for doc in documents:
                observer_docs.append(RetrievedDocument(chunk_id=doc.metadata.get("chunk_id"), source=doc.metadata.get("source"), document_type=doc.metadata.get("corpus"), retriever_score=None))

            observer.record_stage("BM25 Retriever", (end_time - start_time), self._bm25_index.k, observer_docs)

        return documents