from langchain_core.documents import Document
from retrieval.base import BaseRetriever
from langchain_chroma import Chroma
from evaluation.models import RetrievedDocument
from evaluation.observer import PipelineObserver
from time import perf_counter

class DenseRetriever(BaseRetriever):
    def __init__(self, vector_store: Chroma, top_k: int):
        self._vector_store = vector_store
        self._top_k = top_k

    def retrieve(self, query: str, observer: PipelineObserver | None = None) -> list[Document]:
        """
        Retrieve the most relevant documents using Dense Retriever

        Args:
            query: Query string
            observer: The pipeline observer which records the metrics

        Returns:
            List of retrieved documents
        """

        retriever = self._vector_store.as_retriever(
            search_type = "similarity",
            search_kwargs = {"k": self._top_k},
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

         
