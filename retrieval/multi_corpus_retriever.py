from retrieval.base import BaseRetriever
from evaluation.observer import PipelineObserver
from models.retrieval_plan import MetadataFilters

from langchain_core.documents import Document

from concurrent.futures import ThreadPoolExecutor

class MultiCorpusRetriever(BaseRetriever):

    def __init__(self, retrievers: list[BaseRetriever]):
        self._retrievers = retrievers

    def retrieve(self, query: str, observer: PipelineObserver | None = None, filters: MetadataFilters | None = None, gap_top_k: int | None = None) -> list[Document]:
        """
        Retrieve the combined most relevant documents for given query from all retrievers

        Args:
            query: Query string

        Returns:
            List of retrieved documents
        """
        combined_docs = []

        # for retriever in self._retrievers:
        #     combined_docs.extend(retriever.retrieve(query, observer=observer, filters=filters, gap_top_k=gap_top_k))

        with ThreadPoolExecutor(max_workers=len(self._retrievers)) as executor:
            futures = [
                executor.submit(
                    retriever.retrieve,
                    query=query,
                    observer=observer,
                    filters=filters,
                    gap_top_k=gap_top_k,
                )
                for retriever in self._retrievers
            ]

            for future in futures:
                combined_docs.extend(future.result())
            
        return combined_docs