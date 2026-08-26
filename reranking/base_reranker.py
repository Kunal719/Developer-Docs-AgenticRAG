from abc import abstractmethod, ABC
from langchain_core.documents import Document

from evaluation.observer import PipelineObserver

class BaseReranker(ABC):
    """
    The base class for all rerankers
    """

    @abstractmethod
    def rerank(self, query: str, documents: list[Document], observer: PipelineObserver | None = None, top_k: int | None = None) -> list[Document]:
        ...