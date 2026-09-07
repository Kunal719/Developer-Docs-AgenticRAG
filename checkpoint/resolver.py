from typing import Protocol

from langchain_core.documents import Document
from langchain_chroma import Chroma

class EvidenceResolver(Protocol):
    def __call__(self, corpus: str, chunk_id: str) -> Document:
        ...

class ChromaEvidenceResolver:
    def __init__(self, doc_store: Chroma, code_store: Chroma):
        self._doc_store = doc_store
        self._code_store = code_store

    def resolve(self, corpus: str, chunk_id: str) -> Document:
        """
        Get document from the store based on corpus and chunk_id
        """
        if corpus == "implementation":
             store = self._code_store
        elif corpus == "documentation":
            store = self._doc_store
        else:
            raise ValueError(f"Unknown corpus: {corpus}")

        result = store.get(
            where = {"chunk_id": chunk_id}
        )

        if not result["ids"]:
            raise LookupError(f"Could not find document with chunk_id: {chunk_id}")

        return Document(
            page_content = result["documents"][0],
            metadata = result["metadatas"][0]
        )


    def __call__(self, corpus: str, chunk_id: str) -> Document:
        return self.resolve(corpus, chunk_id)