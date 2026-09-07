import pytest

from langchain_core.documents import Document

from checkpoint.resolver import ChromaEvidenceResolver


class FakeChromaStore:
    def __init__(self, documents: list[Document]):
        self._documents = documents

    def get(self, where: dict):
        chunk_id = where["chunk_id"]

        matches = [
            doc
            for doc in self._documents
            if doc.metadata["chunk_id"] == chunk_id
        ]

        return {
            "ids": [f"chroma-id-{i}" for i in range(len(matches))],
            "documents": [doc.page_content for doc in matches],
            "metadatas": [doc.metadata for doc in matches],
        }


@pytest.fixture
def resolver():
    doc_store = FakeChromaStore(
        [
            Document(
                page_content="Documentation chunk",
                metadata={
                    "chunk_id": "doc-123",
                    "corpus": "documentation",
                },
            )
        ]
    )

    code_store = FakeChromaStore(
        [
            Document(
                page_content="Implementation chunk",
                metadata={
                    "chunk_id": "code-456",
                    "corpus": "implementation",
                },
            )
        ]
    )

    return ChromaEvidenceResolver(
        doc_store=doc_store,
        code_store=code_store,
    )


def test_resolve_documentation_chunk(resolver):
    document = resolver.resolve(
        corpus="documentation",
        chunk_id="doc-123",
    )

    assert document.page_content == "Documentation chunk"
    assert document.metadata["chunk_id"] == "doc-123"
    assert document.metadata["corpus"] == "documentation"


def test_resolve_implementation_chunk(resolver):
    document = resolver.resolve(
        corpus="implementation",
        chunk_id="code-456",
    )

    assert document.page_content == "Implementation chunk"
    assert document.metadata["chunk_id"] == "code-456"
    assert document.metadata["corpus"] == "implementation"


def test_resolve_unknown_corpus(resolver):
    with pytest.raises(ValueError, match="Unknown corpus"):
        resolver.resolve(
            corpus="unknown",
            chunk_id="doc-123",
        )


def test_resolve_missing_chunk(resolver):
    with pytest.raises(
        LookupError,
        match="Could not find document",
    ):
        resolver.resolve(
            corpus="documentation",
            chunk_id="does-not-exist",
        )