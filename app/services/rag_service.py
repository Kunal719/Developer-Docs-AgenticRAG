from retrieval.base import BaseRetriever
from reranking.base_reranker import BaseReranker
from langchain_core.language_models.chat_models import BaseChatModel

from ingestion.embedding_model import get_embedding_model
from pipeline.initialization import initialize_resources
from pipeline.prepare_retrieval_resources import prepare_retrieval_resources
from pipeline.build_rag_pipeline import build_rag_pipeline

from ingestion.markdown_parser import load_documents, load_selected_documents
from ingestion.metadata_extractor import extract_metadata

from ingestion.python_loader import load_python_documents, load_selected_python_documents

from ingestion.chunker import chunk_documents
from ingestion.code_chunker import chunk_python_documents

from ingestion.code_metadata_extractor import extract_code_metadata

from dotenv import load_dotenv
from collections.abc import Iterator

from prompt.generation_prompt import generation_prompt
from utils.document_formatter import format_retrieved_context

from app.schemas.query import Source, QueryResponse

load_dotenv()

from config import (
    GITHUB_DOC_REPO_URL,
    GITHUB_CODE_REPO_URL,
    LOCAL_DOC_REPO_PATH,
    LOCAL_CODE_REPO_PATH,
    DOCUMENTATION_SPARSE_PATHS,
    IMPLEMENTATION_SPARSE_PATHS,
    DOC_PERSISTENT_DIRECTORY,
    CODE_PERSISTENT_DIRECTORY,
    DOC_COLLECTION_NAME,
    CODE_COLLECTION_NAME,
    DOCUMENTATION_PATH,
    IMPLEMENTATION_PATH,
    DOC_BM25_PATH,
    CODE_BM25_PATH,
    DOC_INDEX_PATH,
    CODE_INDEX_PATH,
)

class RAGService:
    def __init__(self):
        self._retriever: BaseRetriever | None = None
        self._reranker: BaseReranker | None = None
        self._llm: BaseChatModel | None = None

    def initialize(self) -> None:
        embedding_model = get_embedding_model()

        # Documentation corpus
        doc_repository_status, doc_vector_store = initialize_resources(
            GITHUB_DOC_REPO_URL, 
            LOCAL_DOC_REPO_PATH, 
            DOCUMENTATION_SPARSE_PATHS, 
            DOC_PERSISTENT_DIRECTORY, 
            DOC_COLLECTION_NAME, 
            embedding_model)

        doc_vector_store, doc_bm25_index = prepare_retrieval_resources(
            repository_status=doc_repository_status,
            source_path=DOCUMENTATION_PATH,
            glob_pattern="**/*.mdx",
            vector_store=doc_vector_store,
            bm25_path=DOC_BM25_PATH,
            index_path=DOC_INDEX_PATH,
            loader=load_documents,
            selected_loader=load_selected_documents,
            metadata_extractor=extract_metadata,
            chunker=chunk_documents
        )

        # Implementation corpus
        code_repository_status, code_vector_store = initialize_resources(
            GITHUB_CODE_REPO_URL, 
            LOCAL_CODE_REPO_PATH, 
            IMPLEMENTATION_SPARSE_PATHS, 
            CODE_PERSISTENT_DIRECTORY, 
            CODE_COLLECTION_NAME, 
            embedding_model)

        code_vector_store, code_bm25_index = prepare_retrieval_resources(
            repository_status=code_repository_status,
            source_path=IMPLEMENTATION_PATH,
            glob_pattern="**/*.py",
            vector_store=code_vector_store,
            bm25_path=CODE_BM25_PATH,
            index_path=CODE_INDEX_PATH,
            loader=load_python_documents,
            selected_loader=load_selected_python_documents,
            metadata_extractor=extract_code_metadata,
            chunker=chunk_python_documents
        )

        # Initialize runtime objects
        _, self._retriever, self._reranker, self._llm = build_rag_pipeline(
            doc_vector_store=doc_vector_store,
            doc_bm25_index=doc_bm25_index,
            code_vector_store=code_vector_store,
            code_bm25_index=code_bm25_index
        )

    def stream(self, question: str) -> Iterator[tuple[str, str | list[Source]]]:

        if self._retriever is None or self._reranker is None or self._llm is None:
            raise RuntimeError("RAG service is not initialized")

        yield ("status", "🔎 Searching the docs...")
        documents = self._retriever.retrieve(question)

        yield ("status", "🎯 Hunting down the most relevant sources...")
        reranked_documents = self._reranker.rerank(question, documents)


        sources = []
        seen_sources = set()
        for rank, doc in enumerate(reranked_documents, start=1):
            source = doc.metadata.get("source", "unknown")

            if source in seen_sources:
                rank -= 1
                continue

            seen_sources.add(source)

            sources.append(
                Source(
                    source=source, 
                    corpus=doc.metadata.get("corpus","unknown"), 
                    rank=len(sources) + 1
                )
            )

        # Format context
        context = format_retrieved_context(reranked_documents)

        # Build prompt using question and context
        prompt = generation_prompt.format(question=question, context=context)

        # Stream the response
        yield ("status", "🤓 Making sure the answer isn't hallucinating...")
        for chunk in self._llm.stream(prompt):
            if chunk.content:
                yield("token", chunk.content)

        yield("sources", sources)

    def query(self, question: str) -> QueryResponse:
    
            if self._retriever is None or self._reranker is None or self._llm is None:
                raise RuntimeError("RAG service is not initialized")
    
            documents = self._retriever.retrieve(question)
            reranked_documents = self._reranker.rerank(question, documents)
    
            sources = []
            seen_sources = set()
            for rank, doc in enumerate(reranked_documents, start=1):
                source = doc.metadata.get("source", "unknown")
    
                if source in seen_sources:
                    rank -= 1
                    continue
    
                seen_sources.add(source)
    
                sources.append(
                    Source(
                        source=source, 
                        corpus=doc.metadata.get("corpus","unknown"), 
                        rank=len(sources) + 1
                    )
                )
    
            # Format context
            context = format_retrieved_context(reranked_documents)
    
            # Build prompt using question and context
            prompt = generation_prompt.format(question=question, context=context)
    
            # Get full response
            response = self._llm.invoke(prompt)

            return QueryResponse(
                answer = response.content,
                sources = sources
            )
    
if __name__ == "__main__":
    service = RAGService()
    service.initialize()

    for event_type, data in service.stream(
        "How do I create a checkpoint?"
    ):
        if event_type == "sources":
            print("\n\nSources:")
            for source in data:
                print(f"{source.rank}. {source.source} ({source.corpus})")

        elif event_type == "token":
            print(data, end="", flush=True)

    print()