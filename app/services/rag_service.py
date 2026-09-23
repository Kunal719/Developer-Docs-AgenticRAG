from uuid import uuid4
from dotenv import load_dotenv
from dataclasses import dataclass
from collections.abc import Iterator
from queue import Queue
from threading import Thread
from typing import Literal

from ingestion.embedding_model import get_embedding_model
from ingestion.markdown_parser import load_documents, load_selected_documents
from ingestion.metadata_extractor import extract_metadata
from ingestion.python_loader import load_python_documents, load_selected_python_documents
from ingestion.code_chunker import chunk_python_documents
from ingestion.chunker import chunk_documents
from ingestion.code_metadata_extractor import extract_code_metadata

from pipeline.initialization import initialize_resources
from pipeline.prepare_retrieval_resources import prepare_retrieval_resources
from pipeline.agentic_loop import AgenticRAG
from pipeline.input_normalizer import InputNormalizer

from retrieval.dense_retriever import DenseRetriever
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.multi_corpus_retriever import MultiCorpusRetriever
from retrieval.multi_query_retrieval_strategy import MultiQueryRetrievalStrategy

from reranking.cross_encoder import CrossEncoderReranker

from router.query_router import QueryRouter

from checkpoint.database import create_connection_pool
from checkpoint.store import CheckpointStore
from checkpoint.postgres_store import PostgresCheckpointStore
from checkpoint.resolver import ChromaEvidenceResolver
from checkpoint.models import RunStatus

from evidence.rerank_selector import RerankEvidenceSelector

from app.llm import get_llm
from app.auth.user_store import UserStore

from models.generation import GeneratedAnswer
from checkpoint.models import RunStatus

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
    MODEL_NAME,
    MODEL_PROVIDER,
    MODEL_TEMP,
    RRF_K,
    DENSE_TOP_K,
    MULTI_QUERY_TOP_K,
    DOC_HYBRID_TOP_K,
    CODE_HYBRID_TOP_K,
    RERANK_TOP_K,
    NORMALIZER_MODEL_NAME,
    NORMALIZER_MODEL_PROVIDER
)

load_dotenv()

@dataclass
class RAGRunResult:
    run_id: str
    answer: GeneratedAnswer

@dataclass
class RAGStreamEvent:
    type: Literal["run_started", "run_resumed", "generation_started", "answer_delta", "completed", "error"]
    run_id: str
    data: str | None = None
    generation_round: int | None = None


class RAGService:
    def __init__(self):
        self._agentic_rag: AgenticRAG | None = None
        self._checkpoint_store: CheckpointStore | None = None

    def initialize(self) -> None:
        embedding_model = get_embedding_model()

        # Documentation corpus
        doc_repository_status, doc_vector_store = initialize_resources(
            GITHUB_DOC_REPO_URL,
            LOCAL_DOC_REPO_PATH,
            DOCUMENTATION_SPARSE_PATHS,
            DOC_PERSISTENT_DIRECTORY,
            DOC_COLLECTION_NAME,
            embedding_model,
        )

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
            chunker=chunk_documents,
        )

        # Implementation corpus
        code_repository_status, code_vector_store = initialize_resources(
            GITHUB_CODE_REPO_URL,
            LOCAL_CODE_REPO_PATH,
            IMPLEMENTATION_SPARSE_PATHS,
            CODE_PERSISTENT_DIRECTORY,
            CODE_COLLECTION_NAME,
            embedding_model,
        )

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
            chunker=chunk_python_documents,
        )

        # ---------------------------------------------------------
        # Runtime retrievers
        # ---------------------------------------------------------
        doc_dense_retriever = DenseRetriever(
            vector_store=doc_vector_store,
            top_k=DENSE_TOP_K,
        )

        doc_bm25_retriever = BM25Retriever(
            bm25_index=doc_bm25_index,
        )

        doc_hybrid_retriever = HybridRetriever(
            bm25_retriever=doc_bm25_retriever,
            dense_retriever=doc_dense_retriever,
            top_k=DOC_HYBRID_TOP_K,
            rrf_k=RRF_K,
        )

        code_dense_retriever = DenseRetriever(
            vector_store=code_vector_store,
            top_k=DENSE_TOP_K,
        )

        code_bm25_retriever = BM25Retriever(
            bm25_index=code_bm25_index,
        )

        code_hybrid_retriever = HybridRetriever(
            bm25_retriever=code_bm25_retriever,
            dense_retriever=code_dense_retriever,
            top_k=CODE_HYBRID_TOP_K,
            rrf_k=RRF_K,
        )

        multi_corpus_retriever = MultiCorpusRetriever(
            retrievers=[
                doc_hybrid_retriever,
                code_hybrid_retriever,
            ]
        )

        multi_query_strategy = MultiQueryRetrievalStrategy(
            top_k=MULTI_QUERY_TOP_K,
            rrf_k=RRF_K,
        )

        reranker = CrossEncoderReranker(
            rerank_model="cross-encoder/ettin-reranker-32m-v1",
            top_k=RERANK_TOP_K,
            batch_size=16,
        )

        evidence_selector = RerankEvidenceSelector(reranker)

        # Checkpoint persistence
        checkpoint_pool = create_connection_pool()
        checkpoint_store = PostgresCheckpointStore(checkpoint_pool)

        # User store



        evidence_resolver = ChromaEvidenceResolver(
            doc_store=doc_vector_store,
            code_store=code_vector_store,
        )

        # LLMs
        router_llm = get_llm(
            model_name=MODEL_NAME,
            model_provider=MODEL_PROVIDER,
            temperature=MODEL_TEMP,
            max_tokens=3000,
        )

        # Normalizer LLM
        input_normalizer_llm = get_llm(
            model_name=NORMALIZER_MODEL_NAME,
            model_provider=NORMALIZER_MODEL_PROVIDER,
            temperature=0
        )

        self._input_normalizer = InputNormalizer(input_normalizer_llm)

        self._generation_llm = get_llm(
            model_name=MODEL_NAME,
            model_provider=MODEL_PROVIDER,
            temperature=MODEL_TEMP
        )

        # Initialize Agentic RAG
        self._router_llm = router_llm
        self._doc_hybrid_retriever = doc_hybrid_retriever
        self._code_hybrid_retriever = code_hybrid_retriever
        self._multi_corpus_retriever = multi_corpus_retriever
        self._multi_query_strategy = multi_query_strategy
        self._reranker = reranker
        self._code_vector_store = code_vector_store
        self._evidence_selector = evidence_selector
        self._checkpoint_store = checkpoint_store
        self._user_store = UserStore(self.checkpoint_store.pool)
        self._evidence_resolver = evidence_resolver

        self._agentic_rag = AgenticRAG(
            query_router=QueryRouter(self._router_llm),
            generation_llm=self._generation_llm,
            doc_hybrid_retriever=self._doc_hybrid_retriever,
            code_hybrid_retriever=self._code_hybrid_retriever,
            multi_corpus_retriever=self._multi_corpus_retriever,
            multi_query_strategy=self._multi_query_strategy,
            reranker=self._reranker,
            code_vector_store=self._code_vector_store,
            evidence_selector=self._evidence_selector,
            checkpoint_store=self._checkpoint_store,
            evidence_resolver=self._evidence_resolver,
        )

    def _require_initialized(self) -> AgenticRAG:
        if self._agentic_rag is None:
            raise RuntimeError("RAG service is not initialized")

        return self._agentic_rag

    @property
    def checkpoint_store(self) -> PostgresCheckpointStore:
        if self._checkpoint_store is None:
            raise RuntimeError("Checkpoint store is not initialized")

        return self._checkpoint_store

    @property
    def user_store(self) -> UserStore:
        if self._user_store is None:
            raise RuntimeError("User store is not initialized")

        return self._user_store

    def run(self, question: str, user_id: str) -> Iterator[RAGStreamEvent]:
        agentic_rag = self._require_initialized()

        run_id = uuid4().hex
        query_input = self._input_normalizer.normalize(question)

        answer_queue: Queue[RAGStreamEvent | None] = Queue()

        current_generation_round = 0

        def on_generation_started(round_number: int) -> None:
            nonlocal current_generation_round
            current_generation_round = round_number

            answer_queue.put(
                RAGStreamEvent(
                    type="generation_started",
                    run_id=run_id,
                    generation_round=round_number
                )
            )

        def on_answer_delta(delta: str) -> None:
            answer_queue.put(
                RAGStreamEvent(
                    type="answer_delta",
                    run_id=run_id,
                    data=delta,
                    generation_round=current_generation_round
                )
            )

        def run_agent() -> None:
            try:
                agentic_rag.run(
                    query_input=query_input,
                    run_id=run_id,
                    user_id=user_id,
                    on_answer_delta=on_answer_delta,
                    on_generation_started=on_generation_started
                )

                answer_queue.put(
                    RAGStreamEvent(
                        type="run_completed",
                        run_id=run_id
                    )
                )

            except Exception as e:
                try:
                    self._checkpoint_store.update_status(run_id, RunStatus.FAILED, str(e))
                except KeyError:
                    # If the checkpoint is not found, this is a new run. We can ignore this.
                    pass

                answer_queue.put(
                    RAGStreamEvent(
                        type="error",
                        run_id=run_id,
                        data=str(e)
                    )
                )

            finally:
                answer_queue.put(None)


        answer_queue.put(
            RAGStreamEvent(
                type="run_started",
                run_id=run_id
            )
        )

        worker = Thread(
            target=run_agent,
            daemon=True
        )
        worker.start()

        while True:
            item = answer_queue.get()

            if item is None:
                break

            yield item

     

    def resume(self, run_id: str, user_id: str) -> Iterator[RAGStreamEvent]:
        agentic_rag = self._require_initialized()

        checkpoint_store = self._checkpoint_store
        if checkpoint_store is None:
            raise RuntimeError("Checkpoint store is not initialized")

        checkpoint = checkpoint_store.load(run_id, user_id)
        status = self._checkpoint_store.get_status(run_id)

        if status != RunStatus.INTERRUPTED:
            raise ValueError(f"Run {run_id} cannot be resumed from status {status.value}")

        answer_queue: Queue[RAGStreamEvent | None] = Queue()

        # Get current generation round
        current_generation_round = checkpoint.generation_round

        def on_generation_started(round_number: int) -> None:
            nonlocal current_generation_round
            current_generation_round = round_number

            answer_queue.put(
                RAGStreamEvent(
                    type="generation_started",
                    run_id=run_id,
                    generation_round=round_number
                )
            )

        def on_answer_delta(delta: str) -> None:
            answer_queue.put(
                RAGStreamEvent(
                    type="answer_delta",
                    run_id=run_id,
                    data=delta,
                    generation_round=current_generation_round
                )
            )

        def run_agent() -> None:
 
            try:
                agentic_rag.run(
                    query_input=checkpoint.query_input,
                    run_id=run_id,
                    user_id=user_id,
                    resume=True,
                    on_answer_delta=on_answer_delta,
                    on_generation_started=on_generation_started
                )

                answer_queue.put(
                    RAGStreamEvent(
                        type="run_completed",
                        run_id=run_id
                    )
                )

            except Exception as e:
                try:
                    self.checkpoint_store.update_status(run_id, RunStatus.FAILED, str(e))
                except KeyError:
                    pass

                answer_queue.put(
                    RAGStreamEvent(
                        type="error",
                        run_id=run_id,
                        data=str(e)
                    )
                )

            finally:
                answer_queue.put(None)

        answer_queue.put(
            RAGStreamEvent(
                type="run_resumed",
                run_id=run_id
            )
        )

        worker = Thread(
            target=run_agent,
            daemon=True
        )

        worker.start()

        while True:
            item = answer_queue.get()

            if item is None:
                break

            yield item

    def get_run_status(self, run_id: str, user_id: str) -> RunStatus:
        checkpoint_store = self._checkpoint_store

        if checkpoint_store is None:
            raise RuntimeError("Checkpoint store is not initialized")

        return checkpoint_store.get_status(run_id, user_id)

    def list_runs(self, user_id: str) -> list[dict]:
        checkpoint_store = self._checkpoint_store

        if checkpoint_store is None:
            raise RuntimeError("Checkpoint store is not initialized")

        return checkpoint_store.list_runs(user_id)


if __name__ == "__main__":
    service = RAGService()
    service.initialize()

    run_id, answer = service.run(
        "How do I create a checkpoint?"
    )

    print(f"Run ID: {run_id}")
    print(answer)
