from langchain_core.language_models.chat_models import BaseChatModel
from langchain_chroma import Chroma

from evaluation.agentic_models import AgenticEvaluationResult
from evaluation.models import OverallMetrics, ExperimentMetadata
from evaluation.observer import PipelineObserver
from evaluation.benchmark_loader import load_questions
from evaluation.retrieval_evaluator import RetrievalEvaluator
from evaluation.judge import Judge

from retrieval.hybrid_retriever import HybridRetriever
from retrieval.multi_corpus_retriever import MultiCorpusRetriever
from retrieval.multi_query_retrieval_strategy import MultiQueryRetrievalStrategy

from router.query_router import QueryRouter

from reranking.base_reranker import BaseReranker

from models.query_input import to_query_input

from pipeline.agentic_loop import AgenticRAG

from time import perf_counter
from pathlib import Path
from datetime import datetime

from config import (
    EMBEDDING_MODEL,
    MODEL_NAME,
    MODEL_PROVIDER,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DENSE_TOP_K,
    BM25_TOP_K,
    CODE_HYBRID_TOP_K,
    RRF_K,
    RERANK_TOP_K,
    RERANKER_MODEL,
    PROJECT_VERSION,
    JUDGE_MODEL_NAME,
    JUDGE_MODEL_PROVIDER,
)

class AgenticBenchmarkRunner:
    def __init__(
            self, 
            questions_path: Path, 
            router_llm: BaseChatModel, 
            generation_llm: BaseChatModel,
            judge_llm: BaseChatModel,
            doc_hybrid_retriever: HybridRetriever, 
            code_hybrid_retriever: HybridRetriever,
            multi_corpus_retriever: MultiCorpusRetriever,
            multi_query_strategy: MultiQueryRetrievalStrategy,
            code_vector_store: Chroma,
            reranker: BaseReranker
            ):
        self._questions_path = questions_path
        self._router_llm = router_llm
        self._generation_llm = generation_llm
        self._judge_llm = judge_llm
        self._doc_hybrid_retriever = doc_hybrid_retriever
        self._code_hybrid_retriever = code_hybrid_retriever
        self._multi_corpus_retriever = multi_corpus_retriever
        self._multi_query_strategy = multi_query_strategy
        self._code_vector_store = code_vector_store
        self._reranker = reranker

    def run(self) -> list[AgenticEvaluationResult]:

        # Load the questions
        questions = load_questions(self._questions_path)

        # Initiate the observer and retrieval evaluator
        observer = PipelineObserver()
        retrieval_evaluator = RetrievalEvaluator()

        # results list
        results: list[AgenticEvaluationResult] = []

        # Initiate the agentic RAG
        agentic_rag = AgenticRAG(
            query_router=QueryRouter(self._router_llm),
            generation_llm=self._generation_llm,
            doc_hybrid_retriever=self._doc_hybrid_retriever,
            code_hybrid_retriever=self._code_hybrid_retriever,
            multi_corpus_retriever=self._multi_corpus_retriever,
            multi_query_strategy=self._multi_query_strategy,
            reranker=self._reranker,
            code_vector_store=self._code_vector_store
        )

        # Inititate the judge
        judge = Judge(self._judge_llm)

        # Evaluate for each question
        for question in questions:
            # Reset observer for each question
            observer.reset()

            # Convert question to QueryInput
            query_input = to_query_input(question)

            # Run the Agentic RAG
            start_time = perf_counter()
            generated_answer = agentic_rag.run(
                query_input=query_input,
                observer=observer
            )
            end_time = perf_counter()

            # Evaluate the retrieved context
            retrieval_metrics = retrieval_evaluator.evaluate(
                question=question,
                stages=observer.get_pipeline_stages()
            )

            retrieval_plan = observer.get_retrieval_plan()

            # Evaluate the generated response

            judge_evaluation, judge_cost = judge.evaluate(
                question=question,
                context=observer.get_context(),
                answer=generated_answer.answer
            )

            # Total cost calculation
            all_generation_metrics = observer.get_generation_metrics()
            total_generation_cost = sum([metric.estimated_cost for metric in all_generation_metrics])
            router_metrics_cost = observer.get_router_metrics().estimated_cost

            overall_metrics = OverallMetrics(
                total_latency = (end_time - start_time),
                total_cost=(router_metrics_cost + total_generation_cost + judge_cost["estimated_cost"]),
                recall_at_k=retrieval_metrics["recall_at_k"],
                precision_at_k=retrieval_metrics["precision_at_k"],
                hit_rate=retrieval_metrics["hit_rate"],
                mrr=retrieval_metrics["mrr"]
            )

            experiment_metadata = ExperimentMetadata(
                version=PROJECT_VERSION,
                timestamp=datetime.now(),
                embedding_model=EMBEDDING_MODEL,
                llm=MODEL_NAME,
                llm_provider=MODEL_PROVIDER,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                dense_top_k=DENSE_TOP_K,
                bm25_top_k=BM25_TOP_K,
                hybrid_top_k=CODE_HYBRID_TOP_K,
                rrf_k=RRF_K,
                reranker_top_k=RERANK_TOP_K,
                reranker_name=RERANKER_MODEL,
                judge_model=JUDGE_MODEL_NAME,
                judge_llm_provider=JUDGE_MODEL_PROVIDER,
                router_llm=MODEL_NAME,
                router_llm_provider=MODEL_PROVIDER
            )


            results.append(
                AgenticEvaluationResult(
                    question=question,
                    final_answer=generated_answer.answer,
                    experiment_metadata=experiment_metadata,
                    router_metrics=observer.get_router_metrics(),
                    retrieval_plan=retrieval_plan,
                    pipeline_stages=observer.get_pipeline_stages(),
                    generation_metrics=observer.get_generation_metrics(),
                    judge_evaluation=judge_evaluation,
                    overall_metrics=overall_metrics
                )
            )

        return results


