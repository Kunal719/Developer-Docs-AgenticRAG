from retrieval.rag_chain import build_rag_chain, invoke_chain
from retrieval.rag_chain_v3 import build_hybrid_rag_chain
from retrieval.base import BaseRetriever
from prompt.rag_prompt import rag_prompt
from reranking.base_reranker import BaseReranker
from evaluation.observer import PipelineObserver
from langchain_core.language_models.chat_models import BaseChatModel

from evaluation.benchmark_loader import load_questions
from evaluation.models import EvaluationResult, ExperimentMetadata, OverallMetrics
from evaluation.retrieval_evaluator import RetrievalEvaluator
from evaluation.judge import Judge

from utils.helper import build_evaluation_query

from time import perf_counter

from config import (
    EMBEDDING_MODEL, 
    MODEL_NAME,
    MODEL_PROVIDER,
    CHUNK_OVERLAP, 
    CHUNK_SIZE, 
    DENSE_TOP_K, 
    BM25_TOP_K, 
    HYBRID_TOP_K, 
    RRF_K, 
    RERANK_TOP_K,
    RERANKER_MODEL,
    PROJECT_VERSION,
    JUDGE_MODEL_NAME,
    JUDGE_MODEL_PROVIDER,
    )

from pathlib import Path
from datetime import datetime

class BenchmarkRunner():
    def __init__(self, retriever: BaseRetriever, llm: BaseChatModel, judge_llm: BaseChatModel, questions_path: Path, reranker: BaseReranker | None = None):
        self._retriever = retriever
        self._reranker = reranker
        self._llm = llm
        self._judge_llm = judge_llm
        self._questions_path = questions_path

    def run(self) -> list[EvaluationResult]:
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
            hybrid_top_k=HYBRID_TOP_K,
            rrf_k=RRF_K,
            reranker_top_k=RERANK_TOP_K if self._reranker is not None else None,
            reranker_name=RERANKER_MODEL if self._reranker is not None else None,
            judge_model=JUDGE_MODEL_NAME if self._judge_llm is not None else None,
            judge_provider=JUDGE_MODEL_PROVIDER if self._judge_llm is not None else None
        )

        questions = load_questions(self._questions_path)

        observer = PipelineObserver()
        if self._reranker is None:
            chain = build_hybrid_rag_chain(self._retriever, self._llm, rag_prompt, observer=observer)
        else:
            chain = build_rag_chain(self._retriever, self._reranker, self._llm, rag_prompt, observer=observer)

        results : list[EvaluationResult] = []
        retrieval_evaluator = RetrievalEvaluator()

        judge = Judge(self._judge_llm)

        for question in questions:
            observer.reset()
            evaluation_query = build_evaluation_query(question)

            start = perf_counter()
            answer = invoke_chain(chain, evaluation_query, observer)
            end = perf_counter()

            generation_metrics = observer.get_generation_metrics()
            pipeline_stages = observer.get_pipeline_stages()

            retrieval_metrics = retrieval_evaluator.evaluate(question, pipeline_stages)

            judge_evaluation, judge_cost = judge.evaluate(question, observer.get_context(), generation_metrics.answer)

            overall_metrics = OverallMetrics(
                total_latency=(end - start),
                total_cost=generation_metrics.estimated_cost + judge_cost["estimated_cost"],
                recall_at_k=retrieval_metrics["recall_at_k"],
                precision_at_k=retrieval_metrics["precision_at_k"],
                hit_rate=retrieval_metrics["hit_rate"],
                mrr=retrieval_metrics["mrr"]
            )

            results.append(EvaluationResult(
                question=question,
                pipeline_stages=observer.get_pipeline_stages(),
                generation_metrics=generation_metrics,
                overall_metrics=overall_metrics,
                experiment_metadata=experiment_metadata,
                judge_evaluation=judge_evaluation
            ))

        return results

        