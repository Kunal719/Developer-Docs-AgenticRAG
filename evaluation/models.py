from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from typing import Literal

from evaluation.judge_models import JudgeEvaluation
from models.query_input import BenchmarkQuestion

# @dataclass
# class BenchmarkQuestion:
#     id: int
#     category: str
#     intent: str
#     difficulty: str
#     question: str
#     expected_files: list[str]
#     concepts: list[str]
#     tags: Optional[list[str]] = None
#     code_snippet: str | None = None
#     observed_behavior: str | None = None
#     expected_behavior: str | None = None
#     error: str | None = None
#     logs: str | None = None
#     evidence: list[dict] | None = None
#     queries: list[str] | None = None

@dataclass
class RetrievedDocument:
    chunk_id: str
    source: str
    document_type: Literal["documentation", "implementation"]
    retriever_score: Optional[float] = None
    reranker_score: Optional[float] = None
    rank: Optional[int] = None

@dataclass
class PipelineStage:
    name: str
    latency: float
    top_k: int
    documents: list[RetrievedDocument]

@dataclass
class ExperimentMetadata:
    version: str
    timestamp: datetime
    embedding_model: str
    llm_provider: str
    llm: str
    chunk_size: int
    chunk_overlap: int
    dense_top_k: int
    bm25_top_k: int
    hybrid_top_k: int
    rrf_k: int
    reranker_top_k: Optional[int] = None
    reranker_name: Optional[str] = None
    judge_model: Optional[str] = None
    judge_llm_provider: Optional[str] = None
    router_llm: Optional[str] = None
    router_llm_provider: Optional[str] = None

@dataclass
class GenerationMetrics:
    name: str
    answer: str
    answer_length: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float
    generation_latency: float
    is_complete: bool 
    round_number: int | None = None
    gaps_detected: list[str] | None = None                       

@dataclass
class OverallMetrics:
    total_latency: float
    total_cost: float
    recall_at_k: Optional[float] = None
    precision_at_k: Optional[float] = None
    hit_rate: Optional[float] = None
    mrr: Optional[float] = None

@dataclass
class RouterMetrics:
    name: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float
    router_latency: float

@dataclass
class EvaluationResult:
    question: BenchmarkQuestion
    experiment_metadata: ExperimentMetadata
    pipeline_stages: list[PipelineStage]
    overall_metrics: OverallMetrics
    generation_metrics: GenerationMetrics | None
    judge_evaluation: JudgeEvaluation | None