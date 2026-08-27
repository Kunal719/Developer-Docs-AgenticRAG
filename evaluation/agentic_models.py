from dataclasses import dataclass
from evaluation.models import (
    ExperimentMetadata,
    PipelineStage,
    GenerationMetrics,
    OverallMetrics,
    JudgeEvaluation,
    RouterMetrics
)
from models.query_input import BenchmarkQuestion


@dataclass
class AgenticEvaluationResult:
    question: BenchmarkQuestion
    final_answer: str
    experiment_metadata: ExperimentMetadata
    router_metrics: RouterMetrics | None
    pipeline_stages: list[PipelineStage]
    generation_metrics: list[GenerationMetrics]
    judge_evaluation: JudgeEvaluation | None
    overall_metrics: OverallMetrics