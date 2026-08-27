from evaluation.models import PipelineStage, RetrievedDocument, GenerationMetrics, RouterMetrics

class PipelineObserver():
    def __init__(self):
        self._pipeline_stages: list[PipelineStage] = []
        self._generation_metrics: list[GenerationMetrics] = []
        self._context = None
        self._router_metrics: RouterMetrics | None = None


    def record_stage(self, name: str, latency: float, top_k: int, documents: list[RetrievedDocument]) -> None:
        """
        Record the metrics of PipelineStage
        """
        # Create a new pipeline
        current_pipeline = PipelineStage(name=name, latency=latency, top_k=top_k, documents=documents)
        self._pipeline_stages.append(current_pipeline)

    def record_generation(self, name: str, round_number: int | None, answer: str, answer_length: int, prompt_tokens: int, completion_tokens: int, total_tokens: int, estimated_cost: float, generation_latency: float, gaps_detected: list[str] | None, is_complete: bool) -> None:
        """
        Record the metrics of Generation
        """

        current_generation_metrics = GenerationMetrics(
            name=name,
            round_number=round_number,
            answer=answer, 
            answer_length=answer_length,
            prompt_tokens=prompt_tokens, 
            completion_tokens=completion_tokens, 
            total_tokens=total_tokens, 
            estimated_cost=estimated_cost, 
            generation_latency=generation_latency,
            gaps_detected=gaps_detected,
            is_complete=is_complete
        )
        self._generation_metrics.append(current_generation_metrics)

    def record_router_metrics(self, name: str, prompt_tokens: int, completion_tokens: int, total_tokens: int, estimated_cost: float, router_latency: float) -> None:
        """
        Record the metrics of Query Router
        """
        self._router_metrics = RouterMetrics(
            name=name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
            router_latency=router_latency
        )

    def record_context(self, context: str) -> None:
        self._context = context

    def get_pipeline_stages(self) -> list[PipelineStage]:
        return self._pipeline_stages

    def get_generation_metrics(self) -> list[GenerationMetrics]:
        return self._generation_metrics

    def get_router_metrics(self) -> RouterMetrics:
        return self._router_metrics
    
    def get_context(self) -> str:
        return self._context

    def reset(self) -> None:
        """
        Reset the observer state before a new pipeline execution
        """
        self._pipeline_stages = []
        self._generation_metrics = []
        self._context = None
        self._router_metrics = None

    
        