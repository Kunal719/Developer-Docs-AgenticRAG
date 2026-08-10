from evaluation.models import PipelineStage, RetrievedDocument, GenerationMetrics

class PipelineObserver():
    def __init__(self):
        self._pipeline_stages: list[PipelineStage] = []
        self._generation_metrics: GenerationMetrics | None = None
        self._context = None

    def record_stage(self, name: str, latency: float, top_k: int, documents: list[RetrievedDocument]) -> None:
        """
        Record the metrics of PipelineStage
        """
        # Create a new pipeline
        current_pipeline = PipelineStage(name=name, latency=latency, top_k=top_k, documents=documents)
        self._pipeline_stages.append(current_pipeline)

    def record_generation(self, answer: str, answer_length: int, prompt_tokens: int, completion_tokens: int, total_tokens: int, estimated_cost: float, generation_latency: float) -> None:
        """
        Record the metrics of Generation
        """
        self._generation_metrics = GenerationMetrics(
            answer=answer, 
            answer_length=answer_length,
            prompt_tokens=prompt_tokens, 
            completion_tokens=completion_tokens, 
            total_tokens=total_tokens, 
            estimated_cost=estimated_cost, 
            generation_latency=generation_latency)

    def record_context(self, context: str) -> None:
        self._context = context

    def get_pipeline_stages(self) -> list[PipelineStage]:
        return self._pipeline_stages

    def get_generation_metrics(self) -> GenerationMetrics:
        return self._generation_metrics

    def get_context(self) -> str:
        return self._context

    def reset(self) -> None:
        """
        Reset the observer state before a new pipeline execution
        """
        self._pipeline_stages = []
        self._generation_metrics = None
        self._context = None

    
        