from dataclasses import dataclass

@dataclass(kw_only=True)
class QueryInput:
    """
    A real inference time request
    """
    question: str
    code_snippet: str | None = None
    error: str | None = None
    logs: str | None = None
    observed_behavior: str | None = None
    expected_behavior: str | None = None

@dataclass(kw_only=True)
class BenchmarkQuestion(QueryInput):
    """
    For Evaluation only. Adds reference retrievals. Used by evaluation using the BenchmarkRunner
    """
    id: int
    category: str
    intent: str
    difficulty: str
    expected_files: list[str]
    concepts: list[str]
    tags: list[str] | None = None
    evidence: list[dict] | None = None
    # The queries are LLM generated reference queries only made manually for evaluation purpose
    queries: list[str] | None = None

def to_query_input(question: BenchmarkQuestion) -> QueryInput:
    """
    Strips out the reference retrievals so they don't get sent to the LLM for inference
    """
    return QueryInput(
        question=question.question,
        code_snippet=question.code_snippet,
        error=question.error,
        logs=question.logs,
        observed_behavior=question.observed_behavior,
        expected_behavior=question.expected_behavior
    )