from evaluation.models import BenchmarkQuestion

def calculate_estimated_cost(
    input_tokens: int,
    output_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float,
) -> float:
    """
    Calculate the estimated API cost in USD.

    Args:
        input_tokens: Number of input (prompt) tokens.
        output_tokens: Number of output (completion) tokens.
        input_price_per_million: USD cost per 1M input tokens.
        output_price_per_million: USD cost per 1M output tokens.

    Returns:
        Estimated cost in USD.
    """
    input_cost = (input_tokens / 1_000_000) * input_price_per_million
    output_cost = (output_tokens / 1_000_000) * output_price_per_million

    return round(input_cost + output_cost, 8)

def build_evaluation_query(question: BenchmarkQuestion, include_queries: bool = True) -> str:
    parts = [question.question]

    if include_queries and question.queries:
        parts.append("Queries:")
        for query in question.queries:
            parts.append(f"- {query}")
    
    if question.code_snippet:
        parts.append(f"Code:\n```python\n{question.code_snippet}\n```")

    if question.error:
        parts.append(f"Error:\n{question.error}")

    if question.observed_behavior:
        parts.append(f"Observed behavior:\n{question.observed_behavior}")

    if question.expected_behavior:
        parts.append(f"Expected behavior:\n{question.expected_behavior}")

    if question.logs:
        parts.append(f"Logs:\n{question.logs}")

    if question.evidence:
        parts.append("Evidence:")
        for item in question.evidence:
            parts.append(
                f"EVIDENCE {item['id']} — {item['title']}\n"
                f"{item['content']}"
            )
        
    return "\n\n".join(parts)