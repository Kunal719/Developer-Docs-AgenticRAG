from unittest.mock import MagicMock

from models.query_input import QueryInput
from pipeline.input_normalizer import InputNormalizer


def make_normalizer(parsed: QueryInput) -> InputNormalizer:
    llm = MagicMock()

    llm.with_structured_output.return_value.bind.return_value.invoke.return_value = {
        "parsed": vars(parsed),
        "raw": None,
        "parsing_error": None,
    }

    return InputNormalizer(llm)


def test_normalizes_simple_question():
    normalized = QueryInput(
        question="What does checkpointing do in LangGraph?"
    )

    normalizer = make_normalizer(normalized)

    result = normalizer.normalize(
        "What does checkpointing do in LangGraph?"
    )

    assert result == normalized
    assert result.question == "What does checkpointing do in LangGraph?"
    assert result.code_snippet is None
    assert result.error is None
    assert result.logs is None
    assert result.observed_behavior is None
    assert result.expected_behavior is None


def test_normalizes_question_with_observed_and_expected_behavior():
    normalized = QueryInput(
        question="Why is my graph not resuming after a restart?",
        observed_behavior="After restarting, the graph starts from the beginning.",
        expected_behavior="The graph should resume from the saved checkpoint.",
    )

    normalizer = make_normalizer(normalized)

    result = normalizer.normalize(
        "why my graph is not resuming after restart? "
        "I saved checkpoint but after restart it starts from beginning. "
        "The graph should resume from the saved checkpoint."
    )

    assert result.question == normalized.question
    assert result.observed_behavior == normalized.observed_behavior
    assert result.expected_behavior == normalized.expected_behavior

    assert result.code_snippet is None
    assert result.error is None
    assert result.logs is None


def test_normalizes_error():
    normalized = QueryInput(
        question="How do I resolve the KeyError: thread_id when invoking the graph?",
        error="KeyError: thread_id",
    )

    normalizer = make_normalizer(normalized)

    result = normalizer.normalize(
        "getting KeyError: thread_id when invoking graph"
    )

    assert result.question == normalized.question
    assert result.error == "KeyError: thread_id"

    assert result.code_snippet is None
    assert result.logs is None
    assert result.observed_behavior is None
    assert result.expected_behavior is None


def test_preserves_code_snippet():
    code = """result = graph.invoke(state)
print(result)"""

    normalized = QueryInput(
        question="Why does my graph return None instead of the generated answer?",
        code_snippet=code,
        observed_behavior="The graph returns None instead of the generated answer.",
        expected_behavior="The graph should return the generated answer.",
    )

    normalizer = make_normalizer(normalized)

    result = normalizer.normalize(
        """
        My graph returns None instead of the generated answer.
        Expected it to return the answer.

        Here is the code:

        result = graph.invoke(state)
        print(result)
        """
    )

    assert result.question == normalized.question
    assert result.code_snippet == code
    assert result.observed_behavior == normalized.observed_behavior
    assert result.expected_behavior == normalized.expected_behavior

    assert result.error is None
    assert result.logs is None


def test_preserves_logs():
    logs = """INFO Starting graph
ERROR KeyError: thread_id
INFO Graph execution stopped"""

    normalized = QueryInput(
        question="Why did the graph execution stop?",
        logs=logs,
        error="KeyError: thread_id",
    )

    normalizer = make_normalizer(normalized)

    result = normalizer.normalize(
        f"""
        Why did the graph execution stop?

        Logs:
        {logs}
        """
    )

    assert result.question == normalized.question
    assert result.logs == logs
    assert result.error == "KeyError: thread_id"

    assert result.code_snippet is None
    assert result.observed_behavior is None
    assert result.expected_behavior is None