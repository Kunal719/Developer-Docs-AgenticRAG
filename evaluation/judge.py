from evaluation.judge_models import JudgeEvaluation
from models.query_input import BenchmarkQuestion
from prompt.judge_prompt import judge_prompt
from utils.helper import calculate_estimated_cost

from langchain_core.language_models.chat_models import BaseChatModel

from config import JUDGE_INPUT_PRICE, JUDGE_OUTPUT_PRICE


class Judge:
    def __init__(self, llm: BaseChatModel):
        self._llm = llm.with_structured_output(
            JudgeEvaluation,
            method="json_schema",
            include_raw=True
        ).bind(
            prompt_cache_key="judge_prompt",
            prompt_cache_options={"mode": "explicit", "ttl": "30m"}
        )

    def evaluate(self, question: BenchmarkQuestion, context: str, answer: str) -> tuple[JudgeEvaluation, dict]:

        response = self._llm.invoke(judge_prompt.format(question=question.question, context=context, answer=answer, concepts=question.concepts, expected_files=question.expected_files))

        judge_evaluation = response["parsed"]
        raw_response = response["raw"]
        parsing_error = response["parsing_error"]

        # Never silently continue when structured output fails.
        if judge_evaluation is None:
            print("\n========== JUDGE ERROR ==========")
            print(f"Question: {question.question}")

            print("\nParsing error:")
            print(parsing_error)

            print("\nRaw response:")
            print(raw_response)

            print("\nRaw content:")
            print(raw_response.content)

            print("\nResponse metadata:")
            print(raw_response.response_metadata)

            print("\nUsage metadata:")
            print(raw_response.usage_metadata)

            print("=================================\n")

            raise RuntimeError(
                f"Judge failed to produce a valid JudgeEvaluation "
                f"for question: {question.question}\n"
                f"Parsing error: {parsing_error}"
            )

        usage = raw_response.usage_metadata

        judge_cost = calculate_estimated_cost(
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            input_price_per_million=JUDGE_INPUT_PRICE,
            output_price_per_million=JUDGE_OUTPUT_PRICE
        )

        judge_cost_metrics = {
            "input_tokens": usage["input_tokens"],
            "output_tokens": usage["output_tokens"],
            "total_tokens": usage["total_tokens"],
            "estimated_cost": judge_cost
        }

        return judge_evaluation, judge_cost_metrics