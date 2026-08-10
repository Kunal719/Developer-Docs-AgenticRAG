from evaluation.judge_models import JudgeEvaluation
from evaluation.models import BenchmarkQuestion
from prompt.judge_prompt import judge_prompt
from utils.helper import calculate_estimated_cost

from config import JUDGE_INPUT_PRICE, JUDGE_OUTPUT_PRICE

class Judge:
    def __init__(self, llm):
        self._llm = llm.with_structured_output(JudgeEvaluation, include_raw=True)

    def evaluate(self, question: BenchmarkQuestion, context: str, answer: str) -> tuple[JudgeEvaluation, dict]:
        response = self._llm.invoke(judge_prompt.format(question=question.question, context=context, answer=answer, concepts=question.concepts, expected_files=question.expected_files))

        # JudgeEvaluation
        judge_evaluation = response["parsed"]

        # Raw response for cost calculation
        raw_response = response["raw"]

        # Calculate the cost of the API call here
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