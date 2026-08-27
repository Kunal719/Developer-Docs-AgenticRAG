from models.query_input import QueryInput
from models.retrieval_plan import RetrievalPlan

from evaluation.observer import PipelineObserver

from prompt.router_prompt import router_prompt

from langchain_core.language_models.chat_models import BaseChatModel

from utils.helper import calculate_estimated_cost
from config import ROUTER_INPUT_PRICE, ROUTER_OUTPUT_PRICE

from time import perf_counter

class QueryRouter:
    def __init__(self, llm: BaseChatModel):
        self._llm = llm.with_structured_output(
            RetrievalPlan, method="json_schema", include_raw=True
            ).bind(
                prompt_cache_key="query_router",
                prompt_cache_options={"mode": "explicit", "ttl": "30m"}
            )

    @staticmethod
    def _optional_router_prompt_block(label: str, value: str | None) -> str:
        if not value:
            return ""
        return f"{label}: {value}\n"
    
    def route(self, query: QueryInput, observer: PipelineObserver | None = None) -> RetrievalPlan:

        router_prompt_formatted = router_prompt.format(
            question=query.question,
            code_snippet=self._optional_router_prompt_block("Code", query.code_snippet),
            error=self._optional_router_prompt_block("Error", query.error),
            observed_behavior=self._optional_router_prompt_block("Observed behavior", query.observed_behavior),
            expected_behavior=self._optional_router_prompt_block("Expected behavior", query.expected_behavior),
            logs=self._optional_router_prompt_block("Logs", query.logs)
        )

        start_time = perf_counter()
        response = self._llm.invoke(router_prompt_formatted)
        end_time = perf_counter()

        plan = response["parsed"]
        raw_response = response["raw"]
        parsing_error = response["parsing_error"]

        if plan is None:
            print("\n========== ROUTER ERROR ==========")
            print(f"Question: {query.question}")

            print("\nParsing error:")
            print(parsing_error)

            print("\nRaw response:")
            print(raw_response)

            print("\nRaw content:")
            print(raw_response.content)

            print("===================================\n")

            raise RuntimeError(
                f"Router failed to produce a valid RetrievalPlan "
                f"for question: {query.question}\n"
                f"Parsing error: {parsing_error}"
            )

        usage = raw_response.usage_metadata

        router_cost = calculate_estimated_cost(
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            input_price_per_million=ROUTER_INPUT_PRICE,
            output_price_per_million=ROUTER_OUTPUT_PRICE,
        )

        if observer is not None:
            observer.record_router_metrics(
                name="Query Router",
                prompt_tokens=usage["input_tokens"],
                completion_tokens=usage["output_tokens"],
                total_tokens=usage["total_tokens"],
                estimated_cost=router_cost,
                router_latency=(end_time - start_time)
            )

        return plan