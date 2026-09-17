from langchain_core.language_models.chat_models import BaseChatModel

from models.query_input import QueryInput
from prompt.normalize_question_prompt import input_normalizer_prompt


class InputNormalizer:
    def __init__(self, llm: BaseChatModel):
        self._llm = llm.with_structured_output(
            QueryInput, method="json_schema", include_raw=True
            ).bind(
                prompt_cache_key="input_normalizer_prompt", 
                prompt_cache_options={"mode": "explicit", "ttl": "30m"}
            )

    def normalize(self, raw_input: str) -> QueryInput:
        
        prompt = input_normalizer_prompt.format(
            question=raw_input
        )

        result = self._llm.invoke(prompt)

        return QueryInput(**result["parsed"])