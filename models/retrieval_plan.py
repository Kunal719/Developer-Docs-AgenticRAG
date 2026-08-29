from typing import Literal
from pydantic import BaseModel, Field, model_validator

class MetadataFilters(BaseModel):
    """
    Filters explored by the router from the question

    Applied in every retrieval mode:
     - Dense retriever: `where` clause pre-filter.
     - BM25 retriever: post-filter after retrieval (no native filter
       support in a plain token index).
     - If function_name or class_name is set, the exact match retriever
       (retrieval/metadata_retriever.py) also fires once and its result
       is a guaranteed include in the candidate pool, not rank-competed.
    """

    corpus: Literal["implementation", "documentation"] | None = None
    module_names: list[str] | None = Field(default_factory=list)
    class_names: list[str] | None = Field(default_factory=list)
    function_names: list[str] | None = Field(default_factory=list)
    category: Literal["general", "errors", "frontend"] | None = None

    @model_validator(mode="after")
    def _category_requires_doc_corpus(self) -> "MetadataFilters":
        if self.category and self.corpus != "documentation":
            raise ValueError(
                "category requires corpus='documentation'"
            )

        return self

class RetrievalPlan(BaseModel):
    """
    Router's decision for how to retrieve for a given QueryInput
    """

    retrieval_mode: Literal["direct", "combined_query", "multi_query_fusion"]
    reasoning: str = Field(
        description="Short internal/debug-only justification for the" \
        " mode and filters chosen. Not shown to the end user."
    )
    sub_queries: list[str] | None = Field(
        default=None,
        description="Required if retrieval_mode in ('combined_query', "
        "'multi_query_fusion'). 2-6 distinct sub-questions targeting "
        "different information needs. In 'combined_query' mode these are "
        "merged with question/code_snippet/other key terms into ONE retrieval "
        "string (see combined_query field). In 'multi_query_fusion' mode "
        "each is retrieved separately, then RRF-fused.",
    )
    combined_query: str | None = Field(
        default=None,
        description="Required if retrieval_mode == 'combined_query'. The "
        "single merged string — original question + sub_queries + "
        "code_snippet + key terms — combined into one retrieval call.",
    )
    metadata_filters: MetadataFilters | None = None

    @model_validator(mode="after")
    def _check_mode_fields(self) -> "RetrievalPlan":
        if self.retrieval_mode == "combined_query":
            if not self.combined_query:
                raise ValueError(
                    "combined_query must be set when retrieval_mode == 'combined_query'"
                )
            if not self.sub_queries or not (2 <= len(self.sub_queries) <= 6):
                raise ValueError(
                    "sub_queries must contain 2-6 items when "
                    "retrieval_mode == 'combined_query' (used to build combined_query)"
                )
        if self.retrieval_mode == "multi_query_fusion":
            if not self.sub_queries or not (2 <= len(self.sub_queries) <= 6):
                raise ValueError(
                    "sub_queries must contain 2-6 items when "
                    "retrieval_mode == 'multi_query_fusion'"
                )
        return self