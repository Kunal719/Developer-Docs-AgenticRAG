from models.query_input import QueryInput
from models.generation import GeneratedAnswer
from models.retrieval_plan import RetrievalPlan

from router.query_router import QueryRouter

from prompt.generation_prompt import generation_prompt

from reranking.base_reranker import BaseReranker

from retrieval.retrieval_orchestrator import retrieve_candidates, build_rerank_query, _dedupe
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.multi_corpus_retriever import MultiCorpusRetriever
from retrieval.multi_query_retrieval_strategy import MultiQueryRetrievalStrategy

from evaluation.observer import PipelineObserver

from utils.document_formatter import format_retrieved_context
from utils.helper import calculate_estimated_cost

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import (
    MAX_GENERATION_ROUNDS, 
    GAP_RERANK_TOP_K, 
    GENERATION_INPUT_PRICE, 
    GENERATION_OUTPUT_PRICE
)
from time import perf_counter

class AgenticRAG():
    def __init__(
        self, 
        query_input: QueryInput, 
        reranker: BaseReranker, 
        router_llm: BaseChatModel,
        generation_llm: BaseChatModel, 
        doc_hybrid_retriever: HybridRetriever,
        code_hybrid_retriever: HybridRetriever,
        multi_corpus_retriever: MultiCorpusRetriever,
        multi_query_strategy: MultiQueryRetrievalStrategy,
        code_vector_store: Chroma,
        observer: PipelineObserver | None = None):

        self._query_input = query_input
        self._reranker = reranker
        self._router_llm = router_llm
        self._generation_llm = generation_llm
        self._doc_hybrid_retriever = doc_hybrid_retriever
        self._code_hybrid_retriever = code_hybrid_retriever
        self._multi_corpus_retriever = multi_corpus_retriever
        self._multi_query_strategy = multi_query_strategy
        self._code_vector_store = code_vector_store
        self._observer = observer

    def _generate(self, llm: BaseChatModel, prompt: str, stage_name: str) -> GeneratedAnswer:

        start_time = perf_counter()
        response = llm.invoke(prompt)
        end_time = perf_counter()

        raw_response = response["raw"].usage_metadata
        generated_answer : GeneratedAnswer = response["parsed"]
        usage = raw_response.usage_metadata

        if self._observer is not None:
            self._observer.record_generation(
                name=stage_name,
                round_number=str(generated_answer.round_number),
                answer=generated_answer.answer,
                answer_length=len(generated_answer.answer),
                prompt_tokens=usage["input_tokens"],
                completion_tokens=usage["output_tokens"],
                total_tokens=usage["total_tokens"],
                estimated_cost=calculate_estimated_cost(
                    input_tokens=usage["input_tokens"],
                    output_tokens=usage["output_tokens"],
                    input_price_per_million=GENERATION_INPUT_PRICE,
                    output_price_per_million=GENERATION_OUTPUT_PRICE,
                ),
                generation_latency=(end_time - start_time),
                gaps_detected=[gap.missing_topic for gap in generated_answer.gaps],
                is_complete=generated_answer.is_complete
            )

        return generated_answer
    
    def run(self):
        # Get retrieval plan
        query_router = QueryRouter(self._router_llm)
        retrieval_plan = query_router.route(self._query_input)

        # Get retrieval candidates
        retrieval_candidates = retrieve_candidates(
                query_input=self._query_input,
                plan=retrieval_plan,
                doc_hybrid_retriever=self._doc_hybrid_retriever,
                code_hybrid_retriever=self._code_hybrid_retriever,
                multi_corpus_retriever=self._multi_corpus_retriever,
                multi_query_strategy=self._multi_query_strategy,
                code_vector_store=self._code_vector_store,
                observer=self._observer
        )

        # Rerank retrieval candidates
        reranker_query = build_rerank_query(self._query_input)
        reranked_candidates = self._reranker.rerank(
            query=reranker_query,
            documents=retrieval_candidates,
            observer=self._observer
        )

        all_candidates : list[Document] = reranked_candidates

        # Generate response
        generation_llm_prompt = generation_prompt.format(
            context=format_retrieved_context(reranked_candidates),
            question=reranker_query
        )

        # Get structured response from generation llm
        structured_generation_llm = self._generation_llm.with_structured_output(
            GeneratedAnswer, method="json_schema", include_raw=True
        ).bind(
            prompt_cache_key="generation_prompt",
            prompt_cache_options={"mode": "explicit", "ttl": "30m"}
        )

        generation_response = structured_generation_llm.invoke(
            generation_llm_prompt
        )

        generated_answer : GeneratedAnswer = generation_response["parsed"]

        # Check for gaps in answer

        print(f"Initial candidates: {len(all_candidates)}")

        for round_number in range(MAX_GENERATION_ROUNDS):
            print(f"\n========== GAP ROUND {round_number + 1} ==========")
            print(f"Processing {len(generated_answer.gaps)} gaps")

            if generated_answer.is_complete:
                print("Answer already complete. Stopping.")
                return generated_answer

            if not generated_answer.gaps:
                print("No gaps found. Stopping.")
                return generated_answer

            # Run the gap loop
            for i, gap in enumerate(generated_answer.gaps):

                print(f"\n--- Gap {i} ---")
                print("Topic:", gap.missing_topic)
                print("Query:", gap.suggested_query)
                print("Filters:", gap.suggested_filters)

                gap_retrieval_plan = RetrievalPlan(
                    retrieval_mode="direct",
                    reasoning="Gap retrieval loop",
                    metadata_filters=gap.suggested_filters
                )

                gap_retrieval_candidates = retrieve_candidates(
                    query_input=None,
                    plan=gap_retrieval_plan,
                    doc_hybrid_retriever=self._doc_hybrid_retriever,
                    code_hybrid_retriever=self._code_hybrid_retriever,
                    multi_corpus_retriever=self._multi_corpus_retriever,
                    multi_query_strategy=self._multi_query_strategy,
                    code_vector_store=self._code_vector_store,
                    query_override=gap.suggested_query,
                    observer=self._observer
                )

                reranked_gap_candidates = self._reranker.rerank(
                    query=gap.suggested_query,
                    documents=gap_retrieval_candidates,
                    observer=self._observer,
                    top_k=GAP_RERANK_TOP_K
                )

                print(f"Gap {i} retrieved {len(reranked_gap_candidates)} documents")

                all_candidates.extend(reranked_gap_candidates)

            print(f"Candidates before dedupe (gap round {round_number}): {len(all_candidates)}")
            # Remove duplicates
            all_candidates = _dedupe(all_candidates)
            print(f"Combined candidates after gap round {round_number}: {len(all_candidates)}")

            # Format generation prompt for gap
            gap_generation_llm_prompt = generation_prompt.format(
                context=format_retrieved_context(all_candidates),
                question=reranker_query
            )

            # Get structured response from generation llm
            gap_generation_response = structured_generation_llm.invoke(
                gap_generation_llm_prompt
            )

            generated_answer = gap_generation_response["parsed"]
            print("\n========== GAP ROUND RESULT ==========")
            print("Is complete:", generated_answer.is_complete)
            print("Remaining gaps:", len(generated_answer.gaps))

        return generated_answer
    