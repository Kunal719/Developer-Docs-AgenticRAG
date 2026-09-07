from models.query_input import QueryInput
from models.generation import GeneratedAnswer, RetrievalGap
from models.retrieval_plan import RetrievalPlan
from models.rag_state import RAGState, RAGStage
from models.evidence import EvidenceItem

from router.query_router import QueryRouter

from prompt.generation_prompt import generation_prompt

from reranking.base_reranker import BaseReranker

from retrieval.retrieval_orchestrator import retrieve_candidates, build_rerank_query, _dedupe
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.multi_corpus_retriever import MultiCorpusRetriever
from retrieval.multi_query_retrieval_strategy import MultiQueryRetrievalStrategy

from evidence.base_selector import EvidenceSelector

from evaluation.observer import PipelineObserver

from checkpoint.store import CheckpointStore
from checkpoint.serialization import create_checkpoint

from utils.document_formatter import format_retrieved_context
from utils.helper import calculate_estimated_cost

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_chroma import Chroma
from langchain_core.documents import Document

from time import perf_counter
from uuid import uuid4

from config import (
    MAX_GENERATION_ROUNDS, 
    GAP_RERANK_TOP_K, 
    GENERATION_INPUT_PRICE, 
    GENERATION_OUTPUT_PRICE,
    MAX_GAPS_PER_ROUND,
    RERANK_TOP_K
)

from concurrent.futures import ThreadPoolExecutor, as_completed
from time import perf_counter

class AgenticRAG():
    def __init__(
        self,
        query_router: QueryRouter,
        generation_llm: BaseChatModel, 
        doc_hybrid_retriever: HybridRetriever,
        code_hybrid_retriever: HybridRetriever,
        multi_corpus_retriever: MultiCorpusRetriever,
        multi_query_strategy: MultiQueryRetrievalStrategy,
        reranker: BaseReranker,
        code_vector_store: Chroma,
        evidence_selector: EvidenceSelector,
        checkpoint_store: CheckpointStore):

        self._query_router = query_router
        self._doc_hybrid_retriever = doc_hybrid_retriever
        self._code_hybrid_retriever = code_hybrid_retriever
        self._multi_corpus_retriever = multi_corpus_retriever
        self._multi_query_strategy = multi_query_strategy
        self._reranker = reranker
        self._code_vector_store = code_vector_store
        self._evidence_selector = evidence_selector
        self._checkpoint_store = checkpoint_store

        self._generation_llm = generation_llm.with_structured_output(
            GeneratedAnswer,
            method="json_schema",
            include_raw=True
        ).bind(
            prompt_cache_key="agentic_rag_generation",
            prompt_cache_options={"mode": "explicit", "ttl": "30m"}
        )

    def _select_gaps(self, gaps: list[RetrievalGap]) -> list[RetrievalGap]:
        priority = {
            "critical": 1,
            "important": 2,
            "optional": 3
        }

        actionable_gaps = [gap for gap in gaps if gap.priority in ("critical", "important")]

        actionable_gaps.sort(key=lambda gap: priority[gap.priority])

        return actionable_gaps[:MAX_GAPS_PER_ROUND]

        
    def _generate(
            self, 
            llm: BaseChatModel, 
            prompt: str, 
            stage_name: str, 
            observer: PipelineObserver, 
            round_number: int | None = None, 
    ) -> GeneratedAnswer:
        
        start_time = perf_counter()
        response = llm.invoke(prompt)
        end_time = perf_counter()

        raw_response = response["raw"]
        generated_answer : GeneratedAnswer = response["parsed"]
        usage = raw_response.usage_metadata

        if observer is not None:
            observer.record_generation(
                name=stage_name,
                round_number=round_number,
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

    def _process_gap(self, gap: RetrievalGap) -> tuple[list[Document], PipelineObserver]:
        # A local observer for each gap which will merge with the global observer
        gap_observer = PipelineObserver()

        gap_start = perf_counter()

        # Make retrieval plan
        gap_retrieval_plan = RetrievalPlan(
            retrieval_mode="direct",
            reasoning="Gap retrieval loop",
            metadata_filters=gap.suggested_filters
        )

        retrieval_start = perf_counter()
        # Fetch retrieval candidates
        gap_retrieval_candidates = retrieve_candidates(
            query_input=None,
            plan=gap_retrieval_plan,
            doc_hybrid_retriever=self._doc_hybrid_retriever,
            code_hybrid_retriever=self._code_hybrid_retriever,
            multi_corpus_retriever=self._multi_corpus_retriever,
            multi_query_strategy=self._multi_query_strategy,
            code_vector_store=self._code_vector_store,
            query_override=gap.suggested_query,
            observer=gap_observer
        )

        retrieval_end = perf_counter()

        # Rerank
        rerank_start = perf_counter()
        reranked_gap_candidates = self._reranker.rerank(
            query=gap.suggested_query,
            documents=gap_retrieval_candidates,
            observer=gap_observer,
            top_k=GAP_RERANK_TOP_K
        )
        rerank_end = perf_counter()
        gap_end = perf_counter()

        reranked_gap_candidates = [
            EvidenceItem(
                document=document,
                source="gap_rerank",
                query=gap.suggested_query,
                gap=gap
            )
            for document in reranked_gap_candidates
        ]


        print(
        f"[GAP] {gap.missing_topic}\n"
        f"  total:     {gap_end - gap_start:.3f}s\n"
        f"  retrieval: {retrieval_end - retrieval_start:.3f}s\n"
        f"  rerank:    {rerank_end - rerank_start:.3f}s"
        )

        return reranked_gap_candidates, gap_observer
    
    def run(self, query_input: QueryInput, observer: PipelineObserver | None = None) -> GeneratedAnswer:
        # Initiate run_id
        run_id = uuid4().hex

        # Initialize RAG state
        state = RAGState(query_input=query_input)

        # Get retrieval plan
        state.retrieval_plan = self._query_router.route(query=query_input, observer=observer)
        state.stage = RAGStage.ROUTED

        # Save checkpoint after ROUTED
        self._checkpoint_store.save(run_id, create_checkpoint(state))

        if observer is not None:
            observer.record_retrieval_plan(state.retrieval_plan)

        # Get retrieval candidates
        retrieval_candidates = retrieve_candidates(
                query_input=state.query_input,
                plan=state.retrieval_plan,
                doc_hybrid_retriever=self._doc_hybrid_retriever,
                code_hybrid_retriever=self._code_hybrid_retriever,
                multi_corpus_retriever=self._multi_corpus_retriever,
                multi_query_strategy=self._multi_query_strategy,
                code_vector_store=self._code_vector_store,
                observer=observer
        )

        # Rerank retrieval candidates
        state.reranker_query = build_rerank_query(state.query_input)
        initial_evidence = self._reranker.rerank(
            query=state.reranker_query,
            documents=retrieval_candidates,
            observer=observer
        )

        state.initial_evidence = [
            EvidenceItem(
                document=doc,
                source="initial",
                query=state.reranker_query,
                gap=None
            )
            for doc in initial_evidence
        ]

        state.evidence_pool = list(state.initial_evidence)
        state.stage = RAGStage.INITIAL_EVIDENCE_READY

        # Save checkpoint after INITIAL_EVIDENCE_READY
        self._checkpoint_store.save(run_id, create_checkpoint(state))

        # Format retrieved context and store it
        formatted_context = format_retrieved_context(
            [evidence_item.document for evidence_item in state.evidence_pool]
        )
        if state.evidence_pool and observer is not None:
            observer.record_context(formatted_context)

        # Generate response
        generation_llm_prompt = generation_prompt.format(
            context=formatted_context,
            question=state.reranker_query,
            previous_answer="No previous draft. This is the initial answer."
        )

        state.generated_answer = self._generate(
            llm=self._generation_llm,
            prompt=generation_llm_prompt,
            stage_name="Initial Generation",
            observer=observer
        )
        state.stage = RAGStage.DRAFT_GENERATED

        # Save checkpoint after DRAFT_GENERATED
        self._checkpoint_store.save(run_id, create_checkpoint(state))

        if state.generated_answer.is_complete:
            state.stage = RAGStage.COMPLETED
            # Delete checkpoint after COMPLETED
            self._checkpoint_store.delete(run_id)
            return state.generated_answer

        # Check for gaps in answer

        print(f"Initial candidates: {len(state.evidence_pool)}")

        while state.generation_round < MAX_GENERATION_ROUNDS:
            print(f"\n========== GAP ROUND {state.generation_round + 1} ==========")

            if state.generated_answer.is_complete:
                state.stage = RAGStage.COMPLETED

                # Delete checkpoint after COMPLETED
                self._checkpoint_store.delete(run_id)
                print("Answer already complete. Stopping.")
                return state.generated_answer
            
            state.gaps = self._select_gaps(state.generated_answer.gaps)
            print(f"Selected {len(state.gaps)} gaps for retrieval")

            if not state.gaps:
                print("No gaps found. Stopping.")
                state.stage = RAGStage.COMPLETED

                # Delete checkpoint after COMPLETED
                self._checkpoint_store.delete(run_id)
                return state.generated_answer

            # Run the gap loop parallely
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(self._process_gap, gap) for gap in state.gaps]
                gap_results = [f.result() for f in futures]

            for i, (gap_candidates, gap_observer) in enumerate(gap_results, start=1):
                print(f"Gap {i} retrieved {len(gap_candidates)} documents")
                if observer is not None:
                    observer.get_pipeline_stages().extend(gap_observer.get_pipeline_stages())

                state.evidence_pool.extend(gap_candidates)

            state.stage = RAGStage.GAPS_PROCESSED

            # Save checkpoint after GAPS_PROCESSED
            self._checkpoint_store.save(run_id, create_checkpoint(state))

            # Generate final context
            state.final_evidence = self._evidence_selector.select_evidence(
                query=state.reranker_query,
                evidence=state.evidence_pool,
                top_k=RERANK_TOP_K,
                gaps=state.gaps,
                observer=observer
            )

            formatted_context = format_retrieved_context(
                [evidence_item.document for evidence_item in state.final_evidence]
            )
            if state.final_evidence and observer is not None:
                observer.record_context(formatted_context)

            state.stage = RAGStage.FINAL_EVIDENCE_READY

            # Save checkpoint after FINAL_EVIDENCE_READY
            self._checkpoint_store.save(run_id, create_checkpoint(state))

            # Format generation prompt for gap
            gap_generation_llm_prompt = generation_prompt.format(
                context=formatted_context,
                question=state.reranker_query,
                previous_answer=state.generated_answer.answer
            )

            # Get structured response from generation llm
            state.generated_answer = self._generate(
                llm=self._generation_llm,
                prompt=gap_generation_llm_prompt,
                stage_name=f"Gap Generation Round {state.generation_round + 1}",
                round_number=state.generation_round + 1,
                observer=observer
            )
            state.generation_round += 1

            print("\n========== GAP ROUND RESULT ==========")
            print("Is complete:", state.generated_answer.is_complete)
            print("Remaining gaps:", len(state.generated_answer.gaps))

        # print("\n========== GENERATION PROMPT ==========")
        # print(gap_generation_llm_prompt if gap_generation_llm_prompt else generation_llm_prompt)
        state.stage = RAGStage.COMPLETED

        # Delete checkpoint after COMPLETED
        self._checkpoint_store.delete(run_id)
        return state.generated_answer
    