from models.query_input import QueryInput
from models.generation import GeneratedAnswer, RetrievalGap
from models.retrieval_plan import RetrievalPlan
from models.rag_state import RAGState, RAGStage
from models.evidence import EvidenceItem
from models.streaming import AnswerStreamParser

from router.query_router import QueryRouter

from prompt.generation_prompt import generation_prompt

from reranking.base_reranker import BaseReranker

from retrieval.retrieval_orchestrator import retrieve_candidates, build_rerank_query
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.multi_corpus_retriever import MultiCorpusRetriever
from retrieval.multi_query_retrieval_strategy import MultiQueryRetrievalStrategy

from evidence.base_selector import EvidenceSelector

from evaluation.observer import PipelineObserver

from checkpoint.store import CheckpointStore
from checkpoint.serialization import create_checkpoint, restore_state
from checkpoint.resolver import EvidenceResolver
from checkpoint.models import RunStatus

from utils.document_formatter import format_retrieved_context
from utils.helper import calculate_estimated_cost
from utils.structured_generation_output import make_openai_strict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_chroma import Chroma
from langchain_core.documents import Document

from time import perf_counter
import threading
import json
from collections.abc import Callable
import logging

from config import (
    MAX_GENERATION_ROUNDS, 
    GAP_RERANK_TOP_K, 
    GENERATION_INPUT_PRICE, 
    GENERATION_OUTPUT_PRICE,
    MAX_GAPS_PER_ROUND,
    RERANK_TOP_K
)

from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

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
        checkpoint_store: CheckpointStore,
        evidence_resolver: EvidenceResolver):

        self._query_router = query_router
        self._doc_hybrid_retriever = doc_hybrid_retriever
        self._code_hybrid_retriever = code_hybrid_retriever
        self._multi_corpus_retriever = multi_corpus_retriever
        self._multi_query_strategy = multi_query_strategy
        self._reranker = reranker
        self._code_vector_store = code_vector_store
        self._evidence_selector = evidence_selector
        self._checkpoint_store = checkpoint_store
        self._evidence_resolver = evidence_resolver

        # self._generation_llm = generation_llm.with_structured_output(
        #     GeneratedAnswer,
        #     method="json_schema",
        #     include_raw=True
        # ).bind(
        #     prompt_cache_key="agentic_rag_generation",
        #     prompt_cache_options={"mode": "explicit", "ttl": "30m"}
        # )

        generation_schema = GeneratedAnswer.model_json_schema()
        make_openai_strict(generation_schema)

        self._generation_llm = generation_llm.bind(
            response_format={
                "type": "json_schema",
                "json_schema":{
                    "name": "generated_answer",
                    "schema": generation_schema,
                    "strict": True
                }
            },
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
            on_answer_delta: Callable[[str], None] | None = None,
            on_generation_started: Callable[[int], None] | None = None
    ) -> GeneratedAnswer:

        if on_generation_started is not None and round_number is not None:
            on_generation_started(round_number)
        
        start_time = perf_counter()

        parser = AnswerStreamParser()
        json_buffer = ""
        usage = None

        for chunk in llm.stream(prompt):
            delta = chunk.content

            if delta:
                json_buffer += delta

                answer_delta = parser.feed(delta)

                if answer_delta and on_answer_delta is not None:
                    on_answer_delta(answer_delta)

            if chunk.usage_metadata:
                usage = chunk.usage_metadata

        end_time = perf_counter()

        if usage is None:
            raise RuntimeError(
                "Generation stream completed without usage metadata"
            )

        generated_answer = GeneratedAnswer.model_validate(
            json.loads(json_buffer)
        )

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
                gaps_detected=[
                    gap.missing_topic
                    for gap in generated_answer.gaps
                ],
                is_complete=generated_answer.is_complete,
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

    def _generate_gap_round(
            self, 
            state: RAGState, 
            observer: PipelineObserver | None, 
            on_answer_delta: Callable[[str], None] | None = None,
            on_generation_started: Callable[[int], None] | None = None
    ) -> None:
        # Format retrieved context and store it
        formatted_context = format_retrieved_context(
            [evidence_item.document for evidence_item in state.final_evidence]
        )
        if observer is not None:
            observer.record_context(formatted_context)
        
        gap_generation_llm_prompt = generation_prompt.format(
            context=formatted_context,
            question=state.reranker_query,
            previous_answer=state.generated_answer.answer,
        )

        state.generated_answer = self._generate(
            llm=self._generation_llm,
            prompt=gap_generation_llm_prompt,
            stage_name=f"Gap Generation Round {state.generation_round + 1}",
            round_number=state.generation_round + 1,
            observer=observer,
            on_answer_delta=on_answer_delta,
            on_generation_started=on_generation_started
        )

        state.generation_round += 1

    def _retrieve_initial_evidence(self, state: RAGState, observer: PipelineObserver | None) -> None:

        # Retrieve initial evidence
        retrieval_candidates = retrieve_candidates(
            query_input=state.query_input,
            plan=state.retrieval_plan,
            doc_hybrid_retriever=self._doc_hybrid_retriever,
            code_hybrid_retriever=self._code_hybrid_retriever,
            multi_corpus_retriever=self._multi_corpus_retriever,
            multi_query_strategy=self._multi_query_strategy,
            code_vector_store=self._code_vector_store,
            observer=observer,
        )

        state.reranker_query = build_rerank_query(state.query_input)

        # Rerank initial evidence
        initial_evidence = self._reranker.rerank(
            query=state.reranker_query,
            documents=retrieval_candidates,
            observer=observer,
        )

        state.initial_evidence = [
            EvidenceItem(
                document=doc,
                source="initial",
                query=state.reranker_query,
                gap=None,
            )
            for doc in initial_evidence
        ]

        state.evidence_pool = list(state.initial_evidence)

    def _select_final_evidence(self, state: RAGState, observer: PipelineObserver | None) -> None:
        state.final_evidence = self._evidence_selector.select_evidence(
            query=state.reranker_query,
            evidence=state.evidence_pool,
            top_k=RERANK_TOP_K,
            gaps=state.gaps,
            observer=observer,
        )

    def _heartbeat_loop(self, run_id: str, stop_event: threading.Event) -> None:
        while not stop_event.wait(5):
            try:
                self._checkpoint_store.heartbeat(run_id)
            except Exception:
                # Heartbeat failure should not kill RAG run
                pass

    
    def run(
        self, 
        query_input: QueryInput, 
        run_id: str,
        user_id: str,
        observer: PipelineObserver | None = None,
        resume: bool = False,
        on_answer_delta: Callable[[str], None] | None = None,
        on_generation_started: Callable[[int], None] | None = None
    ) -> GeneratedAnswer:

        # Initiate run_id and rag_state
        if resume:
            checkpoint = self._checkpoint_store.load(run_id, user_id)
            state = restore_state(checkpoint, self._evidence_resolver)

            if state.stage == RAGStage.INITIALIZED:
                state.query_input = query_input

            logger.info(
                "Resuming run %s from stage=%s, generation_round=%s.",
                run_id,
                state.stage.value,
                state.generation_round,
            )

            # self._checkpoint_store.resume_run(run_id, user_id)
        else: 
            state = RAGState(query_input=query_input)
            # Get retrieval plan
            state.retrieval_plan = self._query_router.route(query=query_input, observer=observer)
            state.stage = RAGStage.ROUTED

            # Save checkpoint after ROUTED
            self._checkpoint_store.save(run_id, user_id, create_checkpoint(state))

            if observer is not None:
                observer.record_retrieval_plan(state.retrieval_plan)

        # Initialize heartbeat
        stop_event = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(run_id, stop_event),
            daemon=True,
        )
        heartbeat_thread.start()

        try: 

            if state.stage == RAGStage.INITIALIZED:
                state.retrieval_plan = self._query_router.route(query=state.query_input, observer=observer)
                state.stage = RAGStage.ROUTED

                self._checkpoint_store.save(
                    run_id,
                    user_id,
                    create_checkpoint(state),
                )

            if state.stage == RAGStage.ROUTED:
                logger.info(
                    "Run %s continuing from ROUTED, executing initial retrieval.",
                    run_id,
                )
                self._retrieve_initial_evidence(state, observer)
                state.stage = RAGStage.INITIAL_EVIDENCE_READY

                # Save checkpoint after INITIAL_EVIDENCE_READY
                self._checkpoint_store.save(run_id, user_id, create_checkpoint(state))

            if state.stage == RAGStage.INITIAL_EVIDENCE_READY:
                logger.info(
                    "Run %s continuing from INITIAL_EVIDENCE_READY, starting initial generation.",
                    run_id,
                )
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
                    observer=observer,
                    round_number=state.generation_round,
                    on_answer_delta=on_answer_delta,
                    on_generation_started=on_generation_started
                )
                state.stage = RAGStage.DRAFT_GENERATED

                # Save checkpoint after DRAFT_GENERATED
                self._checkpoint_store.save(run_id, user_id, create_checkpoint(state))

            if state.stage == RAGStage.DRAFT_GENERATED:
                logger.info(f"Run {run_id} continuing from DRAFT_GENERATED.")

                if state.generated_answer.is_complete:
                    state.stage = RAGStage.COMPLETED
                    self._checkpoint_store.save(run_id, user_id, create_checkpoint(state))
                    self._checkpoint_store.update_status(run_id, RunStatus.COMPLETED)
                    return state.generated_answer

                # Check for gaps in answer

                print(f"Initial candidates: {len(state.evidence_pool)}")

            if state.stage == RAGStage.GAPS_PROCESSED:
                logger.info(f"Run {run_id} continuing from GAPS_PROCESSED.")
                # Generate final context
                self._select_final_evidence(state, observer)

                state.stage = RAGStage.FINAL_EVIDENCE_READY

            if state.stage == RAGStage.FINAL_EVIDENCE_READY:
                logger.info(f"Run {run_id} continuing from FINAL_EVIDENCE_READY.")
                self._generate_gap_round(state, observer, on_answer_delta, on_generation_started)
                state.stage = RAGStage.GAP_GENERATION_COMPLETED
                # Save checkpoint after gap round
                self._checkpoint_store.save(run_id, user_id, create_checkpoint(state))

                if state.generated_answer.is_complete:
                    state.stage = RAGStage.COMPLETED

                    self._checkpoint_store.save(run_id, user_id, create_checkpoint(state))
                    self._checkpoint_store.update_status(run_id, RunStatus.COMPLETED)
                    return state.generated_answer

                # Save checkpoint after FINAL_EVIDENCE_READY
                self._checkpoint_store.save(run_id, user_id, create_checkpoint(state))

            while state.generation_round < MAX_GENERATION_ROUNDS:
                print(f"\n========== GAP ROUND {state.generation_round + 1} ==========")

                if state.generated_answer.is_complete:
                    state.stage = RAGStage.COMPLETED
                    self._checkpoint_store.save(run_id, user_id, create_checkpoint(state))
                    self._checkpoint_store.update_status(run_id, RunStatus.COMPLETED)
                    print("Answer already complete. Stopping.")
                    return state.generated_answer
                
                state.gaps = self._select_gaps(state.generated_answer.gaps)
                print(f"Selected {len(state.gaps)} gaps for retrieval")

                if not state.gaps:
                    print("No gaps found. Stopping.")
                    state.stage = RAGStage.COMPLETED
                    self._checkpoint_store.save(run_id, user_id, create_checkpoint(state))
                    self._checkpoint_store.update_status(run_id, RunStatus.COMPLETED)
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
                self._checkpoint_store.save(run_id, user_id, create_checkpoint(state))

                # Generate final context
                self._select_final_evidence(state, observer)
                state.stage = RAGStage.FINAL_EVIDENCE_READY

                # Save checkpoint after FINAL_EVIDENCE_READY
                self._checkpoint_store.save(run_id, user_id, create_checkpoint(state))

                self._generate_gap_round(state, observer, on_answer_delta, on_generation_started)
                state.stage = RAGStage.GAP_GENERATION_COMPLETED

                # Save checkpoint after GAP_GENERATION_COMPLETED
                self._checkpoint_store.save(run_id, user_id, create_checkpoint(state))  

                print("\n========== GAP ROUND RESULT ==========")
                print("Is complete:", state.generated_answer.is_complete)
                print("Remaining gaps:", len(state.generated_answer.gaps))

            # print("\n========== GENERATION PROMPT ==========")
            # print(gap_generation_llm_prompt if gap_generation_llm_prompt else generation_llm_prompt)
            state.stage = RAGStage.COMPLETED

            self._checkpoint_store.save(run_id, user_id, create_checkpoint(state))
            self._checkpoint_store.update_status(run_id, RunStatus.COMPLETED)
            return state.generated_answer

        finally:
            # Stop heartbeat when execution ends
            stop_event.set()
            heartbeat_thread.join(timeout=1)