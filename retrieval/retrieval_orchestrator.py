from langchain_core.documents import Document
from langchain_chroma import Chroma

from models.query_input import QueryInput
from models.retrieval_plan import RetrievalPlan

from retrieval.base import BaseRetriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.multi_query_retrieval_strategy import MultiQueryRetrievalStrategy
from retrieval.metadata_retrieval import exact_match_retrieve, has_entity_filter
from retrieval.multi_corpus_retriever import MultiCorpusRetriever

from evaluation.observer import PipelineObserver

from config import GAP_HYBRID_TOP_K


def build_rerank_query(q: QueryInput) -> str:
    parts = [q.question]
    if q.code_snippet:
        parts.append(f"Code:\n{q.code_snippet}")
    if q.observed_behavior:
        parts.append(f"Observed: {q.observed_behavior}")
    if q.expected_behavior:
        parts.append(f"Expected: {q.expected_behavior}")
    if q.error:
        parts.append(f"Error: {q.error}")
    if q.logs:
        parts.append(f"Logs: {q.logs}")
    return "\n\n".join(parts)


def _dedupe(docs: list[Document]) -> list[Document]:
    seen = set()
    deduped = []
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id")
        if chunk_id not in seen:
            seen.add(chunk_id)
            deduped.append(doc)
    return deduped

def _select_retriever(
    plan: RetrievalPlan,
    doc_hybrid_retriever: HybridRetriever,
    code_hybrid_retriever: HybridRetriever,
    multi_corpus_retriever: MultiCorpusRetriever,
) -> BaseRetriever:

    filters = plan.metadata_filters

    if filters is None or filters.corpus is None:
        return multi_corpus_retriever

    if filters.corpus == "documentation":
        return doc_hybrid_retriever

    if filters.corpus == "implementation":
        return code_hybrid_retriever

    raise ValueError(f"Unknown corpus: {filters.corpus}")

def retrieve_candidates(
    query_input: QueryInput | None,
    plan: RetrievalPlan,
    doc_hybrid_retriever: HybridRetriever,
    code_hybrid_retriever: HybridRetriever,
    multi_corpus_retriever: MultiCorpusRetriever,
    multi_query_strategy: MultiQueryRetrievalStrategy,
    code_vector_store: Chroma,
    query_override: str | None = None,
    observer: PipelineObserver | None = None,
) -> list[Document]:

    retriever = _select_retriever(plan, doc_hybrid_retriever, code_hybrid_retriever, multi_corpus_retriever)

    if plan.retrieval_mode == "direct":
        query = query_override or build_rerank_query(query_input)
        if plan.reasoning == "Gap retrieval loop":
            candidates = retriever.retrieve(query, observer, plan.metadata_filters, GAP_HYBRID_TOP_K)
        else:
            candidates = retriever.retrieve(query, observer, plan.metadata_filters)

    elif plan.retrieval_mode == "combined_query":
        candidates = retriever.retrieve(
            plan.combined_query, observer, plan.metadata_filters
        )

    elif plan.retrieval_mode == "multi_query_fusion":
        candidates = multi_query_strategy.retrieve(
            plan.sub_queries, retriever, plan.metadata_filters, observer
        )

    else:
        raise ValueError(f"Unknown retrieval_mode: {plan.retrieval_mode}")

    if has_entity_filter(plan.metadata_filters):
        exact_matches = exact_match_retrieve(plan.metadata_filters, code_vector_store)
        candidates = exact_matches + candidates

    return _dedupe(candidates)