from evidence.base_selector import EvidenceSelector
from evidence.dedupe import dedupe_evidence

from models.evidence import EvidenceItem
from models.generation import RetrievalGap

from reranking.base_reranker import BaseReranker

from evaluation.observer import PipelineObserver

class RerankEvidenceSelector(EvidenceSelector):
    def __init__(self, reranker: BaseReranker) -> None:
        self._reranker = reranker


    def _select_with_gap_coverage(
            self, 
            ranked_evidence: list[EvidenceItem], 
            top_k: int, 
            gaps: list[RetrievalGap]
    ) -> list[EvidenceItem]:

        top_k_evidence = ranked_evidence[:top_k]
        missing_topics = [item.missing_topic for item in gaps]

        # Identify missing topics in evidence
        for item in top_k_evidence:
            if item.gap is not None and item.gap.missing_topic in missing_topics:
                missing_topics.remove(item.gap.missing_topic)

        # Add missing evidence
        missing_gap_evidence = []
        for item in ranked_evidence[top_k:]:
            if (
                item.gap is not None and 
                missing_topics and
                item.gap.missing_topic in missing_topics
            ):
                missing_gap_evidence.append(item)
                missing_topics.remove(item.gap.missing_topic)       



        missing_gap_evidence_length = len(missing_gap_evidence)
        index_to_replace_from = top_k - missing_gap_evidence_length

        # Replace missing evidence
        for item in missing_gap_evidence:
            top_k_evidence[index_to_replace_from] = item
            index_to_replace_from += 1

        return top_k_evidence

    def select_evidence(self, query: str, evidence: list[EvidenceItem], top_k: int, gaps: list[RetrievalGap] | None = None, observer: PipelineObserver | None = None) -> list[EvidenceItem]:

        unique_evidence = dedupe_evidence(evidence)
        documents = [item.document for item in unique_evidence]
        evidence_by_id = {
            item.document.metadata.get("chunk_id") : item for item in unique_evidence
        }

        reranker_query = self._build_reranker_query(query, gaps)

        selected_documents = self._reranker.rerank(
            query=reranker_query,
            documents=documents,
            top_k=len(documents),
            observer=observer
        )

        reranked_evidence = [evidence_by_id[doc.metadata.get("chunk_id")] for doc in selected_documents]

        final_documents = self._select_with_gap_coverage(
            ranked_evidence=reranked_evidence,
            top_k=top_k,
            gaps=gaps
        )

        return final_documents