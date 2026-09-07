from abc import ABC, abstractmethod

from models.evidence import EvidenceItem
from evaluation.observer import PipelineObserver
from models.generation import RetrievalGap

class EvidenceSelector(ABC):

    def _build_reranker_query(self, query: str, gaps: list[RetrievalGap] | None) -> str:

        important_gaps = [
            gap for gap in (gaps or [])
            if gap.priority in {"critical", "important"}
        ]

        if not important_gaps:
            return query

        gap_text = "\n".join(
            f"- {gap.missing_topic}: {gap.reason}"
            for gap in important_gaps
        )

        return (
            f"Original question:\n{query}\n\n"
            f"Important unresolved evidence gaps:\n{gap_text}"
        )

    @abstractmethod
    def select_evidence(
        self, 
        query:str,  
        evidence: list[EvidenceItem], 
        top_k: int,
        gaps: list[RetrievalGap] | None = None,
        observer: PipelineObserver | None = None) -> list[EvidenceItem]:
        pass