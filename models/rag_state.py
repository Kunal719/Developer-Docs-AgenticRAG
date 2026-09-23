from dataclasses import dataclass, field
from enum import Enum

from langchain_core.documents import Document

from models.query_input import QueryInput
from models.generation import GeneratedAnswer, RetrievalGap
from models.retrieval_plan import RetrievalPlan
from models.evidence import EvidenceItem

class RAGStage(str, Enum):
    INITIALIZED = "initialized"
    ROUTED = "routed"
    INITIAL_EVIDENCE_READY = "initial_evidence_ready"
    DRAFT_GENERATED = "draft_generated"
    GAPS_PROCESSED = "gaps_processed"
    FINAL_EVIDENCE_READY = "final_evidence_ready"
    GAP_GENERATION_COMPLETED = "gap_generation_completed"
    COMPLETED = "completed"


@dataclass
class RAGState:
    """
    Represents the state of the RAG execution.
    """
    # Initial request
    query_input: QueryInput

    # State of the RAG
    stage: RAGStage = RAGStage.INITIALIZED
    generation_round: int = 0

    # Routing
    retrieval_plan: RetrievalPlan | None = None
    reranker_query: str | None = None

    # Evidence
    initial_evidence: list[EvidenceItem] = field(default_factory=list)
    evidence_pool: list[EvidenceItem] = field(default_factory=list)
    final_evidence: list[EvidenceItem] = field(default_factory=list)

    # Generation
    generated_answer: GeneratedAnswer | None = None

    # Gaps
    gaps: list[RetrievalGap] = field(default_factory=list)
