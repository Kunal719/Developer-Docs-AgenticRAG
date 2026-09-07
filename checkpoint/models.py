from pydantic import BaseModel, Field

from models.generation import GeneratedAnswer, RetrievalGap
from models.query_input import QueryInput
from models.rag_state import RAGStage
from models.retrieval_plan import RetrievalPlan

class EvidenceItemCheckpoint(BaseModel):
    """
    Represents an evidence item for checkpointing
    """
    corpus: str
    chunk_id: str
    source: str
    query: str
    gap: RetrievalGap | None = None

class RAGCheckpoint(BaseModel):
    """
    The RAGCheckpoint class represents a checkpoint of the RAG state.
    """
    schema_version: int = 1

    query_input: QueryInput
    stage: RAGStage
    generation_round: int

    retrieval_plan: RetrievalPlan | None = None
    reranker_query: str | None = None

    initial_evidence: list[EvidenceItemCheckpoint] = Field(default_factory=list)
    evidence_pool: list[EvidenceItemCheckpoint] = Field(default_factory=list)
    final_evidence: list[EvidenceItemCheckpoint] = Field(default_factory=list)

    generated_answer: GeneratedAnswer | None = None
    gaps: list[RetrievalGap] = Field(default_factory=list)


