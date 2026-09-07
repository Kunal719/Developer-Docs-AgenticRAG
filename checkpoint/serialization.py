from models.rag_state import RAGState
from models.evidence import EvidenceItem

from checkpoint.resolver import EvidenceResolver
from checkpoint.models import RAGCheckpoint, EvidenceItemCheckpoint

def evidence_items_to_checkpoints(evidence_items: list[EvidenceItem]) -> list[EvidenceItemCheckpoint]:
    evidence_checkpoint_list : list[EvidenceItemCheckpoint] = []

    for item in evidence_items:
        corpus = item.document.metadata.get("corpus")
        chunk_id = item.document.metadata.get("chunk_id")

        evidence_checkpoint_list.append(
            EvidenceItemCheckpoint(
                corpus = corpus,
                chunk_id = chunk_id,
                source = item.source,
                query = item.query,
                gap = item.gap
            )
        )

    return evidence_checkpoint_list

def checkpoints_to_evidence_items(evidence_checkpoint_items: list[EvidenceItemCheckpoint], resolver: EvidenceResolver) -> list[EvidenceItem]:

    evidence_item_list : list[EvidenceItem] = []

    for item in evidence_checkpoint_items:
        evidence_item_list.append(
            EvidenceItem(
                document = resolver(item.corpus, item.chunk_id),
                source = item.source,
                query = item.query,
                gap = item.gap
            )
        )

    return evidence_item_list

def create_checkpoint(rag_state: RAGState) -> RAGCheckpoint:
    initial_evidence = evidence_items_to_checkpoints(rag_state.initial_evidence)
    evidence_pool = evidence_items_to_checkpoints(rag_state.evidence_pool)
    final_evidence = evidence_items_to_checkpoints(rag_state.final_evidence)

    return RAGCheckpoint(
        schema_version = 1,
        query_input = rag_state.query_input,
        stage = rag_state.stage,
        generation_round = rag_state.generation_round,
        retrieval_plan = rag_state.retrieval_plan,
        reranker_query = rag_state.reranker_query,
        initial_evidence = initial_evidence,
        evidence_pool = evidence_pool,
        final_evidence = final_evidence,
        generated_answer = rag_state.generated_answer,
        gaps = rag_state.gaps
    )

def restore_state(checkpoint: RAGCheckpoint, resolver: EvidenceResolver) -> RAGState:
    """
    Restore state from checkpoint
    """
    initial_evidence = checkpoints_to_evidence_items(checkpoint.initial_evidence, resolver)
    evidence_pool = checkpoints_to_evidence_items(checkpoint.evidence_pool, resolver)
    final_evidence = checkpoints_to_evidence_items(checkpoint.final_evidence, resolver)

    return RAGState(
        query_input = checkpoint.query_input,
        stage = checkpoint.stage,
        generation_round = checkpoint.generation_round,
        retrieval_plan = checkpoint.retrieval_plan,
        reranker_query = checkpoint.reranker_query,
        initial_evidence = initial_evidence,
        evidence_pool = evidence_pool,
        final_evidence = final_evidence,
        generated_answer = checkpoint.generated_answer,
        gaps = checkpoint.gaps
    )