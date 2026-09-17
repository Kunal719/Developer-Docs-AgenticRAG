from langchain_chroma import Chroma
from langchain_core.documents import Document

from models.retrieval_plan import MetadataFilters

from config import METADATA_DEFAULT_RETRIEVAL_LIMIT

def _build_entity_where(filters: MetadataFilters) -> dict | None:
    """
    Build where clause using (function_name, class_name, module_name) for metadata filtering.

    Returns None if no where clauses are needed, otherwise returns a dictionary of the where clauses
    """
    or_clauses = []
    if filters.function_names:
        or_clauses.append({"function_name": {"$in": filters.function_names}})
    if filters.class_names:
        or_clauses.append({"class_name": {"$in": filters.class_names}})

    if not or_clauses:
        return None

    entity_clauses = or_clauses[0] if len(or_clauses) == 1 else {"$or": or_clauses}
    
    return {"$and": [entity_clauses, {"corpus": {"$eq": "implementation"}}]}

def exact_match_retrieve(
    filters: MetadataFilters, 
    vector_store: Chroma, 
    limit: int = METADATA_DEFAULT_RETRIEVAL_LIMIT
    ) -> list[Document]:

    if filters is None:
        return []

    where_clauses = _build_entity_where(filters)

    if where_clauses is None:
        return []

    raw = vector_store.get(where=where_clauses, limit=limit, include=["documents", "metadatas"])

    docs = raw.get("documents") or []
    metas = raw.get("metadatas") or []

    return [Document(page_content=doc, metadata=meta or {}) for doc, meta in zip(docs, metas)]

def has_entity_filter(filters: MetadataFilters | None) -> bool:
    if filters is None:
        return False

    return bool(filters.function_names or filters.class_names)