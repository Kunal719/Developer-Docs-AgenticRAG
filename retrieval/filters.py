from langchain_core.documents import Document
from models.retrieval_plan import MetadataFilters

_FIELD_NAMES = ("corpus", "category")

def build_chroma_where(filters: MetadataFilters | None) -> dict | None:
    """
    Build the where clause using corpus/category only. Returns None if neither is set
    """
    if filters is None:
        return None

    clauses = []

    for field in _FIELD_NAMES:
        value = getattr(filters, field)
        if value:
            clauses.append({field: {"$eq": value}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]

    return {"$and": clauses}

def matches_filter(doc: Document, filters: MetadataFilters | None) -> bool:
    """
    Post filter check for BM25 retrieval since no where clause support exists for that.
    A document passes if every field (corpus/category) matches the document's metadata
    """
    if filters is None:
        return True

    for field in _FIELD_NAMES:
        value = getattr(filters, field)
        if value and doc.metadata.get(field) != value:
            return False

    return True