from dataclasses import dataclass
from langchain_core.documents import Document

from models.generation import RetrievalGap

@dataclass
class EvidenceItem:
    document: Document
    source: str
    query: str
    gap: RetrievalGap | None = None