from pydantic import BaseModel

class Source(BaseModel):
    source: str
    corpus: str
    rank: int

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]

class QueryRequest(BaseModel):
    question: str