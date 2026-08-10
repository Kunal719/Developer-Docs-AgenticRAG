from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.schemas.query import QueryRequest

import json

router = APIRouter()

@router.post("/query")
def query(request: Request, query_request: QueryRequest):
    rag_service = request.app.state.rag_service

    def stream_result():
        for event_type, data in rag_service.stream(query_request.question):
            if event_type == "token":
                yield f"event: token\ndata: {json.dumps(data)}\n\n"

            elif event_type == "sources":
                sources = [source.model_dump() for source in data]

                yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

            elif event_type == "status":
                yield f"event: status\ndata: {json.dumps(data)}\n\n"


    return StreamingResponse(stream_result(), media_type="text/event-stream")