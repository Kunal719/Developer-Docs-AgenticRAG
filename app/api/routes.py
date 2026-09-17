from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.schemas.query import QueryRequest

import json

router = APIRouter()

@router.post("/runs")
def create_run(request: Request, query_request: QueryRequest):
    rag_service = request.app.state.rag_service

    result = rag_service.run(query_request.question)

    return {
        "run_id": result.run_id,
        "answer": result.answer.model_dump(),
    }

@router.get("/runs")
def list_runs(request: Request):
    rag_service = request.app.state.rag_service

    return rag_service.list_runs()

@router.get("/runs/{run_id}")
def get_run_status(request: Request, run_id: str):
    rag_service = request.app.state.rag_service

    status = rag_service.get_run_status(run_id)

    return {
        "run_id": run_id,
        "status": status
    }


@router.post("/runs/{run_id}/resume")
def resume_run(request: Request, run_id: str):
    rag_service = request.app.state.rag_service

    result = rag_service.resume(run_id)

    return {
        "run_id": result.run_id,
        "answer": result.answer.model_dump(),
    }



# @router.post("/query")
# def query(request: Request, query_request: QueryRequest):
#     rag_service = request.app.state.rag_service

#     def stream_result():
#         for event_type, data in rag_service.stream(query_request.question):
#             if event_type == "token":
#                 yield f"event: token\ndata: {json.dumps(data)}\n\n"

#             elif event_type == "sources":
#                 sources = [source.model_dump() for source in data]

#                 yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

#             elif event_type == "status":
#                 yield f"event: status\ndata: {json.dumps(data)}\n\n"


#     return StreamingResponse(stream_result(), media_type="text/event-stream")