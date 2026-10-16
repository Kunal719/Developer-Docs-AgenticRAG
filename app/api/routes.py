from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.schemas.query import QueryRequest
from app.auth.dependencies import get_current_user

from checkpoint.models import RunStatus

import json
from dataclasses import asdict

router = APIRouter()

@router.post("/runs")
def create_run(request: Request, query_request: QueryRequest, user_id: str = Depends(get_current_user)):
    rag_service = request.app.state.rag_service

    try:
        run_id = rag_service.create_run(query_request.question, user_id)

    except ValueError as e:
        raise HTTPException(
            status_code=429,
            detail=str(e)
        )

    def event_stream():
        for event in rag_service.run(query_request.question, user_id, run_id):
            yield(
                f"data: {json.dumps(asdict(event))}\n\n"
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.get("/runs")
def list_runs(request: Request, user_id: str = Depends(get_current_user)):
    rag_service = request.app.state.rag_service

    return rag_service.list_runs(user_id)

@router.get("/runs/{run_id}")
def get_run_status(request: Request, run_id: str, user_id: str = Depends(get_current_user)):
    rag_service = request.app.state.rag_service

    try:
        status = rag_service.get_run_status(run_id, user_id)
        checkpoint = rag_service.get_run_checkpoint(run_id, user_id)

        if status == RunStatus.COMPLETED:
            current_state = {
                "question": checkpoint.original_question,
                "stage": checkpoint.stage,
                "generation_round": checkpoint.generation_round,
                "generated_answer": (
                    checkpoint.generated_answer.answer
                    if checkpoint.generated_answer
                    else None
                ),
            }
        else:
            state = rag_service.get_run_state(run_id, user_id)
            current_state = {
                "question": state.original_question,
                "stage": state.stage,
                "generation_round": state.generation_round,
                "generated_answer": (
                    state.generated_answer.answer
                    if state.generated_answer
                    else None
                ),
            }

    except KeyError:
        raise HTTPException(
            status_code= 404,
            detail="Run not found"
        )

    return {
        "run_id": run_id,
        "status": status,
        "current_state": current_state
    }


@router.post("/runs/{run_id}/resume")
def resume_run(request: Request, run_id: str, user_id: str = Depends(get_current_user)):
    rag_service = request.app.state.rag_service

    try:
        _ = rag_service.get_run_status(run_id, user_id)
    except KeyError:
        raise HTTPException(
            status_code= 404,
            detail="Run not found"
        )

    def event_stream():
        for event in rag_service.resume(run_id, user_id):
            yield(
                f"data: {json.dumps(asdict(event))}\n\n"
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")



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