from fastapi import FastAPI
from contextlib import asynccontextmanager

import threading
import os

from app.services.rag_service import RAGService
from app.api.routes import router
from app.api.auth_routes import router as auth_router

from checkpoint.stale_run_monitor import stale_run_monitor

from starlette.middleware.sessions import SessionMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_service = RAGService()
    rag_service.initialize()

    app.state.rag_service = rag_service

    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=stale_run_monitor,
        args=(rag_service.checkpoint_store, stop_event),
        daemon=True
    )

    monitor_thread.start()

    app.state.stale_run_monitor_stop_event = stop_event
    app.state.stale_run_monitor_thread = monitor_thread

    try:
        yield

    finally:
        stop_event.set()
        monitor_thread.join(timeout=1)

app = FastAPI(title="Developer Docs RAG", lifespan=lifespan)

app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ["SESSION_SECRET_KEY"],
    https_only=False,
    same_site='lax'
)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

app.include_router(router)
app.include_router(auth_router)

