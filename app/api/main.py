from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.services.rag_service import RAGService
from app.api.routes import router

@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_service = RAGService()
    rag_service.initialize()

    app.state.rag_service = rag_service

    yield

app = FastAPI(title="Developer Docs RAG", lifespan=lifespan)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

app.include_router(router)

