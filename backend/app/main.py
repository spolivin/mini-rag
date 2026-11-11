"""FastAPI service on top of the ``rag`` package.

Run from backend/ with ``uv run fastapi dev`` (needs the database:
``docker compose up -d db``). Interactive docs are served at /docs.
"""

from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.routers import documents, health, query
from rag import LLMUnavailableError, RAGPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Models are loaded once, at startup
    app.state.pipeline = RAGPipeline()
    yield


app = FastAPI(title="MiniRAG", lifespan=lifespan)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)


@app.exception_handler(psycopg.OperationalError)
async def database_unavailable(request: Request, exc: psycopg.OperationalError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "The database is unavailable."},
    )


@app.exception_handler(LLMUnavailableError)
async def llm_unavailable(request: Request, exc: LLMUnavailableError):
    # e.g. no API key configured, invalid key, rate limit, or the API is down
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY, content={"detail": str(exc)}
    )
