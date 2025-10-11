import os
import shutil
import tempfile
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, Request, UploadFile

from mini_rag.pipeline import RAGPipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.rag_pipeline = RAGPipeline()
    yield
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def read_root() -> dict[str, str]:
    return {
        "app_name": "RAG Service",
        "version": "1.0.0",
        "description": "A simple Retrieval-Augmented Generation (RAG) service for document retrieval and question answering.",
    }


@app.post("/inquire")
async def send_inquiry(
    request: Request, file: UploadFile = File(...), query: str = "", top_k: int = 3
) -> dict[str, str]:
    # Temporarily saving the uploaded file
    filename_suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=filename_suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        answer, _, _ = request.app.state.rag_pipeline(
            source_doc_path=tmp_path,
            query=query,
            top_k=top_k,
        )
        return {"query": query, "answer": answer}
    except RuntimeError as err:
        return {"query": query, "answer": str(err)}
    finally:
        # Cleaning up the temp file after use
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        # Freeing up GPU memory if used
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
