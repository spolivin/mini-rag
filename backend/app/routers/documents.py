import shutil
import tempfile
from pathlib import Path

import psycopg
from fastapi import APIRouter, File, HTTPException, Response, UploadFile, status

from app.dependencies import PipelineDep
from app.schemas import DocumentOut, UploadOut

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

router = APIRouter(prefix="/documents", tags=["documents"])

# Endpoints are plain ``def`` functions: FastAPI runs them in a thread pool, so the
# CPU-heavy embedding work does not block the event loop.


@router.post("", response_model=UploadOut)
def upload_document(pipeline: PipelineDep, file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    if Path(filename).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # The upload is saved under its original name: the document ID is derived from it
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / filename
        with path.open("wb") as out:
            shutil.copyfileobj(file.file, out)
        try:
            doc_id, created = pipeline.ingest(path)
        except psycopg.errors.UniqueViolation:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A different document with the same name and chunk settings already exists.",
            )

    return UploadOut(doc_id=doc_id, filename=filename, created=created)


@router.get("", response_model=list[DocumentOut])
def list_documents(pipeline: PipelineDep):
    return pipeline.list_documents()


@router.delete("/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(doc_id: str, pipeline: PipelineDep):
    if not pipeline.delete_document(doc_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found."
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
