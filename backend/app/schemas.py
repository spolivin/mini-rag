from datetime import datetime

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    chunk_size: int
    overlap: int
    created_at: datetime
    num_chunks: int


class UploadOut(BaseModel):
    doc_id: str
    filename: str
    created: bool = Field(description="False if the same file had been ingested before")


class QueryIn(BaseModel):
    query: str = Field(min_length=1)
    document_ids: list[str] | None = Field(
        default=None, description="Documents to search in; all documents if omitted"
    )
    top_k: int | None = Field(default=None, ge=1, le=50)


class ChunkOut(BaseModel):
    text: str
    score: float
    doc_id: str
    filename: str


class QueryOut(BaseModel):
    answer: str
    chunks: list[ChunkOut]
