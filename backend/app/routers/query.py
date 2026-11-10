from fastapi import APIRouter

from app.dependencies import PipelineDep
from app.schemas import ChunkOut, QueryIn, QueryOut
from rag import NoRelevantResultsError

NO_RESULTS_MESSAGE = "No relevant passages were found for this question."

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryOut)
def query(body: QueryIn, pipeline: PipelineDep):
    try:
        answer, chunks, scores = pipeline.ask(
            body.query, document_ids=body.document_ids or None, top_k=body.top_k
        )
    except NoRelevantResultsError:
        return QueryOut(answer=NO_RESULTS_MESSAGE, chunks=[])

    return QueryOut(
        answer=answer,
        chunks=[
            ChunkOut(
                text=chunk.text,
                score=float(score),
                doc_id=chunk.doc_id,
                filename=chunk.filename,
            )
            for chunk, score in zip(chunks, scores)
        ],
    )
