from typing import Annotated

from fastapi import Depends, Request

from rag import RAGPipeline


def get_pipeline(request: Request) -> RAGPipeline:
    return request.app.state.pipeline


PipelineDep = Annotated[RAGPipeline, Depends(get_pipeline)]
