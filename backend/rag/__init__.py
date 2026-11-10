"""The RAG library: ingestion, retrieval, re-ranking and answer generation.

It knows nothing about HTTP; the FastAPI app in ``app/`` imports it.
"""

from .config import RAGConfig, TextGenerationConfig
from .pipeline import RAGPipeline
from .stages import LLMUnavailableError, NoRelevantResultsError, RetrievedChunk

__all__ = [
    "LLMUnavailableError",
    "NoRelevantResultsError",
    "RAGConfig",
    "RAGPipeline",
    "RetrievedChunk",
    "TextGenerationConfig",
]
