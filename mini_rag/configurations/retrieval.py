from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingsGenerationModel:
    """Default model for embedding generation."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class RerankerModel:
    """Default model for re-ranking retrieved chunks."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass(frozen=True)
class ChunkingConfig:
    """Default parameter config for text chunking."""

    chunk_size: int = 500
    overlap: int = 100
