from mini_rag.configurations import (
    ChunkingConfig,
    EmbeddingsGenerationModel,
    RerankerModel,
)


def test_embeddings_generation_model():
    assert (
        EmbeddingsGenerationModel.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    )


def test_reranker_model():
    assert RerankerModel.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_chunking_config():
    assert ChunkingConfig.chunk_size == 500
    assert ChunkingConfig.overlap == 100
