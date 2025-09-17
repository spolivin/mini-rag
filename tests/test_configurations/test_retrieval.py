from mini_rag.configurations import RAGConfig


def test_rag_config():
    config = RAGConfig()
    assert config.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.reranker_model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert config.generation_model_name == "google/flan-t5-large"
    assert config.chunk_size == 500
    assert config.overlap == 100
    assert config.max_vectors == 30
