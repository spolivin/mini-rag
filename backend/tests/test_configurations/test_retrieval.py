from rag.config import RAGConfig


def test_rag_config(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    config = RAGConfig()
    assert config.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.reranker_model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert config.gen_model_name == "claude-haiku-5-5"
    assert config.chunk_size == 500
    assert config.overlap == 100
    assert config.max_vectors == 30
    assert config.database_url == "postgresql://rag:rag@localhost:5432/rag"


def test_database_url_from_env(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@db:5432/x")
    assert RAGConfig().database_url == "postgresql://u:p@db:5432/x"
