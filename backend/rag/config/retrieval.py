import os
from dataclasses import dataclass, field


@dataclass
class RAGConfig:
    """Default parameters for RAG pipeline.

    Attributes:
        embedding_model_name (str): Name of the embedding model. Default is "sentence-transformers/all-MiniLM-L6-v2".
        reranker_model_name (str): Name of the ranking model. Default is "cross-encoder/ms-marco-MiniLM-L-6-v2".
        gen_model_name (str): Claude model ID used for generation. Default is "claude-haiku-5-5".
        chunk_size (int): Size of a chunk into which document text is separated. Default is 500.
        overlap (int): Word overlap across chunks. Default is 100.
        top_k (int): Number of top documents to consider for generation. Default is 5.
        max_vectors (int): Maximum number of vectors to retrieve during similarity search. Default is 30.
        database_url (str): PostgreSQL connection string. Read from the DATABASE_URL environment
            variable, defaulting to the local docker-compose database.
    """

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    gen_model_name: str = "claude-haiku-5-5"
    chunk_size: int = 500
    overlap: int = 100
    max_vectors: int = 30
    top_k: int = 5
    database_url: str = field(
        default_factory=lambda: os.environ.get(
            "DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag"
        )
    )

    def __post_init__(self):
        if self.chunk_size <= self.overlap:
            raise ValueError("chunk_size must be larger than overlap")
        if self.max_vectors <= 0:
            raise ValueError("max_vectors must be a positive integer")
        if self.top_k <= 0:
            raise ValueError("top_k must be a positive integer")
