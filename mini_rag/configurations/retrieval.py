from dataclasses import dataclass


@dataclass
class RAGConfig:
    """Default parameters for RAG pipeline.

    Attributes:
        embedding_model_name (str): Name of the embedding model. Default is "sentence-transformers/all-MiniLM-L6-v2".
        reranker_model_name (str): Name of the ranking model. Default is "cross-encoder/ms-marco-MiniLM-L-6-v2".
        gen_model_name (str): Name of the generation model. Default is "meta-llama/Llama-2-7b-chat-hf".
        chunk_size (int): Size of a chunk into which document text is separated. Default is 500.
        overlap (int): Word overlap across chunks. Default is 100.
        max_vectors (int): Maximum number of vectors to retrieve during similarity search. Default is 30.
    """

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    gen_model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    chunk_size: int = 500
    overlap: int = 100
    max_vectors: int = 30

    def __post_init__(self):
        if self.chunk_size <= self.overlap:
            raise ValueError("chunk_size must be larger than overlap")
        if self.max_vectors <= 0:
            raise ValueError("max_vectors must be a positive integer")
