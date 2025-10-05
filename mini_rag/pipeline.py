from pathlib import Path

from .configurations import RAGConfig
from .pipeline_components import FaissDB, LLMWrapper, Reranker


class RAGPipeline:
    """A retrieval pipeline for retrieving relevant chunks from a document and generating
    and answer for the query.

    This pipeline loads a document (PDF, TXT, or MD), splits it into chunks,
    generates embeddings, retrieves relevant chunks using FAISS, and reranks them
    with a cross-encoder model. The re-ranked chunks are used to create a prompt for an
    LLM which then uses it to generate an answer to the query.

    Args:
        source_doc_path (str | Path): Path to the source document.
        generation_model_family (str): The family of the generation model to use ("llama" or "mistral").
    """

    def __init__(self, source_doc_path: str | Path, generation_model_family: str):
        """Initializes the RAGPipeline with the specified document and generation model."""

        # Setting path to the document
        self.source_doc_path = source_doc_path

        # Instantiating fixed RAG parameters
        self.rag_config = RAGConfig()

        # Reranker model for ordering retrieved chunks by relevance to the query
        self.cross_encoder = Reranker(model_name=self.rag_config.reranker_model_name)

        # LLM for answering the question
        if generation_model_family == "llama":
            self.llm = LLMWrapper(
                model_name=self.rag_config.llama_model_name,
            )
        elif generation_model_family == "mistral":
            self.llm = LLMWrapper(
                model_name=self.rag_config.mistral_model_name,
            )
        else:
            raise ValueError(
                f"Unknown generation model family: {generation_model_family}"
            )

    def __call__(self, query: str, top_k: int) -> tuple[str, list[str], list[float]]:
        """Runs the RAG pipeline for a given query.

        Args:
            query (str): The search query.
            top_k (int): Number of top relevant chunks to retrieve.

        Returns:
            tuple[str, list[str], list[float]]: The generated answer from the LLM, the ranked candidates, and their scores.
        """
        # ---------------------------
        # RETRIEVAL OF CHUNKS
        # ---------------------------
        with FaissDB(
            embedding_model_name=self.rag_config.embedding_model_name,
            chunk_size=self.rag_config.chunk_size,
            overlap=self.rag_config.overlap,
            max_vectors=self.rag_config.max_vectors,
            base_dir="vector_store",
            db_filename="processed_documents.db",
            index_filename="vectors.faiss",
        ) as faiss_db:

            # Ingesting a new document to the DB
            faiss_db.ingest_document(source_doc_path=self.source_doc_path)

            # Retrieving the most similar chunks to the query
            results = faiss_db.run_similarity_search(query=query)

        # ------------------------------------------------------------
        # RE-RANKING RETRIEVED CHUNKS AND CHOOSING TOP-K RELEVANT ONES
        # ------------------------------------------------------------
        ranked_candidates, ranked_scores = self.cross_encoder(
            query=query,
            candidates=results,
            top_k=top_k,
        )

        # ---------------------------
        # GENERATING A REPLY FROM LLM
        # ---------------------------
        answer = self.llm.generate(user_query=query, context=ranked_candidates)

        return answer, ranked_candidates, ranked_scores
