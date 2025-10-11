from dataclasses import asdict
from pathlib import Path

from .configurations import RAGConfig, TextGenerationConfig
from .pipeline_components import FaissDB, LLMWrapper, ReRanker


class RAGPipeline:
    """A retrieval pipeline for retrieving relevant chunks from a document and generating
    and answer for the query.

    This pipeline loads a document (PDF, TXT, or MD), splits it into chunks,
    generates embeddings, retrieves relevant chunks using FAISS, and reranks them
    with a cross-encoder model. The re-ranked chunks are used to create a prompt for an
    LLM which then uses it to generate an answer to the query.

    Args:
        rag_config (RAGConfig, optional): Configuration for the RAG pipeline.
            Defaults to None.
        textgen_config (TextGenerationConfig, optional): Configuration for text generation.
            Defaults to None.
    """

    def __init__(
        self,
        rag_config: RAGConfig | None = None,
        textgen_config: TextGenerationConfig | None = None,
    ):
        """Initializes the RAGPipeline with the specified document and generation model."""

        self.config = rag_config or RAGConfig()
        self.textgen_params = asdict(textgen_config or TextGenerationConfig())

        # ReRanker model for ordering retrieved chunks by relevance to the query
        self.cross_encoder = ReRanker(model_name=self.config.reranker_model_name)

        # LLM for answering the question
        self.llm = LLMWrapper(
            model_name=self.config.gen_model_name, textgen_params=self.textgen_params
        )

    def __call__(
        self,
        source_doc_path: str | Path,
        query: str,
        top_k: int | None = None,
    ) -> tuple[str, list[str], list[float]]:
        """Runs the RAG pipeline for a given query.

        Args:
            source_doc_path (str | Path): Path to the source document (PDF, TXT, or MD).
            query (str): The search query.
            top_k (int, optional): Number of top relevant chunks to retrieve. Defaults to None.

        Returns:
            tuple[str, list[str], list[float]]: The generated answer from the LLM, the ranked candidates, and their scores.
        """
        # ---------------------------
        # RETRIEVAL OF CHUNKS
        # ---------------------------
        with FaissDB(
            embedding_model_name=self.config.embedding_model_name,
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap,
            max_vectors=self.config.max_vectors,
            base_dir=self.config.vector_store_dir,
            db_filename=self.config.sqlite_db_filename,
            index_filename=self.config.faiss_index_filename,
        ) as faiss_db:

            # Ingesting a new document to the DB
            faiss_db.ingest_document(source_doc_path=source_doc_path)

            # Retrieving the most similar chunks to the query
            results = faiss_db.run_similarity_search(query=query)

        # ------------------------------------------------------------
        # RE-RANKING RETRIEVED CHUNKS AND CHOOSING TOP-K RELEVANT ONES
        # ------------------------------------------------------------
        ranked_candidates, ranked_scores = self.cross_encoder(
            query=query,
            candidates=results,
            top_k=top_k or self.config.top_k,
        )

        # ---------------------------
        # GENERATING A REPLY FROM LLM
        # ---------------------------
        answer = self.llm.generate(user_query=query, context=ranked_candidates)

        return answer, ranked_candidates, ranked_scores
