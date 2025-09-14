from pathlib import Path

from .configurations import (
    ChunkingConfig,
    EmbeddingsGenerationModel,
    RerankerModel,
    TextGenerationConfig,
    TextGenerationModel,
)
from .pipeline_components import AnswerGenerator, FaissDB, PromptBuilder, Reranker


class RAGPipeline:
    """A retrieval pipeline for retieving relevant chunks from a document and generating
    and answer for the query.

    This pipeline loads a document (PDF, TXT, or MD), splits it into chunks,
    generates embeddings, retrieves relevant chunks using FAISS, and reranks them
    with a cross-encoder model. The re-ranked chunks are used to create a prompt for an
    LLM which then uses it to generate an answer to the query.

    Args:
        source_doc_path (str | Path): Path to the source document.
    """

    def __init__(self, source_doc_path: str | Path):
        self.source_doc_path = source_doc_path

        # Reranker model for ordering retrieved chunks by relevance to the query
        self.cross_encoder = Reranker(model_name=RerankerModel.model_name)

        # Builder for the RAG-prompt to the LLM
        self.prompt_builder = PromptBuilder()

        # LLM for answering the question
        self.gen_model = AnswerGenerator(
            model_name=TextGenerationModel.model_name, gen_params=TextGenerationConfig()
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
            embedding_model_name=EmbeddingsGenerationModel.model_name,
            chunk_size=ChunkingConfig.chunk_size,
            overlap=ChunkingConfig.overlap,
            base_dir="vector_store",
            db_filename="processed_documents.db",
            index_filename="vectors.faiss",
        ) as faiss_db:
            faiss_db.ingest_document(source_doc_path=self.source_doc_path)

            results = faiss_db.run_similarity_search(query=query, k=top_k)

        # ---------------------------
        # RE-RANKING RETRIEVED CHUNKS
        # ---------------------------
        ranked_candidates, ranked_scores = self.cross_encoder(
            query=query, candidates=results
        )

        # ---------------------------
        # BUILDING A RAG-PROMPT
        # ---------------------------
        prompt = self.prompt_builder(context_chunks=ranked_candidates, query=query)

        # ---------------------------
        # GENERATING A REPLY FROM LLM
        # ---------------------------
        answer = self.gen_model(prompt)

        return answer, ranked_candidates, ranked_scores
