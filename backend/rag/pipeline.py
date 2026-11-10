from dataclasses import asdict
from pathlib import Path

from sentence_transformers import SentenceTransformer

from .config import RAGConfig, TextGenerationConfig
from .stages import LLMWrapper, PgVectorStore, ReRanker, RetrievedChunk


class RAGPipeline:
    """A retrieval pipeline for retrieving relevant chunks from a document and generating
    and answer for the query.

    This pipeline loads a document (PDF, TXT, or MD), splits it into chunks,
    generates embeddings, stores them in PostgreSQL (pgvector), retrieves relevant chunks
    and reranks them with a cross-encoder model. The re-ranked chunks are used to create
    a prompt for an LLM which then uses it to generate an answer to the query.

    Models are loaded once, in the constructor. Use ``ingest`` to add a document and
    ``ask`` to query one or several stored documents; calling the pipeline does both for
    a single document.

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

        # Embedding model, loaded once and shared by every vector store connection
        self.embedder = SentenceTransformer(self.config.embedding_model_name)

        # ReRanker model for ordering retrieved chunks by relevance to the query
        self.cross_encoder = ReRanker(model_name=self.config.reranker_model_name)

        # LLM for answering the question
        self.llm = LLMWrapper(
            model_name=self.config.gen_model_name, textgen_params=self.textgen_params
        )

    def _open_store(self) -> PgVectorStore:
        """Creates a vector store (one database connection) sharing the loaded embedder."""
        return PgVectorStore(
            embedding_model=self.embedder,
            chunk_size=self.config.chunk_size,
            overlap=self.config.overlap,
            max_vectors=self.config.max_vectors,
            database_url=self.config.database_url,
        )

    def ingest(self, source_doc_path: str | Path) -> tuple[str, bool]:
        """Ingests a document (PDF, TXT, or MD) unless it has been ingested already.

        Args:
            source_doc_path (str | Path): Path to the source document.

        Returns:
            tuple[str, bool]: The document ID and whether it was newly ingested.
        """
        with self._open_store() as vector_store:
            created = vector_store.ingest_document(str(source_doc_path))
            return vector_store.doc_id, created

    def list_documents(self) -> list[dict]:
        """Lists the stored documents."""
        with self._open_store() as vector_store:
            return vector_store.list_documents()

    def delete_document(self, doc_id: str) -> bool:
        """Deletes a stored document. Returns False if it does not exist."""
        with self._open_store() as vector_store:
            return vector_store.delete_document(doc_id)

    def ask(
        self,
        query: str,
        document_ids: list[str] | None = None,
        top_k: int | None = None,
    ) -> tuple[str, list[RetrievedChunk], list[float]]:
        """Answers a query using stored documents.

        Args:
            query (str): The search query.
            document_ids (list[str], optional): Documents to search in. Defaults to all.
            top_k (int, optional): Number of top relevant chunks to use. Defaults to the config value.

        Raises:
            NoRelevantResultsError: If no relevant chunk has been found.

        Returns:
            tuple[str, list[RetrievedChunk], list[float]]: The generated answer from the LLM,
                the ranked chunks (with their source documents), and their scores.
        """
        # ---------------------------
        # RETRIEVAL OF CHUNKS
        # ---------------------------
        with self._open_store() as vector_store:
            retrieved = vector_store.search_chunks(query=query, doc_ids=document_ids)

        # The re-ranker works on texts: identical texts from different documents are
        # collapsed into the first (best matching) one.
        chunk_by_text: dict[str, RetrievedChunk] = {}
        for chunk in retrieved:
            chunk_by_text.setdefault(chunk.text, chunk)

        # ------------------------------------------------------------
        # RE-RANKING RETRIEVED CHUNKS AND CHOOSING TOP-K RELEVANT ONES
        # ------------------------------------------------------------
        ranked_texts, ranked_scores = self.cross_encoder(
            query=query,
            candidates=list(chunk_by_text),
            top_k=top_k or self.config.top_k,
        )

        # ---------------------------
        # GENERATING A REPLY FROM LLM
        # ---------------------------
        answer = self.llm.generate(user_query=query, context=ranked_texts)

        return answer, [chunk_by_text[text] for text in ranked_texts], ranked_scores

    def __call__(
        self,
        source_doc_path: str | Path,
        query: str,
        top_k: int | None = None,
    ) -> tuple[str, list[str], list[float]]:
        """Ingests a document and answers a query about it.

        Args:
            source_doc_path (str | Path): Path to the source document (PDF, TXT, or MD).
            query (str): The search query.
            top_k (int, optional): Number of top relevant chunks to retrieve. Defaults to None.

        Returns:
            tuple[str, list[str], list[float]]: The generated answer from the LLM, the ranked candidates, and their scores.
        """
        doc_id, _ = self.ingest(source_doc_path)
        answer, chunks, scores = self.ask(query, document_ids=[doc_id], top_k=top_k)
        return answer, [chunk.text for chunk in chunks], scores
