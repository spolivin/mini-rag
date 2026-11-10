import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psycopg
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from ..preprocessing import clean_text


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk of text returned by a similarity search, with its source document."""

    doc_id: str
    filename: str
    text: str


class PgVectorStore:
    """Implementation of a PostgreSQL/pgvector store for document chunks and embeddings.

    The class acts as a context manager which ingests a document into PostgreSQL, saving
    chunks of text together with their embeddings and the document metadata. It also allows
    running similarity search to retrieve the most relevant chunks of the ingested document
    to the passed query.

    The database schema is created by ``db/init.sql`` (see docker-compose.yml).

    Args:
        embedding_model (str | SentenceTransformer): Name of the SentenceTransformer embedding
            model, or an already loaded model (lets callers load it once and share it).
        chunk_size (int): Chunk size into which to split the document text.
        overlap (int): Number of overlapping words to consider during text chunking.
        max_vectors (int): Number of chunks to retrieve from the similarity search.
        database_url (str): PostgreSQL connection string.
    """

    def __init__(
        self,
        embedding_model: str | SentenceTransformer,
        chunk_size: int,
        overlap: int,
        max_vectors: int,
        database_url: str,
    ):
        """Initializes the vector store."""

        # Instantiating embedding model (unless an instance was passed)
        self.model = (
            SentenceTransformer(embedding_model)
            if isinstance(embedding_model, str)
            else embedding_model
        )

        # Setting chunk size and overlap values
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Setting the maximum number of chunks to retrieve during similarity search
        self.max_vectors = max_vectors

        # Connection settings; the connection itself is opened by the context manager
        self.database_url = database_url
        self.conn = None

        # ID of the document the search is scoped to (set by ingest_document)
        self.doc_id = None

    # ---------------------------
    # CONTEXT MANAGER SETTINGS
    # ---------------------------
    def __enter__(self):
        # Connection to the database upon entering the context manager
        self.conn = psycopg.connect(self.database_url)
        register_vector(self.conn)
        # Filtered searches over an approximate (HNSW) index may return too few rows,
        # because the filter is applied after the index scan. Iterative scans keep
        # scanning the index until enough rows pass the filter.
        self.conn.execute("SET hnsw.iterative_scan = strict_order")
        self.conn.commit()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Closing the DB connection upon exiting context manager or error
        # (an uncommitted transaction is rolled back)
        if self.conn:
            self.conn.close()
            self.conn = None

    # ---------------------------
    # DOCUMENT INGESTION
    # ---------------------------
    def ingest_document(self, source_doc_path: str) -> bool:
        """Ingests the document into PostgreSQL.

        The method checks if the document has already been processed (as identified
        by a document hash) and skips processing.

        Otherwise, document is loaded, chunked and embedded. The document metadata and all
        of its chunks are then inserted in a single transaction, so a failure never leaves
        a document registered without its chunks.

        Args:
            source_doc_path (str): Path to the document to be ingested.

        Returns:
            bool: True if the document was ingested, False if it was already present.
        """
        # Computing document hash
        doc_hash = self._get_file_hash(source_doc_path)

        # ---------------------------
        # Processed document check
        # ---------------------------
        row = self.conn.execute(
            "SELECT doc_id FROM documents WHERE hash = %s", (doc_hash,)
        ).fetchone()
        if row:
            self.doc_id = row[0]
            print(f"Document already ingested (hash={doc_hash}). Skipping.")
            return False

        # ------------------------------------------
        # Chunks retrieval and embeddings generation
        # ------------------------------------------
        print(f"Document with hash={doc_hash} not found. Document ingestion initiated.")
        docs = self._load_document(source_doc_path=source_doc_path)
        chunks = self._chunk_documents(docs=docs)
        embeddings = self._generate_embeddings(texts=chunks)

        # --------------------------------------------
        # Inserting document metadata and chunks
        # --------------------------------------------
        filename = Path(source_doc_path).stem.lower()
        doc_id = f"{filename}_{self.chunk_size}_{self.overlap}"
        with self.conn.transaction():
            self.conn.execute(
                "INSERT INTO documents (doc_id, hash, filename, chunk_size, overlap) "
                "VALUES (%s, %s, %s, %s, %s)",
                (doc_id, doc_hash, Path(source_doc_path).name, self.chunk_size, self.overlap),
            )
            with self.conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (doc_id, chunk_text, embedding) VALUES (%s, %s, %s)",
                    [(doc_id, chunk, emb) for chunk, emb in zip(chunks, embeddings)],
                )
        self.conn.commit()
        self.doc_id = doc_id
        return True

    # ---------------------------
    # SIMILARITY SEARCH
    # ---------------------------
    def search_chunks(
        self, query: str, doc_ids: list[str] | None = None
    ) -> list[RetrievedChunk]:
        """Retrieves the chunks most similar to the passed query.

        Method generates an embedding of a query to be then compared to the saved
        embeddings (cosine distance) in order to find the most similar ones.

        The search is restricted to ``doc_ids`` if given, otherwise to the document ingested
        through this store, and otherwise covers all documents.

        Args:
            query (str): Query to the document.
            doc_ids (list[str], optional): Documents to search in.

        Returns:
            list[RetrievedChunk]: At most ``max_vectors`` chunks, best match first.
        """
        if doc_ids is None and self.doc_id is not None:
            doc_ids = [self.doc_id]

        # Computing embedding of a query
        query_emb = self._generate_embeddings(texts=[query])[0]

        # Running similarity search (optionally restricted to the target documents)
        where = "WHERE doc_id = ANY(%s)" if doc_ids is not None else ""
        params = [query_emb, *([doc_ids] if doc_ids is not None else []), self.max_vectors]
        rows = self.conn.execute(
            f"""
            SELECT t.doc_id, d.filename, t.chunk_text
            FROM (
                SELECT doc_id, chunk_text, embedding <=> %s AS dist
                FROM chunks {where}
                ORDER BY dist
                LIMIT %s
            ) t
            JOIN documents d USING (doc_id)
            ORDER BY t.dist
            """,
            params,
        ).fetchall()

        return [RetrievedChunk(doc_id=r[0], filename=r[1], text=r[2]) for r in rows]

    def run_similarity_search(self, query: str) -> list[str]:
        """Same as ``search_chunks`` but returns just the chunk texts."""
        return [chunk.text for chunk in self.search_chunks(query)]

    # ---------------------------
    # DOCUMENT MANAGEMENT
    # ---------------------------
    def list_documents(self) -> list[dict]:
        """Lists the stored documents, newest first, with their number of chunks."""
        rows = self.conn.execute(
            """
            SELECT d.doc_id, d.filename, d.chunk_size, d.overlap, d.created_at,
                   count(c.chunk_id)
            FROM documents d
            LEFT JOIN chunks c USING (doc_id)
            GROUP BY d.doc_id
            ORDER BY d.created_at DESC, d.doc_id
            """
        ).fetchall()
        keys = ("doc_id", "filename", "chunk_size", "overlap", "created_at", "num_chunks")
        return [dict(zip(keys, row)) for row in rows]

    def delete_document(self, doc_id: str) -> bool:
        """Deletes a document and its chunks. Returns False if it does not exist."""
        deleted = self.conn.execute(
            "DELETE FROM documents WHERE doc_id = %s", (doc_id,)
        ).rowcount
        self.conn.commit()
        return deleted > 0

    # ---------------------------
    # UTILITIES
    # ---------------------------
    def _get_file_hash(self, path: str) -> str:
        """Computing a hash for the passed document.

        Args:
            path (str): Path to the document.

        Returns:
            str: Hash value.
        """
        hasher = hashlib.sha1()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)

        return hasher.hexdigest()

    def _load_document(self, source_doc_path: str) -> list[Document]:
        """Loads document text.

        Args:
            source_doc_path (str): Path to the document.

        Raises:
            ValueError: Error raised in case unsupported file type is provided.

        Returns:
            list[Document]: Collection of Langchain Document objects.
        """
        ext = Path(source_doc_path).suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(source_doc_path)
        elif ext in {".txt", ".md"}:
            loader = TextLoader(source_doc_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return loader.load()

    def _chunk_documents(self, docs: list[Document]) -> list[str]:
        """Chunks a document.

        Args:
            docs (list[Document]): Collection of Langchain Document objects.

        Returns:
            list[str]: List of collected chunks of text.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.overlap,
        )
        chunks = splitter.split_documents(docs)
        return [clean_text(chunk.page_content) for chunk in chunks]

    def _generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generates embeddings of texts.

        Args:
            texts (list[str]): List of collected chunks of text.

        Returns:
            np.ndarray: Generated embeddings.
        """
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
