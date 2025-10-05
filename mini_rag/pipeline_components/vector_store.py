import hashlib
import sqlite3
from pathlib import Path

import faiss
import numpy as np
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from sentence_transformers import SentenceTransformer

from ..preprocessing import clean_text


class FaissDB:
    """Implementation of FAISS/SQLite integration for managing embedding vectors and document metadata.

    The class acts as a context manager which ingests a document into FAISS index with saving
    chunks of text and other document information to a SQLite database. It also allows
    running similarity search to retrieve the most relevant chunks of text to the passed query.

    Args:
        embedding_model_name (str): Name of the SentenceTransformer embedding model.
        chunk_size (int): Chunk size into which to split the document text.
        overlap (int): Number of overlapping words to consider during text chunking.
        max_vectors (int): Number of vectors to retrieve from FAISS index.
        base_dir (str, optional): Directory where to save FAISS index and SQLite database.
            Defaults to "vector_store".
        db_filename (str, optional): Name of a SQLite database. Defaults to "chunks.db".
        index_filename (str, optional): Name of a FAISS index. Defaults to "index.faiss".
    """

    def __init__(
        self,
        embedding_model_name: str,
        chunk_size: int,
        overlap: int,
        max_vectors: int,
        base_dir: str = "vector_store",
        db_filename: str = "chunks.db",
        index_filename: str = "index.faiss",
    ):
        """Initializes the FAISS database."""

        # Instantiating embedding model and retrieving embedding dimension
        self.model = SentenceTransformer(embedding_model_name)
        embedding_dim = self.model.get_sentence_embedding_dimension()

        # Setting chunk size and overlap values
        self.chunk_size = chunk_size
        self.overlap = overlap

        # Setting the maximum number of vectors to retrieve from FAISS index
        self.max_vectors = max_vectors

        # Creating or loading existing FAISS index
        self.base_dir = base_dir
        self.index_path = self._get_path(filename=index_filename)
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)

        # Setting path and initializing connection to SQLite database
        self.db_path = self._get_path(filename=db_filename)
        self.conn = None

    # ---------------------------
    # CONTEXT MANAGER SETTINGS
    # ---------------------------
    def __enter__(self):
        # Connection to the database upon entering the context manager
        self.conn = sqlite3.connect(str(self.db_path))
        # Initializing the database
        self._init_db()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Closing the DB connection upon exiting context manager or error
        if self.conn:
            self.conn.close()
            self.conn = None

    # ---------------------------
    # DOCUMENT INGESTION
    # ---------------------------
    def ingest_document(self, source_doc_path: str):
        """Ingests the document to FAISS index and SQLite database.

        The method checks if the document has already been processed (as identified
        by a document hash) and skips processing.

        Otherwise, document is loaded and chunked to be used later for embeddings generation.
        Retrieved chunks are added to "chunks" table of the SQLite database along with embeddings
        being added to FAISS index. Document metadata is added to "documents" table.

        Args:
            source_doc_path (str): Path to the document to be ingested.
        """
        # Computing document hash
        doc_hash = self._get_file_hash(source_doc_path)
        self.doc_hash = doc_hash

        # ---------------------------
        # Processed document check
        # ---------------------------

        cur = self.conn.cursor()
        cur.execute("SELECT 1 FROM documents WHERE hash=?", (doc_hash,))
        if cur.fetchone():
            print(f"Document already ingested (hash={doc_hash}). Skipping.")
            return

        # ---------------------------
        # Ingesting a new document
        # ---------------------------
        print(f"Document with hash={doc_hash} not found. Document ingestion initiated.")
        # Adding document metadata to "documents" table
        filename = Path(source_doc_path).stem.lower()
        doc_id = f"{filename}_{self.chunk_size}_{self.overlap}"
        cur.execute(
            "INSERT INTO documents (doc_id, hash, filename, chunk_size, overlap) VALUES (?, ?, ?, ?, ?)",
            (
                doc_id,
                doc_hash,
                Path(source_doc_path).name,
                self.chunk_size,
                self.overlap,
            ),
        )
        self.conn.commit()

        # ------------------------------------------
        # Chunks retrieval and embeddings generation
        # ------------------------------------------

        docs = self._load_document(source_doc_path=source_doc_path)
        chunks = self._chunk_documents(docs=docs)
        embeddings = self._generate_embeddings(texts=chunks)

        # --------------------------------------------
        # Inserting chunks/embeddings to FAISS/SQLite
        # --------------------------------------------

        for i, chunk in enumerate(chunks):
            cur.execute(
                "INSERT INTO chunks (doc_id, chunk_text) VALUES (?, ?)", (doc_id, chunk)
            )
            self.index.add(embeddings[i : i + 1])
        self.conn.commit()
        # Saving inserted embeddings
        faiss.write_index(self.index, str(self.index_path))

    # ---------------------------
    # SIMILARITY SEARCH
    # ---------------------------
    def run_similarity_search(self, query) -> list[str]:
        """Retrieves the most similar chunks to the passed query.

        Method generates an embedding of a query to be then compared to
        saved embeddings in order to find the most similar ones.

        Args:
            query (str): Query to the document.

        Returns:
            list[str]: List of chunks most similar to the query.
        """
        # Computing embedding of a query
        query_emb = self._generate_embeddings(texts=[query])

        # Running similarity search between saved embeddings
        _, I = self.index.search(query_emb, self.max_vectors)
        faiss_indices = [int(idx) + 1 for idx in I[0]]

        # Retrieving chunks in accordance with the retrieved FAISS indices and document hash value
        cur = self.conn.cursor()
        placeholders = ",".join(["?"] * len(faiss_indices))
        query = f"""
            SELECT c.chunk_text
            FROM chunks c
            JOIN documents d ON c.doc_id = d.doc_id
            WHERE c.chunk_id IN ({placeholders}) AND d.hash=?
        """
        cur.execute(query, (*faiss_indices, self.doc_hash))

        retrieved_chunks = [result[0] for result in cur.fetchall()]

        return retrieved_chunks

    # ---------------------------
    # UTILITIES
    # ---------------------------
    def _init_db(self):
        """Initializes a SQLite database."""
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                hash TEXT UNIQUE,
                filename TEXT,
                chunk_size INTEGER,
                overlap INTEGER,
                added_on TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id INTEGER PRIMARY KEY,
                doc_id TEXT,
                chunk_text TEXT,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
        """
        )
        self.conn.commit()

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

    def _get_path(self, filename: str) -> Path:
        """Retrieves a path to FAISS index or SQLite database.

        Args:
            filename (str): Name of a file.

        Returns:
            Path: Path object.
        """
        path = Path(self.base_dir)
        path.mkdir(exist_ok=True)

        return path / filename

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
