import hashlib
import os
from pathlib import Path

import faiss
import numpy as np
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import pipeline

from .preprocessing import clean_text, prettify_answer


class RAGPipeline:
    """A retrieval pipeline for extracting and ranking relevant chunks from a document.

    This pipeline loads a document (PDF, TXT, or MD), splits it into chunks,
    generates embeddings, retrieves relevant chunks using FAISS, and reranks them
    with a cross-encoder model.

    Args:
        source_doc_path (str | Path): Path to the source document.
        gen_model_name (str, optional): Name of the text generation model. Defaults to "google/flan-t5-small".
        embedding_model_name (str, optional): Name of the SentenceTransformer embedding model.
            Defaults to "all-MiniLM-L6-v2".
        cross_encoder_model_name (str, optional): Name of the cross-encoder model for re-ranking.
            Defaults to "cross-encoder/ms-marco-MiniLM-L-6-v2".
    """

    def __init__(
        self,
        source_doc_path: str | Path,
        gen_model_name: str = "google/flan-t5-small",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        cross_encoder_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.source_doc_path = source_doc_path
        self.model = SentenceTransformer(embedding_model_name)
        self.cross_encoder = CrossEncoder(cross_encoder_model_name)
        self.gen_model = pipeline("text2text-generation", model=gen_model_name)

    @staticmethod
    def get_file_hash(path: str | Path, chunk_size: int, overlap: int) -> str:
        """Compute a hash of the file contents and chunking parameters.

        Args:
            path (str | Path): Path to a file.
            chunk_size (int): Size of a chunk.
            overlap (int): Overlap.

        Returns:
            str: Computed hash for a source file.
        """
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        # Adding chunking parameters to the hash
        param_str = f"{chunk_size}_{overlap}".encode()
        hasher.update(param_str)
        return hasher.hexdigest()

    def __call__(
        self,
        query: str,
        top_k: int,
        chunk_size: int,
        overlap: int,
        verbose: bool = False,
    ) -> tuple[str, list[str], list[float]]:
        """Runs the RAG pipeline for a given query.

        Args:
            query (str): The search query.
            top_k (int): Number of top relevant chunks to retrieve.
            chunk_size (int): Size of each text chunk.
            overlap (int): Overlap between consecutive chunks.
            verbose (bool, optional): If True, prints progress information. Defaults to False.

        Returns:
            tuple[str, list[str], list[float]]: The generated answer from the LLM, the ranked candidates, and their scores.
        """
        file_hash = self.get_file_hash(
            path=self.source_doc_path, chunk_size=chunk_size, overlap=overlap
        )
        index_path = self._get_index_path(file_hash=file_hash)

        # Stage 1: Load the document
        docs = self._load_document()
        if verbose:
            print(f"[INFO] Loaded {len(docs)} documents")

        # Stage 2: Chunk the document
        chunks = self._chunk_documents(docs, chunk_size=chunk_size, overlap=overlap)
        if verbose:
            print(f"[INFO] Chunked documents into {len(chunks)} chunks")

        if not os.path.exists(index_path):
            # Stage 3: Generate embeddings
            if verbose:
                print("[INFO] Generating embeddings...")
            embeddings = self._generate_embeddings(chunks)
            if verbose:
                print(f"[INFO] Generated embeddings with shape {embeddings.shape}")

            # Stage 4: Build FAISS index
            index = self._build_faiss_index(embeddings)
            faiss.write_index(index, index_path)
            if verbose:
                print(f"[INFO] Built FAISS index with {index.ntotal} vectors")
        else:
            index = faiss.read_index(index_path)
            if verbose:
                print(f"[INFO] Re-used saved embeddings")

        # Stage 5: Retrieve top-k candidates for a given query
        candidates = self._query_faiss_index(
            index, query=query, chunks=chunks, top_k=top_k
        )
        if verbose:
            print(f"[INFO] Retrieved {len(candidates)} candidate chunks")

        # Stage 6: Rerank candidates with cross-encoder
        ranked_candidates, scores = self._rerank_with_cross_encoder(query, candidates)
        if verbose:
            print(f"[INFO] Reranked candidates")

        # Stage 7: Composing a prompt and generating answer with LLM
        if verbose:
            print(f"[INFO] Generating answer...")
        prompt = self._build_prompt(ranked_candidates, query)
        answer = self._generate_answer(prompt)

        return prettify_answer(answer), ranked_candidates, scores

    def _get_index_path(self, file_hash: str) -> str:
        """Return the path for the FAISS index file."""
        index_dir = Path("indexes")
        index_dir.mkdir(exist_ok=True)
        return str(index_dir / f"{file_hash}.faiss")

    def _load_document(self) -> list[Document]:
        """Loads the source document based on its file extension.

        Returns:
            list[Document]: List of loaded document objects.

        Raises:
            ValueError: If the file type is unsupported.
        """
        ext = Path(self.source_doc_path).suffix.lower()

        if ext == ".pdf":
            loader = PyPDFLoader(self.source_doc_path)
        elif ext in {".txt", ".md"}:
            loader = TextLoader(self.source_doc_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return loader.load()

    def _chunk_documents(
        self, docs: list[Document], chunk_size: int = 500, overlap: int = 50
    ) -> list[str]:
        """Splits documents into overlapping text chunks.

        Args:
            docs (list[Document]): List of loaded document objects.
            chunk_size (int, optional): Size of each chunk. Defaults to 500.
            overlap (int, optional): Overlap between chunks. Defaults to 50.

        Returns:
            list[str]: List of text chunks.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )
        chunks = splitter.split_documents(docs)
        chunks = [clean_text(chunk.page_content) for chunk in chunks]

        return chunks

    def _generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generates embeddings for a list of text chunks.

        Args:
            texts (list[str]): List of text chunks.

        Returns:
            np.ndarray: Embedding vectors.
        """
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    def _build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Builds a FAISS index from embeddings.

        Args:
            embeddings (np.ndarray): Embedding vectors.

        Returns:
            faiss.Index: FAISS index object.
        """
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)  # L2 distance
        index.add(embeddings)

        return index

    def _query_faiss_index(
        self, index: faiss.Index, query: str, chunks: list[str], top_k: int
    ) -> list[str]:
        """Queries the FAISS index to retrieve top-k relevant chunks.

        Args:
            index (faiss.Index): FAISS index object.
            query (str): Search query.
            chunks (list[str]): List of text chunks.
            top_k (int): Number of top chunks to retrieve.

        Returns:
            list[str]: Top-k relevant text chunks.
        """
        query_emb = self.model.encode([query], convert_to_numpy=True).astype("float32")

        _, indices = index.search(query_emb, top_k)

        return [chunks[i] for i in indices[0]]

    def _rerank_with_cross_encoder(
        self,
        query: str,
        candidates: list[str],
    ) -> tuple[list[str], list[float]]:
        """Reranks candidate chunks using a cross-encoder model.

        Args:
            query (str): Search query.
            candidates (list[str]): Candidate text chunks.

        Returns:
            list[str]: Reranked text chunks.
        """
        pairs = [[query, candidate] for candidate in candidates]
        scores = self.cross_encoder.predict(pairs)

        ranked_candidates = [
            candidate for _, candidate in sorted(zip(scores, candidates), reverse=True)
        ]
        ranked_scores = [
            score for score, _ in sorted(zip(scores, candidates), reverse=True)
        ]
        return ranked_candidates, ranked_scores

    def _build_prompt(self, context_chunks: list[str], query: str) -> str:
        """Builds a prompt for the LLM using retrieved context and the user query.

        Args:
            context_chunks (list[str]): Relevant text chunks.
            query (str): User's question.

        Returns:
            str: Prompt for the LLM.
        """
        context = "\n\n".join(context_chunks[::-1])
        prompt = (
            "Use the following context to answer the question:\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )
        return prompt

    def _generate_answer(self, prompt: str) -> str:
        """Generates an answer using the LLM based on the provided prompt.

        Args:
            prompt (str): The prompt containing context and the user's question.

        Returns:
            str: Generated answer.
        """
        answer = self.gen_model(
            prompt,
            max_new_tokens=200,
            min_new_tokens=30,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            no_repeat_ngram_size=3,
        )
        answer = answer[0]["generated_text"]

        return answer
