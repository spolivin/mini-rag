import argparse

from mini_rag.chunking import chunk_documents
from mini_rag.ingestion import load_document
from mini_rag.embeddings import generate_embeddings

parser = argparse.ArgumentParser(description="Process some documents.")
parser.add_argument("path", type=str, help="Path to the document")
parser.add_argument("--chunk-size", type=int, help="Size of each chunk", default=300)
parser.add_argument("--overlap", type=int, help="Overlap between chunks", default=50)
args = parser.parse_args()

docs = load_document(path=args.path)
print(f"[INFO] Loaded {len(docs)} documents.")
chunks = chunk_documents(docs=docs, chunk_size=args.chunk_size, overlap=args.overlap)
print(f"[INFO] Created {len(chunks)} chunks.")
embeddings = generate_embeddings(chunks)
print(f"[INFO] Generated embeddings with shape: {embeddings.shape}")
