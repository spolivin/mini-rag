import argparse

from mini_rag.chunking import chunk_documents
from mini_rag.ingestion import load_document

parser = argparse.ArgumentParser(description="Process some documents.")
parser.add_argument("path", type=str, help="Path to the document")
parser.add_argument("--chunk-size", type=int, help="Size of each chunk", default=300)
parser.add_argument("--overlap", type=int, help="Overlap between chunks", default=50)
args = parser.parse_args()

docs = load_document(path=args.path)
chunks = chunk_documents(docs=docs, chunk_size=args.chunk_size, overlap=args.overlap)

print(f"Loaded {len(docs)} documents.")
print(f"Created {len(chunks)} chunks.\n")
for i, chunk in enumerate(chunks[0:10]):  # Print chunks 1 to 10 as a sample
    print(f"Chunk {i+1}: {chunk.page_content[:args.chunk_size]}\n")  # Print each chunk
