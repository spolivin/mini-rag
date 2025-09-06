import argparse

from mini_rag.pipeline import RetrievalPipeline

parser = argparse.ArgumentParser(description="Process some documents.")
parser.add_argument(
    "--source-doc", required=True, type=str, help="Path to the document"
)
parser.add_argument(
    "--query",
    required=True,
    type=str,
    help="Query string to search in the index",
)
parser.add_argument("--chunk-size", type=int, help="Size of each chunk", default=300)
parser.add_argument("--overlap", type=int, help="Overlap between chunks", default=50)
parser.add_argument(
    "--gen-model",
    type=str,
    help="Text generation model name",
    default="google/flan-t5-small",
)
parser.add_argument(
    "--embedding-model",
    type=str,
    help="SentenceTransformer model name",
    default="sentence-transformers/all-mpnet-base-v2",
)
parser.add_argument(
    "--cross-encoder-model",
    type=str,
    help="Cross-Encoder model name",
    default="cross-encoder/ms-marco-MiniLM-L-6-v2",
)
parser.add_argument(
    "--top-k", type=int, help="Number of top results to retrieve", default=5
)
parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")
args = parser.parse_args()


def main():
    # Initialize the retrieval pipeline
    pipeline = RetrievalPipeline(
        source_doc_path=args.source_doc,
        gen_model_name=args.gen_model,
        embedding_model_name=args.embedding_model,
        cross_encoder_model_name=args.cross_encoder_model,
    )
    # Run the pipeline with the provided query
    answer, candidates, scores = pipeline(
        query=args.query,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        verbose=args.verbose,
    )

    # Print the retrieved candidates
    for i, candidate in enumerate(candidates):
        print(100 * "-")
        print(f"[RETRIEVAL RESULT #{i + 1}, Score: {scores[i]:.4f}]: '{candidate}'")
    print(100 * "-")

    # Print the final answer from the LLM
    print(f"\nQuestion: {args.query}\n")
    print(f"Answer: {answer}\n")


if __name__ == "__main__":
    main()
