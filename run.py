import argparse

import yaml

from mini_rag.configurations import RAGConfig
from mini_rag.pipeline import RAGPipeline

parser = argparse.ArgumentParser(description="RAG script")
parser.add_argument("--source-doc", type=str, help="Path to the document")
parser.add_argument(
    "--query",
    type=str,
    help="Query string to search in the index",
)
parser.add_argument(
    "--top-k", type=int, help="Number of top results to retrieve", default=5
)
parser.add_argument(
    "--config-file", type=str, help="Path to configuration file", default=None
)
args = parser.parse_args()


def main():
    if args.config_file:
        with open(args.config_file) as f:
            config = yaml.safe_load(f)
        source_doc_path = config["source_doc"]
        query = config["query"]
        top_k = config["top_k"]
    else:
        source_doc_path = args.source_doc
        query = args.query
        top_k = args.top_k

    # Running RAG
    config = RAGConfig(gen_model_name="meta-llama/Llama-2-7b-chat-hf", top_k=top_k)
    rag_pipeline = RAGPipeline(rag_config=config)

    # Retieving LLM response, ranked chunks of text and relevance scores for ranked chunks
    answer, ranked_candidates, ranked_scores = rag_pipeline(
        source_doc_path=source_doc_path,
        query=query,
    )

    print("\n" + "=" * 60)
    print("🔎 Retrieved/Ranked Chunks".center(60))
    print("=" * 60)
    for i, (candidate, score) in enumerate(zip(ranked_candidates, ranked_scores), 1):
        print(f"\nRetrieval result #{i} | Relevance score: {score:.4f}")
        print("-" * 60)
        print(candidate.strip())
    print("=" * 60)

    print("\n❓ Question:")
    print(query + "\n")
    print("💡 Answer:")
    print(f"{answer}\n")


if __name__ == "__main__":
    main()
