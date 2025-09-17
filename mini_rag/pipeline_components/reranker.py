from sentence_transformers import CrossEncoder


class Reranker:
    """Class for ranking the results of the retrieval of chunks.

    Args:
        model_name (str): Name of the a CrossEncoder model for re-ranking
    """

    def __init__(self, model_name: str):
        self.cross_encoder = CrossEncoder(model_name_or_path=model_name)

    def __call__(
        self,
        query: str,
        candidates: list[str],
        top_k: int,
    ) -> tuple[list[str], list[float]]:
        """Ranks chunks of texts and computes relevance scores.

        Returns the top-k most relevant text chunks.

        Args:
            query (str): Query to the document.
            candidates (list[str]): List of chunks to be re-ranked.
            top_k (int): Number of top chunks to return.

        Raises:
            RuntimeError: Error raised in case no query-relevant results (chunks) have been found.

        Returns:
            tuple[list[str], list[float]]: Collection of top-k ranked chunks and relevance scores.
        """
        # Computing relevance scores
        pairs = [[query, candidate] for candidate in candidates]
        scores = self.cross_encoder.predict(pairs)

        # Re-ordering chunks and scores in accordance with relevance
        ranking_results = [
            (candidate, score)
            for candidate, score in sorted(
                zip(candidates, scores), key=lambda x: x[1], reverse=True
            )
            if score > 0
        ]

        # Stopping RAG in case no query-relevant results have been extracted
        if ranking_results == []:
            raise RuntimeError(
                "No relevant results have been retrieved for the provided query. RAG aborted."
            )

        # Separating candidate chunks from their relevance scores
        ranked_candidates, ranked_scores = zip(*ranking_results)
        ranked_candidates, ranked_scores = list(ranked_candidates), list(ranked_scores)

        return ranked_candidates[:top_k], ranked_scores[:top_k]
