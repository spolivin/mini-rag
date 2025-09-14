from sentence_transformers import CrossEncoder


class Reranker:
    """Class for ranking the results of the retrieval of chunks.

    Args:
        model_name (str): Name of the a CrossEncoder model for re-ranking
    """

    def __init__(self, model_name: str):
        self.cross_encoder = CrossEncoder(model_name_or_path=model_name)

    def __call__(
        self, query: str, candidates: list[str]
    ) -> tuple[list[str], list[float]]:
        """Ranks chunks of texts and computes relevance scores.

        Args:
            query (str): Query to the document.
            candidates (list[str]): List of chunks to be re-ranked.

        Returns:
            tuple[list[str], list[float]]: Collection of ranked chunks and relevance scores.
        """
        # Computing relevance scores
        pairs = [[query, candidate] for candidate in candidates]
        scores = self.cross_encoder.predict(pairs)

        # Re-ordering chunks and scores in accordance with relevance
        ranked_candidates = [
            candidate for _, candidate in sorted(zip(scores, candidates), reverse=True)
        ]
        ranked_scores = [
            score for score, _ in sorted(zip(scores, candidates), reverse=True)
        ]

        return ranked_candidates, ranked_scores
