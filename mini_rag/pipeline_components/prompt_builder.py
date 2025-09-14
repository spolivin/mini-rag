class PromptBuilder:
    """Builder of a prompt for a LLM."""

    def __call__(self, context_chunks: list[str], query: str) -> str:
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
