import anthropic


class LLMUnavailableError(RuntimeError):
    """Raised when the language model cannot be reached or rejects the request."""


class LLMWrapper:
    """A wrapper for answering questions with Claude through the Anthropic API.

    Builds a prompt from the retrieved context chunks and the user's query and sends it
    to the Messages API. Credentials are resolved by the SDK from the environment
    (``ANTHROPIC_API_KEY``, or a profile created with ``ant auth login``); they are only
    needed when ``generate`` is called, not when the wrapper is created.

    Args:
        model_name (str): Claude model ID, e.g. "claude-haiku-5-5".
        textgen_params (dict): Generation parameters. Recognized keys are ``max_tokens``
            (default 16000) and ``effort`` (default "low"); other keys are ignored.
        client (anthropic.Anthropic, optional): Pre-configured client, mainly for tests.
    """

    SYSTEM_PROMPT = (
        "You are a concise assistant. "
        "Use the provided context to answer in 4-5 sentences. "
        "Do not include introductions or sections. If not in the context, say 'I don't know.'"
    )

    def __init__(
        self,
        model_name: str,
        textgen_params: dict,
        client: anthropic.Anthropic | None = None,
    ):
        """Initializes the LLMWrapper with the specified model."""
        self.model_name = model_name
        self.max_tokens = textgen_params.get("max_tokens", 16000)
        self.effort = textgen_params.get("effort", "low")
        self.client = client or anthropic.Anthropic()

    def _build_user_message(self, context: list[str], user_query: str) -> str:
        """Builds the user message from the context chunks and the query.

        Chunks arrive ordered from most to least relevant. They are injected in reverse
        order so that the most relevant one sits closest to the question.

        Args:
            context (list[str]): Chunks of context to include.
            user_query (str): User's query.

        Returns:
            str: The formatted user message.
        """
        context_text = "\n\n".join(context[::-1])
        return f"Context:\n{context_text}\n\nQuestion:\n{user_query}"

    def generate(self, user_query: str, context: list[str]) -> str:
        """Generates a response from Claude.

        Args:
            user_query (str): The user's query.
            context (list[str]): Chunks of context to include.

        Raises:
            LLMUnavailableError: If the API request fails or no credentials are configured.

        Returns:
            str: The generated response.
        """
        print("Generating a response from LLM...")

        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                output_config={"effort": self.effort},
                system=self.SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": self._build_user_message(context, user_query),
                    }
                ],
            )
        except anthropic.APIError as error:
            raise LLMUnavailableError(
                f"The language model request failed: {error.message}"
            ) from error
        except TypeError as error:
            # The SDK raises a TypeError at request time when no credentials are configured
            if "authentication method" not in str(error):
                raise
            raise LLMUnavailableError(
                "No Anthropic credentials are configured. Set ANTHROPIC_API_KEY."
            ) from error

        if response.stop_reason == "refusal":
            return "The model declined to answer this request."

        # The response can also contain thinking blocks: keep only the text
        answer = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        return answer or "The model returned an empty answer."
