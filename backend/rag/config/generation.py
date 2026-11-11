from dataclasses import dataclass

EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


@dataclass
class TextGenerationConfig:
    """Default parameter config for answer generation with the Anthropic API.

    Sampling parameters (temperature, top_p, top_k) are intentionally absent: current
    Claude models reject non-default values.

    Attributes:
        max_tokens (int): Upper bound on tokens generated per response. Thinking tokens
            count towards it, so keep it generous. Default is 16000.
        effort (str): Thinking depth and token spend, one of "low", "medium", "high",
            "xhigh", "max". Default is "low", which suits short answers.
    """

    max_tokens: int = 16000
    effort: str = "low"

    def __post_init__(self):
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if self.effort not in EFFORT_LEVELS:
            raise ValueError(f"effort must be one of {', '.join(EFFORT_LEVELS)}")
