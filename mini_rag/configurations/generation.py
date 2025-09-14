from dataclasses import dataclass


@dataclass(frozen=True)
class TextGenerationModel:
    """Default model for answer generation."""

    model_name: str = "google/flan-t5-large"


@dataclass(frozen=True)
class TextGenerationConfig:
    """Default parameter config for answer generation."""

    max_new_tokens: int = 450
    min_new_tokens: int = 30
    do_sample: int = True
    top_p: float = 0.9
    temperature: float = 0.7
    no_repeat_ngram_size: int = 3
