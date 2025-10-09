from dataclasses import dataclass


@dataclass
class TextGenerationConfig:
    """Default parameter config for answer generation.

    Attributes:
        max_new_tokens (int): Maximum number of tokens to generate in the output sequence. Default is 250.
        min_new_tokens (int): Minimum number of tokens to generate in the output sequence. Default is 30.
        do_sample (int): Whether to use sampling for generation (1 for True, 0 for False). Default is True.
        top_p (float): Cumulative probability for nucleus sampling. Default is 0.9.
        temperature (float): Sampling temperature for diversity. Default is 0.7.
        no_repeat_ngram_size (int): Size of n-grams that should not be repeated in the output. Default is 3.
    """

    max_new_tokens: int = 250
    min_new_tokens: int = 30
    do_sample: int = True
    top_p: float = 0.9
    temperature: float = 0.7
    no_repeat_ngram_size: int = 3

    def __post_init__(self):
        if self.min_new_tokens >= self.max_new_tokens:
            raise ValueError("min_new_tokens must be smaller than max_new_tokens")
        if self.no_repeat_ngram_size <= 0:
            raise ValueError("no_repeat_ngram_size must be a positive integer")
        if not (0.0 < self.top_p <= 1.0):
            raise ValueError("top_p must be in the range (0.0, 1.0]")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be a positive float")
        if not isinstance(self.do_sample, bool):
            raise ValueError("do_sample must be a boolean value")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        if self.min_new_tokens < 0:
            raise ValueError("min_new_tokens must be a non-negative integer")
