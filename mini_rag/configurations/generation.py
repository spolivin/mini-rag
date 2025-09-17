from dataclasses import dataclass


@dataclass(frozen=True)
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
