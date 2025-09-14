import re

import nltk

from .downloads import ensure_nltk_resource


def clean_text(text: str) -> str:
    """
    Cleans an input text from irrelevant information.

    Args:
        text (str): Raw text of an article.

    Returns:
        str: Cleaned text.
    """
    # Removing numeric citations: e.g. [4], [3,5,8]
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
    # Removing author-year citations: e.g (Doe et al., 2020)
    text = re.sub(r"\(\s*[A-Z][a-z]+(?:\s+et al\.)?,\s*\d{4}\s*\)", "", text)
    # Removing extra spaces left after cleanup
    text = re.sub(r"\s{2,}", " ", text)
    # Removing non-word characters
    text = re.sub(r"\s+", " ", text)
    # Removing spaces left before punctuation
    text = re.sub(r"\s+([.,!?])", r"\1", text).strip()

    return text


def prettify_answer(summary: str) -> str:
    """
    Corrects the format problems of LLM-generated response.

    Args:
        summary (str): Text of a response.

    Returns:
        str: Prettified text.
    """
    ensure_nltk_resource(resource_id="tokenizers/punkt")
    ensure_nltk_resource(resource_id="tokenizers/punkt_tab")
    # Splitting input into sentences and capitalizing each one
    sentences = nltk.tokenize.sent_tokenize(summary)
    prettified_summary = " ".join(s.capitalize() for s in sentences)
    prettified_summary = re.sub(r"\s+([.,!?])", r"\1", prettified_summary)
    prettified_summary = re.sub(r"\s+", " ", prettified_summary)

    return prettified_summary
