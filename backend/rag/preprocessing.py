import re


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
