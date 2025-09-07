import nltk


def ensure_nltk_resource(resource_id: str) -> None:
    """Looks for NLTK resource and downloads it if not present.

    Args:
        resource_id (str): Name of NLTK resource.
    """

    try:
        nltk.data.find(resource_id)
    except LookupError:
        nltk.download(resource_id.split("/")[1], quiet=True)
