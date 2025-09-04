from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_document(path):
    """Extract text from a document based on its file extension.

    Parameters
    ----------
    path : str
        The file path of the document to be loaded.

    Returns
    -------
    list[Document]
        A list of Document objects containing the extracted text.

    Raises
    ------
    ValueError
        If the file type is unsupported.
    """
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext in {".txt", ".md"}:
        loader = TextLoader(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()
