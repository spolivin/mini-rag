from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_documents(docs, chunk_size=500, overlap=50):
    """Chunks a list of Document objects into smaller pieces using a recursive character text splitter.

    Parameters
    ----------
    docs : list[Document]
        The list of Document objects to be chunked.
    chunk_size : int, optional
        The size of each chunk. Default is 500.
    overlap : int, optional
        The overlap between chunks. Default is 50.

    Returns
    -------
    list[Document]
        A list of Document objects containing the chunked text.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return splitter.split_documents(docs)
