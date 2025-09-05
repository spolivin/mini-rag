from sentence_transformers import SentenceTransformer

def generate_embeddings(texts, model_name='all-MiniLM-L6-v2'):
    """
    Generates embeddings for a list of texts using a specified SentenceTransformer model.

    Parameters
    ----------
    texts : list[str]
        List of texts to be embedded.
    model_name : str
        Name of the SentenceTransformer model to use.

    Returns
    -------
        nd.array, shape (n_texts, embedding_dim)
            Array of embeddings corresponding to the input texts.
    """
    model = SentenceTransformer(model_name)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    return embeddings
