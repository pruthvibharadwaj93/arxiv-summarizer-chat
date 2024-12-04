import faiss

def calculate_novelty_scores(baseline_text_embedding, rag_abstract_embeddings):
    """
    Calculate conceptual novelty scores for all paper embeddings in the FAISS index.

    Parameters:
        conceptual_embedding (numpy.ndarray): Embedding vector for the conceptual novelty text.
        rag_abstract_embeddings (faiss.Index): FAISS index containing embeddings for RAG abstracts.

    Returns:
        list: List of dictionaries containing 'paper_id', 'novelty_score'.
    """
    # Search the FAISS index using the conceptual novelty embedding
    k = rag_abstract_embeddings.ntotal  # Query against all abstracts
    distances, indices = rag_abstract_embeddings.search(baseline_text_embedding, k)

    # Map distances to novelty scores
    novelty_scores = [
        {'paper_id': paper_id, 'novelty_score': distance}
        for paper_id, distance in zip(indices[0], distances[0])
    ]
    
    return novelty_scores