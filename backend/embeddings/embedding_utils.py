import faiss
import numpy as np
import json
#from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import time
import os

from langchain_community.vectorstores import FAISS
from langchain.schema import Document



def generate_gemini_embeddings(data, model_name="models/text-embedding-004", retries=3, retry_delay=2):
    """
    Generate embeddings for the given text data using Gemini's embedding model with retry logic.

    Parameters:
        data (list): List of dictionaries with 'paper_id', 'title', and 'abstracts'.
        model_name (str): Gemini model name for embedding generation.
        retries (int): Number of retry attempts if an embedding generation fails.
        retry_delay (int): Delay in seconds between retry attempts.

    Returns:
        list: List of dictionaries containing 'paper_id', 'embedding', and 'title'.
    """
    embeddings = []

    for paper in data:
        combined_text = f"{paper['title']} {paper['abstracts']}".replace("\n", " ")
        success = False  # Flag to track success
        attempt = 0

        while not success and attempt < retries:
            try:
                # Generate embedding using the Gemini API
                response = genai.embed_content(
                    model_name,
                    content=combined_text
                )

                # Extract embedding vector
                embedding = response['embedding']

                embeddings.append({
                    "paper_id": paper['paper_id'],
                    "title": paper['title'],
                    "embedding": np.array(embedding)  # Convert to NumPy array for consistency
                })

                print(f"Generated embedding for: {paper['title']}")
                success = True  # Mark as successful

            except Exception as e:
                attempt += 1
                print(f"Error generating embedding for {paper['title']} (Attempt {attempt}/{retries}): {e}")
                if attempt < retries:
                    time.sleep(retry_delay)  # Wait before retrying

        if not success:
            print(f"Failed to generate embedding for {paper['title']} after {retries} attempts.")

    return embeddings



def generate_gemini_chunk_embeddings(data, model_name="models/text-embedding-004", retries=3, retry_delay=2):
    """
    Generate embeddings for chunks of text data using Gemini's embedding model with retry logic.

    Parameters:
        data (list): List of dictionaries with 'paper_id' and 'chunks' (list of text chunks).
        model_name (str): Gemini model name for embedding generation.
        retries (int): Number of retry attempts if an embedding generation fails.
        retry_delay (int): Delay in seconds between retry attempts.

    Returns:
        list: List of dictionaries containing 'paper_id' and 'chunk_embeddings', where each 'chunk_embeddings'
              is a list of dictionaries with 'chunk' and 'embedding'.
    """
    embeddings = []

    for paper in data:
        paper_id = paper["paper_id"]
        chunk_embeddings = []

        for chunk in paper["chunks"]:
            success = False  # Flag to track success
            attempt = 0

            while not success and attempt < retries:
                try:
                    # Generate embedding using the Gemini API
                    response = genai.embed_content(
                        model_name,
                        content=chunk
                    )

                    # Extract embedding vector
                    embedding = response['embedding']

                    chunk_embeddings.append({
                        "chunk": chunk,
                        "embedding": np.array(embedding)  # Convert to NumPy array for consistency
                    })

                    print(f"Generated embedding for chunk in paper {paper_id}")
                    success = True  # Mark as successful

                except Exception as e:
                    attempt += 1
                    print(f"Error generating embedding for chunk in paper {paper_id} (Attempt {attempt}/{retries}): {e}")
                    if attempt < retries:
                        time.sleep(retry_delay)  # Wait before retrying

            if not success:
                print(f"Failed to generate embedding for a chunk in paper {paper_id} after {retries} attempts.")

        # Append the embeddings for this paper
        embeddings.append({
            "paper_id": paper_id,
            "chunk_embeddings": chunk_embeddings
        })

    return embeddings


def prepare_vector_store(chunk_embeddings, embedding_dim=768):
    """
    Create a FAISS vector store from precomputed chunk embeddings.

    Parameters:
        chunk_embeddings (list): List of dictionaries with 'paper_id' and 'chunk_embeddings'.
        embedding_dim (int): Dimensionality of the embeddings.

    Returns:
        FAISS: A FAISS vector store populated with precomputed embeddings.
    """
    # Initialize FAISS index
    index = faiss.IndexFlatL2(embedding_dim)
    
    # Prepare lists for texts and their metadata
    texts = []
    metadatas = []
    embeddings_list = []

    # Process each paper and its chunks
    for paper in chunk_embeddings:
        for chunk_data in paper["chunk_embeddings"]:
            # Store the embedding
            embeddings_list.append(chunk_data["embedding"])
            # Store the text
            texts.append(chunk_data["chunk"])
            # Store metadata
            metadatas.append({
                "chunk": chunk_data["chunk"],
                "paper_id": paper["paper_id"]
            })

    # Convert embeddings list to numpy array and add to FAISS index
    embeddings_array = np.array(embeddings_list)
    index.add(embeddings_array)

    # Create the FAISS vector store
    vector_store = FAISS(
        embeddings=None,  # No embedding model needed since we're using precomputed embeddings
        index=index,
        docstore=None,
        index_to_docstore_id={i: str(i) for i in range(len(texts))},
        texts=texts,
        metadatas=metadatas
    )
    
    return vector_store



def generate_prompt_embedding(data, model_name="models/text-embedding-004"):
    """
    Generate embeddings for the given text data using Gemini's embedding model.

    Parameters:
        data (list): List of dictionaries with 'paper_id', 'title', and 'abstracts'.
        model_name (str): Gemini model name for embedding generation.

    Returns:
        list: List of dictionaries containing 'paper_id', 'embedding', and 'title'.
    """


    # Generate embedding using the Gemini API
    response = genai.embed_content(
        model_name,
        content=data)

    # Extract embedding vector
    embedding = response['embedding']

    print(f"Generated embedding")

    return embedding


# def generate_embeddings(data,model_name="avsolatorio/NoInstruct-small-Embedding-v0"):
#     """
#     Generate embeddings for the given text data.

#     Parameters:
#         data (list): List of dictionaries with 'paper_id', 'title', and 'abstracts'.
#         model_name (str): SentenceTransformer model name for embedding generation.

#     Returns:
#         list: List of dictionaries containing 'paper_id', 'embedding', and 'title'.
#     """
#     model = SentenceTransformer(model_name)
#     embeddings = []

#     for paper in data:

#         # Combine title and abstract
#         combined_text = f"{paper['title']} {paper['abstracts']}"

#         # Generate embedding for the abstract
#         embedding = model.encode(combined_text, convert_to_numpy=True)
#         embeddings.append({
#             "paper_id": paper['paper_id'],
#             "title": paper['title'],
#             "embedding": embedding
#         })
    
#     return embeddings



def save_embeddings_npy(embeddings, topic= "RAG", abstract = True):
    """
    Save embeddings to a file along with metadata.

    Parameters:
        embeddings (list): List of dictionaries containing 'paper_id', 'title', and 'embedding'.
        file_path (str): Path to save the embedding vectors.
        metadata_file (str): Path to save the metadata (paper IDs and titles).

    Returns:
        None
    """
    if abstract == True:
        save_path="./data/"+topic+"/embeddings/"+topic+"_abstract_embeddings.npy"
    else:
        save_path="./data/"+topic+"/embeddings/"+topic+"_embeddings.npy"
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Separate embeddings and metadata
    embedding_vectors = []
    metadata = []
    for record in embeddings:
        embedding_vectors.append(record['embedding'])
        metadata.append({
            "paper_id": record['paper_id'],
            "title": record['title']
        })

    metadata_path = save_path.replace(".npy", "_metadata.json")

    # Save embedding vectors as a NumPy array
    np.save(save_path, np.array(embedding_vectors))

    # Save metadata as JSON
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    print(f"Saved {len(embeddings)} embeddings to {save_path} and metadata to {metadata_path}.")

def load_embeddings_npy(topic= "RAG", abstract = True):
    """
    Load embeddings from a file along with metadata.

    Parameters:
        file_path (str): Path to the saved embedding vectors.
        metadata_file (str): Path to the saved metadata (paper IDs and titles).

    Returns:
        list: List of dictionaries containing 'paper_id', 'title', and 'embedding'.
    """
    if abstract == True:
        save_path="./data/"+topic+"/embeddings/"+topic+"_abstract_embeddings.npy"
    else:
        save_path="./data/"+topic+"/embeddings/"+topic+"_embeddings.npy"

    metadata_path = save_path.replace(".npy", "_metadata.json")

    # Load embedding vectors
    embedding_vectors = np.load(save_path)

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Reconstruct the list of embeddings
    embeddings = []
    for i, record in enumerate(metadata):
        embeddings.append({
            "paper_id": record["paper_id"],
            "title": record["title"],
            "embedding": embedding_vectors[i]
        })

    print(f"Loaded {len(embeddings)} embeddings from {save_path}.")
    return embeddings



def store_embeddings_faiss(embeddings, save_path):
    """
    Store embeddings in a FAISS index.

    Parameters:
        embeddings (list): List of dictionaries containing 'paper_id', 'title', and 'embedding'.
        save_path (str): Path to save the FAISS index.
    """
    
    # Extract embeddings and metadata
    embedding_matrix = np.array([item['embedding'] for item in embeddings], dtype='float32')
    metadata = [{"paper_id": item['paper_id'], "title": item['title']} for item in embeddings]

    # Initialize FAISS index
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Use L2 distance metric
    index.add(embedding_matrix)  # Add embeddings to the index

    # Save the FAISS index
    faiss.write_index(index, save_path)

    # Save metadata
    metadata_path = save_path.replace(".faiss", "_metadata.json")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    
    print(f"FAISS index saved to {save_path}")
    print(f"Metadata saved to {metadata_path}")

def process_and_store(data, model_name="models/text-embedding-004", topic = "RAG", abstract = True, retries=3, retry_delay=2):
    """
    Full pipeline to process input data, generate embeddings, and store in FAISS.

    Parameters:
        data (list): List of dictionaries with 'paper_id', 'title', and 'abstracts'.
        model_name (str): SentenceTransformer model name for embedding generation.
        save_path (str): Path to save the FAISS index.
    """
    if abstract == True:
        save_path="./data/"+topic+"/embeddings/"+topic+"_abstract_vector_index.faiss"
    else:
        save_path="./data/"+topic+"/embeddings/"+topic+"_vector_index.faiss"
    # Step 1: Generate embeddings
    embeddings = generate_gemini_embeddings(data, model_name, retries, retry_delay)

    # Step 2: Store embeddings in FAISS
    store_embeddings_faiss(embeddings, save_path)
    print("Pipeline completed successfully!")

    return embeddings

