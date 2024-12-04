import re

def remove_metadata_and_noise(text):
    """
    Remove metadata, email addresses, and other noise from the text.
    """
    text = re.sub(r"\S+@\S+\.\S+", "", text)  # Remove email addresses
    text = re.sub(r"arXiv:\d+\.\d+(v\d+)?\s*\[.*?\]", "", text)  # Remove arXiv IDs
    text = re.sub(r"(Received:.*?|Accepted:.*?|Page \d+)", "", text)  # Remove metadata
    text = "\n".join([line for line in text.splitlines() if len(line.split()) > 3])  # Remove short lines
    return text

def fix_line_breaks(text):
    """
    Fix line breaks and combine broken sentences.
    """
    text = re.sub(r"(?<!\.\s)(\n|\r)+", " ", text)  # Replace newlines with spaces within paragraphs
    text = re.sub(r"\s+", " ", text)  # Normalize multiple spaces
    return text.strip()

def clean_special_characters(text):
    """
    Remove special characters and LaTeX-style notations.
    """
    text = re.sub(r"[‘’“”→]", "", text)  # Remove specific symbols
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove non-ASCII characters
    text = re.sub(r"\s+", " ", text)  # Normalize spaces
    return text.strip()

def split_into_chunks(text, chunk_size=1000):
    """
    Split text into chunks of specified size (by word count).
    """
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks


def preprocess_and_chunk_text(raw_text, chunk_size=1000):
    """
    Comprehensive pipeline to clean and chunk text for embedding.
    """
    # Step 1: Remove metadata and noise
    cleaned_text = remove_metadata_and_noise(raw_text)

    # Step 2: Fix line breaks
    cleaned_text = fix_line_breaks(cleaned_text)

    # Step 3: Remove special characters
    cleaned_text = clean_special_characters(cleaned_text)

    # Step 4: Chunk the cleaned text
    #chunks = split_into_chunks(cleaned_text, chunk_size)

    return cleaned_text


def preprocess_all_papers(papers, chunk_size=1000):
    """
    Processes multiple papers and returns a list of chunks for each paper.
    
    Parameters:
        papers (list): A list of dictionaries, where each dictionary contains raw text of a paper under the key "text".
        chunk_size (int): Number of words per chunk (default: 1000).
        
    Returns:
        list: A list of dictionaries, where each dictionary contains:
              - "paper_id": A unique identifier for the paper.
              - "chunks": A list of text chunks from the paper.
    """
    processed_papers = []
    
    for i, paper in enumerate(papers):
        raw_text = paper.get("text", "")
        paper_id = paper['paper_id']  # Use provided ID or generate one
        title = paper['title']
        
        # Preprocess and chunk the text
        text = preprocess_and_chunk_text(raw_text, chunk_size)
        
        # Append the processed result
        processed_papers.append({
            "paper_id": paper_id,
            "title": title,
            "text": text
        })
    
    return processed_papers

def preprocess_text(raw_text):
    """
    Comprehensive pipeline to clean text for embedding.
    """
    # Step 1: Remove metadata and noise
    cleaned_text = remove_metadata_and_noise(raw_text)

    # Step 2: Fix line breaks
    cleaned_text = fix_line_breaks(cleaned_text)

    # Step 3: Remove special characters
    cleaned_text = clean_special_characters(cleaned_text)

    return cleaned_text

def preprocess_all_paper_abstracts(papers):
    """
    Processes multiple papers and returns a list of chunks for each paper.
    
    Parameters:
        papers (list): A list of dictionaries, where each dictionary contains raw text of a paper under the key "text".
        chunk_size (int): Number of words per chunk (default: 1000).
        
    Returns:
        list: A list of dictionaries, where each dictionary contains:
              - "paper_id": A unique identifier for the paper.
              - "chunks": A list of text chunks from the paper.
    """
    processed_papers = []
    
    for i, paper in enumerate(papers):
        raw_text = paper.get("summary", "")
        title = paper.get("title", "")
        paper_id = paper.get("id", f"paper_{i+1}")  # Use provided ID or generate one
        
        # Preprocess and chunk the text
        abstracts = preprocess_text(raw_text)
        
        # Append the processed result
        processed_papers.append({
            "paper_id": paper_id,
            "title":title,
            "abstracts": abstracts
        })
    
    return processed_papers