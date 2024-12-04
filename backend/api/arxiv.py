import requests
import xml.etree.ElementTree as ET
import os
import re
import json
from requests.sessions import Session

def fetch_and_parse_arxiv(query="RAG", chunk_size=1, year='2024', month='11'):
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"all:{query}"
    start = 0
    papers = []
    time_period = int(year + month)
    paper_id = 1  # Initialize the paper_id counter

    while int(year + month) == time_period:
        params = {
            "search_query": search_query,
            "start": start,
            "max_results": chunk_size,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        response = requests.get(base_url, params=params)

        if response.status_code != 200:
            print(f"Error fetching papers: {response.status_code}")
            break

        xml_data = response.text
        root = ET.fromstring(xml_data)

        # Iterate over each entry in the response
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            paper = {
                "paper_id": paper_id,  # Assign the current ID
                "title": entry.find("{http://www.w3.org/2005/Atom}title").text.strip(),
                "summary": entry.find("{http://www.w3.org/2005/Atom}summary").text.strip(),
                "published": entry.find("{http://www.w3.org/2005/Atom}published").text.strip(),
                "link": entry.find("{http://www.w3.org/2005/Atom}id").text.strip(),
                "authors": [
                    author.find("{http://www.w3.org/2005/Atom}name").text.strip()
                    for author in entry.findall("{http://www.w3.org/2005/Atom}author")
                ]
            }
            paper_id += 1  # Increment the ID for the next paper

            month = paper['published'][5:7]
            year = paper['published'][0:4]

            if int(year + month) == time_period:
                papers.append(paper)
        start += chunk_size  # Increment start to fetch the next chunk

    return papers


def extract_pdf_links(papers):
    for paper in papers:
        paper_id = paper["link"].split("/")[-1]  # Extract arxiv_id
        paper["pdf_link"] = f"http://arxiv.org/pdf/{paper_id}.pdf"
    return papers


def save_data(data, file_path="data/RAG/text/rag_papers.json"):
    """
    Saves a list of dictionaries to a JSON file.

    Parameters:
        data (list): List of dictionaries to save.
        file_path (str): Path to the JSON file.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Data successfully saved to {file_path}")

def load_data(file_path="data.json"):
    """
    Loads a list of dictionaries from a JSON file.

    Parameters:
        file_path (str): Path to the JSON file.

    Returns:
        list: Loaded list of dictionaries.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Data successfully loaded from {file_path}")
    return data


def download_pdfs_with_session(papers, save_dir="data/", topic="RAG"):
    """
    Downloads PDFs using a persistent session with custom headers.

    Parameters:
        papers (list): List of dictionaries with metadata for papers, including "pdf_link".
        save_dir (str): Base directory to save the downloaded PDFs.
        topic (str): Subdirectory name for the topic.

    Returns:
        list: Updated list of papers with local PDF paths added.
    """
    session = Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": "https://arxiv.org/",
    })

    save_path = os.path.join(save_dir, topic, "Papers")
    os.makedirs(save_path, exist_ok=True)

    for paper in papers:
        safe_title = re.sub(r'[^\w\s-]', '', paper['title'])
        safe_title = re.sub(r'\s+', '_', safe_title)
        pdf_path = os.path.join(save_path, f"{paper['paper_id']:03d}_{safe_title}.pdf")

        try:
            response = session.get(paper["pdf_link"], stream=True)
            response.raise_for_status()

            if "pdf" not in response.headers.get("Content-Type", ""):
                print(f"Warning: URL does not point to a PDF. Skipping: {paper['pdf_link']}")
                paper["pdf_path"] = None
                continue

            with open(pdf_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)

            paper["pdf_path"] = pdf_path
            print(f"Downloaded: {paper['title']}")

        except Exception as e:
            print(f"Error downloading {paper['title']}: {e}")
            paper["pdf_path"] = None



def download_pdfs(papers, save_dir="data/", topic="RAG"):
    """
    Downloads PDFs from the provided links and saves them locally.

    Parameters:
        papers (list): List of dictionaries with metadata for papers, including "pdf_link".
        save_dir (str): Base directory to save the downloaded PDFs.
        topic (str): Subdirectory name for the topic.

    Returns:
        list: Updated list of papers with local PDF paths added.
    """
    # Ensure the save directory exists
    save_path = os.path.join(save_dir, topic, "Papers")
    os.makedirs(save_path, exist_ok=True)

    for paper in papers:
        # Derive a safe filename by removing invalid characters
        safe_title = re.sub(r'[^\w\s-]', '', paper['title'])  # Remove special characters
        safe_title = re.sub(r'\s+', '_', safe_title)  # Replace spaces with underscores
        pdf_path = os.path.join(save_path, f"{paper['paper_id']:04d}_{safe_title}.pdf")

        try:
            # Fetch the PDF
            pdf_url = paper["pdf_link"]
            headers = {"User-Agent": "Mozilla/5.0"}  # Add headers to mimic a browser
            response = requests.get(pdf_url, stream=True, headers=headers, allow_redirects=True)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Check content type to ensure it's a PDF
            content_type = response.headers.get("Content-Type", "")
            if "pdf" not in content_type:
                print(f"Warning: URL does not point to a PDF. Skipping: {pdf_url}")
                paper["pdf_path"] = None
                continue

            # Write the PDF to the specified path
            with open(pdf_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)

            # Update the paper metadata with the local PDF path
            paper["pdf_path"] = pdf_path
            print(f"Downloaded: {paper['title']}")

        except requests.exceptions.RequestException as e:
            print(f"Error downloading {paper['title']}: {e}")
            paper["pdf_path"] = None  # Indicate failure to download

