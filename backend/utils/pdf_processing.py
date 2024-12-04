import os
import PyPDF2


def add_text_to_papers(papers):
    """
    Extracts text from PDFs using the `pdf_path` in the papers list 
    and adds the extracted text to the corresponding paper dictionary.

    Parameters:
        papers (list): List of dictionaries containing paper metadata, 
                       including the local `pdf_path` for each paper.

    Returns:
        list: Updated list of papers with extracted text added as `paper["text"]`.
    """
    for paper in papers:
        # Ensure the paper has a valid PDF path
        pdf_path = paper.get("pdf_path")
        if pdf_path and os.path.isfile(pdf_path):  # Check if the file exists
            try:
                # Extract text from the PDF
                with open(pdf_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text()

                # Add the text to the paper dictionary
                paper["text"] = text.strip()
                print(f"Added text for: {paper['title']}")

            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                paper["text"] = None  # Indicate failure to extract text
        else:
            print(f"PDF file not found for: {paper['title']}")
            paper["text"] = None  # Indicate missing file or invalid path

    return papers

