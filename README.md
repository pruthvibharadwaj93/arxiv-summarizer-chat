## ArXiv Paper Ranking, Summarizing and Email Automation

This project aims to fetch all papers related to a certain topic that are posted to Arxiv in a particular month and to build a RAG pipeline to analyse the papers.
Broadly, the following functionalities are implemented in the current version of this repo
- Fetching Papers Data from Arxiv filtered by month and topic
- Vector Embedding and Similarity Scoring for Ranking papers
- Parsing and Cleaning text data from pdfs
- Chunking, Vector Embedding and RAG pipeline based Question-Answering
- Automated Email System for Paper Recommendation 

## Folder Structure

- `backend/`: Contains backend logic for the different modules
- `data/`: Stores embeddings, fetched paper data, and logs
- ranking.ipynb - Contains code for fetching paper data and using that to rank papers
- rag.ipynb - Contains code for RAG based QuestionAnswering
- email.ipynb - Contains code for setting up email recommendation

## Next Steps

- Build a Chatbot to answer specific questions about the papers
- Setup scripts for automating Email paper recommedation for different topics on a monthly basis
- Build a Paper Recommendation website capable of understanding user's interests
