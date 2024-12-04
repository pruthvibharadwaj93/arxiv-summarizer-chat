## ArXiv Paper Ranking, Summarizing and Email Automation

This project aims to fetch all papers related to a certain topic that are posted to Arxiv in a particular month and to build a RAG pipeline to analyse the papers.
Broadly, the following functionalities are implemented in the current version of this repo.

- Fetching Papers Data from Arxiv filtered by month and topic
- Vector Embedding and Similarity Scoring for Ranking papers
- Parsing and Cleaning text data from pdfs
- Chunking, Vector Embedding and RAG pipeline based Question-Answering


For the purpose of this POC, I have selected the Topic as "RAG" and the month as "Nov 2024". There were 112 papers available with these filters on. The Ranking and Question-Answering tasks were performed on this corpus.

## Next Steps

- Include more parameters like Relevance, etc to finetune and personalise Ranking
- Setup scripts for automating paper recommedation Emails for different topics on a monthly basis
- Build a Chatbot to answer specific questions about the papers
- Build a Paper Recommendation website capable of understanding user's interests and providing personalised recommendations based on latest papers

## Folder Structure

- `backend/`: Contains backend logic for the different modules
- `data/`: Stores embeddings, fetched paper data, and logs
- ranking.ipynb - Contains code for fetching paper data and using that to rank papers
- rag.ipynb - Contains code for RAG based QuestionAnswering
- email.ipynb - Contains code for setting up email recommendation
