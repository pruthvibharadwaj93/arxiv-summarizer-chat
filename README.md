## ArXiv Paper Ranking and RAG based Question-Answering

### OBJECTIVE

This project aims to fetch all papers related to a certain topic that are posted to Arxiv in a particular month and to build a RAG pipeline to analyse the papers.

### Current Capabilities

Broadly, the following functionalities are implemented in the current version of this repo.

- Fetching Papers Data from Arxiv filtered by month and topic
- Vector Embedding and Similarity Scoring for Ranking papers
- Parsing and Cleaning text data from pdfs
- Chunking, Vector Embedding and RAG pipeline based Question-Answering

### Details

For the purpose of this POC, I have selected the Topic as "RAG" and the month as "Nov 2024". There were 112 papers available with these filters on. 
The Ranking and Question-Answering tasks were performed on this corpus.
For Ranking the papers, I primarily evaluated the papers on their Novelty as compared to the baseline idea of RAG. In particular I considered 4 different Novelty scores and used a weighted sum of these for ranking. The 4 different Novelty scores are

1. Conceptual Novelty
2. Methodological Novelty
3. Technical Novelty
4. Theoretical Novelty

The details about how each of these are defined can be found in ranking.ipynb.

For the QA task using RAG, I considered the Top 5 papers selected according to the Ranking algorithm and asked 4 questions for each of the papers.
The idea is to use the answers as content for monthly automated mails. The Questions asked for each of the papers are:

1. Provide a comprehensive summary of the paper from the info that you have
2. What specific problem does this paper solve?
3. How does the paper solve this problem?
4. What are the next steps or future work suggested in this paper?

More details can be found in RAG.ipynb.
The output can be found in the top5_rag_nov24.pdf file. 

### Next Steps

- Include more parameters like Relevance, etc to finetune and personalise Ranking
- Setup scripts for automating paper recommedation Emails for different topics on a monthly basis
- Build a Chatbot to answer specific questions about the papers
- Build a Paper Recommendation website capable of understanding user's interests and providing personalised recommendations based on latest papers

### Folder Structure

- `backend/`: Contains backend logic for the different modules
- `data/`: Stores embeddings, fetched paper data, and logs
- ranking.ipynb - Contains code for fetching paper data and using that to rank papers
- rag.ipynb - Contains code for RAG based QuestionAnswering
- email.ipynb - Contains code for setting up email recommendation
