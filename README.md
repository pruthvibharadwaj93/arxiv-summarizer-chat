# ArXiv RAG Bot

This project integrates Retrieval-Augmented Generation (RAG) and Neo4j for:
- Chatbot: Answers specific questions based on last week's papers.
- Email System: Summarizes and emails top papers weekly.

## Folder Structure
- `backend/`: Contains backend logic for chatbot and email workflows.
- `frontend/`: Streamlit-based chatbot interface.
- `data/`: Stores embeddings, fetched paper data, and logs.
- `tests/`: Unit tests for backend workflows.
- `config/`: Configuration files for Neo4j and email settings.
- `scripts/`: Utility scripts for setup and automation.

## Setup
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd arxiv-rag-bot
