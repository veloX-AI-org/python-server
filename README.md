# VeloX - Python Server

The `python-server` is the brain of VeloX. It impliments our advanced AI algorithms, RAG pipelines, Langgraph workflows, handles all the heavy work for document processing and quiz generation.

## Features
* **Advanced RAG Pipeline:** Utilizes LangChain and LangGraph to orchestrate complex, multi-step retrieval algorithms.
* **Vector Database Integration:** Chunks documents and stores embeddings in **Pinecone** for semantic search.
* **"Did I understand the video?":** A module that need YouTube URLs, extracts transcripts, and generates contextual quizzes.
* **Observability:** Fully integrated with **LangSmith** to trace LLM calls, monitor token usage, and debug the LangGraph flows.

## Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-org/python-server.git](https://github.com/your-org/python-server.git)
   cd python-server
   ```

2. Install Dependencies:
    
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your `.env` file.

4. Start server

   ```bash
   uvicorn main:app --reload --port 5000
   ```

That's it! Now start working on it. If you did like to contribute, please open an issue and start working on it.

## Author
~Ankit Ahirwar
