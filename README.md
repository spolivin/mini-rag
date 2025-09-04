# 📖 MiniRAG: Question Answering over PDFs (MVP)

***MiniRAG*** is a lightweight, CPU-friendly Retrieval-Augmented Generation (RAG) system that helps you query your own documents (PDFs, text files, Markdown) using natural language. The goal is to demonstrate how an ML engineer can take raw documents (PDFs), preprocess them into chunks, embed them, and perform semantic search to answer user questions.

The project is in its initial stage so more changes are to be expected.

⚡ **Why this project matters**

* Many people rely on PDFs (papers, manuals, contracts) and struggle with simple keyword search.

* This project shows how to turn those documents into a queryable knowledge base using modern NLP techniques.

* It is built as an MVP (minimum viable product) — simple, reproducible, and extendable.

💡 **Why it’s on GitHub**

* To demonstrate end-to-end system thinking: ingestion → chunking → embeddings → vector search → Q&A.

* To showcase practical trade-offs (fixed chunk size, file hashing for deduplication, FAISS for storage).

* To provide a foundation that can later evolve (UI, multi-document search, different chunking strategies).

This is not meant to be perfect or production-ready. Instead, it’s a clear demonstration of ML engineering skills: building a working system from scratch, making design choices explicit, and leaving room for iteration.

## ✨ Features (Planned & Completed)

- [x] **Document ingestion** (PDF, TXT, MD)
- [x] **Text chunking** (with configurable chunk size and overlap)
- [ ] **Embedding generation** with CPU-friendly models (`sentence-transformers`)
- [ ] **Vector search** using *FAISS*
- [ ] **Basic retrieval pipeline** (query → embeddings → top-k chunk search)
- [ ] **Lightweight LLM integration** (local small models)
- [ ] **Q&A pipeline**
- [ ] **CLI interface** for document querying
- [ ] Optional **Streamlit Web UI**
- [ ] **Docker deployment**

## 🚀 Project Roadmap

### Stage 1 - Ingestion & Chunking

- [x] Implement document loader (PDF → text)
- [x] Implement chunking strategy (with LangChain splitters)

### Stage 2 - Embeddings & Storage

- [ ] Use `sentence-transformers/all-MiniLM-L6-v2` for chunk embeddings
- [ ] Store embeddings in FAISS index with metadata
- [ ] Implement similarity search

### Stage 3 - Retrieval pipeline

- [ ] Take a user query, embed it, and search for top-k chunks
- [ ] Return most relevant context to user
- [ ] Format results into a prompt template

### Stage 4 - Lightweight LLM Integration

- [ ] Integrate a CPU-friendly model
- [ ] Generate answers in retrieved context

### Stage 5 - CLI & Web Interface

- [ ] Add a CLI to query documents
- [ ] Add a simple Streamlit UI for interactive search

### Stage 6 - Deployment

- [ ] Add `Dockerfile` for reproducible deployment
- [ ] Write documentation and examples
- [ ] Add tests for core components

## 🔧 Tech Stack

* **Python** (3.10+)

* **LangChain** (for document loaders & text splitters)

* **Sentence Transformers** (for embeddings)

* **FAISS** (for similarity search)

## Preparation of the environment

We firstly clone the project:

```bash
git clone https://github.com/spolivin/mini-rag.git

cd mini-rag
```

Afterwards, we set up the virtual environment:

```bash
source setup_env.sh
```
> This command uses Conda so make sure you have it installed on your system.

## Current progress

At its starting stage only text retrieval and chunking mechanisms have been implemented which can be tested in this way:

```bash
python run.py articles/article.pdf --chunk-size=300 --overlap=20
```
> Make sure to add some PDF document to `articles` folder first.

This will show how many documents have been extracted as well as how many chunks have been created depending on `--chunk-size` and `--overlap`. Additionally, the first 10 chunks are displayed.

The next step of the project is to generate emdeddings.
