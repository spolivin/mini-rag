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
- [x] **Embedding generation** with CPU-friendly models (`sentence-transformers`)
- [x] **Vector search** using *FAISS*
- [x] **Basic retrieval pipeline** (query → embeddings → top-k chunk search)
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

- [x] Use `sentence-transformers/all-MiniLM-L6-v2` for chunk embeddings
- [x] Store embeddings in FAISS index
- [x] Implement similarity search

### Stage 3 - Retrieval pipeline

- [x] Take a user query, embed it, and search for top-k chunks
- [x] Return most relevant context to user
- [ ] Format results into a prompt template

### Stage 4 - Lightweight LLM Integration

- [ ] Integrate a CPU-friendly model
- [ ] Generate answers in retrieved context

### Stage 5 - CLI & Web Interface

- [x] Add a CLI to query documents
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

The script below will preprocess the PDF-document via extracting its text and chunking it depending on `--chunk-size` and `--overlap`. It will then generate embeddings, store them in FAISS index and then conduct top-k similarity search given the provided `--query`, at the end displaying re-ranked chunks considered to be the most relevant to the query.

```bash
python run.py --source-doc=articles/rnn_paper.pdf --query="What is an RNN?" --chunk-size=800 --overlap=100 --top-k=10 --verbose
```
> Make sure to add some PDF document to `articles` folder first as well as set *HuggingFace* token via `hf auth login <HF-TOKEN>`.

By default, "under the hood" the following models are used:

* `sentence-transformers/all-mpnet-base-v2` => Chunks embedding generation (`--embedding-model` flag)
* `cross-encoder/ms-marco-MiniLM-L-6-v2` => Re-ranker of chunks relevance (`--cross-encoder-model` flag)
