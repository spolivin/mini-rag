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
- [x] **Lightweight LLM integration** (local small models)
- [x] **Q&A pipeline**
- [x] **CLI interface** for document querying
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
- [x] Implement hashing to keep track of documents with embeddings already computed (taking into account chunk size and overlap)
- [] Integrate FAISS index for embeddings with chunks tracking in SQLite DB

### Stage 3 - Retrieval pipeline

- [x] Take a user query, embed it, and search for top-k chunks
- [x] Return most relevant context to user
- [x] Format results into a prompt template

### Stage 4 - Lightweight LLM Integration

- [x] Integrate a CPU-friendly model
- [x] Generate answers in retrieved context

### Stage 5 - CLI & Web Interface

- [x] Add a CLI to query documents
- [ ] Add a simple Streamlit UI for interactive search

### Stage 6 - Deployment

- [ ] Add `Dockerfile` for reproducible deployment
- [ ] Write documentation and examples
- [x] Add tests for core components

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

The script below runs a Q&A pipeline going through the following stages:

1. Retrieve the text from the source document located in the path passed in `--source-doc` flag, separating the pages into a list of *Langchain*'s `Document` objects.
2. Use the resulting list of `Document`-s to separate them into *Regex*-preprocessed *chunks* in accordance with the values passed in `--chunk-size` and `--overlap`. 
3. Generate embeddings for the created chunks. By default `sentence-transformers/all-mpnet-base-v2` model is used (as determined by `--embedding-model` flag).
4. Build a FAISS index for the generated embeddings.
5. Retrieve top-k chunk candidates for a given query (as determined by `--query` flag).
6. Re-rank chunk candidates obtained in the previous stage with cross-encoder. By default `cross-encoder/ms-marco-MiniLM-L-6-v2` model is used (as determined by `--cross-encoder-model` flag).
7. Compose a prompt using the obtained context and generate an answer with LLM. By default `google/flan-t5-small` model is used (as determined by `--gen-model` flag).

> **Design detail**: Through trial and error, it has been observed that chunk order affects answer quality due to recency bias; therefore the most relevant chunks are injected closest to the query in the prompt. The script below additionally prints out the top-k retrieved chunks in order to get a sense of what kind of information is used during answer generation.

```bash
python run.py --source-doc=articles/rnn_paper.pdf --query="What is an RNN?" --chunk-size=800 --overlap=100 --top-k=10 --verbose
```
> Make sure to add some PDF document to `articles` folder first as well as set *HuggingFace* token via `hf auth login <HF-TOKEN>`.

The most recent addition is the implementation of embeddings tracking via checking whether the document (or better to say, its chunks) has already been embedded. Thus, the script will not run embeddings generation and add them to FAISS index (or building one) if these are already present in the index. In this case not only the document hash is taken into account but also chunk size as well as overlap.

## Current limitations

While this project demonstrates a minimal RAG pipeline running fully on CPU, there are several limitations to be aware of:

* **Small language models** (e.g., `flan-t5-small`) often produce repetitive (unless controlled for) or shallow answers. 
* **Context quality** depends heavily on chunking — some questions may retrieve irrelevant or incomplete passages.
* **Limited preprocessing** — while basic regex cleaning reduces noise, more advanced normalization could improve retrieval.
* **Single-document focus** — current pipeline is built for ingesting one document at a time, without multi-document management.
* **Per-document FAISS indices** - initializing indices for each `document-chunk_size-overlap` leads to the accumulation of a large number of files taking space.

## Future work

* **Model improvements**: integrate larger open LLMs (e.g. LLaMA 2, Mistral) or API-based models for better fluency and accuracy.
* **Better retrieval**: experiment with other re-ranking models to improve relevance of top results.
* **UI integration**: add a simple Streamlit or FastAPI interface to make Q&A interactive.
* **Metadata handling**: extend the pipeline with document hashes (SQLite or JSON) to prevent duplicate ingestion.
* **Multi-document RAG**: scale pipeline to handle multiple sources and filter by document ID.
* **FAISS integration with SQLite**: store chunks for the processed documents in a SQLite database to be retrieved based on embeddings stored in a common single FAISS index.
