# 📖 MiniRAG: Question Answering over PDFs (MVP)

***MiniRAG*** is a lightweight Retrieval-Augmented Generation (RAG) system that helps you query your own documents (PDFs, text files, Markdown) using natural language. The goal is to demonstrate how an ML engineer can take raw documents (PDFs), preprocess them into chunks, embed them, and perform semantic search to answer user questions.

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
- [x] **FastAPI deployment**
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
- [x] Integrate FAISS index for embeddings with chunks tracking in SQLite DB

### Stage 3 - Retrieval pipeline

- [x] Take a user query, embed it, and search for top-k chunks
- [x] Return most relevant context to user
- [x] Format results into a prompt template using `apply_chat_template`

### Stage 4 - LLM Integration

- [x] Test on CPU-friendly models (such as `distilgpt2`)
- [x] Generate answers in retrieved context
- [x] Integrate LLaMA 2 7B
- [x] Integrate Mistral 7B Instruct v0.3
- [x] Integrate Gemma 7B

### Stage 5 - CLI & Web Interface

- [x] Add a CLI to query documents
- [ ] Add a simple Streamlit UI for interactive search

### Stage 6 - Deployment

- [x] Create a FastAPI endpoint for sending and querying documents to avoid multiple model reloading 
- [ ] Add `Dockerfile` for reproducible deployment
- [ ] Write documentation and examples
- [x] Add tests for core components

## 🔧 Tech Stack

* **Python** (3.10+)

* **LangChain** (for document loaders & text splitters)

* **Sentence Transformers** (for embeddings)

* **Hugging Face Transformers** (for generating text in RAG-context via LLMs)

* **FAISS** (for similarity search)

* **FastAPI** (for RAG deployment)

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

1. Retrieve the text from the source document, separating the pages into a list of *Langchain*'s `Document` objects.
2. Use the resulting list of `Document`-s to separate them into *Regex*-preprocessed *chunks*.
3. Generate embeddings for the created chunks and record document metadata and chunks in a local SQLite database.
4. Build a FAISS index for the generated embeddings.
5. Retrieve a number of chunk candidates which are most similar for the given query.
6. Re-rank and select top-k chunk candidates obtained in the previous stage with a cross-encoder to single out the most query-relevant chunks.
7. Compose a prompt using the obtained context and generate an answer with LLM.

### Design details

* Through trial and error, it has been observed that chunk order affects answer quality due to *recency bias*; therefore the most relevant chunks are injected closest to the query in the prompt.

* The most useful and important addition to the following project is the integration of FAISS index with SQLite database. More specifically, when running the pipeline, document/chunks metadata as well as the embeddings generated thereof are stored and tracked using a local database and FAISS index. The following directory structure is created:

```
mini-rag/
├── vector_store/
    ├── processed_documents.db
    └── vectors.faiss
```
> Local DB named `processed_documents.db` stores information about processed documents in `documents` table as well as the retrieved chunks in `chunks` table. FAISS index `vectors.faiss` stores embeddings of all documents' chunks that have gone through the pipeline and are used during similarity search.

* Avoiding storing multiple FAISS indices is solved by having one common FAISS index. While supporting working with multiple documents (distinguished by hashes), a problem can easily occur during similarity search due to algorithm retrieving embeddings belonging to other documents. The issue is solved by using "overfetching" to retrieve multiple similar chunks and then filtering stage where only chunks belonging to a specific unique document hash are used.

* In order to keep the RAG system as simplistic as possible and not to overcomplicate it with more functionality (such as support for different embedding dimensions and chunking configurations), the system currently employs fixed default chunking parameters as well as embedding, re-ranking and text generation models which can be found in this [module](./mini_rag/configurations/).

### Pipeline

The pipeline can be launched in one of the following ways. The script below additionally prints out the top-k retrieved chunks in order to get a sense of what kind of information is used during answer generation.

```bash
python run.py --source-doc=articles/rnn_paper.pdf --query="What is an RNN?" --top-k=10
```

or using the config:

```bash
python run.py --config-file=config.yaml
```
> Make sure to add some PDF document to `articles` folder first as well as set *HuggingFace* token via `hf auth login --token <HF-TOKEN>`.

Running the pipeline will create the above directory structure including a SQLite database called `processed_documents.db`. We could optionally take a look at what information has been saved there in `documents` and `chunks` tables:

```bash
# Listing all documents that have gone through the pipeline
sqlite3 vector_store/processed_documents.db < sql_scripts/show_documents.sql

# Showing the first five rows with chunks for each document
sqlite3 vector_store/processed_documents.db < sql_scripts/show_chunks.sql
```

## Current limitations

While this project demonstrates a minimal RAG pipeline running fully on CPU, there are several limitations to be aware of:

* **Context quality** depends heavily on chunking — some questions may retrieve irrelevant or incomplete passages.
* **Limited preprocessing** — while basic regex cleaning reduces noise, more advanced normalization could improve retrieval.

## Future work

* **Better retrieval**: experiment with other re-ranking models to improve relevance of top results.
* **UI integration**: add a simple Streamlit or FastAPI interface to make Q&A interactive.

## Completed goals

* **Metadata handling**: extend the pipeline with document hashes (SQLite or JSON) to prevent duplicate ingestion.
* **Multi-document RAG**: scale pipeline to handle multiple sources and filter by document ID.
* **FAISS integration with SQLite**: store chunks for the processed documents in a SQLite database to be retrieved based on embeddings stored in a common single FAISS index.
* **Model improvements**: integrate larger open LLMs (e.g. LLaMA 2, Mistral) or API-based models for better fluency and accuracy.
