-- Runs once, when the database volume is first created.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id     TEXT PRIMARY KEY,           -- {filename_stem}_{chunk_size}_{overlap}
    hash       TEXT NOT NULL UNIQUE,       -- SHA-1 of the file, used to skip re-ingestion
    filename   TEXT NOT NULL,
    chunk_size INTEGER NOT NULL,
    overlap    INTEGER NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id   BIGSERIAL PRIMARY KEY,
    doc_id     TEXT NOT NULL REFERENCES documents (doc_id) ON DELETE CASCADE,
    chunk_text TEXT NOT NULL,
    embedding  vector(384) NOT NULL        -- all-MiniLM-L6-v2 produces 384 dimensions
);

CREATE INDEX IF NOT EXISTS chunks_doc_id_idx ON chunks (doc_id);

-- Approximate nearest-neighbour index, cosine distance (query with the <=> operator)
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
