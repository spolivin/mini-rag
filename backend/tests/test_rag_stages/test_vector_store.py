"""Integration tests for PgVectorStore.

They need the pgvector database from docker-compose (``docker compose up -d db``) and are
skipped when it is unreachable. The embedding model is replaced with a fast deterministic
fake, so no model is downloaded: equal texts get equal vectors.
"""

import pytest

from rag.config import RAGConfig
from rag.stages import vector_store as vector_store_module
from rag.stages.vector_store import PgVectorStore
from tests.helpers import (
    PARAGRAPHS_A,
    PARAGRAPHS_B,
    FakeSentenceTransformer,
    write_doc,
)

DATABASE_URL = RAGConfig().database_url


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    monkeypatch.setattr(vector_store_module, "SentenceTransformer", FakeSentenceTransformer)


def make_store():
    return PgVectorStore(
        embedding_model="fake",
        chunk_size=60,
        overlap=0,
        max_vectors=10,
        database_url=DATABASE_URL,
    )


def ingest(path, created):
    with make_store() as store:
        store.ingest_document(str(path))
        created.append(store.doc_id)
        return store.doc_id


def test_search_returns_best_match_first(tmp_path, db):
    _, created = db
    path = write_doc(tmp_path, PARAGRAPHS_A)
    with make_store() as store:
        store.ingest_document(str(path))
        created.append(store.doc_id)
        results = store.run_similarity_search(PARAGRAPHS_A[1])
    assert results[0] == PARAGRAPHS_A[1]
    assert len(results) >= len(PARAGRAPHS_A)


def test_reingesting_same_file_is_skipped(tmp_path, db):
    conn, created = db
    path = write_doc(tmp_path, PARAGRAPHS_A)
    doc_id = ingest(path, created)
    count = lambda: conn.execute(
        "SELECT count(*) FROM chunks WHERE doc_id = %s", (doc_id,)
    ).fetchone()[0]
    before = count()
    assert ingest(path, created) == doc_id
    assert count() == before > 0


def test_search_is_scoped_to_the_ingested_document(tmp_path, db):
    _, created = db
    ingest(write_doc(tmp_path, PARAGRAPHS_B), created)
    path_a = write_doc(tmp_path, PARAGRAPHS_A)
    with make_store() as store:
        store.ingest_document(str(path_a))
        created.append(store.doc_id)
        results = store.run_similarity_search(PARAGRAPHS_B[0])
    assert PARAGRAPHS_B[0] not in results
    assert all(any(p in r for p in PARAGRAPHS_A + ["ref "]) for r in results)


def test_search_without_ingest_covers_all_documents(tmp_path, db):
    _, created = db
    path = write_doc(tmp_path, PARAGRAPHS_A)
    doc_id = ingest(path, created)
    with make_store() as store:
        results = store.search_chunks(PARAGRAPHS_A[0])
    assert results[0].text == PARAGRAPHS_A[0]
    assert results[0].doc_id == doc_id
    assert results[0].filename == path.name


def test_search_can_be_restricted_to_given_documents(tmp_path, db):
    _, created = db
    doc_a = ingest(write_doc(tmp_path, PARAGRAPHS_A), created)
    ingest(write_doc(tmp_path, PARAGRAPHS_B), created)
    with make_store() as store:
        results = store.search_chunks(PARAGRAPHS_B[0], doc_ids=[doc_a])
    assert results
    assert {chunk.doc_id for chunk in results} == {doc_a}


def test_ingest_list_and_delete_document(tmp_path, db):
    _, created = db
    path = write_doc(tmp_path, PARAGRAPHS_A)
    with make_store() as store:
        assert store.ingest_document(str(path)) is True
        doc_id = store.doc_id
        created.append(doc_id)
        assert store.ingest_document(str(path)) is False

        docs = {d["doc_id"]: d for d in store.list_documents()}
        assert docs[doc_id]["filename"] == path.name
        assert docs[doc_id]["num_chunks"] >= len(PARAGRAPHS_A)

        assert store.delete_document(doc_id) is True
        assert store.delete_document(doc_id) is False
        assert doc_id not in {d["doc_id"] for d in store.list_documents()}
