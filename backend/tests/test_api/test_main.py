"""Tests for the FastAPI service.

The embedding and re-ranking models are replaced with fakes (no downloads) and the LLM is
the simulated one. Tests that touch documents need the pgvector database
(``docker compose up -d db``) and are skipped without it.
"""

import pytest
from fastapi.testclient import TestClient

import rag.pipeline as rag_pipeline
from app.main import app
from app.routers import query as query_router
from rag import LLMUnavailableError
from tests.helpers import (
    PARAGRAPHS_A,
    PARAGRAPHS_B,
    FakeLLMWrapper,
    FakeReRanker,
    FakeSentenceTransformer,
    write_doc,
)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(rag_pipeline, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(rag_pipeline, "ReRanker", FakeReRanker)
    monkeypatch.setattr(rag_pipeline, "LLMWrapper", FakeLLMWrapper)
    with TestClient(app) as test_client:  # runs the lifespan (builds the pipeline)
        yield test_client


def upload(client, path, name=None):
    return client.post(
        "/documents",
        files={"file": (name or path.name, path.read_bytes(), "text/plain")},
    )


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_document_lifecycle_and_query(client, db, tmp_path):
    _, created = db
    path = write_doc(tmp_path, PARAGRAPHS_A)

    # upload
    response = upload(client, path)
    assert response.status_code == 200
    body = response.json()
    doc_id = body["doc_id"]
    created.append(doc_id)
    assert body["created"] is True
    assert body["filename"] == path.name

    # uploading the same file again is a no-op
    again = upload(client, path).json()
    assert again["created"] is False
    assert again["doc_id"] == doc_id

    # listing
    documents = {doc["doc_id"]: doc for doc in client.get("/documents").json()}
    assert documents[doc_id]["filename"] == path.name
    assert documents[doc_id]["num_chunks"] >= 1

    # query
    response = client.post(
        "/query", json={"query": PARAGRAPHS_A[0], "document_ids": [doc_id], "top_k": 3}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["answer"].startswith("[Simulated answer]")
    assert 1 <= len(result["chunks"]) <= 3
    top = result["chunks"][0]
    assert PARAGRAPHS_A[0] in top["text"]
    assert top["doc_id"] == doc_id
    assert top["filename"] == path.name
    assert isinstance(top["score"], float)

    # deletion
    assert client.delete(f"/documents/{doc_id}").status_code == 204
    assert client.delete(f"/documents/{doc_id}").status_code == 404
    assert doc_id not in {doc["doc_id"] for doc in client.get("/documents").json()}


def test_query_can_be_restricted_to_documents(client, db, tmp_path):
    _, created = db
    doc_a = upload(client, write_doc(tmp_path, PARAGRAPHS_A)).json()["doc_id"]
    doc_b = upload(client, write_doc(tmp_path, PARAGRAPHS_B)).json()["doc_id"]
    created.extend([doc_a, doc_b])

    response = client.post(
        "/query", json={"query": PARAGRAPHS_B[0], "document_ids": [doc_a]}
    )
    chunks = response.json()["chunks"]
    assert chunks
    assert {chunk["doc_id"] for chunk in chunks} == {doc_a}


def test_query_without_matches_returns_message(client, db):
    response = client.post(
        "/query", json={"query": "anything", "document_ids": ["does-not-exist"]}
    )
    assert response.status_code == 200
    assert response.json() == {"answer": query_router.NO_RESULTS_MESSAGE, "chunks": []}


def test_same_name_different_content_conflicts(client, db, tmp_path):
    _, created = db
    first = write_doc(tmp_path, PARAGRAPHS_A)
    second = write_doc(tmp_path, PARAGRAPHS_B)

    response = upload(client, first)
    created.append(response.json()["doc_id"])
    assert upload(client, second, name=first.name).status_code == 409


def test_upload_rejects_unsupported_file_type(client):
    response = client.post("/documents", files={"file": ("malware.exe", b"x")})
    assert response.status_code == 400


@pytest.mark.parametrize("body", [{"query": ""}, {"query": "q", "top_k": 0}])
def test_query_validation(client, body):
    assert client.post("/query", json=body).status_code == 422


def test_llm_failure_returns_502(client, monkeypatch):
    def fail(*args, **kwargs):
        raise LLMUnavailableError("The language model request failed: boom")

    monkeypatch.setattr(client.app.state.pipeline, "ask", fail)
    response = client.post("/query", json={"query": "q"})
    assert response.status_code == 502
    assert "boom" in response.json()["detail"]


def test_database_unavailable_returns_503(client, monkeypatch):
    pipeline = client.app.state.pipeline
    monkeypatch.setattr(pipeline.config, "database_url", "postgresql://rag:rag@localhost:1/rag")
    response = client.get("/documents")
    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]
