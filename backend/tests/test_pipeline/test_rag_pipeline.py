"""Tests for RAGPipeline with fake embedding/re-ranking models and the real database.

They need the pgvector database (``docker compose up -d db``) and are skipped without it.
The LLM is the simulated one, so no API is called.
"""

import pytest

import rag.pipeline as rag_pipeline
from rag.config import RAGConfig
from rag import NoRelevantResultsError
from tests.helpers import (
    PARAGRAPHS_A,
    PARAGRAPHS_B,
    FakeLLMWrapper,
    FakeReRanker,
    FakeSentenceTransformer,
    write_doc,
)


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(rag_pipeline, "SentenceTransformer", FakeSentenceTransformer)
    monkeypatch.setattr(rag_pipeline, "ReRanker", FakeReRanker)
    monkeypatch.setattr(rag_pipeline, "LLMWrapper", FakeLLMWrapper)
    return rag_pipeline.RAGPipeline(RAGConfig(chunk_size=60, overlap=0))


def test_ingest_then_ask(pipeline, db, tmp_path):
    _, created = db
    path = write_doc(tmp_path, PARAGRAPHS_A)

    doc_id, was_created = pipeline.ingest(path)
    created.append(doc_id)
    assert was_created is True
    assert pipeline.ingest(path) == (doc_id, False)

    answer, chunks, scores = pipeline.ask(PARAGRAPHS_A[0])
    assert answer.startswith("[Simulated answer]")
    assert chunks[0].text == PARAGRAPHS_A[0]
    assert chunks[0].doc_id == doc_id
    assert chunks[0].filename == path.name
    assert len(scores) == len(chunks)


def test_ask_can_be_restricted_to_documents(pipeline, db, tmp_path):
    _, created = db
    doc_a, _ = pipeline.ingest(write_doc(tmp_path, PARAGRAPHS_A))
    doc_b, _ = pipeline.ingest(write_doc(tmp_path, PARAGRAPHS_B))
    created.extend([doc_a, doc_b])

    _, chunks, _ = pipeline.ask(PARAGRAPHS_B[0], document_ids=[doc_a])
    assert chunks
    assert {chunk.doc_id for chunk in chunks} == {doc_a}


def test_ask_for_unknown_document_raises(pipeline, db):
    with pytest.raises(NoRelevantResultsError):
        pipeline.ask("anything", document_ids=["does-not-exist"])


def test_call_ingests_and_answers_about_that_document(pipeline, db, tmp_path):
    _, created = db
    path = write_doc(tmp_path, PARAGRAPHS_A)

    answer, texts, scores = pipeline(path, PARAGRAPHS_A[1], top_k=2)
    created.append(f"{path.stem.lower()}_{pipeline.config.chunk_size}_{pipeline.config.overlap}")

    assert answer.startswith("[Simulated answer]")
    assert texts[0] == PARAGRAPHS_A[1]
    assert len(texts) == len(scores) == 2


def test_list_and_delete_documents(pipeline, db, tmp_path):
    _, created = db
    doc_id, _ = pipeline.ingest(write_doc(tmp_path, PARAGRAPHS_A))
    created.append(doc_id)

    assert doc_id in {doc["doc_id"] for doc in pipeline.list_documents()}
    assert pipeline.delete_document(doc_id) is True
    assert pipeline.delete_document(doc_id) is False
    assert doc_id not in {doc["doc_id"] for doc in pipeline.list_documents()}
