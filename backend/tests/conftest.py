import psycopg
import pytest

from rag.config import RAGConfig


@pytest.fixture
def db():
    """Connection to the pgvector database; skips the test when it is unreachable.

    Yields ``(connection, created)``. Tests append the IDs of the documents they create to
    ``created`` and they are deleted afterwards (chunks go with them, ON DELETE CASCADE).
    """
    try:
        conn = psycopg.connect(
            RAGConfig().database_url, connect_timeout=2, autocommit=True
        )
    except psycopg.OperationalError:
        pytest.skip("pgvector database is not reachable (docker compose up -d db)")
    created: list[str] = []
    yield conn, created
    for doc_id in created:
        conn.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
    conn.close()
