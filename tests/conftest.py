"""Shared fixtures for tests."""

import pytest
from unittest.mock import MagicMock, patch

from retrico.models.document import Chunk, Document
from retrico.models.entity import EntityMention
from retrico.models.relation import Relation


@pytest.fixture
def sample_texts():
    return [
        "Albert Einstein was born in Ulm, Germany in 1879.",
        "Marie Curie worked at the University of Paris.",
    ]


@pytest.fixture
def sample_documents(sample_texts):
    return [Document(text=t, source=f"doc_{i}") for i, t in enumerate(sample_texts)]


@pytest.fixture
def sample_chunks():
    return [
        Chunk(id="c1", document_id="d1", text="Albert Einstein was born in Ulm, Germany in 1879.", index=0),
        Chunk(id="c2", document_id="d1", text="Marie Curie worked at the University of Paris.", index=1),
    ]


@pytest.fixture
def sample_mentions():
    return [
        [
            EntityMention(text="Albert Einstein", label="person", start=0, end=15, score=0.95, chunk_id="c1"),
            EntityMention(text="Ulm", label="location", start=29, end=32, score=0.85, chunk_id="c1"),
            EntityMention(text="Germany", label="location", start=34, end=41, score=0.9, chunk_id="c1"),
        ],
        [
            EntityMention(text="Marie Curie", label="person", start=0, end=11, score=0.92, chunk_id="c2"),
            EntityMention(text="University of Paris", label="organization", start=27, end=46, score=0.88, chunk_id="c2"),
        ],
    ]


@pytest.fixture
def sample_relations():
    return [
        [
            Relation(
                head_text="Albert Einstein", tail_text="Ulm",
                relation_type="born in", score=0.8, chunk_id="c1",
                head_label="person", tail_label="location",
            ),
        ],
        [
            Relation(
                head_text="Marie Curie", tail_text="University of Paris",
                relation_type="works at", score=0.75, chunk_id="c2",
                head_label="person", tail_label="organization",
            ),
        ],
    ]


@pytest.fixture
def mock_neo4j_store():
    """A mocked Neo4jGraphStore."""
    with patch("retrico.store.graph.neo4j_store.GraphDatabase") as mock_gdb:
        mock_driver = MagicMock()
        mock_session = MagicMock()
        mock_result = MagicMock()
        mock_result.__iter__ = lambda self: iter([])

        mock_session.run.return_value = mock_result
        mock_session.__enter__ = lambda self: self
        mock_session.__exit__ = MagicMock(return_value=False)
        mock_driver.session.return_value = mock_session
        mock_gdb.driver.return_value = mock_driver

        from retrico.store.graph.neo4j_store import Neo4jGraphStore
        store = Neo4jGraphStore()
        yield store


@pytest.fixture
def mock_falkordb_store():
    """A mocked FalkorDBGraphStore."""
    from retrico.store.graph.falkordb_store import FalkorDBGraphStore
    store = FalkorDBGraphStore(host="localhost", port=6379, graph="test")

    mock_graph = MagicMock()
    mock_result = MagicMock()
    mock_result.result_set = []
    mock_graph.query.return_value = mock_result

    # Bypass lazy FalkorDB import by setting internals directly
    store._db = MagicMock()
    store._graph = mock_graph
    yield store
