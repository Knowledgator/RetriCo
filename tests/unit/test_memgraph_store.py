"""Tests for Memgraph graph store (inherits from Neo4jGraphStore)."""

import pytest
from unittest.mock import MagicMock

from grapsit.models.entity import Entity


@pytest.fixture
def mock_memgraph_store():
    """A mocked MemgraphGraphStore."""
    from grapsit.store.memgraph_store import MemgraphGraphStore
    store = MemgraphGraphStore(uri="bolt://localhost:7687")

    mock_session = MagicMock()
    mock_result = MagicMock()
    mock_result.__iter__ = MagicMock(return_value=iter([]))
    mock_session.run.return_value = mock_result
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=False)

    mock_driver = MagicMock()
    mock_driver.session.return_value = mock_session

    store._driver = mock_driver

    yield store, mock_session


class TestMemgraphInheritance:
    def test_is_subclass_of_neo4j(self):
        from grapsit.store.neo4j_store import Neo4jGraphStore
        from grapsit.store.memgraph_store import MemgraphGraphStore
        assert issubclass(MemgraphGraphStore, Neo4jGraphStore)

    def test_is_subclass_of_base(self):
        from grapsit.store.base import BaseGraphStore
        from grapsit.store.memgraph_store import MemgraphGraphStore
        assert issubclass(MemgraphGraphStore, BaseGraphStore)


class TestMemgraphDefaults:
    def test_default_params(self):
        from grapsit.store.memgraph_store import MemgraphGraphStore
        store = MemgraphGraphStore()
        assert store.uri == "bolt://localhost:7687"
        assert store.user == ""
        assert store.password == ""
        assert store.database == "memgraph"

    def test_lazy_connection(self):
        from grapsit.store.memgraph_store import MemgraphGraphStore
        store = MemgraphGraphStore()
        assert store._driver is None


class TestMemgraphOverrides:
    """Test the methods that differ from Neo4jGraphStore."""

    def test_setup_indexes_uses_memgraph_syntax(self, mock_memgraph_store):
        store, mock_session = mock_memgraph_store
        store.setup_indexes()
        assert mock_session.run.call_count == 6
        # Verify Memgraph CREATE INDEX ON syntax (not Neo4j named constraints)
        first_query = mock_session.run.call_args_list[0][0][0]
        assert "CREATE INDEX ON" in first_query
        # Should NOT contain Neo4j-style IF NOT EXISTS or CONSTRAINT
        for call in mock_session.run.call_args_list:
            query = call[0][0]
            assert "IF NOT EXISTS" not in query
            assert "CONSTRAINT" not in query

    def test_setup_indexes_includes_community(self, mock_memgraph_store):
        store, mock_session = mock_memgraph_store
        store.setup_indexes()
        queries = [c[0][0] for c in mock_session.run.call_args_list]
        assert any("Community" in q for q in queries)

    def test_setup_indexes_tolerates_errors(self, mock_memgraph_store):
        store, mock_session = mock_memgraph_store
        mock_session.run.side_effect = Exception("Index already exists")
        store.setup_indexes()  # should not raise

    def test_write_entity_no_apoc(self, mock_memgraph_store):
        store, mock_session = mock_memgraph_store
        entity = Entity(id="e1", label="Einstein", entity_type="person")
        store.write_entity(entity)
        mock_session.run.assert_called_once()
        query = mock_session.run.call_args[0][0]
        assert "MERGE" in query
        assert "apoc" not in query.lower()
        params = mock_session.run.call_args[0][1]
        assert params["id"] == "e1"
        assert params["label"] == "Einstein"
        assert params["entity_type"] == "person"


class TestMemgraphCommunityDetection:
    """Test detect_communities using MAGE."""

    def test_detect_communities_calls_mage(self, mock_memgraph_store):
        store, mock_session = mock_memgraph_store
        # Mock the MAGE result: list of records with .data() dicts
        record1 = MagicMock()
        record1.data.return_value = {"entity_id": "e1", "community_id": "0"}
        record2 = MagicMock()
        record2.data.return_value = {"entity_id": "e2", "community_id": "0"}
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([record1, record2]))
        mock_session.run.return_value = mock_result

        result = store.detect_communities()
        assert isinstance(result, dict)
        query = mock_session.run.call_args[0][0]
        assert "community_detection.get" in query
        assert "Entity" in query

    def test_detect_communities_leiden_falls_back(self, mock_memgraph_store, caplog):
        """Leiden is not supported by MAGE; should fall back to Louvain with a warning."""
        import logging
        store, mock_session = mock_memgraph_store
        mock_result = MagicMock()
        mock_result.__iter__ = MagicMock(return_value=iter([]))
        mock_session.run.return_value = mock_result

        with caplog.at_level(logging.WARNING, logger="grapsit.store.memgraph_store"):
            result = store.detect_communities(method="leiden")
        assert isinstance(result, dict)
        assert "Leiden" in caplog.text
        query = mock_session.run.call_args[0][0]
        assert "community_detection.get" in query


class TestMemgraphInheritedBehavior:
    """Spot-check that inherited Neo4j methods work through Memgraph."""

    def test_write_document(self, mock_memgraph_store):
        from grapsit.models.document import Document
        store, mock_session = mock_memgraph_store
        doc = Document(id="d1", source="test.txt")
        store.write_document(doc)
        mock_session.run.assert_called_once()
        assert "MERGE" in mock_session.run.call_args[0][0]

    def test_write_relation(self, mock_memgraph_store):
        from grapsit.models.relation import Relation
        store, mock_session = mock_memgraph_store
        rel = Relation(
            id="r1", head_text="Einstein", tail_text="Ulm",
            relation_type="born in", score=0.8,
        )
        store.write_relation(rel, "e1", "e2")
        mock_session.run.assert_called_once()
        assert "BORN_IN" in mock_session.run.call_args[0][0]

    def test_write_community_inherited(self, mock_memgraph_store):
        store, mock_session = mock_memgraph_store
        store.write_community("com1", level=0, title="Science", summary="Scientific entities")
        mock_session.run.assert_called_once()
        query = mock_session.run.call_args[0][0]
        assert "MERGE" in query
        assert "Community" in query

    def test_get_community_members_inherited(self, mock_memgraph_store):
        store, mock_session = mock_memgraph_store
        result = store.get_community_members("com1")
        assert result == []

    def test_clear_all(self, mock_memgraph_store):
        store, mock_session = mock_memgraph_store
        store.clear_all()
        assert "DETACH DELETE" in mock_session.run.call_args[0][0]

    def test_close(self, mock_memgraph_store):
        store, _ = mock_memgraph_store
        assert store._driver is not None
        store.close()
        assert store._driver is None


class TestCreateStoreMemgraph:
    def test_create_store_memgraph(self):
        from grapsit.store import create_store
        from grapsit.store.memgraph_store import MemgraphGraphStore
        store = create_store({
            "store_type": "memgraph",
            "memgraph_uri": "bolt://myhost:7687",
            "memgraph_user": "testuser",
            "memgraph_password": "testpass",
            "memgraph_database": "testdb",
        })
        assert isinstance(store, MemgraphGraphStore)
        assert store.uri == "bolt://myhost:7687"
        assert store.user == "testuser"
        assert store.password == "testpass"
        assert store.database == "testdb"

    def test_create_store_memgraph_defaults(self):
        from grapsit.store import create_store
        store = create_store({"store_type": "memgraph"})
        assert store.uri == "bolt://localhost:7687"
        assert store.user == ""
        assert store.password == ""
        assert store.database == "memgraph"

    def test_create_store_error_message_includes_memgraph(self):
        from grapsit.store import create_store
        with pytest.raises(ValueError, match="memgraph"):
            create_store({"store_type": "sqlite"})
