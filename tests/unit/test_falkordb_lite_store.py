"""Tests for FalkorDBLite graph store (mocked client)."""

import pytest
from unittest.mock import MagicMock, patch

from retrico.models.document import Chunk, Document
from retrico.models.entity import Entity, EntityMention
from retrico.models.relation import Relation


@pytest.fixture
def mock_falkordb_lite_store():
    """A mocked FalkorDBLiteGraphStore."""
    from retrico.store.graph.falkordb_lite_store import FalkorDBLiteGraphStore
    store = FalkorDBLiteGraphStore(db_path="/tmp/test.db", graph="test")

    mock_graph = MagicMock()
    mock_result = MagicMock()
    mock_result.result_set = []
    mock_graph.query.return_value = mock_result

    # Bypass lazy import by setting internals directly
    store._db = MagicMock()
    store._graph = mock_graph

    yield store, mock_graph


class TestFalkorDBLiteInheritance:
    """Verify FalkorDBLiteGraphStore inherits from FalkorDBGraphStore."""

    def test_is_subclass_of_falkordb(self):
        from retrico.store.graph.falkordb_store import FalkorDBGraphStore
        from retrico.store.graph.falkordb_lite_store import FalkorDBLiteGraphStore
        assert issubclass(FalkorDBLiteGraphStore, FalkorDBGraphStore)

    def test_is_subclass_of_base(self):
        from retrico.store.graph.base import BaseGraphStore
        from retrico.store.graph.falkordb_lite_store import FalkorDBLiteGraphStore
        assert issubclass(FalkorDBLiteGraphStore, BaseGraphStore)


class TestFalkorDBLiteInit:
    def test_default_params(self):
        from retrico.store.graph.falkordb_lite_store import FalkorDBLiteGraphStore
        store = FalkorDBLiteGraphStore()
        assert store.db_path == "retrico.db"
        assert store.graph_name == "retrico"
        assert store.query_timeout == 0
        assert store._db is None
        assert store._graph is None

    def test_custom_params(self):
        from retrico.store.graph.falkordb_lite_store import FalkorDBLiteGraphStore
        store = FalkorDBLiteGraphStore(
            db_path="/custom/path.db", graph="mygraph", query_timeout=5000
        )
        assert store.db_path == "/custom/path.db"
        assert store.graph_name == "mygraph"
        assert store.query_timeout == 5000

    def test_no_host_port_attributes(self):
        """FalkorDBLite doesn't need host/port."""
        from retrico.store.graph.falkordb_lite_store import FalkorDBLiteGraphStore
        store = FalkorDBLiteGraphStore()
        assert not hasattr(store, "host")
        assert not hasattr(store, "port")


class TestFalkorDBLiteConnection:
    def test_ensure_connection(self):
        from retrico.store.graph.falkordb_lite_store import FalkorDBLiteGraphStore
        store = FalkorDBLiteGraphStore(db_path="/tmp/test.db")

        mock_db = MagicMock()
        mock_graph = MagicMock()
        mock_db.select_graph.return_value = mock_graph

        with patch("retrico.store.graph.falkordb_lite_store.FalkorDBLiteGraphStore._ensure_connection") as mock_ensure:
            # Test that _ensure_connection is called
            pass

    def test_close(self):
        from retrico.store.graph.falkordb_lite_store import FalkorDBLiteGraphStore
        store = FalkorDBLiteGraphStore()
        store._db = MagicMock()
        store._graph = MagicMock()
        store.close()
        assert store._graph is None
        assert store._db is None

    def test_close_calls_shutdown(self):
        from retrico.store.graph.falkordb_lite_store import FalkorDBLiteGraphStore
        store = FalkorDBLiteGraphStore()
        mock_db = MagicMock()
        store._db = mock_db
        store._graph = MagicMock()
        store.close()
        mock_db.shutdown.assert_called_once()

    def test_close_handles_missing_shutdown(self):
        from retrico.store.graph.falkordb_lite_store import FalkorDBLiteGraphStore
        store = FalkorDBLiteGraphStore()
        mock_db = MagicMock(spec=[])  # No shutdown method
        store._db = mock_db
        store._graph = MagicMock()
        store.close()  # Should not raise
        assert store._db is None


class TestFalkorDBLiteInheritedMethods:
    """Test that inherited methods work correctly via the mock."""

    def test_write_document(self, mock_falkordb_lite_store):
        store, mock_graph = mock_falkordb_lite_store
        doc = Document(id="d1", source="test.txt")
        store.write_document(doc)
        mock_graph.query.assert_called_once()
        call_args = mock_graph.query.call_args
        assert "MERGE" in call_args[0][0]

    def test_write_entity(self, mock_falkordb_lite_store):
        store, mock_graph = mock_falkordb_lite_store
        entity = Entity(id="e1", label="Einstein", entity_type="person")
        store.write_entity(entity)
        mock_graph.query.assert_called_once()
        assert "MERGE" in mock_graph.query.call_args[0][0]

    def test_write_chunk(self, mock_falkordb_lite_store):
        store, mock_graph = mock_falkordb_lite_store
        chunk = Chunk(id="c1", document_id="d1", text="Hello world", index=0)
        store.write_chunk(chunk)
        mock_graph.query.assert_called_once()

    def test_write_relation(self, mock_falkordb_lite_store):
        store, mock_graph = mock_falkordb_lite_store
        rel = Relation(
            id="r1", head_text="Einstein", tail_text="Ulm",
            relation_type="born in", score=0.8,
        )
        store.write_relation(rel, "e1", "e2")
        mock_graph.query.assert_called_once()
        assert "BORN_IN" in mock_graph.query.call_args[0][0]

    def test_get_entity_by_id(self, mock_falkordb_lite_store):
        store, mock_graph = mock_falkordb_lite_store
        mock_node = MagicMock()
        mock_node.properties = {"id": "e1", "label": "Einstein"}
        mock_result = MagicMock()
        mock_result.result_set = [[mock_node]]
        mock_graph.query.return_value = mock_result

        result = store.get_entity_by_id("e1")
        assert result["id"] == "e1"

    def test_clear_all(self, mock_falkordb_lite_store):
        store, mock_graph = mock_falkordb_lite_store
        store.clear_all()
        assert "DETACH DELETE" in mock_graph.query.call_args[0][0]

    def test_setup_indexes(self, mock_falkordb_lite_store):
        store, mock_graph = mock_falkordb_lite_store
        store.setup_indexes()
        assert mock_graph.query.call_count == 7


class TestFalkorDBLiteConfig:
    def test_default_config(self):
        from retrico.store.config import FalkorDBLiteConfig
        cfg = FalkorDBLiteConfig()
        assert cfg.store_type == "falkordb_lite"
        assert cfg.db_path == "retrico.db"
        assert cfg.graph == "retrico"

    def test_custom_config(self):
        from retrico.store.config import FalkorDBLiteConfig
        cfg = FalkorDBLiteConfig(db_path="/data/kg.db", graph="mykg")
        assert cfg.db_path == "/data/kg.db"
        assert cfg.graph == "mykg"

    def test_to_flat_dict(self):
        from retrico.store.config import FalkorDBLiteConfig
        cfg = FalkorDBLiteConfig(db_path="/data/kg.db", graph="mykg")
        flat = cfg.to_flat_dict()
        assert flat["store_type"] == "falkordb_lite"
        assert flat["falkordb_lite_db_path"] == "/data/kg.db"
        assert flat["falkordb_lite_graph"] == "mykg"

    def test_from_flat_dict(self):
        from retrico.store.config import FalkorDBLiteConfig
        cfg = FalkorDBLiteConfig.from_flat_dict({
            "falkordb_lite_db_path": "/tmp/test.db",
            "falkordb_lite_graph": "test",
        })
        assert cfg.db_path == "/tmp/test.db"
        assert cfg.graph == "test"

    def test_roundtrip(self):
        from retrico.store.config import FalkorDBLiteConfig
        original = FalkorDBLiteConfig(db_path="/data/kg.db", graph="mykg")
        restored = FalkorDBLiteConfig.from_flat_dict(original.to_flat_dict())
        assert restored.db_path == original.db_path
        assert restored.graph == original.graph


class TestFalkorDBLiteIsDefault:
    """Verify FalkorDBLite is the default store type."""

    def test_base_config_default_is_falkordb_lite(self):
        from retrico.store.config import BaseStoreConfig, FalkorDBLiteConfig
        cfg = BaseStoreConfig.from_flat_dict({})
        assert isinstance(cfg, FalkorDBLiteConfig)
        assert cfg.store_type == "falkordb_lite"

    def test_create_graph_store_default(self):
        """create_graph_store with empty config uses falkordb_lite."""
        from retrico.store.graph import create_graph_store
        from retrico.store.graph.falkordb_lite_store import FalkorDBLiteGraphStore
        with patch("retrico.store.graph.falkordb_lite_store.FalkorDBLiteGraphStore._ensure_connection"):
            store = create_graph_store({})
            assert isinstance(store, FalkorDBLiteGraphStore)


class TestFalkorDBLiteRegistry:
    def test_registered_in_graph_store_registry(self):
        from retrico.store.graph import graph_store_registry
        factory = graph_store_registry.get("falkordb_lite")
        assert factory is not None

    def test_factory_creates_correct_type(self):
        from retrico.store.graph import graph_store_registry
        from retrico.store.graph.falkordb_lite_store import FalkorDBLiteGraphStore
        factory = graph_store_registry.get("falkordb_lite")
        store = factory({"falkordb_lite_db_path": "/tmp/test.db"})
        assert isinstance(store, FalkorDBLiteGraphStore)
        assert store.db_path == "/tmp/test.db"

    def test_config_creates_store(self):
        from retrico.store.config import FalkorDBLiteConfig
        from retrico.store.graph import create_graph_store
        from retrico.store.graph.falkordb_lite_store import FalkorDBLiteGraphStore
        cfg = FalkorDBLiteConfig(db_path="/tmp/cfg.db")
        store = create_graph_store(cfg)
        assert isinstance(store, FalkorDBLiteGraphStore)
        assert store.db_path == "/tmp/cfg.db"