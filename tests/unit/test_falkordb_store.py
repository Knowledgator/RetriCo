"""Tests for FalkorDB graph store (mocked client)."""

import pytest
from unittest.mock import MagicMock, patch

from grapsit.models.document import Chunk, Document
from grapsit.models.entity import Entity, EntityMention
from grapsit.models.relation import Relation


@pytest.fixture
def mock_falkordb_store():
    """A mocked FalkorDBGraphStore."""
    from grapsit.store.falkordb_store import FalkorDBGraphStore
    store = FalkorDBGraphStore(host="localhost", port=6379, graph="test")

    mock_graph = MagicMock()
    mock_result = MagicMock()
    mock_result.result_set = []
    mock_graph.query.return_value = mock_result

    # Bypass lazy FalkorDB import by setting internals directly
    store._db = MagicMock()
    store._graph = mock_graph

    yield store, mock_graph


class TestFalkorDBGraphStore:
    def test_write_document(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        doc = Document(id="d1", source="test.txt")
        store.write_document(doc)
        mock_graph.query.assert_called_once()
        call_args = mock_graph.query.call_args
        assert "MERGE" in call_args[0][0]
        params = call_args[0][1]
        assert params == {"id": "d1", "source": "test.txt", "metadata": "{}"}

    def test_write_chunk(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        chunk = Chunk(id="c1", document_id="d1", text="Hello world", index=0)
        store.write_chunk(chunk)
        mock_graph.query.assert_called_once()
        call_args = mock_graph.query.call_args
        assert "MERGE" in call_args[0][0]
        params = call_args[0][1]
        assert params["id"] == "c1"

    def test_write_chunk_document_link(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        store.write_chunk_document_link("c1", "d1")
        mock_graph.query.assert_called_once()
        call_args = mock_graph.query.call_args
        assert "PART_OF" in call_args[0][0]

    def test_write_entity(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        entity = Entity(id="e1", label="Einstein", entity_type="person")
        store.write_entity(entity)
        mock_graph.query.assert_called_once()
        call_args = mock_graph.query.call_args
        assert "MERGE" in call_args[0][0]
        # No APOC — just Entity label
        assert "apoc" not in call_args[0][0].lower()

    def test_write_mention_link(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        mention = EntityMention(text="Einstein", label="person", start=0, end=8, score=0.95)
        store.write_mention_link("e1", "c1", mention)
        mock_graph.query.assert_called_once()
        call_args = mock_graph.query.call_args
        assert "MENTIONED_IN" in call_args[0][0]

    def test_write_relation(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        rel = Relation(
            id="r1", head_text="Einstein", tail_text="Ulm",
            relation_type="born in", score=0.8,
        )
        store.write_relation(rel, "e1", "e2")
        mock_graph.query.assert_called_once()
        call_args = mock_graph.query.call_args
        assert "BORN_IN" in call_args[0][0]

    def test_get_entity_by_id_found(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        mock_node = MagicMock()
        mock_node.properties = {"id": "e1", "label": "Einstein", "entity_type": "person"}
        mock_result = MagicMock()
        mock_result.result_set = [[mock_node]]
        mock_graph.query.return_value = mock_result

        result = store.get_entity_by_id("e1")
        assert result["id"] == "e1"
        assert result["label"] == "Einstein"

    def test_get_entity_by_id_not_found(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        mock_result = MagicMock()
        mock_result.result_set = []
        mock_graph.query.return_value = mock_result

        result = store.get_entity_by_id("e999")
        assert result is None

    def test_get_all_entities(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        node1 = MagicMock()
        node1.properties = {"id": "e1", "label": "Einstein"}
        node2 = MagicMock()
        node2.properties = {"id": "e2", "label": "Curie"}
        mock_result = MagicMock()
        mock_result.result_set = [[node1], [node2]]
        mock_graph.query.return_value = mock_result

        result = store.get_all_entities()
        assert len(result) == 2
        assert result[0]["label"] == "Einstein"
        assert result[1]["label"] == "Curie"

    def test_get_entity_by_label(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        mock_node = MagicMock()
        mock_node.properties = {"id": "e1", "label": "Einstein"}
        mock_result = MagicMock()
        mock_result.result_set = [[mock_node]]
        mock_graph.query.return_value = mock_result

        result = store.get_entity_by_label("Einstein")
        assert result["id"] == "e1"

    def test_get_entity_relations(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        mock_result = MagicMock()
        mock_result.result_set = [
            ["BORN_IN", 0.8, "e2", "Ulm", "c1"],
        ]
        mock_graph.query.return_value = mock_result

        result = store.get_entity_relations("e1")
        assert len(result) == 1
        assert result[0]["relation_type"] == "BORN_IN"
        assert result[0]["target_id"] == "e2"

    def test_get_chunks_for_entity(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        mock_node = MagicMock()
        mock_node.properties = {"id": "c1", "text": "Einstein was born...", "document_id": "d1"}
        mock_result = MagicMock()
        mock_result.result_set = [[mock_node]]
        mock_graph.query.return_value = mock_result

        result = store.get_chunks_for_entity("e1")
        assert len(result) == 1
        assert result[0]["text"] == "Einstein was born..."

    def test_get_subgraph_empty(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        mock_result = MagicMock()
        mock_result.result_set = []
        mock_graph.query.return_value = mock_result

        result = store.get_subgraph(["e1"], max_hops=1)
        assert result == {"entities": [], "relations": []}

    def test_get_subgraph_with_data(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        node1 = MagicMock()
        node1.properties = {"id": "e1", "label": "Einstein"}
        node2 = MagicMock()
        node2.properties = {"id": "e2", "label": "Ulm"}
        mock_result = MagicMock()
        mock_result.result_set = [
            [[node1, node2], [["e1", "e2", "BORN_IN", 0.8]]]
        ]
        mock_graph.query.return_value = mock_result

        result = store.get_subgraph(["e1"], max_hops=1)
        assert len(result["entities"]) == 2
        assert len(result["relations"]) == 1
        assert result["relations"][0]["type"] == "BORN_IN"

    def test_clear_all(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        store.clear_all()
        mock_graph.query.assert_called_once()
        assert "DETACH DELETE" in mock_graph.query.call_args[0][0]

    def test_setup_indexes(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        store.setup_indexes()
        # Should create 5 indexes
        assert mock_graph.query.call_count == 5

    def test_setup_indexes_tolerates_errors(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        mock_graph.query.side_effect = Exception("Index already exists")
        # Should not raise
        store.setup_indexes()

    def test_close(self, mock_falkordb_store):
        store, _ = mock_falkordb_store
        assert store._graph is not None
        store.close()
        assert store._graph is None
        assert store._db is None

    def test_node_to_dict_with_dict(self):
        from grapsit.store.falkordb_store import _node_to_dict
        d = {"id": "e1", "label": "test"}
        assert _node_to_dict(d) == d

    def test_node_to_dict_with_node(self):
        from grapsit.store.falkordb_store import _node_to_dict
        node = MagicMock()
        node.properties = {"id": "e1", "label": "test"}
        result = _node_to_dict(node)
        assert result == {"id": "e1", "label": "test"}

    def test_sanitize_label(self):
        from grapsit.store.falkordb_store import _sanitize_label
        assert _sanitize_label("born in") == "BORN_IN"
        assert _sanitize_label("works-at") == "WORKS_AT"
        assert _sanitize_label("123test") == "_123TEST"
        assert _sanitize_label("") == "UNKNOWN"

    def test_lazy_connection(self):
        """Verify connection is not established until first query."""
        from grapsit.store.falkordb_store import FalkorDBGraphStore
        store = FalkorDBGraphStore()
        assert store._graph is None
        assert store._db is None


class TestFalkorDBIsBaseGraphStore:
    def test_isinstance(self):
        from grapsit.store.base import BaseGraphStore
        from grapsit.store.falkordb_store import FalkorDBGraphStore
        assert issubclass(FalkorDBGraphStore, BaseGraphStore)

    def test_neo4j_isinstance(self):
        from grapsit.store.base import BaseGraphStore
        from grapsit.store.neo4j_store import Neo4jGraphStore
        assert issubclass(Neo4jGraphStore, BaseGraphStore)
