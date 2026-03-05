"""Tests for FalkorDB graph store (mocked client)."""

import pytest
from unittest.mock import MagicMock, patch

from grapsit.models.document import Chunk, Document
from grapsit.models.entity import Entity, EntityMention
from grapsit.models.relation import Relation


@pytest.fixture
def mock_falkordb_store():
    """A mocked FalkorDBGraphStore."""
    from grapsit.store.graph.falkordb_store import FalkorDBGraphStore
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
        # Should create 7 indexes (5 original + Community + FTS)
        assert mock_graph.query.call_count == 7

    def test_setup_indexes_includes_community(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        store.setup_indexes()
        queries = [c[0][0] for c in mock_graph.query.call_args_list]
        assert any("Community" in q for q in queries)

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
        from grapsit.store.graph.falkordb_store import _node_to_dict
        d = {"id": "e1", "label": "test"}
        assert _node_to_dict(d) == d

    def test_node_to_dict_with_node(self):
        from grapsit.store.graph.falkordb_store import _node_to_dict
        node = MagicMock()
        node.properties = {"id": "e1", "label": "test"}
        result = _node_to_dict(node)
        assert result == {"id": "e1", "label": "test"}

    def test_sanitize_label(self):
        from grapsit.store.graph.falkordb_store import _sanitize_label
        assert _sanitize_label("born in") == "BORN_IN"
        assert _sanitize_label("works-at") == "WORKS_AT"
        assert _sanitize_label("123test") == "_123TEST"
        assert _sanitize_label("") == "UNKNOWN"

    def test_lazy_connection(self):
        """Verify connection is not established until first query."""
        from grapsit.store.graph.falkordb_store import FalkorDBGraphStore
        store = FalkorDBGraphStore()
        assert store._graph is None
        assert store._db is None


class TestFalkorDBChunkLookups:
    def test_get_entities_for_chunk(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        mock_node = MagicMock()
        mock_node.properties = {"id": "e1", "label": "Einstein"}
        mock_result = MagicMock()
        mock_result.result_set = [[mock_node]]
        mock_graph.query.return_value = mock_result

        result = store.get_entities_for_chunk("c1")
        assert len(result) == 1
        assert result[0]["id"] == "e1"
        query = mock_graph.query.call_args[0][0]
        assert "MENTIONED_IN" in query
        assert "Chunk" in query

    def test_get_entities_for_chunk_empty(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        result = store.get_entities_for_chunk("c999")
        assert result == []

    def test_get_chunk_by_id_found(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        mock_node = MagicMock()
        mock_node.properties = {"id": "c1", "text": "Hello", "document_id": "d1"}
        mock_result = MagicMock()
        mock_result.result_set = [[mock_node]]
        mock_graph.query.return_value = mock_result

        result = store.get_chunk_by_id("c1")
        assert result["id"] == "c1"
        assert result["text"] == "Hello"

    def test_get_chunk_by_id_not_found(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        result = store.get_chunk_by_id("c999")
        assert result is None


class TestFalkorDBPathQueries:
    def test_get_shortest_paths_empty(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        result = store.get_shortest_paths("e1", "e2")
        assert result == []
        query = mock_graph.query.call_args[0][0]
        assert "MATCH" in query

    def test_get_shortest_paths_directed(self, mock_falkordb_store):
        """Directed path found: existence check returns rows, then chain query returns path."""
        store, mock_graph = mock_falkordb_store
        node1 = MagicMock()
        node1.properties = {"id": "e1", "label": "Einstein"}
        node2 = MagicMock()
        node2.properties = {"id": "e2", "label": "Ulm"}

        # First call: existence check (RETURN s, t) -> found
        existence_result = MagicMock()
        existence_result.result_set = [[node1, node2]]
        # Second call: _get_entity_chain length=1 (RETURN s, t, type(r), r.score)
        chain_result = MagicMock()
        chain_result.result_set = [[node1, node2, "BORN_IN", 0.8]]
        mock_graph.query.side_effect = [existence_result, chain_result]

        result = store.get_shortest_paths("e1", "e2")
        assert len(result) == 1
        assert len(result[0]["nodes"]) == 2
        assert result[0]["rels"][0]["type"] == "BORN_IN"
        assert result[0]["rels"][0]["score"] == 0.8

    def test_get_shortest_paths_via_intermediate(self, mock_falkordb_store):
        """No directed path; fallback finds shared intermediate entity."""
        store, mock_graph = mock_falkordb_store
        node1 = MagicMock()
        node1.properties = {"id": "e1", "label": "Einstein"}
        node2 = MagicMock()
        node2.properties = {"id": "e2", "label": "Ulm"}
        mid = MagicMock()
        mid.properties = {"id": "e3", "label": "Germany"}

        empty_result = MagicMock()
        empty_result.result_set = []
        # Directed checks (both directions) return empty
        # Then first fallback query returns intermediate node
        fallback_result = MagicMock()
        fallback_result.result_set = [[node1, mid, node2, "LOCATED_IN", 0.9, "LOCATED_IN", 0.7]]
        mock_graph.query.side_effect = [empty_result, empty_result, fallback_result]

        result = store.get_shortest_paths("e1", "e2")
        assert len(result) == 1
        assert len(result[0]["nodes"]) == 3
        assert result[0]["rels"][0]["type"] == "LOCATED_IN"


class TestFalkorDBCommunityCRUD:
    def test_write_community(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        store.write_community("com1", level=0, title="Science", summary="Scientific entities")
        query = mock_graph.query.call_args[0][0]
        assert "MERGE" in query
        assert "Community" in query
        params = mock_graph.query.call_args[0][1]
        assert params["id"] == "com1"
        assert params["level"] == 0
        assert params["title"] == "Science"

    def test_write_community_membership(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        store.write_community_membership("e1", "com1", level=0)
        query = mock_graph.query.call_args[0][0]
        assert "MEMBER_OF" in query
        assert "Entity" in query
        assert "Community" in query

    def test_get_community_members(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        mock_node = MagicMock()
        mock_node.properties = {"id": "e1", "label": "Einstein"}
        mock_result = MagicMock()
        mock_result.result_set = [[mock_node]]
        mock_graph.query.return_value = mock_result

        result = store.get_community_members("com1")
        assert len(result) == 1
        assert result[0]["id"] == "e1"

    def test_get_all_communities(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        mock_node = MagicMock()
        mock_node.properties = {"id": "com1", "level": 0, "title": "Science"}
        mock_result = MagicMock()
        mock_result.result_set = [[mock_node]]
        mock_graph.query.return_value = mock_result

        result = store.get_all_communities()
        assert len(result) == 1
        assert result[0]["id"] == "com1"

    def test_detect_communities_cdlp(self, mock_falkordb_store):
        store, mock_graph = mock_falkordb_store
        mock_result = MagicMock()
        mock_result.result_set = [["e1", "0"], ["e2", "0"], ["e3", "1"]]
        mock_graph.query.return_value = mock_result

        result = store.detect_communities()

        assert result == {"e1": "0", "e2": "0", "e3": "1"}
        call_args = mock_graph.query.call_args
        assert "algo.labelPropagation" in call_args[0][0]
        assert call_args[0][1]["max_iter"] == 10


class TestFalkorDBMutations:
    """Test FalkorDB mutation methods with controlled side effects."""

    def _make_store_with_run(self, side_effects):
        from grapsit.store.graph.falkordb_store import FalkorDBGraphStore
        store = FalkorDBGraphStore.__new__(FalkorDBGraphStore)
        store._db = MagicMock()
        store._graph = MagicMock()
        store._run = MagicMock(side_effect=side_effects)
        return store

    def _mock_node(self, props):
        node = MagicMock()
        node.properties = props
        return node

    # -- delete_entity -------------------------------------------------------

    def test_delete_entity_found(self):
        store = self._make_store_with_run([
            [self._mock_node({"id": "e1", "label": "Einstein"})],  # get_entity_by_id row
            [],  # DETACH DELETE
        ])
        # FalkorDB get_entity_by_id checks rows[0][0], so wrap in list
        store._run.side_effect = [
            [[self._mock_node({"id": "e1", "label": "Einstein"})]],
            [],
        ]
        assert store.delete_entity("e1") is True

    def test_delete_entity_not_found(self):
        store = self._make_store_with_run([
            [],  # empty result
        ])
        assert store.delete_entity("e999") is False

    # -- delete_relation -----------------------------------------------------

    def test_delete_relation_found(self):
        store = self._make_store_with_run([
            [[1]],  # count = 1
        ])
        assert store.delete_relation("r1") is True

    def test_delete_relation_not_found(self):
        store = self._make_store_with_run([
            [[0]],  # count = 0
        ])
        assert store.delete_relation("r999") is False

    # -- delete_chunk --------------------------------------------------------

    def test_delete_chunk_found(self):
        store = self._make_store_with_run([
            [[self._mock_node({"id": "c1", "text": "hello"})]],  # get_chunk_by_id
            [],  # DETACH DELETE
        ])
        assert store.delete_chunk("c1") is True

    def test_delete_chunk_not_found(self):
        store = self._make_store_with_run([
            [],  # empty
        ])
        assert store.delete_chunk("c999") is False

    # -- update_entity -------------------------------------------------------

    def test_update_entity_label(self):
        node = self._mock_node({"id": "e1", "label": "Old", "properties": "{}"})
        store = self._make_store_with_run([
            [[node]],  # get_entity_by_id
            [],  # SET
        ])
        assert store.update_entity("e1", label="New") is True
        set_call = store._run.call_args_list[1]
        assert "e.label = $label" in set_call[0][0]

    def test_update_entity_not_found(self):
        store = self._make_store_with_run([
            [],  # not found
        ])
        assert store.update_entity("e999", label="X") is False

    # -- add_entity ----------------------------------------------------------

    def test_add_entity_with_id(self):
        store = self._make_store_with_run([
            [],  # CREATE
        ])
        result = store.add_entity("Einstein", "person", id="e-custom")
        assert result == "e-custom"
        query = store._run.call_args[0][0]
        assert "CREATE" in query

    def test_add_entity_generated_id(self):
        store = self._make_store_with_run([
            [],  # CREATE
        ])
        result = store.add_entity("Einstein")
        import uuid
        uuid.UUID(result)  # validates format

    # -- add_relation --------------------------------------------------------

    def test_add_relation_success(self):
        node1 = self._mock_node({"id": "e1"})
        node2 = self._mock_node({"id": "e2"})
        store = self._make_store_with_run([
            [[node1]],  # head exists
            [[node2]],  # tail exists
            [],  # CREATE
        ])
        result = store.add_relation("e1", "e2", "born in", id="r1")
        assert result == "r1"

    def test_add_relation_head_missing(self):
        store = self._make_store_with_run([
            [],  # head not found
        ])
        with pytest.raises(ValueError, match="Head entity"):
            store.add_relation("e1", "e2", "born in")

    # -- merge_entities ------------------------------------------------------

    def test_merge_entities_same_id(self):
        store = self._make_store_with_run([])
        assert store.merge_entities("e1", "e1") is True

    def test_merge_entities_source_missing(self):
        store = self._make_store_with_run([
            [],  # source not found
        ])
        assert store.merge_entities("e1", "e2") is False

    def test_merge_entities_success(self):
        src = self._mock_node({"id": "e1", "label": "A", "properties": "{'x': 1}"})
        tgt = self._mock_node({"id": "e2", "label": "B", "properties": "{'y': 2}"})
        store = self._make_store_with_run([
            [[src]],  # source
            [[tgt]],  # target
            [],  # move MENTIONED_IN
            [],  # move MEMBER_OF
            [],  # outgoing rels (none)
            [],  # incoming rels (none)
            [],  # merge properties
            [],  # delete source
        ])
        assert store.merge_entities("e1", "e2") is True


class TestFalkorDBIsBaseGraphStore:
    def test_isinstance(self):
        from grapsit.store.graph.base import BaseGraphStore
        from grapsit.store.graph.falkordb_store import FalkorDBGraphStore
        assert issubclass(FalkorDBGraphStore, BaseGraphStore)

    def test_neo4j_isinstance(self):
        from grapsit.store.graph.base import BaseGraphStore
        from grapsit.store.graph.neo4j_store import Neo4jGraphStore
        assert issubclass(Neo4jGraphStore, BaseGraphStore)
