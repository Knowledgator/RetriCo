"""Tests for Neo4j graph store (mocked driver)."""

import pytest
from unittest.mock import MagicMock, patch, call

from grapsit.models.document import Chunk, Document
from grapsit.models.entity import Entity, EntityMention
from grapsit.models.relation import Relation


class TestNeo4jGraphStore:
    def test_write_document(self, mock_neo4j_store):
        doc = Document(id="d1", source="test.txt")
        mock_neo4j_store.write_document(doc)

    def test_write_chunk(self, mock_neo4j_store):
        chunk = Chunk(id="c1", document_id="d1", text="Hello world", index=0)
        mock_neo4j_store.write_chunk(chunk)

    def test_write_entity(self, mock_neo4j_store):
        entity = Entity(id="e1", label="Einstein", entity_type="person")
        mock_neo4j_store.write_entity(entity)

    def test_write_relation(self, mock_neo4j_store):
        rel = Relation(
            id="r1", head_text="Einstein", tail_text="Ulm",
            relation_type="born in", score=0.8,
        )
        mock_neo4j_store.write_relation(rel, "e1", "e2")

    def test_sanitize_label(self):
        from grapsit.store.graph.neo4j_store import _sanitize_label
        assert _sanitize_label("born in") == "BORN_IN"
        assert _sanitize_label("works-at") == "WORKS_AT"
        assert _sanitize_label("123test") == "_123TEST"
        assert _sanitize_label("") == "UNKNOWN"


class TestNeo4jChunkLookups:
    def test_get_entities_for_chunk(self, mock_neo4j_store):
        mock_neo4j_store.get_entities_for_chunk("c1")

    def test_get_chunk_by_id(self, mock_neo4j_store):
        result = mock_neo4j_store.get_chunk_by_id("c1")
        assert result is None  # empty mock returns no records


class TestNeo4jPathQueries:
    def test_get_shortest_paths(self, mock_neo4j_store):
        result = mock_neo4j_store.get_shortest_paths("e1", "e2")
        assert result == []

    def test_get_shortest_paths_custom_max_length(self, mock_neo4j_store):
        mock_neo4j_store.get_shortest_paths("e1", "e2", max_length=3)


class TestNeo4jCommunityCRUD:
    def test_write_community(self, mock_neo4j_store):
        mock_neo4j_store.write_community("com1", level=0, title="Science", summary="Scientific entities")

    def test_write_community_membership(self, mock_neo4j_store):
        mock_neo4j_store.write_community_membership("e1", "com1", level=0)

    def test_get_community_members(self, mock_neo4j_store):
        result = mock_neo4j_store.get_community_members("com1")
        assert result == []

    def test_get_all_communities(self, mock_neo4j_store):
        result = mock_neo4j_store.get_all_communities()
        assert result == []

    def test_detect_communities(self, mock_neo4j_store):
        # detect_communities calls GDS; just verify it runs without error on mock
        result = mock_neo4j_store.detect_communities()
        assert isinstance(result, dict)

    def test_detect_communities_leiden(self, mock_neo4j_store):
        result = mock_neo4j_store.detect_communities(method="leiden")
        assert isinstance(result, dict)

    def test_setup_indexes_includes_community(self, mock_neo4j_store):
        mock_neo4j_store.setup_indexes()


class TestBaseGraphStoreDefaults:
    """Verify new methods raise NotImplementedError on a minimal concrete subclass."""

    def _make_store(self):
        from grapsit.store.graph.base import BaseGraphStore

        class MinimalStore(BaseGraphStore):
            def setup_indexes(self): pass
            def close(self): pass
            def write_document(self, doc): pass
            def write_chunk(self, chunk): pass
            def write_chunk_document_link(self, chunk_id, document_id): pass
            def write_entity(self, entity): pass
            def write_mention_link(self, entity_id, chunk_id, mention): pass
            def write_relation(self, relation, head_entity_id, tail_entity_id): pass
            def get_entity_by_id(self, entity_id): pass
            def get_all_entities(self): pass
            def get_entity_by_label(self, label): pass
            def get_entity_neighbors(self, entity_id, max_hops=1): pass
            def get_entity_relations(self, entity_id): pass
            def get_chunks_for_entity(self, entity_id): pass
            def get_subgraph(self, entity_ids, max_hops=1): pass
            def clear_all(self): pass

        return MinimalStore()

    def test_get_entities_for_chunk_raises(self):
        with pytest.raises(NotImplementedError):
            self._make_store().get_entities_for_chunk("c1")

    def test_get_chunk_by_id_raises(self):
        with pytest.raises(NotImplementedError):
            self._make_store().get_chunk_by_id("c1")

    def test_get_shortest_paths_raises(self):
        with pytest.raises(NotImplementedError):
            self._make_store().get_shortest_paths("e1", "e2")

    def test_write_community_raises(self):
        with pytest.raises(NotImplementedError):
            self._make_store().write_community("com1", 0, "title", "summary")

    def test_write_community_membership_raises(self):
        with pytest.raises(NotImplementedError):
            self._make_store().write_community_membership("e1", "com1", 0)

    def test_get_community_members_raises(self):
        with pytest.raises(NotImplementedError):
            self._make_store().get_community_members("com1")

    def test_get_all_communities_raises(self):
        with pytest.raises(NotImplementedError):
            self._make_store().get_all_communities()

    def test_detect_communities_raises(self):
        with pytest.raises(NotImplementedError):
            self._make_store().detect_communities()

    def test_delete_entity_raises(self):
        with pytest.raises(NotImplementedError):
            self._make_store().delete_entity("e1")

    def test_delete_relation_raises(self):
        with pytest.raises(NotImplementedError):
            self._make_store().delete_relation("r1")

    def test_delete_chunk_raises(self):
        with pytest.raises(NotImplementedError):
            self._make_store().delete_chunk("c1")

    def test_update_entity_raises(self):
        with pytest.raises(NotImplementedError):
            self._make_store().update_entity("e1", label="new")

    def test_add_entity_raises(self):
        with pytest.raises(NotImplementedError):
            self._make_store().add_entity("Einstein", "person")

    def test_add_relation_raises(self):
        with pytest.raises(NotImplementedError):
            self._make_store().add_relation("e1", "e2", "knows")

    def test_merge_entities_raises(self):
        with pytest.raises(NotImplementedError):
            self._make_store().merge_entities("e1", "e2")


class TestNeo4jMutations:
    """Test Neo4j mutation methods with controlled _run() side effects."""

    def _make_store_with_run(self, side_effects):
        """Create a Neo4jGraphStore with _run mocked to return successive values."""
        from grapsit.store.graph.neo4j_store import Neo4jGraphStore
        store = Neo4jGraphStore.__new__(Neo4jGraphStore)
        store._driver = MagicMock()
        store._run = MagicMock(side_effect=side_effects)
        return store

    # -- delete_entity -------------------------------------------------------

    def test_delete_entity_found(self):
        store = self._make_store_with_run([
            [{"e": {"id": "e1", "label": "Einstein"}}],  # get_entity_by_id
            [],  # DETACH DELETE
        ])
        assert store.delete_entity("e1") is True
        assert store._run.call_count == 2

    def test_delete_entity_not_found(self):
        store = self._make_store_with_run([
            [],  # get_entity_by_id returns nothing
        ])
        assert store.delete_entity("e999") is False
        assert store._run.call_count == 1

    # -- delete_relation -----------------------------------------------------

    def test_delete_relation_found(self):
        store = self._make_store_with_run([
            [{"cnt": 1}],  # DELETE + RETURN count
        ])
        assert store.delete_relation("r1") is True

    def test_delete_relation_not_found(self):
        store = self._make_store_with_run([
            [{"cnt": 0}],
        ])
        assert store.delete_relation("r999") is False

    # -- delete_chunk --------------------------------------------------------

    def test_delete_chunk_found(self):
        store = self._make_store_with_run([
            [{"c": {"id": "c1", "text": "hello"}}],  # get_chunk_by_id
            [],  # DETACH DELETE
        ])
        assert store.delete_chunk("c1") is True

    def test_delete_chunk_not_found(self):
        store = self._make_store_with_run([
            [],  # get_chunk_by_id returns nothing
        ])
        assert store.delete_chunk("c999") is False

    # -- update_entity -------------------------------------------------------

    def test_update_entity_label(self):
        store = self._make_store_with_run([
            [{"e": {"id": "e1", "label": "Old", "entity_type": "person", "properties": "{}"}}],
            [],  # SET
        ])
        assert store.update_entity("e1", label="New") is True
        set_call = store._run.call_args_list[1]
        assert "e.label = $label" in set_call[0][0]
        assert set_call[0][1]["label"] == "New"

    def test_update_entity_properties_merged(self):
        store = self._make_store_with_run([
            [{"e": {"id": "e1", "label": "X", "properties": "{'a': 1}"}}],
            [],  # SET
        ])
        assert store.update_entity("e1", properties={"b": 2}) is True
        set_call = store._run.call_args_list[1]
        props_str = set_call[0][1]["properties"]
        import ast
        merged = ast.literal_eval(props_str)
        assert merged == {"a": 1, "b": 2}

    def test_update_entity_not_found(self):
        store = self._make_store_with_run([
            [],  # get_entity_by_id returns nothing
        ])
        assert store.update_entity("e999", label="X") is False

    def test_update_entity_no_changes(self):
        store = self._make_store_with_run([
            [{"e": {"id": "e1", "label": "X", "properties": "{}"}}],
        ])
        assert store.update_entity("e1") is True
        # Only the get_entity_by_id call, no SET
        assert store._run.call_count == 1

    # -- add_entity ----------------------------------------------------------

    def test_add_entity_with_id(self):
        store = self._make_store_with_run([
            [],  # CREATE
        ])
        result = store.add_entity("Einstein", "person", id="e-custom")
        assert result == "e-custom"
        query = store._run.call_args[0][0]
        assert "CREATE" in query
        assert "MERGE" not in query

    def test_add_entity_generated_id(self):
        store = self._make_store_with_run([
            [],  # CREATE
        ])
        result = store.add_entity("Einstein", "person")
        # Should be a valid UUID
        import uuid
        uuid.UUID(result)

    def test_add_entity_with_properties(self):
        store = self._make_store_with_run([
            [],  # CREATE
        ])
        store.add_entity("Einstein", "person", properties={"birth_year": 1879})
        params = store._run.call_args[0][1]
        assert "1879" in params["properties"]

    # -- add_relation --------------------------------------------------------

    def test_add_relation_success(self):
        store = self._make_store_with_run([
            [{"e": {"id": "e1"}}],  # head exists
            [{"e": {"id": "e2"}}],  # tail exists
            [],  # CREATE
        ])
        result = store.add_relation("e1", "e2", "born in", id="r-custom")
        assert result == "r-custom"
        query = store._run.call_args[0][0]
        assert "BORN_IN" in query
        assert "CREATE" in query

    def test_add_relation_head_missing(self):
        store = self._make_store_with_run([
            [],  # head not found
        ])
        with pytest.raises(ValueError, match="Head entity"):
            store.add_relation("e1", "e2", "born in")

    def test_add_relation_tail_missing(self):
        store = self._make_store_with_run([
            [{"e": {"id": "e1"}}],  # head exists
            [],  # tail not found
        ])
        with pytest.raises(ValueError, match="Tail entity"):
            store.add_relation("e1", "e2", "born in")

    # -- merge_entities ------------------------------------------------------

    def test_merge_entities_same_id(self):
        store = self._make_store_with_run([])
        assert store.merge_entities("e1", "e1") is True
        store._run.assert_not_called()

    def test_merge_entities_source_missing(self):
        store = self._make_store_with_run([
            [],  # source not found
        ])
        assert store.merge_entities("e1", "e2") is False

    def test_merge_entities_target_missing(self):
        store = self._make_store_with_run([
            [{"e": {"id": "e1", "label": "A", "properties": "{}"}}],  # source found
            [],  # target not found
        ])
        assert store.merge_entities("e1", "e2") is False

    def test_merge_entities_success(self):
        store = self._make_store_with_run([
            [{"e": {"id": "e1", "label": "A", "properties": "{'x': 1}"}}],  # source
            [{"e": {"id": "e2", "label": "B", "properties": "{'y': 2}"}}],  # target
            [],  # move MENTIONED_IN
            [],  # move MEMBER_OF
            [],  # outgoing rels (none)
            [],  # incoming rels (none)
            [],  # merge properties
            [],  # delete source
        ])
        assert store.merge_entities("e1", "e2") is True
        # Verify properties merge call
        props_call = store._run.call_args_list[6]
        import ast
        merged = ast.literal_eval(props_call[0][1]["properties"])
        assert merged == {"x": 1, "y": 2}  # target wins
