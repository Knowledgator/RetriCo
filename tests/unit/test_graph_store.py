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
        from grapsit.store.neo4j_store import _sanitize_label
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
        from grapsit.store.base import BaseGraphStore

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
