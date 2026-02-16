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
