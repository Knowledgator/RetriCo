"""Tests for chunk retriever processor."""

import pytest
from unittest.mock import MagicMock

from retrico.query.chunk_retriever import ChunkRetrieverProcessor
from retrico.models.entity import Entity
from retrico.models.relation import Relation
from retrico.models.graph import Subgraph


class TestChunkRetrieverProcessor:
    def _make_proc(self, **config_overrides):
        config = {
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j",
            "neo4j_password": "password",
        }
        config.update(config_overrides)
        proc = ChunkRetrieverProcessor(config)
        proc._store = MagicMock()
        return proc

    def test_basic_chunk_retrieval(self):
        proc = self._make_proc()
        proc._store.get_chunks_for_entity.side_effect = [
            [{"id": "c1", "document_id": "d1", "text": "Einstein was born in Ulm.", "index": 0}],
            [{"id": "c2", "document_id": "d1", "text": "Ulm is in Germany.", "index": 1}],
        ]

        subgraph = Subgraph(
            entities=[
                Entity(id="e1", label="Einstein"),
                Entity(id="e2", label="Ulm"),
            ],
            relations=[],
        )

        result = proc(subgraph=subgraph)

        sg = result["subgraph"]
        assert len(sg.chunks) == 2
        assert sg.chunks[0].id == "c1"
        assert sg.chunks[1].id == "c2"
        # Entities and relations preserved
        assert len(sg.entities) == 2

    def test_chunk_deduplication(self):
        proc = self._make_proc()
        # Both entities mention the same chunk
        proc._store.get_chunks_for_entity.side_effect = [
            [{"id": "c1", "document_id": "d1", "text": "Einstein was born in Ulm.", "index": 0}],
            [{"id": "c1", "document_id": "d1", "text": "Einstein was born in Ulm.", "index": 0}],
        ]

        subgraph = Subgraph(
            entities=[
                Entity(id="e1", label="Einstein"),
                Entity(id="e2", label="Ulm"),
            ],
        )

        result = proc(subgraph=subgraph)
        assert len(result["subgraph"].chunks) == 1

    def test_empty_subgraph(self):
        proc = self._make_proc()
        result = proc(subgraph=Subgraph())
        assert result["subgraph"].chunks == []
        assert result["subgraph"].entities == []

    def test_max_chunks_limiting(self):
        proc = self._make_proc(max_chunks=1)
        proc._store.get_chunks_for_entity.return_value = [
            {"id": "c1", "document_id": "d1", "text": "Chunk 1", "index": 0},
            {"id": "c2", "document_id": "d1", "text": "Chunk 2", "index": 1},
        ]

        subgraph = Subgraph(entities=[Entity(id="e1", label="X")])
        result = proc(subgraph=subgraph)
        assert len(result["subgraph"].chunks) == 1

    def test_accepts_dict_subgraph(self):
        proc = self._make_proc()
        proc._store.get_chunks_for_entity.return_value = [
            {"id": "c1", "document_id": "d1", "text": "Test.", "index": 0},
        ]

        subgraph_dict = {
            "entities": [{"id": "e1", "label": "X", "entity_type": ""}],
            "relations": [],
            "chunks": [],
        }
        result = proc(subgraph=subgraph_dict)
        assert len(result["subgraph"].chunks) == 1


class TestChunkEntitySource:
    def _make_proc(self, chunk_entity_source="all", **config_overrides):
        config = {
            "neo4j_uri": "bolt://localhost:7687",
            "neo4j_user": "neo4j",
            "neo4j_password": "password",
            "chunk_entity_source": chunk_entity_source,
        }
        config.update(config_overrides)
        proc = ChunkRetrieverProcessor(config)
        proc._store = MagicMock()
        return proc

    def _make_subgraph(self):
        return Subgraph(
            entities=[
                Entity(id="e1", label="Einstein", entity_type="person"),
                Entity(id="e2", label="Ulm", entity_type="location"),
                Entity(id="e3", label="Physics", entity_type="field"),
            ],
            relations=[
                Relation(head_text="Einstein", tail_text="Ulm", relation_type="BORN_IN"),
            ],
        )

    def test_all_entities_default(self):
        proc = self._make_proc(chunk_entity_source="all")
        proc._store.get_chunks_for_entity.return_value = [
            {"id": "c1", "document_id": "d1", "text": "test", "index": 0},
        ]

        result = proc(subgraph=self._make_subgraph())
        # Should query all 3 entities
        assert proc._store.get_chunks_for_entity.call_count == 3

    def test_head_entities_only(self):
        proc = self._make_proc(chunk_entity_source="head")
        proc._store.get_chunks_for_entity.return_value = [
            {"id": "c1", "document_id": "d1", "text": "test", "index": 0},
        ]

        scored_triples = [
            {"head": "Einstein", "relation": "BORN_IN", "tail": "Ulm", "score": 0.9},
        ]

        result = proc(subgraph=self._make_subgraph(), scored_triples=scored_triples)
        # Should only query Einstein (head)
        assert proc._store.get_chunks_for_entity.call_count == 1
        proc._store.get_chunks_for_entity.assert_called_with("e1")

    def test_tail_entities_only(self):
        proc = self._make_proc(chunk_entity_source="tail")
        proc._store.get_chunks_for_entity.return_value = [
            {"id": "c1", "document_id": "d1", "text": "test", "index": 0},
        ]

        scored_triples = [
            {"head": "Einstein", "relation": "BORN_IN", "tail": "Ulm", "score": 0.9},
        ]

        result = proc(subgraph=self._make_subgraph(), scored_triples=scored_triples)
        # Should only query Ulm (tail)
        assert proc._store.get_chunks_for_entity.call_count == 1
        proc._store.get_chunks_for_entity.assert_called_with("e2")

    def test_fallback_to_all_without_scored_triples(self):
        proc = self._make_proc(chunk_entity_source="head")
        proc._store.get_chunks_for_entity.return_value = []

        result = proc(subgraph=self._make_subgraph())
        # No scored_triples → falls back to all entities
        assert proc._store.get_chunks_for_entity.call_count == 3

    def test_both_alias(self):
        proc = self._make_proc(chunk_entity_source="both")
        proc._store.get_chunks_for_entity.return_value = []

        result = proc(subgraph=self._make_subgraph())
        # "both" is same as "all"
        assert proc._store.get_chunks_for_entity.call_count == 3
