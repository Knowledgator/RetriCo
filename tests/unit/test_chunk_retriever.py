"""Tests for chunk retriever processor."""

import pytest
from unittest.mock import MagicMock

from grapsit.query.chunk_retriever import ChunkRetrieverProcessor
from grapsit.models.entity import Entity
from grapsit.models.relation import Relation
from grapsit.models.graph import Subgraph


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
