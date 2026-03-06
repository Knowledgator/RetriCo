"""Tests for chunk embedding retriever processor."""

import pytest
from unittest.mock import MagicMock

from retrico.query.chunk_embedding_retriever import ChunkEmbeddingRetrieverProcessor


class TestChunkEmbeddingRetrieverProcessor:
    def _make_proc(self, **config_overrides):
        config = {
            "neo4j_uri": "bolt://localhost:7687",
            "top_k": 3,
            "max_hops": 1,
            "vector_index_name": "chunk_embeddings",
        }
        config.update(config_overrides)
        proc = ChunkEmbeddingRetrieverProcessor(config)
        proc._store = MagicMock()
        proc._embedding_model = MagicMock()
        proc._vector_store = MagicMock()
        return proc

    def test_basic_retrieval(self):
        proc = self._make_proc()
        proc._embedding_model.encode.return_value = [[0.1, 0.2, 0.3]]
        proc._vector_store.search_similar.return_value = [
            ("chunk_1", 0.95),
            ("chunk_2", 0.80),
        ]
        proc._store.get_entities_for_chunk.side_effect = [
            [{"id": "e1"}, {"id": "e2"}],
            [{"id": "e2"}, {"id": "e3"}],
        ]
        proc._store.get_subgraph.return_value = {
            "entities": [
                {"id": "e1", "label": "A", "entity_type": "person"},
                {"id": "e2", "label": "B", "entity_type": "person"},
                {"id": "e3", "label": "C", "entity_type": "org"},
            ],
            "relations": [
                {"head": "e1", "tail": "e2", "type": "WORKS_AT", "score": 0.7},
            ],
        }

        result = proc(query="Who works where?")

        sg = result["subgraph"]
        assert len(sg.entities) == 3
        assert len(sg.relations) == 1
        proc._vector_store.search_similar.assert_called_once_with(
            "chunk_embeddings", [0.1, 0.2, 0.3], top_k=3
        )

    def test_no_vector_matches(self):
        proc = self._make_proc()
        proc._embedding_model.encode.return_value = [[0.1]]
        proc._vector_store.search_similar.return_value = []

        result = proc(query="Unknown")
        assert len(result["subgraph"].entities) == 0

    def test_chunks_with_no_entities(self):
        proc = self._make_proc()
        proc._embedding_model.encode.return_value = [[0.1]]
        proc._vector_store.search_similar.return_value = [("c1", 0.9)]
        proc._store.get_entities_for_chunk.return_value = []

        result = proc(query="query")
        assert len(result["subgraph"].entities) == 0

    def test_dedup_entity_ids_across_chunks(self):
        proc = self._make_proc()
        proc._embedding_model.encode.return_value = [[0.1]]
        proc._vector_store.search_similar.return_value = [
            ("c1", 0.9),
            ("c2", 0.8),
        ]
        proc._store.get_entities_for_chunk.side_effect = [
            [{"id": "e1"}],
            [{"id": "e1"}],  # duplicate
        ]
        proc._store.get_subgraph.return_value = {
            "entities": [{"id": "e1", "label": "A", "entity_type": ""}],
            "relations": [],
        }

        result = proc(query="q")
        call_args = proc._store.get_subgraph.call_args
        entity_ids = call_args[0][0]
        assert entity_ids == ["e1"]

    def test_config_defaults(self):
        proc = ChunkEmbeddingRetrieverProcessor({})
        assert proc.top_k == 5
        assert proc.max_hops == 1
        assert proc.vector_index_name == "chunk_embeddings"
