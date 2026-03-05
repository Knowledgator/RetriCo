"""Tests for entity embedding retriever processor."""

import pytest
from unittest.mock import MagicMock

from retrico.query.entity_embedding_retriever import EntityEmbeddingRetrieverProcessor
from retrico.models.entity import EntityMention


class TestEntityEmbeddingRetrieverProcessor:
    def _make_proc(self, **config_overrides):
        config = {
            "neo4j_uri": "bolt://localhost:7687",
            "top_k": 3,
            "max_hops": 2,
            "vector_index_name": "entity_embeddings",
        }
        config.update(config_overrides)
        proc = EntityEmbeddingRetrieverProcessor(config)
        proc._store = MagicMock()
        proc._embedding_model = MagicMock()
        proc._vector_store = MagicMock()
        return proc

    def test_basic_retrieval(self):
        proc = self._make_proc()
        proc._embedding_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
        proc._vector_store.search_similar.side_effect = [
            [("e1", 0.95), ("e2", 0.80)],
            [("e3", 0.90)],
        ]
        proc._store.get_subgraph.return_value = {
            "entities": [
                {"id": "e1", "label": "Einstein", "entity_type": "person"},
                {"id": "e2", "label": "Feynman", "entity_type": "person"},
                {"id": "e3", "label": "MIT", "entity_type": "org"},
            ],
            "relations": [
                {"head": "e2", "tail": "e3", "type": "WORKS_AT", "score": 0.7},
            ],
        }

        entities = [
            EntityMention(text="Einstein", label="person"),
            EntityMention(text="physics", label="field"),
        ]
        result = proc(entities=entities)

        sg = result["subgraph"]
        assert len(sg.entities) == 3
        assert len(sg.relations) == 1
        proc._embedding_model.encode.assert_called_once_with(["Einstein", "physics"])

    def test_empty_entities(self):
        proc = self._make_proc()
        result = proc(entities=[])
        assert len(result["subgraph"].entities) == 0
        proc._embedding_model.encode.assert_not_called()

    def test_no_vector_matches(self):
        proc = self._make_proc()
        proc._embedding_model.encode.return_value = [[0.1]]
        proc._vector_store.search_similar.return_value = []

        entities = [EntityMention(text="Unknown", label="")]
        result = proc(entities=entities)
        assert len(result["subgraph"].entities) == 0

    def test_dedup_entity_ids(self):
        proc = self._make_proc()
        proc._embedding_model.encode.return_value = [[0.1], [0.2]]
        # Both mentions return the same entity
        proc._vector_store.search_similar.side_effect = [
            [("e1", 0.9)],
            [("e1", 0.8)],
        ]
        proc._store.get_subgraph.return_value = {
            "entities": [{"id": "e1", "label": "A", "entity_type": ""}],
            "relations": [],
        }

        entities = [
            EntityMention(text="A", label=""),
            EntityMention(text="A variant", label=""),
        ]
        result = proc(entities=entities)
        call_args = proc._store.get_subgraph.call_args
        entity_ids = call_args[0][0]
        assert entity_ids == ["e1"]

    def test_config_defaults(self):
        proc = EntityEmbeddingRetrieverProcessor({})
        assert proc.top_k == 5
        assert proc.max_hops == 2
        assert proc.vector_index_name == "entity_embeddings"
