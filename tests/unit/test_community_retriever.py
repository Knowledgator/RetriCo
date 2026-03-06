"""Tests for community retriever processor."""

import pytest
from unittest.mock import MagicMock

from retrico.query.community_retriever import CommunityRetrieverProcessor


class TestCommunityRetrieverProcessor:
    def _make_proc(self, **config_overrides):
        config = {
            "neo4j_uri": "bolt://localhost:7687",
            "top_k": 2,
            "max_hops": 1,
            "vector_index_name": "community_embeddings",
        }
        config.update(config_overrides)
        proc = CommunityRetrieverProcessor(config)
        proc._store = MagicMock()
        proc._embedding_model = MagicMock()
        proc._vector_store = MagicMock()
        return proc

    def test_basic_retrieval(self):
        proc = self._make_proc()
        proc._embedding_model.encode.return_value = [[0.1, 0.2, 0.3]]
        proc._vector_store.search_similar.return_value = [
            ("comm_1", 0.95),
            ("comm_2", 0.80),
        ]
        proc._store.get_community_members.side_effect = [
            [{"id": "e1"}, {"id": "e2"}],
            [{"id": "e3"}],
        ]
        proc._store.get_subgraph.return_value = {
            "entities": [
                {"id": "e1", "label": "A", "entity_type": "person"},
                {"id": "e2", "label": "B", "entity_type": "person"},
                {"id": "e3", "label": "C", "entity_type": "org"},
            ],
            "relations": [
                {"head": "e1", "tail": "e2", "type": "KNOWS", "score": 0.9},
            ],
        }

        result = proc(query="Tell me about group A")

        assert "subgraph" in result
        sg = result["subgraph"]
        assert len(sg.entities) == 3
        assert len(sg.relations) == 1
        proc._embedding_model.encode.assert_called_once_with(["Tell me about group A"])
        proc._vector_store.search_similar.assert_called_once_with(
            "community_embeddings", [0.1, 0.2, 0.3], top_k=2
        )

    def test_no_vector_matches(self):
        proc = self._make_proc()
        proc._embedding_model.encode.return_value = [[0.1, 0.2]]
        proc._vector_store.search_similar.return_value = []

        result = proc(query="Unknown topic")

        sg = result["subgraph"]
        assert len(sg.entities) == 0
        assert len(sg.relations) == 0

    def test_empty_community_members(self):
        proc = self._make_proc()
        proc._embedding_model.encode.return_value = [[0.1]]
        proc._vector_store.search_similar.return_value = [("comm_1", 0.9)]
        proc._store.get_community_members.return_value = []

        result = proc(query="Some query")
        assert len(result["subgraph"].entities) == 0

    def test_dedup_entity_ids(self):
        proc = self._make_proc()
        proc._embedding_model.encode.return_value = [[0.1]]
        proc._vector_store.search_similar.return_value = [
            ("comm_1", 0.9),
            ("comm_2", 0.8),
        ]
        # e1 appears in both communities
        proc._store.get_community_members.side_effect = [
            [{"id": "e1"}, {"id": "e2"}],
            [{"id": "e1"}, {"id": "e3"}],
        ]
        proc._store.get_subgraph.return_value = {
            "entities": [
                {"id": "e1", "label": "A", "entity_type": ""},
                {"id": "e2", "label": "B", "entity_type": ""},
                {"id": "e3", "label": "C", "entity_type": ""},
            ],
            "relations": [],
        }

        result = proc(query="query")

        # Should have called get_subgraph with deduped IDs
        call_args = proc._store.get_subgraph.call_args
        entity_ids = call_args[0][0]
        assert entity_ids == ["e1", "e2", "e3"]

    def test_config_defaults(self):
        proc = CommunityRetrieverProcessor({})
        assert proc.top_k == 3
        assert proc.max_hops == 1
        assert proc.vector_index_name == "community_embeddings"
