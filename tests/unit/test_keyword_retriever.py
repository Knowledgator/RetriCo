"""Tests for keyword retriever processor."""

import pytest
from unittest.mock import MagicMock, patch

from retrico.query.keyword_retriever import KeywordRetrieverProcessor


class TestKeywordRetrieverChunksOnly:
    """Tests for default mode (expand_entities=False) — returns chunks only."""

    def _make_proc(self, **config_overrides):
        config = {
            "relational_store_type": "sqlite",
            "top_k": 10,
            "chunk_table": "chunks",
        }
        config.update(config_overrides)
        proc = KeywordRetrieverProcessor(config)
        proc._relational_store = MagicMock()
        return proc

    def test_basic_retrieval(self):
        proc = self._make_proc(top_k=5)
        proc._relational_store.search.return_value = [
            {"id": "chunk_1", "document_id": "d1", "text": "Einstein was born in Ulm", "index": 0},
            {"id": "chunk_2", "document_id": "d1", "text": "Einstein worked at Princeton", "index": 1},
        ]

        result = proc(query="Where was Einstein born?")

        sg = result["subgraph"]
        assert len(sg.chunks) == 2
        assert sg.chunks[0].id == "chunk_1"
        assert sg.chunks[0].text == "Einstein was born in Ulm"
        assert sg.chunks[1].id == "chunk_2"
        assert len(sg.entities) == 0
        assert len(sg.relations) == 0
        proc._relational_store.search.assert_called_once_with(
            "chunks", "Where was Einstein born?", top_k=5
        )

    def test_no_search_results(self):
        proc = self._make_proc()
        proc._relational_store.search.return_value = []

        result = proc(query="Unknown topic")
        assert len(result["subgraph"].chunks) == 0

    def test_records_without_id_skipped(self):
        proc = self._make_proc()
        proc._relational_store.search.return_value = [
            {"id": "c1", "text": "text1"},
            {"text": "text2"},  # no id — skipped
            {"id": "c3", "text": "text3"},
        ]

        result = proc(query="q")
        assert len(result["subgraph"].chunks) == 2
        assert [c.id for c in result["subgraph"].chunks] == ["c1", "c3"]

    def test_chunk_fields_populated(self):
        proc = self._make_proc()
        proc._relational_store.search.return_value = [
            {
                "id": "c1",
                "document_id": "doc_42",
                "text": "Hello world",
                "index": 3,
                "start_char": 100,
                "end_char": 111,
            },
        ]

        result = proc(query="hello")
        chunk = result["subgraph"].chunks[0]
        assert chunk.id == "c1"
        assert chunk.document_id == "doc_42"
        assert chunk.text == "Hello world"
        assert chunk.index == 3
        assert chunk.start_char == 100
        assert chunk.end_char == 111

    def test_custom_chunk_table(self):
        proc = self._make_proc(chunk_table="my_chunks")
        proc._relational_store.search.return_value = [
            {"id": "c1", "text": "text"},
        ]

        proc(query="test")
        proc._relational_store.search.assert_called_once_with(
            "my_chunks", "test", top_k=10
        )

    def test_no_graph_store_needed(self):
        """Chunks-only mode should not touch a graph store."""
        proc = self._make_proc()
        proc._relational_store.search.return_value = [
            {"id": "c1", "text": "some text"},
        ]

        result = proc(query="test")
        assert len(result["subgraph"].chunks) == 1
        assert proc._store is None


class TestKeywordRetrieverExpandEntities:
    """Tests for expand_entities=True — returns chunks + entities + relations."""

    def _make_proc(self, **config_overrides):
        config = {
            "relational_store_type": "sqlite",
            "neo4j_uri": "bolt://localhost:7687",
            "top_k": 10,
            "chunk_table": "chunks",
            "expand_entities": True,
            "max_hops": 1,
        }
        config.update(config_overrides)
        proc = KeywordRetrieverProcessor(config)
        proc._relational_store = MagicMock()
        proc._store = MagicMock()
        return proc

    def test_basic_entity_expansion(self):
        proc = self._make_proc(top_k=5)
        proc._relational_store.search.return_value = [
            {"id": "chunk_1", "text": "Einstein was born in Ulm"},
            {"id": "chunk_2", "text": "Einstein worked at Princeton"},
        ]
        proc._store.get_entities_for_chunk.side_effect = [
            [{"id": "e1"}, {"id": "e2"}],
            [{"id": "e2"}, {"id": "e3"}],
        ]
        proc._store.get_subgraph.return_value = {
            "entities": [
                {"id": "e1", "label": "Einstein", "entity_type": "person"},
                {"id": "e2", "label": "Ulm", "entity_type": "location"},
                {"id": "e3", "label": "Princeton", "entity_type": "org"},
            ],
            "relations": [
                {"head": "e1", "tail": "e2", "type": "BORN_IN", "score": 0.9},
            ],
        }

        result = proc(query="Where was Einstein born?")

        sg = result["subgraph"]
        assert len(sg.chunks) == 2
        assert len(sg.entities) == 3
        assert len(sg.relations) == 1
        proc._store.get_subgraph.assert_called_once_with(
            ["e1", "e2", "e3"], max_hops=1
        )

    def test_entity_dedup_across_chunks(self):
        proc = self._make_proc()
        proc._relational_store.search.return_value = [
            {"id": "c1", "text": "text1"},
            {"id": "c2", "text": "text2"},
        ]
        proc._store.get_entities_for_chunk.side_effect = [
            [{"id": "e1"}, {"id": "e2"}],
            [{"id": "e1"}, {"id": "e3"}],  # e1 is duplicate
        ]
        proc._store.get_subgraph.return_value = {
            "entities": [
                {"id": "e1", "label": "A", "entity_type": ""},
                {"id": "e2", "label": "B", "entity_type": ""},
                {"id": "e3", "label": "C", "entity_type": ""},
            ],
            "relations": [],
        }

        proc(query="q")
        entity_ids = proc._store.get_subgraph.call_args[0][0]
        assert entity_ids == ["e1", "e2", "e3"]

    def test_chunks_with_no_entities_still_returns_chunks(self):
        proc = self._make_proc()
        proc._relational_store.search.return_value = [
            {"id": "c1", "text": "Some text"},
        ]
        proc._store.get_entities_for_chunk.return_value = []

        result = proc(query="query")
        sg = result["subgraph"]
        assert len(sg.chunks) == 1
        assert len(sg.entities) == 0
        proc._store.get_subgraph.assert_not_called()

    def test_no_search_results(self):
        proc = self._make_proc()
        proc._relational_store.search.return_value = []

        result = proc(query="nothing")
        assert len(result["subgraph"].chunks) == 0
        assert len(result["subgraph"].entities) == 0
        proc._store.get_entities_for_chunk.assert_not_called()


class TestKeywordRetrieverConfig:
    def test_defaults(self):
        proc = KeywordRetrieverProcessor({})
        assert proc.top_k == 10
        assert proc.chunk_table == "chunks"
        assert proc.expand_entities is False
        assert proc.max_hops == 1

    def test_custom(self):
        proc = KeywordRetrieverProcessor({
            "top_k": 20,
            "chunk_table": "documents",
            "expand_entities": True,
            "max_hops": 3,
        })
        assert proc.top_k == 20
        assert proc.chunk_table == "documents"
        assert proc.expand_entities is True
        assert proc.max_hops == 3

    def test_ensure_relational_store_lazy_init(self):
        config = {
            "relational_store_type": "sqlite",
            "sqlite_path": ":memory:",
        }
        proc = KeywordRetrieverProcessor(config)
        assert proc._relational_store is None

        mock_store = MagicMock()
        with patch(
            "retrico.store.pool.resolve_from_pool_or_create",
            return_value=mock_store,
        ) as mock_resolve:
            proc._ensure_relational_store()
            mock_resolve.assert_called_once_with(config, "relational")
            assert proc._relational_store is mock_store

    def test_ensure_relational_store_not_recreated(self):
        proc = KeywordRetrieverProcessor({})
        proc._relational_store = MagicMock()
        existing = proc._relational_store

        with patch(
            "retrico.store.pool.resolve_from_pool_or_create"
        ) as mock_resolve:
            proc._ensure_relational_store()
            mock_resolve.assert_not_called()
            assert proc._relational_store is existing


class TestKeywordRetrieverRegistration:
    def test_registered(self):
        from retrico.core.registry import processor_registry
        factory = processor_registry.get("keyword_retriever")
        assert factory is not None
