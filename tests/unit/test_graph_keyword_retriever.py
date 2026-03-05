"""Tests for keyword retriever with graph DB native full-text search source."""

import pytest
from unittest.mock import MagicMock, patch

from grapsit.query.keyword_retriever import KeywordRetrieverProcessor
from grapsit.models.graph import Subgraph
from grapsit.core.builders import QueryConfigBuilder


class TestKeywordRetrieverGraphSourceConfig:
    """Test config defaults and overrides for graph search source."""

    def test_graph_source_defaults(self):
        proc = KeywordRetrieverProcessor({"search_source": "graph"})
        assert proc.search_source == "graph"
        assert proc.top_k == 10
        assert proc.expand_entities is True  # default for graph
        assert proc.max_hops == 1
        assert proc.fulltext_index == "chunk_text_idx"

    def test_relational_source_defaults(self):
        proc = KeywordRetrieverProcessor({})
        assert proc.search_source == "relational"
        assert proc.expand_entities is False  # default for relational

    def test_explicit_expand_entities_overrides_default(self):
        proc = KeywordRetrieverProcessor({
            "search_source": "graph",
            "expand_entities": False,
        })
        assert proc.expand_entities is False

        proc2 = KeywordRetrieverProcessor({
            "search_source": "relational",
            "expand_entities": True,
        })
        assert proc2.expand_entities is True

    def test_custom_config(self):
        proc = KeywordRetrieverProcessor({
            "search_source": "graph",
            "top_k": 5,
            "expand_entities": False,
            "max_hops": 2,
            "fulltext_index": "my_index",
        })
        assert proc.top_k == 5
        assert proc.expand_entities is False
        assert proc.max_hops == 2
        assert proc.fulltext_index == "my_index"


class TestKeywordRetrieverGraphChunksOnly:
    """Test chunks-only mode with graph source (expand_entities=False)."""

    def test_returns_chunks_only(self):
        mock_store = MagicMock()
        mock_store.fulltext_search_chunks.return_value = [
            {"id": "c1", "text": "Einstein was born in Ulm.", "document_id": "d1",
             "index": 0, "start_char": 0, "end_char": 25, "score": 1.5},
            {"id": "c2", "text": "He studied physics.", "document_id": "d1",
             "index": 1, "start_char": 26, "end_char": 45, "score": 0.8},
        ]

        proc = KeywordRetrieverProcessor({
            "search_source": "graph",
            "expand_entities": False,
        })
        proc._store = mock_store

        result = proc(query="Einstein born")
        sg = result["subgraph"]

        assert isinstance(sg, Subgraph)
        assert len(sg.chunks) == 2
        assert sg.chunks[0].id == "c1"
        assert sg.chunks[1].id == "c2"
        assert len(sg.entities) == 0
        assert len(sg.relations) == 0
        mock_store.fulltext_search_chunks.assert_called_once_with(
            "Einstein born", top_k=10, index_name="chunk_text_idx"
        )

    def test_empty_results(self):
        mock_store = MagicMock()
        mock_store.fulltext_search_chunks.return_value = []

        proc = KeywordRetrieverProcessor({
            "search_source": "graph",
            "expand_entities": False,
        })
        proc._store = mock_store

        result = proc(query="nonexistent")
        sg = result["subgraph"]

        assert isinstance(sg, Subgraph)
        assert len(sg.chunks) == 0
        assert len(sg.entities) == 0

    def test_skips_records_without_id(self):
        mock_store = MagicMock()
        mock_store.fulltext_search_chunks.return_value = [
            {"id": "c1", "text": "Valid chunk"},
            {"text": "No id chunk"},
        ]

        proc = KeywordRetrieverProcessor({
            "search_source": "graph",
            "expand_entities": False,
        })
        proc._store = mock_store

        result = proc(query="test")
        assert len(result["subgraph"].chunks) == 1


class TestKeywordRetrieverGraphEntityExpansion:
    """Test entity expansion mode with graph source (expand_entities=True)."""

    def test_expands_entities(self):
        mock_store = MagicMock()
        mock_store.fulltext_search_chunks.return_value = [
            {"id": "c1", "text": "Einstein was born in Ulm.", "document_id": "d1",
             "index": 0, "start_char": 0, "end_char": 25},
        ]
        mock_store.get_entities_for_chunk.return_value = [
            {"id": "e1", "label": "Einstein", "entity_type": "person"},
            {"id": "e2", "label": "Ulm", "entity_type": "location"},
        ]
        mock_store.get_subgraph.return_value = {
            "entities": [
                {"id": "e1", "label": "Einstein", "entity_type": "person"},
                {"id": "e2", "label": "Ulm", "entity_type": "location"},
            ],
            "relations": [
                {"head": "e1", "tail": "e2", "type": "BORN_IN", "score": 0.9},
            ],
        }

        proc = KeywordRetrieverProcessor({"search_source": "graph"})
        proc._store = mock_store

        result = proc(query="Einstein born")
        sg = result["subgraph"]

        assert len(sg.entities) == 2
        assert len(sg.relations) == 1
        assert len(sg.chunks) == 1
        assert sg.relations[0].relation_type == "BORN_IN"
        mock_store.get_subgraph.assert_called_once_with(["e1", "e2"], max_hops=1)

    def test_chunks_found_but_no_entities(self):
        mock_store = MagicMock()
        mock_store.fulltext_search_chunks.return_value = [
            {"id": "c1", "text": "Some text", "document_id": "d1",
             "index": 0, "start_char": 0, "end_char": 9},
        ]
        mock_store.get_entities_for_chunk.return_value = []

        proc = KeywordRetrieverProcessor({"search_source": "graph"})
        proc._store = mock_store

        result = proc(query="test")
        sg = result["subgraph"]

        assert len(sg.chunks) == 1
        assert len(sg.entities) == 0
        assert len(sg.relations) == 0
        mock_store.get_subgraph.assert_not_called()

    def test_deduplicates_entity_ids(self):
        mock_store = MagicMock()
        mock_store.fulltext_search_chunks.return_value = [
            {"id": "c1", "text": "Chunk 1"},
            {"id": "c2", "text": "Chunk 2"},
        ]
        mock_store.get_entities_for_chunk.side_effect = [
            [{"id": "e1"}, {"id": "e2"}],
            [{"id": "e2"}, {"id": "e3"}],
        ]
        mock_store.get_subgraph.return_value = {"entities": [], "relations": []}

        proc = KeywordRetrieverProcessor({"search_source": "graph"})
        proc._store = mock_store

        proc(query="test")
        mock_store.get_subgraph.assert_called_once_with(["e1", "e2", "e3"], max_hops=1)

    def test_custom_max_hops(self):
        mock_store = MagicMock()
        mock_store.fulltext_search_chunks.return_value = [
            {"id": "c1", "text": "Chunk"},
        ]
        mock_store.get_entities_for_chunk.return_value = [{"id": "e1"}]
        mock_store.get_subgraph.return_value = {"entities": [], "relations": []}

        proc = KeywordRetrieverProcessor({
            "search_source": "graph",
            "max_hops": 3,
        })
        proc._store = mock_store

        proc(query="test")
        mock_store.get_subgraph.assert_called_once_with(["e1"], max_hops=3)

    def test_custom_fulltext_index(self):
        mock_store = MagicMock()
        mock_store.fulltext_search_chunks.return_value = []

        proc = KeywordRetrieverProcessor({
            "search_source": "graph",
            "expand_entities": False,
            "fulltext_index": "my_custom_idx",
        })
        proc._store = mock_store

        proc(query="test")
        mock_store.fulltext_search_chunks.assert_called_once_with(
            "test", top_k=10, index_name="my_custom_idx"
        )


class TestKeywordRetrieverGraphEnsureStore:
    """Test lazy store initialization for graph source."""

    def test_ensure_store_called_for_graph_source(self):
        proc = KeywordRetrieverProcessor({
            "search_source": "graph",
            "expand_entities": False,
        })
        mock_store = MagicMock()
        mock_store.fulltext_search_chunks.return_value = []

        with patch.object(proc, "_ensure_store") as mock_ensure:
            proc._store = mock_store
            proc(query="test")
            mock_ensure.assert_called_once()

    def test_relational_source_does_not_call_ensure_store(self):
        proc = KeywordRetrieverProcessor({"search_source": "relational"})
        proc._relational_store = MagicMock()
        proc._relational_store.search.return_value = [{"id": "c1", "text": "x"}]

        result = proc(query="test")
        assert proc._store is None
        assert len(result["subgraph"].chunks) == 1


class TestKeywordRetrieverBuilderGraphSource:
    """Test QueryConfigBuilder.keyword_retriever() with search_source='graph'."""

    def test_builder_graph_source(self):
        builder = QueryConfigBuilder(name="test")
        builder.keyword_retriever(search_source="graph")
        assert builder._retriever_type == "keyword_retriever"
        assert builder._retriever_config["search_source"] == "graph"
        assert builder._retriever_config["expand_entities"] is True  # default for graph

    def test_builder_graph_source_custom(self):
        builder = QueryConfigBuilder(name="test")
        builder.keyword_retriever(
            search_source="graph",
            top_k=5,
            expand_entities=False,
            max_hops=2,
            fulltext_index="my_idx",
        )
        assert builder._retriever_config["search_source"] == "graph"
        assert builder._retriever_config["top_k"] == 5
        assert builder._retriever_config["expand_entities"] is False
        assert builder._retriever_config["max_hops"] == 2
        assert builder._retriever_config["fulltext_index"] == "my_idx"

    def test_builder_graph_source_produces_valid_config(self):
        builder = QueryConfigBuilder(name="test")
        builder.keyword_retriever(
            search_source="graph",
            top_k=5,
            neo4j_uri="bolt://localhost:7687",
        )
        config = builder.get_config()

        nodes = config["nodes"]
        retriever_node = next(n for n in nodes if n["id"] == "retriever")
        assert retriever_node["processor"] == "keyword_retriever"
        assert retriever_node["inputs"]["query"]["source"] == "$input"
        assert retriever_node["config"]["search_source"] == "graph"
        assert retriever_node["config"]["top_k"] == 5
        assert retriever_node["requires"] == []

    def test_builder_relational_source_default(self):
        builder = QueryConfigBuilder(name="test")
        builder.keyword_retriever()
        assert builder._retriever_config["search_source"] == "relational"
        assert builder._retriever_config["expand_entities"] is False

    def test_builder_returns_self(self):
        builder = QueryConfigBuilder(name="test")
        result = builder.keyword_retriever(search_source="graph")
        assert result is builder
