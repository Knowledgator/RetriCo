"""Integration test — full query pipeline with mocked models + Neo4j."""

import json
import pytest
from unittest.mock import MagicMock, patch

from retrico.core.builders import RetriCoSearch
from retrico.models.graph import QueryResult


class TestQueryPipeline:
    def _mock_store(self):
        """Create a mock Neo4j store with realistic data."""
        store = MagicMock()
        store.get_entity_by_label.side_effect = lambda label: {
            "einstein": {"id": "e1", "label": "Einstein", "entity_type": "person"},
            "ulm": {"id": "e2", "label": "Ulm", "entity_type": "location"},
            "germany": {"id": "e3", "label": "Germany", "entity_type": "location"},
        }.get(label.strip().lower())

        store.get_subgraph.return_value = {
            "entities": [
                {"id": "e1", "label": "Einstein", "entity_type": "person"},
                {"id": "e2", "label": "Ulm", "entity_type": "location"},
                {"id": "e3", "label": "Germany", "entity_type": "location"},
            ],
            "relations": [
                {"head": "e1", "tail": "e2", "type": "BORN_IN", "score": 0.8},
                {"head": "e2", "tail": "e3", "type": "LOCATED_IN", "score": 0.9},
            ],
        }
        store.get_chunks_for_entity.side_effect = lambda eid: {
            "e1": [{"id": "c1", "document_id": "d1", "text": "Einstein was born in Ulm.", "index": 0}],
            "e2": [{"id": "c1", "document_id": "d1", "text": "Einstein was born in Ulm.", "index": 0}],
            "e3": [{"id": "c2", "document_id": "d1", "text": "Ulm is in Germany.", "index": 1}],
        }.get(eid, [])

        return store

    @patch("retrico.query.chunk_retriever.ChunkRetrieverProcessor._ensure_store")
    @patch("retrico.query.retriever.RetrieverProcessor._ensure_store")
    def test_pipeline_without_reasoner(self, mock_ret_store, mock_chunk_store):
        """Test query_parser -> retriever -> chunk_retriever (no reasoner)."""
        builder = RetriCoSearch(name="test_query")
        builder.query_parser(method="gliner", labels=["person", "location"])
        builder.retriever(neo4j_uri="bolt://localhost:7687", max_hops=2)
        builder.chunk_retriever()

        executor = builder.build()

        # Mock GLiNER parser via engine
        parser_proc = executor.processors["query_parser"]
        engine = parser_proc._get_gliner_engine()
        engine._model = MagicMock()
        engine._model.inference.return_value = [[
            {"text": "Einstein", "label": "person", "start": 10, "end": 18, "score": 0.95},
        ]]

        # Mock store for retriever + chunk_retriever
        mock_store = self._mock_store()
        executor.processors["retriever"]._store = mock_store
        executor.processors["chunk_retriever"]._store = mock_store

        ctx = executor.run(query="Where was Einstein born?")

        assert ctx.has("parser_result")
        assert ctx.has("retriever_result")
        assert ctx.has("chunk_result")

        chunk_result = ctx.get("chunk_result")
        sg = chunk_result["subgraph"]
        assert len(sg.entities) == 3
        assert len(sg.relations) == 2
        assert len(sg.chunks) == 2  # c1, c2 deduplicated

    @patch("retrico.query.chunk_retriever.ChunkRetrieverProcessor._ensure_store")
    @patch("retrico.query.retriever.RetrieverProcessor._ensure_store")
    def test_pipeline_with_reasoner(self, mock_ret_store, mock_chunk_store):
        """Test full pipeline with LLM reasoner."""
        builder = RetriCoSearch(name="test_query_reasoner")
        builder.query_parser(method="gliner", labels=["person", "location"])
        builder.retriever(neo4j_uri="bolt://localhost:7687")
        builder.chunk_retriever()
        builder.reasoner(api_key="test-key", model="test-model")

        executor = builder.build()

        # Mock parser via engine
        parser_proc = executor.processors["query_parser"]
        engine = parser_proc._get_gliner_engine()
        engine._model = MagicMock()
        engine._model.inference.return_value = [[
            {"text": "Einstein", "label": "person", "start": 10, "end": 18, "score": 0.95},
        ]]

        # Mock store
        mock_store = self._mock_store()
        executor.processors["retriever"]._store = mock_store
        executor.processors["chunk_retriever"]._store = mock_store

        # Mock reasoner LLM
        reasoner_proc = executor.processors["reasoner"]
        reasoner_proc._reasoner = MagicMock()
        reasoner_proc._reasoner.reason.return_value = QueryResult(
            query="Where was Einstein born?",
            answer="Einstein was born in Ulm, Germany.",
            metadata={"inferred_relation_count": 0},
        )

        ctx = executor.run(query="Where was Einstein born?")

        assert ctx.has("reasoner_result")
        result = ctx.get("reasoner_result")["result"]
        assert isinstance(result, QueryResult)
        assert result.answer == "Einstein was born in Ulm, Germany."

    @patch("retrico.query.chunk_retriever.ChunkRetrieverProcessor._ensure_store")
    @patch("retrico.query.retriever.RetrieverProcessor._ensure_store")
    def test_llm_parser_pipeline(self, mock_ret_store, mock_chunk_store):
        """Test pipeline with LLM-based query parser."""
        builder = RetriCoSearch(name="llm_parser_test")
        builder.query_parser(method="llm", api_key="test", labels=["person"])
        builder.retriever(neo4j_uri="bolt://localhost:7687")
        builder.chunk_retriever()

        executor = builder.build()

        # Mock LLM parser via engine
        parser_proc = executor.processors["query_parser"]
        engine = parser_proc._get_llm_engine()
        engine._client = MagicMock()
        engine._client.complete.return_value = json.dumps({"entities": [
            {"text": "Einstein", "label": "person"},
        ]})

        # Mock store
        mock_store = self._mock_store()
        executor.processors["retriever"]._store = mock_store
        executor.processors["chunk_retriever"]._store = mock_store

        ctx = executor.run(query="Tell me about Einstein")

        assert ctx.has("parser_result")
        parser_result = ctx.get("parser_result")
        assert len(parser_result["entities"]) == 1
        assert parser_result["entities"][0].text == "Einstein"

    @patch("retrico.query.chunk_retriever.ChunkRetrieverProcessor._ensure_store")
    @patch("retrico.query.retriever.RetrieverProcessor._ensure_store")
    def test_query_graph_convenience_without_reasoner(self, mock_ret_store, mock_chunk_store):
        """Test query_graph() convenience function without API key (no reasoner)."""
        import retrico

        # We need to mock at the processor level after build
        with patch.object(retrico.RetriCoSearch, "build") as mock_build:
            mock_executor = MagicMock()
            mock_ctx = MagicMock()
            mock_ctx.has.side_effect = lambda key: key == "chunk_result"
            mock_ctx.get.return_value = {
                "subgraph": retrico.Subgraph(
                    entities=[retrico.Entity(id="e1", label="Einstein")],
                ),
            }
            mock_executor.run.return_value = mock_ctx
            mock_build.return_value = mock_executor

            result = retrico.query_graph(
                query="Where was Einstein born?",
                entity_labels=["person", "location"],
            )

            assert isinstance(result, retrico.QueryResult)
            assert result.query == "Where was Einstein born?"

    def test_yaml_config_loading(self, tmp_path):
        """Test saving and loading query pipeline config."""
        builder = RetriCoSearch(name="yaml_test")
        builder.query_parser(method="gliner", labels=["person"])
        builder.retriever(neo4j_uri="bolt://localhost:7687")
        builder.chunk_retriever()

        yaml_path = str(tmp_path / "query.yaml")
        builder.save(yaml_path)

        from retrico.core.factory import ProcessorFactory
        executor = ProcessorFactory.create_pipeline(yaml_path)
        assert executor.pipeline.name == "yaml_test"
        assert len(executor.nodes_map) == 3
