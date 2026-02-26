"""Tests for retrieval strategy wiring in QueryConfigBuilder and query_graph()."""

import pytest
from unittest.mock import MagicMock, patch

from grapsit.core.builders import QueryConfigBuilder


class TestQueryConfigBuilderStrategies:
    """Test that QueryConfigBuilder generates correct DAG wiring per strategy."""

    def test_entity_strategy_requires_parser(self):
        builder = QueryConfigBuilder(name="test")
        builder.retriever(neo4j_uri="bolt://localhost:7687")
        with pytest.raises(ValueError, match="Parser or linker"):
            builder.get_config()

    def test_entity_strategy_dag(self):
        builder = QueryConfigBuilder(name="test")
        builder.query_parser(labels=["person"])
        builder.retriever(neo4j_uri="bolt://localhost:7687")
        builder.chunk_retriever()
        config = builder.get_config()
        node_ids = [n["id"] for n in config["nodes"]]
        assert node_ids == ["query_parser", "retriever", "chunk_retriever"]
        retriever_node = config["nodes"][1]
        assert retriever_node["processor"] == "retriever"
        assert "entities" in retriever_node["inputs"]

    def test_community_strategy_no_parser_needed(self):
        builder = QueryConfigBuilder(name="test")
        builder.community_retriever(neo4j_uri="bolt://localhost:7687")
        builder.chunk_retriever()
        config = builder.get_config()
        node_ids = [n["id"] for n in config["nodes"]]
        assert node_ids == ["retriever", "chunk_retriever"]
        retriever_node = config["nodes"][0]
        assert retriever_node["processor"] == "community_retriever"
        assert "query" in retriever_node["inputs"]

    def test_chunk_embedding_strategy_no_parser_needed(self):
        builder = QueryConfigBuilder(name="test")
        builder.chunk_embedding_retriever(neo4j_uri="bolt://localhost:7687")
        builder.chunk_retriever()
        config = builder.get_config()
        node_ids = [n["id"] for n in config["nodes"]]
        assert node_ids == ["retriever", "chunk_retriever"]
        assert config["nodes"][0]["processor"] == "chunk_embedding_retriever"

    def test_entity_embedding_strategy_requires_parser(self):
        builder = QueryConfigBuilder(name="test")
        builder.entity_embedding_retriever(neo4j_uri="bolt://localhost:7687")
        with pytest.raises(ValueError, match="Parser or linker"):
            builder.get_config()

    def test_entity_embedding_strategy_dag(self):
        builder = QueryConfigBuilder(name="test")
        builder.query_parser(labels=["person"])
        builder.entity_embedding_retriever(neo4j_uri="bolt://localhost:7687")
        builder.chunk_retriever()
        config = builder.get_config()
        node_ids = [n["id"] for n in config["nodes"]]
        assert node_ids == ["query_parser", "retriever", "chunk_retriever"]
        assert config["nodes"][1]["processor"] == "entity_embedding_retriever"
        assert "entities" in config["nodes"][1]["inputs"]

    def test_tool_strategy_no_parser_needed(self):
        builder = QueryConfigBuilder(name="test")
        builder.tool_retriever(
            api_key="key", neo4j_uri="bolt://localhost:7687",
            entity_types=["person"], relation_types=["BORN_IN"],
        )
        builder.chunk_retriever()
        config = builder.get_config()
        node_ids = [n["id"] for n in config["nodes"]]
        assert node_ids == ["retriever", "chunk_retriever"]
        assert config["nodes"][0]["processor"] == "tool_retriever"
        assert "query" in config["nodes"][0]["inputs"]

    def test_path_strategy_requires_parser(self):
        builder = QueryConfigBuilder(name="test")
        builder.path_retriever(neo4j_uri="bolt://localhost:7687")
        with pytest.raises(ValueError, match="Parser or linker"):
            builder.get_config()

    def test_path_strategy_dag(self):
        builder = QueryConfigBuilder(name="test")
        builder.query_parser(labels=["person"])
        builder.path_retriever(neo4j_uri="bolt://localhost:7687")
        builder.chunk_retriever()
        config = builder.get_config()
        node_ids = [n["id"] for n in config["nodes"]]
        assert node_ids == ["query_parser", "retriever", "chunk_retriever"]
        assert config["nodes"][1]["processor"] == "path_retriever"

    def test_community_with_reasoner(self):
        builder = QueryConfigBuilder(name="test")
        builder.community_retriever(neo4j_uri="bolt://localhost:7687")
        builder.chunk_retriever()
        builder.reasoner(api_key="key", model="test")
        config = builder.get_config()
        node_ids = [n["id"] for n in config["nodes"]]
        assert node_ids == ["retriever", "chunk_retriever", "reasoner"]

    def test_store_config_inheritance(self):
        builder = QueryConfigBuilder(name="test")
        builder.community_retriever(
            neo4j_uri="bolt://custom:7687",
            store_type="neo4j",
        )
        builder.chunk_retriever()
        config = builder.get_config()
        # chunk_retriever should inherit store config
        chunk_config = config["nodes"][1]["config"]
        assert chunk_config["neo4j_uri"] == "bolt://custom:7687"

    def test_no_retriever_raises(self):
        builder = QueryConfigBuilder(name="test")
        builder.query_parser(labels=["person"])
        with pytest.raises(ValueError, match="Retriever config required"):
            builder.get_config()


class TestQueryGraphConvenience:
    """Test the query_graph() convenience function with retrieval_strategy param."""

    def test_unknown_strategy_raises(self):
        import grapsit
        with pytest.raises(ValueError, match="Unknown retrieval_strategy"):
            grapsit.query_graph(
                query="test",
                entity_labels=["person"],
                retrieval_strategy="unknown",
            )

    def test_entity_strategy_requires_labels(self):
        import grapsit
        with pytest.raises(ValueError, match="entity_labels required"):
            grapsit.query_graph(
                query="test",
                retrieval_strategy="entity",
            )

    @patch("grapsit.QueryConfigBuilder.build")
    def test_community_strategy_convenience(self, mock_build):
        import grapsit

        mock_executor = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.has.side_effect = lambda key: key == "chunk_result"
        mock_ctx.get.return_value = {"subgraph": grapsit.Subgraph()}
        mock_executor.execute.return_value = mock_ctx
        mock_build.return_value = mock_executor

        result = grapsit.query_graph(
            query="Tell me about community X",
            retrieval_strategy="community",
            retriever_kwargs={"top_k": 5},
        )

        assert isinstance(result, grapsit.QueryResult)

    @patch("grapsit.QueryConfigBuilder.build")
    def test_tool_strategy_inherits_api_key(self, mock_build):
        import grapsit

        mock_executor = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.has.side_effect = lambda key: key == "chunk_result"
        mock_ctx.get.return_value = {"subgraph": grapsit.Subgraph()}
        mock_executor.execute.return_value = mock_ctx
        mock_build.return_value = mock_executor

        result = grapsit.query_graph(
            query="query",
            api_key="sk-test",
            retrieval_strategy="tool",
            retriever_kwargs={"entity_types": ["person"]},
        )
        assert isinstance(result, grapsit.QueryResult)
