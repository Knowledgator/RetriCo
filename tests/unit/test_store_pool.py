"""Tests for StorePool, resolve_from_pool_or_create, and vector store configs."""

import pytest
from unittest.mock import MagicMock, patch

from retrico.store.pool import StorePool, resolve_from_pool_or_create
from retrico.store.config import (
    BaseVectorStoreConfig, InMemoryVectorConfig, FaissVectorConfig,
    QdrantVectorConfig, GraphDBVectorConfig,
    BaseStoreConfig, Neo4jConfig, FalkorDBConfig, MemgraphConfig,
)


# ---------------------------------------------------------------------------
# Vector store config tests
# ---------------------------------------------------------------------------


class TestInMemoryVectorConfig:
    def test_defaults(self):
        cfg = InMemoryVectorConfig()
        assert cfg.vector_store_type == "in_memory"
        assert cfg.name == "default"

    def test_to_flat_dict(self):
        cfg = InMemoryVectorConfig()
        assert cfg.to_flat_dict() == {"vector_store_type": "in_memory"}

    def test_to_flat_dict_with_name(self):
        cfg = InMemoryVectorConfig(name="my_vec")
        flat = cfg.to_flat_dict()
        assert flat["vector_store_name"] == "my_vec"

    def test_roundtrip(self):
        original = InMemoryVectorConfig(name="test")
        rebuilt = InMemoryVectorConfig.from_flat_dict(original.to_flat_dict())
        assert original == rebuilt


class TestFaissVectorConfig:
    def test_defaults(self):
        cfg = FaissVectorConfig()
        assert cfg.vector_store_type == "faiss"
        assert cfg.use_gpu is False

    def test_to_flat_dict(self):
        cfg = FaissVectorConfig(use_gpu=True)
        flat = cfg.to_flat_dict()
        assert flat == {"vector_store_type": "faiss", "use_gpu": True}

    def test_roundtrip(self):
        original = FaissVectorConfig(use_gpu=True, name="gpu_vec")
        rebuilt = FaissVectorConfig.from_flat_dict(original.to_flat_dict())
        assert original == rebuilt


class TestQdrantVectorConfig:
    def test_defaults(self):
        cfg = QdrantVectorConfig()
        assert cfg.vector_store_type == "qdrant"
        assert cfg.url is None
        assert cfg.prefer_grpc is False

    def test_to_flat_dict(self):
        cfg = QdrantVectorConfig(url="http://localhost:6333", api_key="secret")
        flat = cfg.to_flat_dict()
        assert flat["qdrant_url"] == "http://localhost:6333"
        assert flat["qdrant_api_key"] == "secret"

    def test_roundtrip(self):
        original = QdrantVectorConfig(url="http://localhost:6333", prefer_grpc=True)
        rebuilt = QdrantVectorConfig.from_flat_dict(original.to_flat_dict())
        assert original == rebuilt


class TestGraphDBVectorConfig:
    def test_defaults(self):
        cfg = GraphDBVectorConfig()
        assert cfg.vector_store_type == "graph_db"
        assert cfg.graph_store_name == "default"

    def test_to_flat_dict(self):
        cfg = GraphDBVectorConfig(graph_store_name="analytics")
        flat = cfg.to_flat_dict()
        assert flat == {
            "vector_store_type": "graph_db",
            "graph_store_name": "analytics",
        }

    def test_roundtrip(self):
        original = GraphDBVectorConfig(graph_store_name="main", name="vec1")
        rebuilt = GraphDBVectorConfig.from_flat_dict(original.to_flat_dict())
        assert original == rebuilt


class TestBaseVectorStoreConfigDispatch:
    def test_from_flat_dict_in_memory(self):
        cfg = BaseVectorStoreConfig.from_flat_dict({})
        assert isinstance(cfg, InMemoryVectorConfig)

    def test_from_flat_dict_faiss(self):
        cfg = BaseVectorStoreConfig.from_flat_dict({"vector_store_type": "faiss", "use_gpu": True})
        assert isinstance(cfg, FaissVectorConfig)
        assert cfg.use_gpu is True

    def test_from_flat_dict_qdrant(self):
        cfg = BaseVectorStoreConfig.from_flat_dict({"vector_store_type": "qdrant"})
        assert isinstance(cfg, QdrantVectorConfig)

    def test_from_flat_dict_graph_db(self):
        cfg = BaseVectorStoreConfig.from_flat_dict({
            "vector_store_type": "graph_db", "graph_store_name": "main",
        })
        assert isinstance(cfg, GraphDBVectorConfig)
        assert cfg.graph_store_name == "main"

    def test_from_flat_dict_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown vector_store_type"):
            BaseVectorStoreConfig.from_flat_dict({"vector_store_type": "weaviate"})


# ---------------------------------------------------------------------------
# BaseStoreConfig name field tests
# ---------------------------------------------------------------------------


class TestStoreConfigName:
    def test_neo4j_name_default(self):
        cfg = Neo4jConfig()
        assert cfg.name == "default"
        # Default name not included in flat dict
        assert "store_name" not in cfg.to_flat_dict()

    def test_neo4j_name_custom(self):
        cfg = Neo4jConfig(name="main")
        flat = cfg.to_flat_dict()
        assert flat["store_name"] == "main"

    def test_neo4j_roundtrip_with_name(self):
        cfg = Neo4jConfig(uri="bolt://x:7687", name="analytics")
        rebuilt = Neo4jConfig.from_flat_dict(cfg.to_flat_dict())
        assert rebuilt.name == "analytics"
        assert rebuilt.uri == "bolt://x:7687"

    def test_falkordb_name(self):
        cfg = FalkorDBConfig(name="cache_db")
        flat = cfg.to_flat_dict()
        assert flat["store_name"] == "cache_db"

    def test_memgraph_name(self):
        cfg = MemgraphConfig(name="fast_db")
        flat = cfg.to_flat_dict()
        assert flat["store_name"] == "fast_db"

    def test_base_from_flat_dict_preserves_name(self):
        flat = {"store_type": "neo4j", "store_name": "my_store"}
        cfg = BaseStoreConfig.from_flat_dict(flat)
        assert cfg.name == "my_store"


# ---------------------------------------------------------------------------
# StorePool tests
# ---------------------------------------------------------------------------


class TestStorePool:
    def test_register_and_has_graph(self):
        pool = StorePool()
        assert not pool.has_graph("default")
        pool.register_graph("default", {"store_type": "neo4j"})
        assert pool.has_graph("default")

    def test_register_and_has_vector(self):
        pool = StorePool()
        assert not pool.has_vector("default")
        pool.register_vector("default", {"vector_store_type": "in_memory"})
        assert pool.has_vector("default")

    def test_register_and_has_relational(self):
        pool = StorePool()
        assert not pool.has_relational("default")
        pool.register_relational("default", {"relational_store_type": "sqlite"})
        assert pool.has_relational("default")

    def test_get_graph_unknown_raises(self):
        pool = StorePool()
        with pytest.raises(KeyError, match="No graph store"):
            pool.get_graph("nonexistent")

    def test_get_vector_unknown_raises(self):
        pool = StorePool()
        with pytest.raises(KeyError, match="No vector store"):
            pool.get_vector("nonexistent")

    def test_get_relational_unknown_raises(self):
        pool = StorePool()
        with pytest.raises(KeyError, match="No relational store"):
            pool.get_relational("nonexistent")

    @patch("retrico.store.graph.create_graph_store")
    def test_lazy_creation(self, mock_create):
        mock_store = MagicMock()
        mock_create.return_value = mock_store

        pool = StorePool()
        pool.register_graph("default", {"store_type": "neo4j"})

        # Not created yet
        mock_create.assert_not_called()

        # First get creates it
        result = pool.get_graph("default")
        assert result is mock_store
        mock_create.assert_called_once()

    @patch("retrico.store.graph.create_graph_store")
    def test_shared_instance(self, mock_create):
        """get_graph() returns the same instance on repeated calls."""
        mock_store = MagicMock()
        mock_create.return_value = mock_store

        pool = StorePool()
        pool.register_graph("db1", {"store_type": "neo4j"})

        first = pool.get_graph("db1")
        second = pool.get_graph("db1")
        assert first is second
        assert mock_create.call_count == 1

    @patch("retrico.store.graph.create_graph_store")
    def test_multiple_named_stores(self, mock_create):
        """Different names create different instances."""
        stores = {}
        def side_effect(cfg):
            s = MagicMock()
            stores[cfg.get("store_type", "neo4j")] = s
            return s
        mock_create.side_effect = side_effect

        pool = StorePool()
        pool.register_graph("main", {"store_type": "neo4j"})
        pool.register_graph("analytics", {"store_type": "neo4j"})

        s1 = pool.get_graph("main")
        s2 = pool.get_graph("analytics")
        assert s1 is not s2
        assert mock_create.call_count == 2

    @patch("retrico.store.graph.create_graph_store")
    def test_close_all(self, mock_create):
        """close() closes all instantiated stores."""
        mock_store = MagicMock()
        mock_create.return_value = mock_store

        pool = StorePool()
        pool.register_graph("default", {"store_type": "neo4j"})
        pool.get_graph("default")  # trigger creation

        pool.close()
        mock_store.close.assert_called_once()

    @patch("retrico.store.graph.create_graph_store")
    def test_close_only_instantiated(self, mock_create):
        """close() only closes stores that were actually created."""
        pool = StorePool()
        pool.register_graph("lazy", {"store_type": "neo4j"})
        # Never call get_graph — store not created
        pool.close()
        mock_create.assert_not_called()

    def test_to_dict_from_dict_roundtrip(self):
        pool = StorePool()
        pool.register_graph("main", {"store_type": "neo4j", "neo4j_uri": "bolt://main:7687"})
        pool.register_graph("analytics", {"store_type": "neo4j", "neo4j_uri": "bolt://analytics:7687"})
        pool.register_vector("default", {"vector_store_type": "faiss", "use_gpu": True})

        data = pool.to_dict()
        assert "graph" in data
        assert "main" in data["graph"]
        assert "vector" in data

        rebuilt = StorePool.from_dict(data)
        assert rebuilt.has_graph("main")
        assert rebuilt.has_graph("analytics")
        assert rebuilt.has_vector("default")

    def test_to_dict_empty(self):
        pool = StorePool()
        assert pool.to_dict() == {}

    @patch("retrico.store.graph.create_graph_store")
    @patch("retrico.store.vector.create_vector_store")
    def test_graph_db_vector_injects_graph_store(self, mock_cvs, mock_cgs):
        """graph_db vector stores get the shared graph store instance."""
        mock_graph = MagicMock()
        mock_cgs.return_value = mock_graph
        mock_vec = MagicMock()
        mock_cvs.return_value = mock_vec

        pool = StorePool()
        pool.register_graph("default", {"store_type": "neo4j"})
        pool.register_vector("default", {"vector_store_type": "graph_db", "graph_store_name": "default"})

        pool.get_vector("default")

        # Check that create_vector_store was called with __graph_store_instance__ set
        call_config = mock_cvs.call_args[0][0]
        assert call_config["__graph_store_instance__"] is mock_graph


# ---------------------------------------------------------------------------
# resolve_from_pool_or_create tests
# ---------------------------------------------------------------------------


class TestResolveFromPoolOrCreate:

    @patch("retrico.store.graph.create_graph_store")
    def test_no_pool_falls_back_to_create(self, mock_create):
        """Without __store_pool__, falls back to create_graph_store."""
        mock_store = MagicMock()
        mock_create.return_value = mock_store

        result = resolve_from_pool_or_create({"store_type": "neo4j"}, "graph")
        assert result is mock_store
        mock_create.assert_called_once()

    @patch("retrico.store.graph.create_graph_store")
    def test_with_pool_uses_pool(self, mock_create):
        """With pool, uses the shared instance instead of creating new."""
        pool = StorePool()
        mock_shared = MagicMock()
        pool._graph_configs["default"] = {"store_type": "neo4j"}
        pool._graph_instances["default"] = mock_shared

        result = resolve_from_pool_or_create(
            {"store_type": "neo4j", "__store_pool__": pool}, "graph"
        )
        assert result is mock_shared
        mock_create.assert_not_called()

    @patch("retrico.store.graph.create_graph_store")
    def test_with_pool_named_store(self, mock_create):
        """Pool resolves named stores."""
        pool = StorePool()
        mock_main = MagicMock()
        pool._graph_configs["main"] = {"store_type": "neo4j"}
        pool._graph_instances["main"] = mock_main

        result = resolve_from_pool_or_create(
            {"graph_store_name": "main", "__store_pool__": pool}, "graph"
        )
        assert result is mock_main

    @patch("retrico.store.vector.create_vector_store")
    def test_vector_category(self, mock_create):
        """Vector stores work with resolve_from_pool_or_create."""
        mock_vs = MagicMock()
        mock_create.return_value = mock_vs

        result = resolve_from_pool_or_create(
            {"vector_store_type": "in_memory"}, "vector"
        )
        assert result is mock_vs

    def test_unknown_category_raises(self):
        with pytest.raises(ValueError, match="Unknown store category"):
            resolve_from_pool_or_create({}, "invalid")

    @patch("retrico.store.graph.create_graph_store")
    def test_pool_without_matching_name_falls_back(self, mock_create):
        """If pool doesn't have the requested name, falls back to direct creation."""
        pool = StorePool()
        pool.register_graph("other", {"store_type": "neo4j"})

        mock_store = MagicMock()
        mock_create.return_value = mock_store

        result = resolve_from_pool_or_create(
            {"store_type": "neo4j", "__store_pool__": pool, "graph_store_name": "unknown"},
            "graph"
        )
        assert result is mock_store
        mock_create.assert_called_once()


# ---------------------------------------------------------------------------
# Builder pool integration tests
# ---------------------------------------------------------------------------


class TestBuilderPoolIntegration:
    def test_builder_emits_stores_section(self):
        """get_config() includes stores section when graph_store() is called."""
        from retrico.core.builders import RetriCoBuilder

        builder = RetriCoBuilder(name="test")
        builder.graph_store(Neo4jConfig(uri="bolt://custom:7687"))
        builder.ner_gliner(labels=["person"])
        builder.graph_writer()

        config = builder.get_config()
        assert "stores" in config
        assert "graph" in config["stores"]
        assert "default" in config["stores"]["graph"]
        assert config["stores"]["graph"]["default"]["neo4j_uri"] == "bolt://custom:7687"

    def test_builder_multiple_named_stores(self):
        from retrico.core.builders import RetriCoBuilder

        builder = RetriCoBuilder(name="test")
        builder.graph_store(Neo4jConfig(uri="bolt://main:7687"), name="main")
        builder.graph_store(Neo4jConfig(uri="bolt://analytics:7687"), name="analytics")
        builder.ner_gliner(labels=["person"])
        builder.graph_writer()

        config = builder.get_config()
        assert "main" in config["stores"]["graph"]
        assert "analytics" in config["stores"]["graph"]

    def test_builder_vector_store_in_stores_section(self):
        from retrico.core.builders import RetriCoBuilder

        builder = RetriCoBuilder(name="test")
        builder.graph_store(Neo4jConfig())
        builder.vector_store(type="faiss", use_gpu=True)
        builder.ner_gliner(labels=["person"])
        builder.graph_writer()

        config = builder.get_config()
        assert "vector" in config["stores"]
        assert config["stores"]["vector"]["default"]["vector_store_type"] == "faiss"

    def test_builder_vector_store_config_object(self):
        from retrico.core.builders import RetriCoBuilder

        builder = RetriCoBuilder(name="test")
        builder.graph_store(Neo4jConfig())
        builder.vector_store(config=FaissVectorConfig(use_gpu=True, name="gpu_vec"))
        builder.ner_gliner(labels=["person"])
        builder.graph_writer()

        config = builder.get_config()
        assert "gpu_vec" in config["stores"]["vector"]

    def test_builder_no_stores_section_when_empty(self):
        """Without calling graph_store(), no stores section emitted."""
        from retrico.core.builders import RetriCoBuilder

        builder = RetriCoBuilder(name="test")
        builder.ner_gliner(labels=["person"])
        # graph_writer uses defaults internally
        builder._writer_config = {"store_type": "neo4j"}

        config = builder.get_config()
        assert "stores" not in config

    def test_query_builder_emits_stores(self):
        from retrico.core.builders import RetriCoSearch

        builder = RetriCoSearch(name="test")
        builder.graph_store(Neo4jConfig(uri="bolt://custom:7687"))
        builder.query_parser(labels=["person"])
        builder.retriever()

        config = builder.get_config()
        assert "stores" in config
        assert config["stores"]["graph"]["default"]["neo4j_uri"] == "bolt://custom:7687"


# ---------------------------------------------------------------------------
# DAGExecutor pool integration tests
# ---------------------------------------------------------------------------


class TestDAGExecutorPool:
    def test_context_manager(self):
        """DAGExecutor supports with-statement and calls pool.close()."""
        from retrico.core.dag import DAGExecutor, DAGPipeline

        pool = MagicMock()
        pipeline = DAGPipeline(name="test", nodes=[])
        executor = DAGExecutor(pipeline, store_pool=pool)

        with executor:
            pass

        pool.close.assert_called_once()

    def test_close_without_pool(self):
        """close() is safe to call when no pool is set."""
        from retrico.core.dag import DAGExecutor, DAGPipeline

        pipeline = DAGPipeline(name="test", nodes=[])
        executor = DAGExecutor(pipeline)
        executor.close()  # should not raise

    def test_pool_injected_into_processor_config(self):
        """Store pool is injected into each processor's config via __store_pool__."""
        from retrico.core.dag import DAGExecutor, DAGPipeline, PipeNode, OutputConfig

        pool = StorePool()

        # Use the chunker processor which always exists
        node = PipeNode(
            id="chunker",
            processor="chunker",
            inputs={},
            output=OutputConfig(key="chunker_result"),
            config={"method": "sentence"},
        )
        pipeline = DAGPipeline(name="test", nodes=[node])
        executor = DAGExecutor(pipeline, store_pool=pool)

        # Check that the processor got the pool in its config
        proc = executor.processors["chunker"]
        assert proc.config_dict.get("__store_pool__") is pool


class TestProcessorFactoryPool:
    def test_auto_detect_stores_section(self):
        """ProcessorFactory auto-creates pool from 'stores' key in config."""
        from retrico.core.factory import ProcessorFactory

        config = {
            "name": "test",
            "stores": {
                "graph": {
                    "default": {"store_type": "neo4j", "neo4j_uri": "bolt://test:7687"},
                },
            },
            "nodes": [
                {
                    "id": "chunker",
                    "processor": "chunker",
                    "inputs": {},
                    "output": {"key": "chunker_result"},
                    "config": {"method": "sentence"},
                },
            ],
        }

        executor = ProcessorFactory.create_from_dict(config)
        assert executor.store_pool is not None
        assert executor.store_pool.has_graph("default")
