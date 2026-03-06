"""Tests for store config classes and builder store integration."""

import pytest
from retrico.store.config import (
    BaseStoreConfig, Neo4jConfig, FalkorDBConfig, FalkorDBLiteConfig, MemgraphConfig,
    resolve_store_config, extract_store_kwargs, _STORE_FLAT_KEYS,
    BaseRelationalStoreConfig, SqliteRelationalConfig,
    PostgresRelationalConfig, ElasticsearchRelationalConfig,
    resolve_relational_store_config, extract_relational_store_kwargs,
)
from retrico.core.builders import (
    RetriCoBuilder, RetriCoIngest, RetriCoSearch,
    RetriCoCommunity, RetriCoModeling,
)


# ---------------------------------------------------------------------------
# Config class tests
# ---------------------------------------------------------------------------

class TestNeo4jConfig:
    def test_defaults(self):
        cfg = Neo4jConfig()
        assert cfg.store_type == "neo4j"
        assert cfg.uri == "bolt://localhost:7687"
        assert cfg.user == "neo4j"
        assert cfg.password == "password"
        assert cfg.database == "neo4j"

    def test_custom_values(self):
        cfg = Neo4jConfig(uri="bolt://myhost:7688", password="secret")
        assert cfg.uri == "bolt://myhost:7688"
        assert cfg.password == "secret"

    def test_to_flat_dict(self):
        cfg = Neo4jConfig(uri="bolt://x:7687", password="p")
        flat = cfg.to_flat_dict()
        assert flat == {
            "store_type": "neo4j",
            "neo4j_uri": "bolt://x:7687",
            "neo4j_user": "neo4j",
            "neo4j_password": "p",
            "neo4j_database": "neo4j",
        }

    def test_from_flat_dict(self):
        flat = {"neo4j_uri": "bolt://y:7687", "neo4j_password": "s"}
        cfg = Neo4jConfig.from_flat_dict(flat)
        assert cfg.uri == "bolt://y:7687"
        assert cfg.password == "s"
        assert cfg.user == "neo4j"  # default

    def test_roundtrip(self):
        original = Neo4jConfig(uri="bolt://rnd:7687", user="u", password="p", database="d")
        rebuilt = Neo4jConfig.from_flat_dict(original.to_flat_dict())
        assert original == rebuilt


class TestFalkorDBConfig:
    def test_defaults(self):
        cfg = FalkorDBConfig()
        assert cfg.store_type == "falkordb"
        assert cfg.host == "localhost"
        assert cfg.port == 6379
        assert cfg.graph == "retrico"

    def test_to_flat_dict(self):
        cfg = FalkorDBConfig(host="myhost", port=6380, graph="mygraph")
        flat = cfg.to_flat_dict()
        assert flat == {
            "store_type": "falkordb",
            "falkordb_host": "myhost",
            "falkordb_port": 6380,
            "falkordb_graph": "mygraph",
        }

    def test_roundtrip(self):
        original = FalkorDBConfig(host="h", port=1234, graph="g")
        rebuilt = FalkorDBConfig.from_flat_dict(original.to_flat_dict())
        assert original == rebuilt


class TestMemgraphConfig:
    def test_defaults(self):
        cfg = MemgraphConfig()
        assert cfg.store_type == "memgraph"
        assert cfg.uri == "bolt://localhost:7687"
        assert cfg.user == ""
        assert cfg.password == ""

    def test_to_flat_dict(self):
        cfg = MemgraphConfig(uri="bolt://mg:7687")
        flat = cfg.to_flat_dict()
        assert flat["store_type"] == "memgraph"
        assert flat["memgraph_uri"] == "bolt://mg:7687"

    def test_roundtrip(self):
        original = MemgraphConfig(uri="bolt://mg:7687", user="u", password="p", database="d")
        rebuilt = MemgraphConfig.from_flat_dict(original.to_flat_dict())
        assert original == rebuilt


class TestBaseStoreConfigDispatch:
    def test_from_flat_dict_neo4j(self):
        cfg = BaseStoreConfig.from_flat_dict({"store_type": "neo4j", "neo4j_uri": "bolt://x:7687"})
        assert isinstance(cfg, Neo4jConfig)
        assert cfg.uri == "bolt://x:7687"

    def test_from_flat_dict_falkordb(self):
        cfg = BaseStoreConfig.from_flat_dict({"store_type": "falkordb", "falkordb_host": "h"})
        assert isinstance(cfg, FalkorDBConfig)
        assert cfg.host == "h"

    def test_from_flat_dict_memgraph(self):
        cfg = BaseStoreConfig.from_flat_dict({"store_type": "memgraph"})
        assert isinstance(cfg, MemgraphConfig)

    def test_from_flat_dict_default_falkordb_lite(self):
        cfg = BaseStoreConfig.from_flat_dict({})
        assert isinstance(cfg, FalkorDBLiteConfig)

    def test_from_flat_dict_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown store_type"):
            BaseStoreConfig.from_flat_dict({"store_type": "unknown"})


# ---------------------------------------------------------------------------
# resolve_store_config tests
# ---------------------------------------------------------------------------

class TestResolveStoreConfig:
    def test_with_config_object(self):
        cfg = Neo4jConfig(uri="bolt://x:7687")
        result = resolve_store_config(cfg)
        assert result is cfg

    def test_with_kwargs(self):
        result = resolve_store_config(store_type="falkordb", falkordb_host="myhost")
        assert isinstance(result, FalkorDBConfig)
        assert result.host == "myhost"

    def test_config_plus_kwargs_override(self):
        cfg = Neo4jConfig(uri="bolt://original:7687")
        result = resolve_store_config(cfg, neo4j_password="new_pass")
        assert isinstance(result, Neo4jConfig)
        assert result.uri == "bolt://original:7687"
        assert result.password == "new_pass"

    def test_no_args_defaults_to_falkordb_lite(self):
        result = resolve_store_config()
        assert isinstance(result, FalkorDBLiteConfig)


class TestExtractStoreKwargs:
    def test_pops_store_keys(self):
        kwargs = {
            "neo4j_uri": "bolt://x:7687",
            "max_hops": 2,
            "store_type": "neo4j",
        }
        store_kw = extract_store_kwargs(kwargs)
        assert store_kw == {"neo4j_uri": "bolt://x:7687", "store_type": "neo4j"}
        assert kwargs == {"max_hops": 2}

    def test_no_store_keys(self):
        kwargs = {"max_hops": 2, "top_k": 5}
        store_kw = extract_store_kwargs(kwargs)
        assert store_kw == {}
        assert kwargs == {"max_hops": 2, "top_k": 5}


# ---------------------------------------------------------------------------
# Builder store integration tests
# ---------------------------------------------------------------------------

class TestBuilderStoreConfig:
    def test_build_builder_store_propagates(self):
        """builder.store(config) should propagate to graph_writer."""
        builder = RetriCoBuilder(name="test")
        builder.store(Neo4jConfig(uri="bolt://custom:7687", password="secret"))
        builder.ner_gliner(labels=["person"])
        builder.graph_writer()  # no store params — should inherit

        config = builder.get_config()
        writer = [n for n in config["nodes"] if n["id"] == "graph_writer"][0]
        assert writer["config"]["neo4j_uri"] == "bolt://custom:7687"
        assert writer["config"]["neo4j_password"] == "secret"

    def test_build_builder_legacy_store_config_works(self):
        """graph_writer(store_config=Neo4jConfig(...)) works."""
        builder = RetriCoBuilder()
        builder.ner_gliner(labels=["person"])
        builder.graph_writer(store_config=Neo4jConfig(uri="bolt://old:7687", password="oldpass"))

        config = builder.get_config()
        writer = [n for n in config["nodes"] if n["id"] == "graph_writer"][0]
        assert writer["config"]["neo4j_uri"] == "bolt://old:7687"
        assert writer["config"]["neo4j_password"] == "oldpass"

    def test_graph_writer_override_store(self):
        """graph_writer(store_config=...) overrides builder.store()."""
        builder = RetriCoBuilder(name="test")
        builder.store(Neo4jConfig(uri="bolt://builder:7687"))
        builder.ner_gliner(labels=["person"])
        builder.graph_writer(store_config=FalkorDBConfig(host="override"))

        config = builder.get_config()
        writer = [n for n in config["nodes"] if n["id"] == "graph_writer"][0]
        assert writer["config"]["store_type"] == "falkordb"
        assert writer["config"]["falkordb_host"] == "override"

    def test_ingest_builder_store_propagates(self):
        builder = RetriCoIngest(name="test")
        builder.store(Neo4jConfig(uri="bolt://custom:7687"))
        builder.graph_writer()

        config = builder.get_config()
        writer = [n for n in config["nodes"] if n["id"] == "graph_writer"][0]
        assert writer["config"]["neo4j_uri"] == "bolt://custom:7687"

    def test_query_builder_store_propagates(self):
        """builder.store() should propagate to retriever."""
        builder = RetriCoSearch(name="test")
        builder.store(Neo4jConfig(uri="bolt://custom:7687"))
        builder.query_parser(labels=["person"])
        builder.retriever()  # no store params — should inherit

        config = builder.get_config()
        retriever = [n for n in config["nodes"] if n["id"] == "retriever"][0]
        assert retriever["config"]["neo4j_uri"] == "bolt://custom:7687"

    def test_query_retriever_override(self):
        """Retriever-level store_config overrides builder-level."""
        builder = RetriCoSearch(name="test")
        builder.store(Neo4jConfig(uri="bolt://builder:7687"))
        builder.query_parser(labels=["person"])
        builder.retriever(store_config=FalkorDBConfig(host="override"))

        config = builder.get_config()
        retriever = [n for n in config["nodes"] if n["id"] == "retriever"][0]
        assert retriever["config"]["store_type"] == "falkordb"
        assert retriever["config"]["falkordb_host"] == "override"

    def test_chunk_retriever_inherits_from_retriever(self):
        """chunk_retriever should inherit store config from retriever."""
        builder = RetriCoSearch(name="test")
        builder.query_parser(labels=["person"])
        builder.retriever(neo4j_uri="bolt://test:7687", neo4j_password="testpass")

        config = builder.get_config()
        chunk = [n for n in config["nodes"] if n["id"] == "chunk_retriever"][0]
        assert chunk["config"]["neo4j_uri"] == "bolt://test:7687"
        assert chunk["config"]["neo4j_password"] == "testpass"

    def test_community_builder_store_propagates(self):
        builder = RetriCoCommunity(name="test")
        builder.store(Neo4jConfig(uri="bolt://custom:7687"))
        builder.detector(method="louvain")

        config = builder.get_config()
        detector = config["nodes"][0]
        assert detector["config"]["neo4j_uri"] == "bolt://custom:7687"

    def test_kg_modeling_builder_store_propagates(self):
        builder = RetriCoModeling(name="test")
        builder.store(Neo4jConfig(uri="bolt://custom:7687"))
        builder.triple_reader()

        config = builder.get_config()
        reader = config["nodes"][0]
        assert reader["config"]["neo4j_uri"] == "bolt://custom:7687"


class TestBuilderStoreTypes:
    """Test graph_store(), vector_store(), chunk_store() methods."""

    def test_graph_store_alias(self):
        """store() and graph_store() are equivalent."""
        b1 = RetriCoBuilder(name="t1")
        b1.store(Neo4jConfig(uri="bolt://a:7687"))

        b2 = RetriCoBuilder(name="t2")
        b2.graph_store(Neo4jConfig(uri="bolt://a:7687"))

        # Both set the same internal config
        assert b1._store_config == b2._store_config

    def test_graph_store_with_kwargs(self):
        builder = RetriCoBuilder(name="test")
        builder.graph_store(store_type="falkordb", falkordb_host="myhost")
        assert isinstance(builder._store_config, FalkorDBConfig)
        assert builder._store_config.host == "myhost"

    def test_vector_store_sets_config(self):
        builder = RetriCoBuilder(name="test")
        builder.vector_store(type="faiss", use_gpu=True)
        assert builder._vector_store_config == {"vector_store_type": "faiss", "use_gpu": True}

    def test_vector_store_propagates_to_embedder(self):
        """vector_store() config is inherited by chunk_embedder()."""
        builder = RetriCoBuilder(name="test")
        builder.graph_store(Neo4jConfig())
        builder.vector_store(type="faiss")
        builder.ner_gliner(labels=["person"])
        builder.graph_writer()
        builder.chunk_embedder()

        config = builder.get_config()
        embedder = [n for n in config["nodes"] if n["id"] == "chunk_embedder"][0]
        assert embedder["config"]["vector_store_type"] == "faiss"

    def test_vector_store_overridden_by_explicit(self):
        """Explicit vector_store_type in chunk_embedder() wins over builder-level."""
        builder = RetriCoBuilder(name="test")
        builder.vector_store(type="faiss")
        builder.ner_gliner(labels=["person"])
        builder.graph_writer()
        builder.chunk_embedder(vector_store_type="qdrant")

        config = builder.get_config()
        embedder = [n for n in config["nodes"] if n["id"] == "chunk_embedder"][0]
        assert embedder["config"]["vector_store_type"] == "qdrant"

    def test_chunk_store_sets_config(self):
        builder = RetriCoBuilder(name="test")
        builder.chunk_store(type="sqlite", sqlite_path="/tmp/test.db")
        assert builder._chunk_store_config == {"relational_store_type": "sqlite", "sqlite_path": "/tmp/test.db"}

    def test_vector_store_propagates_to_query_retriever(self):
        """vector_store() config is inherited by embedding retrievers."""
        builder = RetriCoSearch(name="test")
        builder.graph_store(Neo4jConfig())
        builder.vector_store(type="faiss")
        builder.query_parser(labels=["person"])
        builder.entity_embedding_retriever()

        config = builder.get_config()
        retriever = [n for n in config["nodes"] if n["id"] == "retriever"][0]
        assert retriever["config"]["vector_store_type"] == "faiss"


class TestCreateStoreWithConfig:
    def test_create_store_with_config_object(self):
        """create_store() accepts BaseStoreConfig objects."""
        from unittest.mock import patch, MagicMock
        from retrico.store import create_store

        with patch("retrico.store.graph.neo4j_store.Neo4jGraphStore") as MockStore:
            MockStore.return_value = MagicMock()
            cfg = Neo4jConfig(uri="bolt://test:7687", password="p")
            store = create_store(cfg)
            MockStore.assert_called_once_with(
                uri="bolt://test:7687", user="neo4j", password="p", database="neo4j",
            )

    def test_create_store_with_dict_still_works(self):
        """create_store(dict) backward compat — default is now falkordb_lite."""
        from unittest.mock import patch, MagicMock
        from retrico.store import create_store

        with patch("retrico.store.graph.falkordb_lite_store.FalkorDBLiteGraphStore") as MockStore:
            MockStore.return_value = MagicMock()
            store = create_store({})
            MockStore.assert_called_once()


# ---------------------------------------------------------------------------
# Relational store config tests
# ---------------------------------------------------------------------------


class TestSqliteRelationalConfig:
    def test_defaults(self):
        cfg = SqliteRelationalConfig()
        assert cfg.relational_store_type == "sqlite"
        assert cfg.path == ":memory:"

    def test_custom_values(self):
        cfg = SqliteRelationalConfig(path="/tmp/test.db")
        assert cfg.path == "/tmp/test.db"

    def test_to_flat_dict(self):
        cfg = SqliteRelationalConfig(path="/data/chunks.db")
        flat = cfg.to_flat_dict()
        assert flat == {
            "relational_store_type": "sqlite",
            "sqlite_path": "/data/chunks.db",
        }

    def test_from_flat_dict(self):
        flat = {"sqlite_path": "/data/chunks.db"}
        cfg = SqliteRelationalConfig.from_flat_dict(flat)
        assert cfg.path == "/data/chunks.db"

    def test_roundtrip(self):
        original = SqliteRelationalConfig(path="/tmp/test.db")
        rebuilt = SqliteRelationalConfig.from_flat_dict(original.to_flat_dict())
        assert original == rebuilt

    def test_named(self):
        cfg = SqliteRelationalConfig(path="/tmp/test.db", name="chunks")
        flat = cfg.to_flat_dict()
        assert flat["relational_store_name"] == "chunks"


class TestPostgresRelationalConfig:
    def test_defaults(self):
        cfg = PostgresRelationalConfig()
        assert cfg.relational_store_type == "postgres"
        assert cfg.host == "localhost"
        assert cfg.port == 5432
        assert cfg.user == "postgres"
        assert cfg.password == ""
        assert cfg.database == "retrico"

    def test_to_flat_dict(self):
        cfg = PostgresRelationalConfig(host="pghost", port=5433, password="secret")
        flat = cfg.to_flat_dict()
        assert flat == {
            "relational_store_type": "postgres",
            "postgres_host": "pghost",
            "postgres_port": 5433,
            "postgres_user": "postgres",
            "postgres_password": "secret",
            "postgres_database": "retrico",
        }

    def test_roundtrip(self):
        original = PostgresRelationalConfig(
            host="h", port=5433, user="u", password="p", database="d",
        )
        rebuilt = PostgresRelationalConfig.from_flat_dict(original.to_flat_dict())
        assert original == rebuilt


class TestElasticsearchRelationalConfig:
    def test_defaults(self):
        cfg = ElasticsearchRelationalConfig()
        assert cfg.relational_store_type == "elasticsearch"
        assert cfg.url == "http://localhost:9200"
        assert cfg.api_key is None
        assert cfg.index_prefix == "retrico_"

    def test_to_flat_dict(self):
        cfg = ElasticsearchRelationalConfig(
            url="http://es:9200", api_key="key", index_prefix="app_",
        )
        flat = cfg.to_flat_dict()
        assert flat == {
            "relational_store_type": "elasticsearch",
            "elasticsearch_url": "http://es:9200",
            "elasticsearch_api_key": "key",
            "elasticsearch_index_prefix": "app_",
        }

    def test_to_flat_dict_omits_none_api_key(self):
        cfg = ElasticsearchRelationalConfig()
        flat = cfg.to_flat_dict()
        assert "elasticsearch_api_key" not in flat

    def test_roundtrip(self):
        original = ElasticsearchRelationalConfig(
            url="http://es:9200", api_key="key", index_prefix="app_",
        )
        rebuilt = ElasticsearchRelationalConfig.from_flat_dict(original.to_flat_dict())
        assert original == rebuilt


class TestBaseRelationalStoreConfigDispatch:
    def test_from_flat_dict_sqlite(self):
        cfg = BaseRelationalStoreConfig.from_flat_dict(
            {"relational_store_type": "sqlite", "sqlite_path": "/tmp/test.db"}
        )
        assert isinstance(cfg, SqliteRelationalConfig)
        assert cfg.path == "/tmp/test.db"

    def test_from_flat_dict_postgres(self):
        cfg = BaseRelationalStoreConfig.from_flat_dict(
            {"relational_store_type": "postgres", "postgres_host": "h"}
        )
        assert isinstance(cfg, PostgresRelationalConfig)
        assert cfg.host == "h"

    def test_from_flat_dict_elasticsearch(self):
        cfg = BaseRelationalStoreConfig.from_flat_dict(
            {"relational_store_type": "elasticsearch"}
        )
        assert isinstance(cfg, ElasticsearchRelationalConfig)

    def test_from_flat_dict_default_sqlite(self):
        cfg = BaseRelationalStoreConfig.from_flat_dict({})
        assert isinstance(cfg, SqliteRelationalConfig)

    def test_from_flat_dict_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown relational_store_type"):
            BaseRelationalStoreConfig.from_flat_dict(
                {"relational_store_type": "unknown"}
            )


class TestResolveRelationalStoreConfig:
    def test_with_config_object(self):
        cfg = SqliteRelationalConfig(path="/tmp/test.db")
        result = resolve_relational_store_config(cfg)
        assert result is cfg

    def test_with_kwargs(self):
        result = resolve_relational_store_config(
            relational_store_type="postgres", postgres_host="myhost",
        )
        assert isinstance(result, PostgresRelationalConfig)
        assert result.host == "myhost"

    def test_no_args_defaults_to_sqlite(self):
        result = resolve_relational_store_config()
        assert isinstance(result, SqliteRelationalConfig)


class TestExtractRelationalStoreKwargs:
    def test_pops_relational_keys(self):
        kwargs = {
            "sqlite_path": "/tmp/test.db",
            "max_hops": 2,
            "relational_store_type": "sqlite",
        }
        rel_kw = extract_relational_store_kwargs(kwargs)
        assert rel_kw == {
            "sqlite_path": "/tmp/test.db",
            "relational_store_type": "sqlite",
        }
        assert kwargs == {"max_hops": 2}

    def test_no_relational_keys(self):
        kwargs = {"max_hops": 2, "top_k": 5}
        rel_kw = extract_relational_store_kwargs(kwargs)
        assert rel_kw == {}
        assert kwargs == {"max_hops": 2, "top_k": 5}
