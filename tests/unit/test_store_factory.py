"""Tests for store factory function."""

import pytest
from unittest.mock import patch, MagicMock


class TestCreateStore:
    def test_default_creates_neo4j(self):
        with patch("grapsit.store.graph.neo4j_store.GraphDatabase"):
            from grapsit.store import create_store
            store = create_store({})
            from grapsit.store.graph.neo4j_store import Neo4jGraphStore
            assert isinstance(store, Neo4jGraphStore)

    def test_neo4j_explicit(self):
        with patch("grapsit.store.graph.neo4j_store.GraphDatabase"):
            from grapsit.store import create_store
            store = create_store({"store_type": "neo4j"})
            from grapsit.store.graph.neo4j_store import Neo4jGraphStore
            assert isinstance(store, Neo4jGraphStore)

    def test_neo4j_with_params(self):
        with patch("grapsit.store.graph.neo4j_store.GraphDatabase"):
            from grapsit.store import create_store
            store = create_store({
                "store_type": "neo4j",
                "neo4j_uri": "bolt://myhost:7687",
                "neo4j_user": "admin",
                "neo4j_password": "secret",
                "neo4j_database": "mydb",
            })
            assert store.uri == "bolt://myhost:7687"
            assert store.user == "admin"
            assert store.password == "secret"
            assert store.database == "mydb"

    def test_falkordb(self):
        from grapsit.store import create_store
        store = create_store({"store_type": "falkordb"})
        from grapsit.store.graph.falkordb_store import FalkorDBGraphStore
        assert isinstance(store, FalkorDBGraphStore)
        # Connection is lazy — no import of falkordb yet
        assert store._graph is None

    def test_falkordb_with_params(self):
        from grapsit.store import create_store
        store = create_store({
            "store_type": "falkordb",
            "falkordb_host": "myredis",
            "falkordb_port": 6380,
            "falkordb_graph": "mygraph",
        })
        assert store.host == "myredis"
        assert store.port == 6380
        assert store.graph_name == "mygraph"

    def test_unknown_store_type(self):
        from grapsit.store import create_store
        with pytest.raises(KeyError, match="Unknown graph_store type"):
            create_store({"store_type": "invalid"})

    def test_base_graph_store_is_abstract(self):
        from grapsit.store.graph.base import BaseGraphStore
        with pytest.raises(TypeError):
            BaseGraphStore()
