"""Graph store submodule — registry-based graph store creation."""

from ..registry import StoreRegistry
from .base import BaseGraphStore

graph_store_registry = StoreRegistry("graph_store", BaseGraphStore)


def _create_falkordb_lite(config: dict):
    from .falkordb_lite_store import FalkorDBLiteGraphStore
    return FalkorDBLiteGraphStore(
        db_path=config.get("falkordb_lite_db_path", "retrico.db"),
        graph=config.get("falkordb_lite_graph", "retrico"),
        query_timeout=config.get("falkordb_lite_query_timeout", 0),
    )


def _create_neo4j(config: dict):
    from .neo4j_store import Neo4jGraphStore
    return Neo4jGraphStore(
        uri=config.get("neo4j_uri", "bolt://localhost:7687"),
        user=config.get("neo4j_user", "neo4j"),
        password=config.get("neo4j_password", "password"),
        database=config.get("neo4j_database", "neo4j"),
    )


def _create_falkordb(config: dict):
    from .falkordb_store import FalkorDBGraphStore
    return FalkorDBGraphStore(
        host=config.get("falkordb_host", "localhost"),
        port=config.get("falkordb_port", 6379),
        graph=config.get("falkordb_graph", "retrico"),
        query_timeout=config.get("falkordb_query_timeout", 0),
    )


def _create_memgraph(config: dict):
    from .memgraph_store import MemgraphGraphStore
    return MemgraphGraphStore(
        uri=config.get("memgraph_uri", "bolt://localhost:7687"),
        user=config.get("memgraph_user", ""),
        password=config.get("memgraph_password", ""),
        database=config.get("memgraph_database", "memgraph"),
    )


graph_store_registry.register("falkordb_lite", _create_falkordb_lite)
graph_store_registry.register("neo4j", _create_neo4j)
graph_store_registry.register("falkordb", _create_falkordb)
graph_store_registry.register("memgraph", _create_memgraph)


def create_graph_store(config) -> BaseGraphStore:
    """Create a graph store from a config dict or BaseStoreConfig.

    The ``store_type`` key selects the backend (default: ``"falkordb_lite"``).

    FalkorDBLite keys: ``falkordb_lite_db_path``, ``falkordb_lite_graph``
    Neo4j keys: ``neo4j_uri``, ``neo4j_user``, ``neo4j_password``, ``neo4j_database``
    FalkorDB keys: ``falkordb_host``, ``falkordb_port``, ``falkordb_graph``
    Memgraph keys: ``memgraph_uri``, ``memgraph_user``, ``memgraph_password``, ``memgraph_database``
    """
    from ..config import BaseStoreConfig
    if isinstance(config, BaseStoreConfig):
        config = config.to_flat_dict()
    store_type = config.get("store_type", "falkordb_lite")
    factory = graph_store_registry.get(store_type)
    return factory(config)


__all__ = [
    "BaseGraphStore",
    "graph_store_registry",
    "create_graph_store",
]