from .base import BaseGraphStore
from .neo4j_store import Neo4jGraphStore
from .falkordb_store import FalkorDBGraphStore
from .memgraph_store import MemgraphGraphStore

__all__ = ["BaseGraphStore", "Neo4jGraphStore", "FalkorDBGraphStore", "MemgraphGraphStore", "create_store"]


def create_store(config: dict) -> BaseGraphStore:
    """Create a graph store from a config dict.

    The ``store_type`` key selects the backend (default: ``"neo4j"``).

    Neo4j keys: ``neo4j_uri``, ``neo4j_user``, ``neo4j_password``, ``neo4j_database``
    FalkorDB keys: ``falkordb_host``, ``falkordb_port``, ``falkordb_graph``
    Memgraph keys: ``memgraph_uri``, ``memgraph_user``, ``memgraph_password``, ``memgraph_database``
    """
    store_type = config.get("store_type", "neo4j")
    if store_type == "neo4j":
        return Neo4jGraphStore(
            uri=config.get("neo4j_uri", "bolt://localhost:7687"),
            user=config.get("neo4j_user", "neo4j"),
            password=config.get("neo4j_password", "password"),
            database=config.get("neo4j_database", "neo4j"),
        )
    elif store_type == "falkordb":
        return FalkorDBGraphStore(
            host=config.get("falkordb_host", "localhost"),
            port=config.get("falkordb_port", 6379),
            graph=config.get("falkordb_graph", "grapsit"),
        )
    elif store_type == "memgraph":
        return MemgraphGraphStore(
            uri=config.get("memgraph_uri", "bolt://localhost:7687"),
            user=config.get("memgraph_user", ""),
            password=config.get("memgraph_password", ""),
            database=config.get("memgraph_database", "memgraph"),
        )
    else:
        raise ValueError(f"Unknown store_type: {store_type!r}. Expected 'neo4j', 'falkordb', or 'memgraph'.")
