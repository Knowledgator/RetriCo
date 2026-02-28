"""Store module — backward-compatible re-exports.

All store types are organized into submodules (graph/, vector/, relational/)
with registry-based dispatch. This top-level ``__init__`` re-exports everything
at the original import paths for backward compatibility.
"""

# -- Graph stores (moved to store/graph/) ------------------------------------
from .graph import create_graph_store, graph_store_registry
from .graph.base import BaseGraphStore
from .graph.neo4j_store import Neo4jGraphStore
from .graph.falkordb_store import FalkorDBGraphStore
from .graph.memgraph_store import MemgraphGraphStore

# -- Vector stores (moved to store/vector/) ----------------------------------
from .vector import create_vector_store, vector_store_registry
from .vector.base import BaseVectorStore

# -- Relational stores (new in store/relational/) ----------------------------
from .relational import create_relational_store, relational_store_registry
from .relational.base import BaseRelationalStore
from .relational.sqlite_store import SqliteRelationalStore
from .relational.postgres_store import PostgresRelationalStore
from .relational.elasticsearch_store import ElasticsearchRelationalStore

# -- Registry class ----------------------------------------------------------
from .registry import StoreRegistry

# -- Config ------------------------------------------------------------------
from .config import (
    BaseStoreConfig, Neo4jConfig, FalkorDBConfig, MemgraphConfig,
    resolve_store_config, extract_store_kwargs, _STORE_FLAT_KEYS,
    BaseVectorStoreConfig, InMemoryVectorConfig, FaissVectorConfig,
    QdrantVectorConfig, GraphDBVectorConfig,
    resolve_vector_store_config, extract_vector_store_kwargs,
    BaseRelationalStoreConfig, SqliteRelationalConfig,
    PostgresRelationalConfig, ElasticsearchRelationalConfig,
    resolve_relational_store_config, extract_relational_store_kwargs,
)

# -- Pool --------------------------------------------------------------------
from .pool import StorePool, resolve_from_pool_or_create

# Legacy alias — create_store() is now create_graph_store()
create_store = create_graph_store

__all__ = [
    # Graph stores
    "BaseGraphStore",
    "Neo4jGraphStore",
    "FalkorDBGraphStore",
    "MemgraphGraphStore",
    "create_store",
    "create_graph_store",
    "graph_store_registry",
    # Vector stores
    "BaseVectorStore",
    "create_vector_store",
    "vector_store_registry",
    # Relational stores
    "BaseRelationalStore",
    "SqliteRelationalStore",
    "PostgresRelationalStore",
    "ElasticsearchRelationalStore",
    "create_relational_store",
    "relational_store_registry",
    # Registry
    "StoreRegistry",
    # Config
    "BaseStoreConfig",
    "Neo4jConfig",
    "FalkorDBConfig",
    "MemgraphConfig",
    "resolve_store_config",
    "extract_store_kwargs",
    # Vector store configs
    "BaseVectorStoreConfig",
    "InMemoryVectorConfig",
    "FaissVectorConfig",
    "QdrantVectorConfig",
    "GraphDBVectorConfig",
    "resolve_vector_store_config",
    "extract_vector_store_kwargs",
    # Relational store configs
    "BaseRelationalStoreConfig",
    "SqliteRelationalConfig",
    "PostgresRelationalConfig",
    "ElasticsearchRelationalConfig",
    "resolve_relational_store_config",
    "extract_relational_store_kwargs",
    # Pool
    "StorePool",
    "resolve_from_pool_or_create",
]
