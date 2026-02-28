"""Per-database store configuration classes.

Provides typed config objects for each supported graph database backend
and vector store backend, replacing the flat-dict-with-prefixed-keys
pattern used throughout the builder and convenience APIs.

Each config class knows how to serialize to/from the legacy flat-dict
format so existing YAML configs and keyword-argument calling patterns
continue to work unchanged.
"""

from typing import Optional, Union

from pydantic import BaseModel


# ============================================================================
# Graph store configs
# ============================================================================


class BaseStoreConfig(BaseModel):
    """Base class for all graph store configurations."""

    store_type: str
    name: str = "default"

    def to_flat_dict(self) -> dict:
        """Serialize to flat dict with prefixed keys for backward compat.

        e.g. Neo4jConfig(uri="bolt://x") -> {"store_type": "neo4j", "neo4j_uri": "bolt://x"}
        """
        raise NotImplementedError

    @classmethod
    def from_flat_dict(cls, d: dict) -> "BaseStoreConfig":
        """Create from a legacy flat dict (auto-detect store_type)."""
        store_type = d.get("store_type", "neo4j")
        name = d.get("store_name", "default")
        if store_type == "neo4j":
            cfg = Neo4jConfig.from_flat_dict(d)
        elif store_type == "falkordb":
            cfg = FalkorDBConfig.from_flat_dict(d)
        elif store_type == "memgraph":
            cfg = MemgraphConfig.from_flat_dict(d)
        else:
            raise ValueError(f"Unknown store_type: {store_type!r}")
        cfg.name = name
        return cfg


class Neo4jConfig(BaseStoreConfig):
    """Configuration for Neo4j graph store."""

    store_type: str = "neo4j"
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"

    def to_flat_dict(self) -> dict:
        d = {
            "store_type": self.store_type,
            "neo4j_uri": self.uri,
            "neo4j_user": self.user,
            "neo4j_password": self.password,
            "neo4j_database": self.database,
        }
        if self.name != "default":
            d["store_name"] = self.name
        return d

    @classmethod
    def from_flat_dict(cls, d: dict) -> "Neo4jConfig":
        return cls(
            uri=d.get("neo4j_uri", "bolt://localhost:7687"),
            user=d.get("neo4j_user", "neo4j"),
            password=d.get("neo4j_password", "password"),
            database=d.get("neo4j_database", "neo4j"),
            name=d.get("store_name", "default"),
        )


class FalkorDBConfig(BaseStoreConfig):
    """Configuration for FalkorDB graph store."""

    store_type: str = "falkordb"
    host: str = "localhost"
    port: int = 6379
    graph: str = "grapsit"

    def to_flat_dict(self) -> dict:
        d = {
            "store_type": self.store_type,
            "falkordb_host": self.host,
            "falkordb_port": self.port,
            "falkordb_graph": self.graph,
        }
        if self.name != "default":
            d["store_name"] = self.name
        return d

    @classmethod
    def from_flat_dict(cls, d: dict) -> "FalkorDBConfig":
        return cls(
            host=d.get("falkordb_host", "localhost"),
            port=d.get("falkordb_port", 6379),
            graph=d.get("falkordb_graph", "grapsit"),
            name=d.get("store_name", "default"),
        )


class MemgraphConfig(BaseStoreConfig):
    """Configuration for Memgraph graph store."""

    store_type: str = "memgraph"
    uri: str = "bolt://localhost:7687"
    user: str = ""
    password: str = ""
    database: str = "memgraph"

    def to_flat_dict(self) -> dict:
        d = {
            "store_type": self.store_type,
            "memgraph_uri": self.uri,
            "memgraph_user": self.user,
            "memgraph_password": self.password,
            "memgraph_database": self.database,
        }
        if self.name != "default":
            d["store_name"] = self.name
        return d

    @classmethod
    def from_flat_dict(cls, d: dict) -> "MemgraphConfig":
        return cls(
            uri=d.get("memgraph_uri", "bolt://localhost:7687"),
            user=d.get("memgraph_user", ""),
            password=d.get("memgraph_password", ""),
            database=d.get("memgraph_database", "memgraph"),
            name=d.get("store_name", "default"),
        )


# All known flat-dict keys for store params
_STORE_FLAT_KEYS = frozenset({
    "store_type", "store_name",
    "neo4j_uri", "neo4j_user", "neo4j_password", "neo4j_database",
    "falkordb_host", "falkordb_port", "falkordb_graph",
    "memgraph_uri", "memgraph_user", "memgraph_password", "memgraph_database",
})


def resolve_store_config(
    store_config: Optional[BaseStoreConfig] = None,
    **kwargs,
) -> BaseStoreConfig:
    """Resolve a store config from explicit config object or legacy kwargs.

    Priority: store_config object > keyword arguments > defaults.
    Supports backward-compat kwargs like neo4j_uri="..." alongside new config objects.
    """
    if store_config is not None and not kwargs:
        return store_config
    # Build from kwargs (legacy flat-key style)
    flat = store_config.to_flat_dict() if store_config else {}
    flat.update({k: v for k, v in kwargs.items() if k in _STORE_FLAT_KEYS})
    return BaseStoreConfig.from_flat_dict(flat)


def extract_store_kwargs(kwargs: dict) -> dict:
    """Pop store-related keys from a kwargs dict, returning them separately."""
    return {k: kwargs.pop(k) for k in list(kwargs) if k in _STORE_FLAT_KEYS}


# ============================================================================
# Vector store configs
# ============================================================================


# All known flat-dict keys for vector store params
_VECTOR_STORE_FLAT_KEYS = frozenset({
    "vector_store_type", "vector_store_name",
    "use_gpu",
    "qdrant_url", "qdrant_api_key", "qdrant_path", "prefer_grpc",
    "graph_store_name",
})


class BaseVectorStoreConfig(BaseModel):
    """Base class for all vector store configurations."""

    vector_store_type: str
    name: str = "default"

    def to_flat_dict(self) -> dict:
        """Serialize to flat dict for backward compat."""
        raise NotImplementedError

    @classmethod
    def from_flat_dict(cls, d: dict) -> "BaseVectorStoreConfig":
        """Create from a flat dict (auto-detect vector_store_type)."""
        vtype = d.get("vector_store_type", "in_memory")
        name = d.get("vector_store_name", "default")
        if vtype == "in_memory":
            cfg = InMemoryVectorConfig.from_flat_dict(d)
        elif vtype == "faiss":
            cfg = FaissVectorConfig.from_flat_dict(d)
        elif vtype == "qdrant":
            cfg = QdrantVectorConfig.from_flat_dict(d)
        elif vtype == "graph_db":
            cfg = GraphDBVectorConfig.from_flat_dict(d)
        else:
            raise ValueError(f"Unknown vector_store_type: {vtype!r}")
        cfg.name = name
        return cfg


class InMemoryVectorConfig(BaseVectorStoreConfig):
    """Configuration for in-memory vector store."""

    vector_store_type: str = "in_memory"

    def to_flat_dict(self) -> dict:
        d: dict = {"vector_store_type": self.vector_store_type}
        if self.name != "default":
            d["vector_store_name"] = self.name
        return d

    @classmethod
    def from_flat_dict(cls, d: dict) -> "InMemoryVectorConfig":
        return cls(name=d.get("vector_store_name", "default"))


class FaissVectorConfig(BaseVectorStoreConfig):
    """Configuration for FAISS vector store."""

    vector_store_type: str = "faiss"
    use_gpu: bool = False

    def to_flat_dict(self) -> dict:
        d: dict = {
            "vector_store_type": self.vector_store_type,
            "use_gpu": self.use_gpu,
        }
        if self.name != "default":
            d["vector_store_name"] = self.name
        return d

    @classmethod
    def from_flat_dict(cls, d: dict) -> "FaissVectorConfig":
        return cls(
            use_gpu=d.get("use_gpu", False),
            name=d.get("vector_store_name", "default"),
        )


class QdrantVectorConfig(BaseVectorStoreConfig):
    """Configuration for Qdrant vector store."""

    vector_store_type: str = "qdrant"
    url: Optional[str] = None
    api_key: Optional[str] = None
    path: Optional[str] = None
    prefer_grpc: bool = False

    def to_flat_dict(self) -> dict:
        d: dict = {"vector_store_type": self.vector_store_type}
        if self.url is not None:
            d["qdrant_url"] = self.url
        if self.api_key is not None:
            d["qdrant_api_key"] = self.api_key
        if self.path is not None:
            d["qdrant_path"] = self.path
        if self.prefer_grpc:
            d["prefer_grpc"] = self.prefer_grpc
        if self.name != "default":
            d["vector_store_name"] = self.name
        return d

    @classmethod
    def from_flat_dict(cls, d: dict) -> "QdrantVectorConfig":
        return cls(
            url=d.get("qdrant_url"),
            api_key=d.get("qdrant_api_key"),
            path=d.get("qdrant_path"),
            prefer_grpc=d.get("prefer_grpc", False),
            name=d.get("vector_store_name", "default"),
        )


class GraphDBVectorConfig(BaseVectorStoreConfig):
    """Configuration for graph-DB-backed vector store.

    References a graph store by pool name so the vector store can share
    the same database connection.
    """

    vector_store_type: str = "graph_db"
    graph_store_name: str = "default"

    def to_flat_dict(self) -> dict:
        d: dict = {
            "vector_store_type": self.vector_store_type,
            "graph_store_name": self.graph_store_name,
        }
        if self.name != "default":
            d["vector_store_name"] = self.name
        return d

    @classmethod
    def from_flat_dict(cls, d: dict) -> "GraphDBVectorConfig":
        return cls(
            graph_store_name=d.get("graph_store_name", "default"),
            name=d.get("vector_store_name", "default"),
        )


def resolve_vector_store_config(
    config: Optional[BaseVectorStoreConfig] = None,
    **kwargs,
) -> BaseVectorStoreConfig:
    """Resolve a vector store config from explicit config object or kwargs."""
    if config is not None and not kwargs:
        return config
    flat = config.to_flat_dict() if config else {}
    flat.update({k: v for k, v in kwargs.items() if k in _VECTOR_STORE_FLAT_KEYS})
    return BaseVectorStoreConfig.from_flat_dict(flat)


def extract_vector_store_kwargs(kwargs: dict) -> dict:
    """Pop vector-store-related keys from a kwargs dict, returning them separately."""
    return {k: kwargs.pop(k) for k in list(kwargs) if k in _VECTOR_STORE_FLAT_KEYS}
