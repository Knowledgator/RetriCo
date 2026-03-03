"""Named store pool — lazily creates and shares store instances.

The ``StorePool`` manages named configs and instances for graph, vector,
and relational stores.  Processors that receive a pool (via the
``__store_pool__`` key injected by ``DAGExecutor``) share a single
connection per named store instead of each creating its own.
"""

from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class StorePool:
    """Named store pool — lazily creates and shares store instances."""

    def __init__(self):
        self._graph_configs: Dict[str, dict] = {}
        self._vector_configs: Dict[str, dict] = {}
        self._relational_configs: Dict[str, dict] = {}
        self._graph_instances: Dict[str, Any] = {}
        self._vector_instances: Dict[str, Any] = {}
        self._relational_instances: Dict[str, Any] = {}

    # -- registration --------------------------------------------------------

    def register_graph(self, name: str, config: dict):
        """Register a named graph store config."""
        self._graph_configs[name] = dict(config)

    def register_vector(self, name: str, config: dict):
        """Register a named vector store config."""
        self._vector_configs[name] = dict(config)

    def register_relational(self, name: str, config: dict):
        """Register a named relational store config."""
        self._relational_configs[name] = dict(config)

    # -- accessors -----------------------------------------------------------

    def has_graph(self, name: str = "default") -> bool:
        return name in self._graph_configs

    def has_vector(self, name: str = "default") -> bool:
        return name in self._vector_configs

    def has_relational(self, name: str = "default") -> bool:
        return name in self._relational_configs

    def get_graph(self, name: str = "default"):
        """Get or lazily create a named graph store instance."""
        if name in self._graph_instances:
            return self._graph_instances[name]
        if name not in self._graph_configs:
            raise KeyError(
                f"No graph store registered with name {name!r}. "
                f"Available: {sorted(self._graph_configs)}"
            )
        from .graph import create_graph_store
        instance = create_graph_store(self._graph_configs[name])
        self._graph_instances[name] = instance
        logger.debug(f"StorePool: created graph store {name!r}")
        return instance

    def get_vector(self, name: str = "default"):
        """Get or lazily create a named vector store instance.

        For ``graph_db`` vector stores, injects the shared graph store
        instance via ``__graph_store_instance__`` so it can reuse the
        existing connection.
        """
        if name in self._vector_instances:
            return self._vector_instances[name]
        if name not in self._vector_configs:
            raise KeyError(
                f"No vector store registered with name {name!r}. "
                f"Available: {sorted(self._vector_configs)}"
            )
        config = dict(self._vector_configs[name])
        # For graph_db vector stores, inject the shared graph store
        if config.get("vector_store_type") == "graph_db":
            graph_name = config.get("graph_store_name", "default")
            if self.has_graph(graph_name):
                config["__graph_store_instance__"] = self.get_graph(graph_name)
        from .vector import create_vector_store
        instance = create_vector_store(config)
        self._vector_instances[name] = instance
        logger.debug(f"StorePool: created vector store {name!r}")
        return instance

    def get_relational(self, name: str = "default"):
        """Get or lazily create a named relational store instance."""
        if name in self._relational_instances:
            return self._relational_instances[name]
        if name not in self._relational_configs:
            raise KeyError(
                f"No relational store registered with name {name!r}. "
                f"Available: {sorted(self._relational_configs)}"
            )
        from .relational import create_relational_store
        instance = create_relational_store(self._relational_configs[name])
        self._relational_instances[name] = instance
        logger.debug(f"StorePool: created relational store {name!r}")
        return instance

    # -- lifecycle -----------------------------------------------------------

    def close(self):
        """Close all instantiated stores."""
        for name, instance in self._graph_instances.items():
            try:
                instance.close()
            except Exception as e:
                logger.debug(f"StorePool: error closing graph store {name!r}: {e}")
        for name, instance in self._vector_instances.items():
            try:
                instance.close()
            except Exception as e:
                logger.debug(f"StorePool: error closing vector store {name!r}: {e}")
        for name, instance in self._relational_instances.items():
            try:
                instance.close()
            except Exception as e:
                logger.debug(f"StorePool: error closing relational store {name!r}: {e}")
        self._graph_instances.clear()
        self._vector_instances.clear()
        self._relational_instances.clear()

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize pool configs for YAML/dict storage."""
        d: dict = {}
        if self._graph_configs:
            d["graph"] = dict(self._graph_configs)
        if self._vector_configs:
            d["vector"] = dict(self._vector_configs)
        if self._relational_configs:
            d["relational"] = dict(self._relational_configs)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "StorePool":
        """Create a pool from a serialized dict."""
        pool = cls()
        for name, config in data.get("graph", {}).items():
            pool.register_graph(name, config)
        for name, config in data.get("vector", {}).items():
            pool.register_vector(name, config)
        for name, config in data.get("relational", {}).items():
            pool.register_relational(name, config)
        return pool


def resolve_from_pool_or_create(config_dict: dict, category: str = "graph"):
    """Try pool first (via ``__store_pool__``), fall back to direct creation.

    Args:
        config_dict: Processor config dict. May contain ``__store_pool__``
            (injected by DAGExecutor) and ``graph_store_name`` /
            ``vector_store_name`` / ``relational_store_name``.
        category: ``"graph"``, ``"vector"``, or ``"relational"``.

    Returns:
        A store instance — shared from pool if available, freshly created otherwise.
    """
    pool: Optional[StorePool] = config_dict.get("__store_pool__")

    if category == "graph":
        name = config_dict.get("graph_store_name", "default")
        if pool is not None and pool.has_graph(name):
            return pool.get_graph(name)
        # Fallback: create directly
        from .graph import create_graph_store
        return create_graph_store(config_dict)

    elif category == "vector":
        name = config_dict.get("vector_store_name", "default")
        if pool is not None and pool.has_vector(name):
            return pool.get_vector(name)
        # Fallback: create directly
        from .vector import create_vector_store
        return create_vector_store(config_dict)

    elif category == "relational":
        name = config_dict.get("relational_store_name", "default")
        if pool is not None and pool.has_relational(name):
            return pool.get_relational(name)
        # Fallback: create directly
        from .relational import create_relational_store
        return create_relational_store(config_dict)

    else:
        raise ValueError(f"Unknown store category: {category!r}")
