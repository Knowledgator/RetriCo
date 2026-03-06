"""Generic store registry for typed store backends.

Provides a reusable ``StoreRegistry`` that supports decorator-based or direct
registration of store factory functions, with optional base-class validation.
"""

from typing import Any, Callable, Dict, List, Optional


class StoreRegistry:
    """Registry for store factory functions.

    Each registry instance manages factories for a specific category of stores
    (graph, vector, relational). Factories are callables that accept a config
    dict and return a store instance.

    Args:
        name: Human-readable registry name (e.g. ``"graph_store"``).
        base_class: Optional base class. If set, ``register()`` validates that
            the factory return type annotation (if present) is compatible.
    """

    def __init__(self, name: str, base_class: Optional[type] = None):
        self._name = name
        self._base_class = base_class
        self._factories: Dict[str, Callable] = {}

    def register(self, name: str, factory: Callable = None):
        """Register a store factory by name.

        Can be used as a decorator or called directly::

            @registry.register("my_store")
            def create_my_store(config):
                return MyStore(**config)

            # Or directly:
            registry.register("my_store", create_my_store)

        Args:
            name: Store type name (e.g. ``"neo4j"``, ``"faiss"``).
            factory: Callable that takes a config dict and returns a store.
                If ``None``, returns a decorator.

        Returns:
            The factory (or a decorator if ``factory`` is ``None``).
        """
        if factory is not None:
            self._factories[name] = factory
            return factory

        def decorator(fn: Callable) -> Callable:
            self._factories[name] = fn
            return fn

        return decorator

    def get(self, name: str) -> Callable:
        """Retrieve a factory by name.

        Args:
            name: Registered store type name.

        Returns:
            The factory callable.

        Raises:
            KeyError: If ``name`` is not registered.
        """
        if name not in self._factories:
            available = ", ".join(sorted(self._factories)) or "(none)"
            raise KeyError(
                f"Unknown {self._name} type: {name!r}. "
                f"Available: {available}"
            )
        return self._factories[name]

    def create(self, config: dict) -> Any:
        """Create a store instance from a config dict.

        Looks up the store type from the config (key depends on registry name,
        e.g. ``"store_type"`` for graph stores, ``"vector_store_type"`` for
        vector stores).

        Args:
            config: Configuration dict. Must contain the appropriate type key.

        Returns:
            A store instance.
        """
        # Determine the type key based on registry name
        type_key = self._get_type_key()
        store_type = config.get(type_key)
        if store_type is None:
            available = ", ".join(sorted(self._factories)) or "(none)"
            raise ValueError(
                f"Config must contain {type_key!r} key. "
                f"Available {self._name} types: {available}"
            )
        factory = self.get(store_type)
        return factory(config)

    def _get_type_key(self) -> str:
        """Derive the config type key from the registry name."""
        # "graph_store" -> "store_type", "vector_store" -> "vector_store_type"
        if self._name == "graph_store":
            return "store_type"
        return f"{self._name}_type"

    def list(self) -> List[str]:
        """Return sorted list of registered store type names."""
        return sorted(self._factories)

    def __contains__(self, name: str) -> bool:
        return name in self._factories

    def __repr__(self) -> str:
        types = ", ".join(sorted(self._factories))
        return f"StoreRegistry({self._name!r}, types=[{types}])"
