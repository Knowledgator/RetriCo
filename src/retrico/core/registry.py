"""Processor registry — maps processor names to factory functions."""

from typing import Any, Callable, Dict, List


class ProcessorRegistry:
    """Registry for pipeline processors within a single category."""

    def __init__(self, name: str = "processor"):
        self.name = name
        self._factories: Dict[str, Callable] = {}

    def register(self, name: str, factory: Callable = None):
        """Register a processor factory.

        Can be used as a decorator::

            @construct_registry.register("chunker")
            def create_chunker(config_dict, pipeline):
                return ChunkerProcessor(config_dict)
        """
        if factory is not None:
            self._factories[name] = factory
            return factory

        def decorator(fn: Callable) -> Callable:
            self._factories[name] = fn
            return fn

        return decorator

    def get(self, name: str) -> Callable:
        if name not in self._factories:
            raise KeyError(
                f"Processor '{name}' not registered in {self.name!r} registry. "
                f"Available: {list(self._factories.keys())}"
            )
        return self._factories[name]

    def list(self) -> List[str]:
        return list(self._factories.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._factories

    def __repr__(self) -> str:
        return f"ProcessorRegistry({self.name!r}, processors={self.list()})"


class CompositeProcessorRegistry:
    """Composite registry that delegates to category-specific sub-registries.

    Provides backward-compatible access to all registered processors across
    construct, query, and modeling categories. Also supports direct registration
    for ad-hoc/test use.
    """

    def __init__(self, *registries: ProcessorRegistry):
        self._registries = list(registries)
        self._direct: Dict[str, Callable] = {}

    @property
    def _factories(self) -> Dict[str, Callable]:
        """Aggregated view of all factories (backward compat for tests)."""
        merged = {}
        for reg in self._registries:
            merged.update(reg._factories)
        merged.update(self._direct)
        return merged

    def register(self, name: str, factory: Callable = None):
        """Register a processor directly (ad-hoc / test registrations)."""
        if factory is not None:
            self._direct[name] = factory
            return factory

        def decorator(fn: Callable) -> Callable:
            self._direct[name] = fn
            return fn

        return decorator

    def get(self, name: str) -> Callable:
        # Search sub-registries first, then direct
        for reg in self._registries:
            if name in reg:
                return reg.get(name)
        if name in self._direct:
            return self._direct[name]
        raise KeyError(
            f"Processor '{name}' not registered. "
            f"Available: {self.list()}"
        )

    def list(self) -> List[str]:
        return list(self._factories.keys())

    def __contains__(self, name: str) -> bool:
        return any(name in reg for reg in self._registries) or name in self._direct

    def __repr__(self) -> str:
        return (
            f"CompositeProcessorRegistry("
            f"registries={[r.name for r in self._registries]}, "
            f"processors={self.list()})"
        )


# Category-specific registries
construct_registry = ProcessorRegistry("construct")
query_registry = ProcessorRegistry("query")
modeling_registry = ProcessorRegistry("modeling")

# Composite registry — backward compatible, delegates to all categories
processor_registry = CompositeProcessorRegistry(
    construct_registry, query_registry, modeling_registry
)
