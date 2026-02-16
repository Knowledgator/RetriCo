"""Processor registry — maps processor names to factory functions."""

from typing import Any, Callable, Dict


class ProcessorRegistry:
    """Global registry for pipeline processors."""

    def __init__(self):
        self._factories: Dict[str, Callable] = {}

    def register(self, name: str, factory: Callable = None):
        """Register a processor factory.

        Can be used as a decorator::

            @processor_registry.register("chunker")
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
                f"Processor '{name}' not registered. "
                f"Available: {list(self._factories.keys())}"
            )
        return self._factories[name]

    def list(self) -> list[str]:
        return list(self._factories.keys())


processor_registry = ProcessorRegistry()
