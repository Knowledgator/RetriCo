"""Relational store submodule — registry-based relational store creation."""

from ..registry import StoreRegistry
from .base import BaseRelationalStore

relational_store_registry = StoreRegistry("relational_store", BaseRelationalStore)


def create_relational_store(config: dict) -> BaseRelationalStore:
    """Create a relational store from config.

    The ``relational_store_type`` key selects the backend.

    Args:
        config: Configuration dict.

    Returns:
        A BaseRelationalStore instance.
    """
    config = dict(config)
    store_type = config.pop("relational_store_type", None)
    if store_type is None:
        raise ValueError(
            "Config must contain 'relational_store_type' key. "
            f"Available types: {relational_store_registry.list() or '(none registered)'}"
        )
    factory = relational_store_registry.get(store_type)
    return factory(config)


__all__ = [
    "BaseRelationalStore",
    "relational_store_registry",
    "create_relational_store",
]
