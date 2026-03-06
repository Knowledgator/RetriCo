"""Relational store submodule — registry-based relational store creation."""

from ..registry import StoreRegistry
from .base import BaseRelationalStore

relational_store_registry = StoreRegistry("relational_store", BaseRelationalStore)


def _create_sqlite(config: dict):
    from .sqlite_store import SqliteRelationalStore
    return SqliteRelationalStore(path=config.get("sqlite_path", ":memory:"))


def _create_postgres(config: dict):
    from .postgres_store import PostgresRelationalStore
    return PostgresRelationalStore(
        host=config.get("postgres_host", "localhost"),
        port=config.get("postgres_port", 5432),
        user=config.get("postgres_user", "postgres"),
        password=config.get("postgres_password", ""),
        database=config.get("postgres_database", "retrico"),
    )


def _create_elasticsearch(config: dict):
    from .elasticsearch_store import ElasticsearchRelationalStore
    return ElasticsearchRelationalStore(
        url=config.get("elasticsearch_url", "http://localhost:9200"),
        api_key=config.get("elasticsearch_api_key"),
        index_prefix=config.get("elasticsearch_index_prefix", "retrico_"),
    )


relational_store_registry.register("sqlite", _create_sqlite)
relational_store_registry.register("postgres", _create_postgres)
relational_store_registry.register("elasticsearch", _create_elasticsearch)


def create_relational_store(config: dict) -> BaseRelationalStore:
    """Create a relational store from config.

    The ``relational_store_type`` key selects the backend.

    Supported types:
    - ``"sqlite"``: SQLite with FTS5 full-text search (no extra deps).
    - ``"postgres"``: PostgreSQL with tsvector FTS (requires ``psycopg``).
    - ``"elasticsearch"``: Elasticsearch (requires ``elasticsearch``).

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
