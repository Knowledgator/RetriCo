"""Vector store submodule — registry-based vector store creation."""

from ..registry import StoreRegistry
from .base import BaseVectorStore

vector_store_registry = StoreRegistry("vector_store", BaseVectorStore)


def _create_in_memory(config: dict):
    from .memory import InMemoryVectorStore
    return InMemoryVectorStore()


def _create_faiss(config: dict):
    from .faiss import FaissVectorStore
    return FaissVectorStore(use_gpu=config.get("use_gpu", False))


def _create_qdrant(config: dict):
    from .qdrant import QdrantVectorStore
    return QdrantVectorStore(
        url=config.get("qdrant_url"),
        api_key=config.get("qdrant_api_key"),
        path=config.get("qdrant_path"),
        prefer_grpc=config.get("prefer_grpc", False),
    )


def _create_graph_db(config: dict):
    from .graph_db import GraphDBVectorStore
    return GraphDBVectorStore(store_config=config)


vector_store_registry.register("in_memory", _create_in_memory)
vector_store_registry.register("faiss", _create_faiss)
vector_store_registry.register("qdrant", _create_qdrant)
vector_store_registry.register("graph_db", _create_graph_db)


def create_vector_store(config: dict) -> BaseVectorStore:
    """Create a vector store from config.

    The ``vector_store_type`` key selects the backend (default: ``"in_memory"``).

    Supported types:
    - ``"in_memory"``: Numpy-based cosine similarity (no extra deps).
    - ``"faiss"``: Facebook AI Similarity Search (requires ``faiss-cpu`` or ``faiss-gpu``).
    - ``"qdrant"``: Qdrant vector database (requires ``qdrant-client``).
    - ``"graph_db"``: Use the graph database's native vector index.

    Args:
        config: Configuration dict.

    Returns:
        A BaseVectorStore instance.
    """
    config = dict(config)
    store_type = config.pop("vector_store_type", "in_memory")
    factory = vector_store_registry.get(store_type)
    return factory(config)


__all__ = [
    "BaseVectorStore",
    "vector_store_registry",
    "create_vector_store",
]
