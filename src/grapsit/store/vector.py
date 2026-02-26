"""Factory for creating vector stores."""

from .vector_base import BaseVectorStore
from .vector_memory import InMemoryVectorStore


def create_vector_store(config: dict) -> BaseVectorStore:
    """Create a vector store from config.

    The ``vector_store_type`` key selects the backend (default: ``"in_memory"``).

    Supported types:
    - ``"in_memory"``: Numpy-based cosine similarity (no extra deps).
    - ``"faiss"``: Facebook AI Similarity Search (requires ``faiss-cpu`` or ``faiss-gpu``).
      Extra keys: ``use_gpu`` (bool, default False).
    - ``"qdrant"``: Qdrant vector database (requires ``qdrant-client``).
      Extra keys: ``qdrant_url``, ``qdrant_api_key``, ``qdrant_path``, ``prefer_grpc``.
    - ``"graph_db"``: Use the graph database's native vector index (Neo4j / FalkorDB).
      Extra keys: same as ``create_store()`` (``store_type``, ``neo4j_uri``, etc.).

    Args:
        config: Configuration dict. Keys beyond ``vector_store_type`` are
            passed to the store constructor.

    Returns:
        A BaseVectorStore instance.
    """
    config = dict(config)
    store_type = config.pop("vector_store_type", "in_memory")

    if store_type == "in_memory":
        return InMemoryVectorStore()
    elif store_type == "faiss":
        from .vector_faiss import FaissVectorStore

        return FaissVectorStore(
            use_gpu=config.get("use_gpu", False),
        )
    elif store_type == "qdrant":
        from .vector_qdrant import QdrantVectorStore

        return QdrantVectorStore(
            url=config.get("qdrant_url"),
            api_key=config.get("qdrant_api_key"),
            path=config.get("qdrant_path"),
            prefer_grpc=config.get("prefer_grpc", False),
        )
    elif store_type == "graph_db":
        from .vector_graph import GraphDBVectorStore

        return GraphDBVectorStore(store_config=config)
    else:
        raise ValueError(
            f"Unknown vector_store_type: {store_type!r}. "
            f"Expected 'in_memory', 'faiss', 'qdrant', or 'graph_db'."
        )