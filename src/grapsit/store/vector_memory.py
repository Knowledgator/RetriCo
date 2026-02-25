"""In-memory vector store using numpy for cosine similarity search."""

from typing import Dict, List, Optional, Tuple

import logging

import numpy as np

from .vector_base import BaseVectorStore

logger = logging.getLogger(__name__)


class _VectorIndex:
    """Internal container for a single named index."""

    __slots__ = ("name", "dimension", "ids", "vectors")

    def __init__(self, name: str, dimension: int):
        self.name = name
        self.dimension = dimension
        self.ids: List[str] = []
        self.vectors: Optional[np.ndarray] = None  # shape (n, dimension)


class InMemoryVectorStore(BaseVectorStore):
    """In-memory vector store with numpy-based cosine similarity search.

    Suitable for development, testing, and small-to-medium datasets.
    All data is lost when the process exits.

    Uses a singleton pattern so that all callers within the same process share
    the same index data (e.g. embeddings created during build are available
    at query time).
    """

    _instance: "InMemoryVectorStore | None" = None
    _shared_indexes: Dict[str, _VectorIndex] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._indexes = self.__class__._shared_indexes

    def create_index(self, name: str, dimension: int):
        if name in self._indexes:
            existing = self._indexes[name]
            if existing.dimension != dimension:
                raise ValueError(
                    f"Index {name!r} already exists with dimension "
                    f"{existing.dimension}, cannot recreate with {dimension}."
                )
            logger.debug(f"In-memory index {name!r} already exists, reusing")
            return
        self._indexes[name] = _VectorIndex(name, dimension)
        logger.info(f"Created in-memory index {name!r} (dim={dimension})")

    def _get_index(self, name: str) -> _VectorIndex:
        if name not in self._indexes:
            raise KeyError(f"Index {name!r} does not exist.")
        return self._indexes[name]

    def store_embeddings(self, index_name: str, items: List[Tuple[str, List[float]]]):
        idx = self._get_index(index_name)

        new_ids = []
        new_vecs = []
        for item_id, vector in items:
            if len(vector) != idx.dimension:
                raise ValueError(
                    f"Vector dimension {len(vector)} != index dimension {idx.dimension}"
                )
            new_ids.append(item_id)
            new_vecs.append(vector)

        if not new_ids:
            return

        new_array = np.array(new_vecs, dtype=np.float32)

        # Handle updates: replace existing IDs
        existing_ids_set = set(idx.ids)
        update_ids = set(new_ids) & existing_ids_set

        if update_ids:
            # Build mapping of id -> new vector
            update_map = {
                item_id: vec
                for item_id, vec in zip(new_ids, new_vecs)
                if item_id in update_ids
            }
            for i, eid in enumerate(idx.ids):
                if eid in update_map:
                    idx.vectors[i] = update_map[eid]

            # Filter to only truly new items
            mask = [item_id not in update_ids for item_id in new_ids]
            new_ids = [item_id for item_id, m in zip(new_ids, mask) if m]
            new_array = new_array[[i for i, m in enumerate(mask) if m]]

        if len(new_ids) == 0:
            return

        idx.ids.extend(new_ids)
        if idx.vectors is None:
            idx.vectors = new_array
        else:
            idx.vectors = np.vstack([idx.vectors, new_array])

    def search_similar(
        self, index_name: str, query_vector: List[float], top_k: int = 10
    ) -> List[Tuple[str, float]]:
        idx = self._get_index(index_name)

        if idx.vectors is None or len(idx.ids) == 0:
            return []

        query = np.array(query_vector, dtype=np.float32)

        # Cosine similarity: dot(a, b) / (|a| * |b|)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        vec_norms = np.linalg.norm(idx.vectors, axis=1)
        # Avoid division by zero
        safe_norms = np.where(vec_norms == 0, 1.0, vec_norms)
        similarities = (idx.vectors @ query) / (safe_norms * query_norm)

        # Get top-k indices
        k = min(top_k, len(idx.ids))
        if k == len(idx.ids):
            # All items requested — just argsort
            top_indices = np.argsort(-similarities)[:k]
        else:
            top_indices = np.argpartition(-similarities, k)[:k]
            top_indices = top_indices[np.argsort(-similarities[top_indices])]

        return [(idx.ids[i], float(similarities[i])) for i in top_indices]

    def get_embedding(self, index_name: str, item_id: str) -> Optional[List[float]]:
        idx = self._get_index(index_name)

        try:
            pos = idx.ids.index(item_id)
        except ValueError:
            return None

        return idx.vectors[pos].tolist()

    def delete_index(self, name: str):
        if name not in self._indexes:
            raise KeyError(f"Index {name!r} does not exist.")
        del self._indexes[name]
        logger.info(f"Deleted in-memory index {name!r}")

    def close(self):
        """No-op for the singleton in-memory store.

        Use ``reset()`` to explicitly clear all indexes.
        """
        pass

    @classmethod
    def reset(cls):
        """Clear all indexes and release the singleton instance."""
        cls._shared_indexes.clear()
        cls._instance = None