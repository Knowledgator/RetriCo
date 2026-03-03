"""Faiss-based vector store with lazy library loading."""

from typing import Dict, List, Optional, Tuple

import logging

import numpy as np

from .base import BaseVectorStore

logger = logging.getLogger(__name__)


class _FaissIndex:
    """Internal container for a single named Faiss index."""

    __slots__ = ("name", "dimension", "ids", "index")

    def __init__(self, name: str, dimension: int, index):
        self.name = name
        self.dimension = dimension
        self.ids: List[str] = []
        self.index = index  # faiss.IndexFlatIP


class FaissVectorStore(BaseVectorStore):
    """Vector store backed by Facebook AI Similarity Search (Faiss).

    Uses ``IndexFlatIP`` (inner product) on L2-normalized vectors to
    compute cosine similarity. Suitable for medium-to-large datasets
    that fit in memory.

    The ``faiss`` package is lazily imported on first use.

    Args:
        use_gpu: If True, attempt to move indexes to GPU via
            ``faiss.index_cpu_to_all_gpus``. Falls back to CPU silently.
    """

    def __init__(self, *, use_gpu: bool = False):
        self._use_gpu = use_gpu
        self._faiss = None
        self._indexes: Dict[str, _FaissIndex] = {}

    def _ensure_faiss(self):
        """Lazily import faiss."""
        if self._faiss is None:
            try:
                import faiss
            except ImportError:
                raise ImportError(
                    "faiss package required for FaissVectorStore. "
                    "Install with: pip install faiss-cpu  (or faiss-gpu)"
                )
            self._faiss = faiss
            logger.info("Faiss library loaded")

    def _get_index(self, name: str) -> _FaissIndex:
        if name not in self._indexes:
            raise KeyError(f"Index {name!r} does not exist.")
        return self._indexes[name]

    def create_index(self, name: str, dimension: int):
        if name in self._indexes:
            raise ValueError(f"Index {name!r} already exists.")

        self._ensure_faiss()
        faiss = self._faiss

        index = faiss.IndexFlatIP(dimension)

        if self._use_gpu:
            try:
                index = faiss.index_cpu_to_all_gpus(index)
                logger.info(f"Faiss index {name!r} moved to GPU")
            except Exception:
                logger.warning(f"GPU not available for Faiss index {name!r}, using CPU")

        self._indexes[name] = _FaissIndex(name, dimension, index)
        logger.info(f"Created Faiss index {name!r} (dim={dimension})")

    def store_embeddings(self, index_name: str, items: List[Tuple[str, List[float]]]):
        idx = self._get_index(index_name)

        if not items:
            return

        new_ids = []
        new_vecs = []
        for item_id, vector in items:
            if len(vector) != idx.dimension:
                raise ValueError(
                    f"Vector dimension {len(vector)} != index dimension {idx.dimension}"
                )
            new_ids.append(item_id)
            new_vecs.append(vector)

        # Handle updates by rebuilding the index
        existing_ids_set = set(idx.ids)
        update_ids = set(new_ids) & existing_ids_set

        if update_ids:
            # Rebuild: collect all vectors, apply updates, then re-add
            update_map = {
                item_id: vec
                for item_id, vec in zip(new_ids, new_vecs)
                if item_id in update_ids
            }

            all_ids = list(idx.ids)
            all_vecs = []
            for i, eid in enumerate(idx.ids):
                if eid in update_map:
                    all_vecs.append(update_map[eid])
                else:
                    all_vecs.append(self._reconstruct(idx, i))

            # Add truly new items
            for item_id, vec in zip(new_ids, new_vecs):
                if item_id not in update_ids:
                    all_ids.append(item_id)
                    all_vecs.append(vec)

            # Rebuild index
            arr = np.array(all_vecs, dtype=np.float32)
            self._faiss.normalize_L2(arr)
            idx.index.reset()
            idx.index.add(arr)
            idx.ids = all_ids
        else:
            arr = np.array(new_vecs, dtype=np.float32)
            self._faiss.normalize_L2(arr)
            idx.index.add(arr)
            idx.ids.extend(new_ids)

    def _reconstruct(self, idx: _FaissIndex, position: int) -> List[float]:
        """Reconstruct a vector from the Faiss index (already normalized)."""
        vec = np.zeros(idx.dimension, dtype=np.float32)
        idx.index.reconstruct(position, vec)
        return vec.tolist()

    def search_similar(
        self, index_name: str, query_vector: List[float], top_k: int = 10
    ) -> List[Tuple[str, float]]:
        idx = self._get_index(index_name)

        if idx.index.ntotal == 0:
            return []

        query = np.array([query_vector], dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        self._faiss.normalize_L2(query)

        k = min(top_k, idx.index.ntotal)
        scores, indices = idx.index.search(query, k)

        results = []
        for score, i in zip(scores[0], indices[0]):
            if i == -1:
                continue
            results.append((idx.ids[i], float(score)))

        return results

    def get_embedding(self, index_name: str, item_id: str) -> Optional[List[float]]:
        idx = self._get_index(index_name)

        try:
            pos = idx.ids.index(item_id)
        except ValueError:
            return None

        # Reconstruct returns the normalized vector; we return it as-is
        # since the original was normalized on insertion
        vec = np.zeros(idx.dimension, dtype=np.float32)
        idx.index.reconstruct(pos, vec)
        return vec.tolist()

    def delete_index(self, name: str):
        if name not in self._indexes:
            raise KeyError(f"Index {name!r} does not exist.")
        del self._indexes[name]
        logger.info(f"Deleted Faiss index {name!r}")

    def close(self):
        self._indexes.clear()