"""Qdrant-based vector store with lazy client loading."""

from typing import Dict, List, Optional, Tuple

import logging
import uuid

from .base import BaseVectorStore

logger = logging.getLogger(__name__)


class QdrantVectorStore(BaseVectorStore):
    """Vector store backed by Qdrant vector database.

    Uses cosine similarity via Qdrant's built-in distance metric.
    Supports both local (in-memory / on-disk) and remote server modes.

    The ``qdrant-client`` package is lazily imported on first use.

    Args:
        url: Qdrant server URL (e.g. ``"http://localhost:6333"``).
            If None, uses in-memory mode.
        api_key: API key for Qdrant Cloud.
        path: Local on-disk storage path. Mutually exclusive with ``url``.
        prefer_grpc: Use gRPC instead of HTTP for server communication.
    """

    def __init__(
        self,
        *,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        path: Optional[str] = None,
        prefer_grpc: bool = False,
    ):
        self._url = url
        self._api_key = api_key
        self._path = path
        self._prefer_grpc = prefer_grpc
        self._client = None
        self._models = None
        # Track dimensions per collection for validation
        self._dimensions: Dict[str, int] = {}
        # Map string item_id -> UUID point_id for each collection
        self._id_maps: Dict[str, Dict[str, str]] = {}

    def _ensure_client(self):
        """Lazily create the Qdrant client."""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client import models
            except ImportError:
                raise ImportError(
                    "qdrant-client package required for QdrantVectorStore. "
                    "Install with: pip install qdrant-client"
                )

            self._models = models

            if self._url is not None:
                self._client = QdrantClient(
                    url=self._url,
                    api_key=self._api_key,
                    prefer_grpc=self._prefer_grpc,
                )
                logger.info(f"Qdrant client connected to {self._url}")
            elif self._path is not None:
                self._client = QdrantClient(path=self._path)
                logger.info(f"Qdrant client using on-disk storage: {self._path}")
            else:
                self._client = QdrantClient(location=":memory:")
                logger.info("Qdrant client using in-memory mode")

    def create_index(self, name: str, dimension: int):
        if name in self._dimensions:
            raise ValueError(f"Index {name!r} already exists.")

        self._ensure_client()

        self._client.create_collection(
            collection_name=name,
            vectors_config=self._models.VectorParams(
                size=dimension,
                distance=self._models.Distance.COSINE,
            ),
        )
        self._dimensions[name] = dimension
        self._id_maps[name] = {}
        logger.info(f"Created Qdrant collection {name!r} (dim={dimension})")

    def _check_index(self, name: str):
        if name not in self._dimensions:
            raise KeyError(f"Index {name!r} does not exist.")

    def store_embeddings(self, index_name: str, items: List[Tuple[str, List[float]]]):
        self._check_index(index_name)

        if not items:
            return

        self._ensure_client()
        dim = self._dimensions[index_name]
        id_map = self._id_maps[index_name]

        points = []
        for item_id, vector in items:
            if len(vector) != dim:
                raise ValueError(
                    f"Vector dimension {len(vector)} != index dimension {dim}"
                )

            # Reuse existing UUID if updating, otherwise generate new one
            if item_id in id_map:
                point_id = id_map[item_id]
            else:
                point_id = str(uuid.uuid4())
                id_map[item_id] = point_id

            points.append(
                self._models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"item_id": item_id},
                )
            )

        self._client.upsert(
            collection_name=index_name,
            wait=True,
            points=points,
        )

    def search_similar(
        self, index_name: str, query_vector: List[float], top_k: int = 10
    ) -> List[Tuple[str, float]]:
        self._check_index(index_name)
        self._ensure_client()

        results = self._client.query_points(
            collection_name=index_name,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        ).points

        return [
            (point.payload["item_id"], point.score)
            for point in results
        ]

    def get_embedding(self, index_name: str, item_id: str) -> Optional[List[float]]:
        self._check_index(index_name)
        self._ensure_client()

        id_map = self._id_maps[index_name]
        if item_id not in id_map:
            return None

        point_id = id_map[item_id]
        points = self._client.retrieve(
            collection_name=index_name,
            ids=[point_id],
            with_vectors=True,
        )

        if not points:
            return None

        return points[0].vector

    def delete_index(self, name: str):
        if name not in self._dimensions:
            raise KeyError(f"Index {name!r} does not exist.")

        self._ensure_client()
        self._client.delete_collection(collection_name=name)
        del self._dimensions[name]
        del self._id_maps[name]
        logger.info(f"Deleted Qdrant collection {name!r}")

    def close(self):
        if self._client is not None:
            self._client.close()
            self._client = None
        self._dimensions.clear()
        self._id_maps.clear()