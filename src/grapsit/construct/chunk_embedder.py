"""Chunk embedding processor — embeds chunks into a vector store after graph writing."""

from typing import Any, Dict, List
import logging

from ..core.base import BaseProcessor
from ..core.registry import processor_registry
from ..store.pool import resolve_from_pool_or_create
from ..modeling.embeddings import create_embedding_model
from ..store.vector.graph_db import GraphDBVectorStore as _GraphDBVectorStore

logger = logging.getLogger(__name__)


class ChunkEmbedderProcessor(BaseProcessor):
    """Embed chunk texts and store in vector store + optionally on graph nodes.

    Runs after ``graph_writer`` so that chunks already exist in the database.

    Config keys:
        embedding_method: str — "sentence_transformer" or "openai" (default: "sentence_transformer").
        model_name: str — embedding model name (default: "all-MiniLM-L6-v2").
        vector_store_type: str — "in_memory", "faiss", or "qdrant" (default: "in_memory").
        vector_index_name: str — index name (default: "chunk_embeddings").
        batch_size: int — batch size for encoding (default: 32).
        device: str — device for sentence-transformers (default: "cpu").
        Store params: store_type, neo4j_uri, etc. (for persisting embeddings on nodes).
        Additional embedding/vector store params passed through.
    """

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self._store = None
        self._embedding_model = None
        self._vector_store = None

    def _ensure_store(self):
        if self._store is None:
            self._store = resolve_from_pool_or_create(self.config_dict, "graph")

    def _ensure_embedding_model(self):
        if self._embedding_model is None:
            embed_config = {
                "embedding_method": self.config_dict.get("embedding_method", "sentence_transformer"),
            }
            for key in ("model_name", "device", "batch_size", "api_key", "base_url",
                        "model", "dimensions", "timeout"):
                if key in self.config_dict:
                    embed_config[key] = self.config_dict[key]
            self._embedding_model = create_embedding_model(embed_config)

    def _ensure_vector_store(self):
        if self._vector_store is None:
            self._vector_store = resolve_from_pool_or_create(self.config_dict, "vector")

    def __call__(self, *, chunks: List = None, **kwargs) -> Dict[str, Any]:
        if not chunks:
            logger.info("No chunks to embed")
            return {"embedded_count": 0, "dimension": 0, "index_name": ""}

        self._ensure_store()
        self._ensure_embedding_model()
        self._ensure_vector_store()

        index_name = self.config_dict.get("vector_index_name", "chunk_embeddings")
        dimension = self._embedding_model.dimension
        self._vector_store.create_index(index_name, dimension)

        # Extract texts from Chunk objects or dicts
        texts = []
        ids = []
        for chunk in chunks:
            text = chunk.text if hasattr(chunk, "text") else chunk.get("text", "")
            cid = chunk.id if hasattr(chunk, "id") else chunk.get("id", "")
            if text and cid:
                texts.append(text)
                ids.append(cid)

        if not texts:
            logger.info("No chunk texts to embed")
            return {"embedded_count": 0, "dimension": dimension, "index_name": index_name}

        embeddings = self._embedding_model.encode(texts)

        # Store in vector store
        items = list(zip(ids, embeddings))
        self._vector_store.store_embeddings(index_name, items)

        # Persist on graph nodes (skip if vector store already writes to graph DB)
        if not isinstance(self._vector_store, _GraphDBVectorStore):
            for cid, emb in items:
                try:
                    self._store.update_chunk_embedding(cid, emb)
                except Exception as e:
                    logger.debug(f"Could not store embedding on Chunk node: {e}")

        logger.info(f"Embedded {len(items)} chunks (dim={dimension})")
        return {
            "embedded_count": len(items),
            "dimension": dimension,
            "index_name": index_name,
        }


@processor_registry.register("chunk_embedder")
def create_chunk_embedder(config_dict: dict, pipeline=None):
    return ChunkEmbedderProcessor(config_dict, pipeline)
