"""Entity embedding processor — embeds entities into a vector store after graph writing."""

from typing import Any, Dict, List
import logging

from ..core.base import BaseProcessor
from ..core.registry import construct_registry
from ..store.pool import resolve_from_pool_or_create
from ..modeling.embeddings import create_embedding_model
from ..store.vector.graph_db import GraphDBVectorStore as _GraphDBVectorStore

logger = logging.getLogger(__name__)


class EntityEmbedderProcessor(BaseProcessor):
    """Embed entity labels and store in vector store + optionally on graph nodes.

    Runs after ``graph_writer`` so that entities already exist in the database.
    Uses ``entity_map`` from ``writer_result`` (Dict[str, Entity]).

    Config keys:
        embedding_method: str — "sentence_transformer" or "openai" (default: "sentence_transformer").
        model_name: str — embedding model name (default: "all-MiniLM-L6-v2").
        vector_store_type: str — "in_memory", "faiss", or "qdrant" (default: "in_memory").
        vector_index_name: str — index name (default: "entity_embeddings").
        batch_size: int — batch size for encoding (default: 32).
        device: str — device for sentence-transformers (default: "cpu").
        Store params: store_type, neo4j_uri, etc. (for persisting embeddings on nodes).
        Additional embedding/vector store params passed through.
    """

    default_inputs = {"entity_map": "writer_result.entity_map"}
    default_output = "entity_embedder_result"

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

    def __call__(self, *, entity_map: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        if not entity_map:
            logger.info("No entities to embed")
            return {"embedded_count": 0, "dimension": 0, "index_name": ""}

        self._ensure_store()
        self._ensure_embedding_model()
        self._ensure_vector_store()

        index_name = self.config_dict.get("vector_index_name", "entity_embeddings")
        dimension = self._embedding_model.dimension
        self._vector_store.create_index(index_name, dimension)

        # Extract labels and IDs from entity_map
        texts = []
        ids = []
        for entity in entity_map.values():
            label = entity.label if hasattr(entity, "label") else entity.get("label", "")
            eid = entity.id if hasattr(entity, "id") else entity.get("id", "")
            if label and eid:
                texts.append(label)
                ids.append(eid)

        if not texts:
            logger.info("No entity labels to embed")
            return {"embedded_count": 0, "dimension": dimension, "index_name": index_name}

        embeddings = self._embedding_model.encode(texts)

        # Store in vector store
        items = list(zip(ids, embeddings))
        self._vector_store.store_embeddings(index_name, items)

        # Persist on graph nodes (skip if vector store already writes to graph DB)
        if not isinstance(self._vector_store, _GraphDBVectorStore):
            for eid, emb in items:
                try:
                    self._store.update_entity_embedding(eid, emb)
                except Exception as e:
                    logger.debug(f"Could not store embedding on Entity node: {e}")

        logger.info(f"Embedded {len(items)} entities (dim={dimension})")
        return {
            "embedded_count": len(items),
            "dimension": dimension,
            "index_name": index_name,
        }


@construct_registry.register("entity_embedder")
def create_entity_embedder(config_dict: dict, pipeline=None):
    return EntityEmbedderProcessor(config_dict, pipeline)
