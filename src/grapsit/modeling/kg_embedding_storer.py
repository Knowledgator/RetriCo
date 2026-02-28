"""KG embedding storer — extract and persist trained KG embeddings."""

from typing import Any, Dict
import json
import logging
import os

from ..core.base import BaseProcessor
from ..core.registry import modeling_registry
from ..store.pool import resolve_from_pool_or_create

logger = logging.getLogger(__name__)


class KGEmbeddingStorerProcessor(BaseProcessor):
    """Extract entity/relation embeddings from a trained model and store them.

    Stores embeddings in:
    1. Vector store (separate entity and relation indexes).
    2. Disk (model weights via torch.save + id mappings as JSON).
    3. Optionally, graph DB entity nodes (via update_entity_embedding).

    Config keys:
        model_path: str — directory to save model weights + mappings (default: "kg_model").
        entity_index_name: str — vector store index name for entity embeddings
            (default: "kg_entity_embeddings").
        relation_index_name: str — vector store index name for relation embeddings
            (default: "kg_relation_embeddings").
        vector_store_type: str — "in_memory", "faiss", or "qdrant" (default: "in_memory").
        store_to_graph: bool — write entity embeddings to graph DB (default: False).
        Store params: store_type, neo4j_uri, etc. (needed if store_to_graph=True).
        Vector store params: use_gpu, qdrant_url, etc.
    """

    default_inputs = {
        "model": "kg_trainer_result.model",
        "entity_to_id": "triple_reader_result.entity_to_id",
        "relation_to_id": "triple_reader_result.relation_to_id",
    }
    default_output = "embedding_storer_result"

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self._store = None
        self._vector_store = None
        self.model_path = config_dict.get("model_path", "kg_model")
        self.entity_index_name = config_dict.get(
            "entity_index_name", "kg_entity_embeddings"
        )
        self.relation_index_name = config_dict.get(
            "relation_index_name", "kg_relation_embeddings"
        )
        self.store_to_graph = config_dict.get("store_to_graph", False)

    def _ensure_store(self):
        if self._store is None:
            self._store = resolve_from_pool_or_create(self.config_dict, "graph")

    def _ensure_vector_store(self):
        if self._vector_store is None:
            self._vector_store = resolve_from_pool_or_create(self.config_dict, "vector")

    def __call__(self, **kwargs) -> Dict[str, Any]:
        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch is required for KG embedding storage. "
                "Install with: pip install torch"
            )

        model = kwargs.get("model")
        if model is None:
            raise ValueError("'model' (trained PyKEEN model) is required.")

        entity_to_id = kwargs.get("entity_to_id", {})
        relation_to_id = kwargs.get("relation_to_id", {})

        # Extract embeddings
        entity_repr = model.entity_representations[0]
        entity_embeddings = entity_repr(indices=None).detach().cpu().numpy()

        relation_repr = model.relation_representations[0]
        relation_embeddings = relation_repr(indices=None).detach().cpu().numpy()

        entity_dim = entity_embeddings.shape[1]
        relation_dim = relation_embeddings.shape[1]

        # Store in vector store
        self._ensure_vector_store()

        self._vector_store.create_index(self.entity_index_name, entity_dim)
        id_to_entity = {v: k for k, v in entity_to_id.items()}
        entity_items = [
            (id_to_entity.get(i, str(i)), entity_embeddings[i].tolist())
            for i in range(len(entity_embeddings))
        ]
        self._vector_store.store_embeddings(self.entity_index_name, entity_items)

        self._vector_store.create_index(self.relation_index_name, relation_dim)
        id_to_relation = {v: k for k, v in relation_to_id.items()}
        relation_items = [
            (id_to_relation.get(i, str(i)), relation_embeddings[i].tolist())
            for i in range(len(relation_embeddings))
        ]
        self._vector_store.store_embeddings(self.relation_index_name, relation_items)

        # Save to disk
        os.makedirs(self.model_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(self.model_path, "model.pt"))
        with open(os.path.join(self.model_path, "entity_to_id.json"), "w") as f:
            json.dump(entity_to_id, f)
        with open(os.path.join(self.model_path, "relation_to_id.json"), "w") as f:
            json.dump(relation_to_id, f)
        # Save model metadata for loading
        with open(os.path.join(self.model_path, "metadata.json"), "w") as f:
            json.dump({
                "model_name": type(model).__name__,
                "entity_dim": entity_dim,
                "relation_dim": relation_dim,
                "num_entities": len(entity_to_id),
                "num_relations": len(relation_to_id),
            }, f)

        # Optionally write to graph DB
        if self.store_to_graph:
            self._ensure_store()
            for label, idx in entity_to_id.items():
                try:
                    # Find entity by label in graph store
                    entity = self._store.get_entity_by_label(label)
                    if entity:
                        self._store.update_entity_embedding(
                            entity.get("id", ""),
                            entity_embeddings[idx].tolist(),
                        )
                except Exception as e:
                    logger.debug(f"Could not store embedding for '{label}': {e}")

        logger.info(
            f"Stored embeddings: entities={entity_embeddings.shape}, "
            f"relations={relation_embeddings.shape}, path={self.model_path}"
        )

        return {
            "entity_embeddings_shape": list(entity_embeddings.shape),
            "relation_embeddings_shape": list(relation_embeddings.shape),
            "model_path": self.model_path,
        }


@modeling_registry.register("kg_embedding_storer")
def create_kg_embedding_storer(config_dict: dict, pipeline=None):
    return KGEmbeddingStorerProcessor(config_dict, pipeline)
