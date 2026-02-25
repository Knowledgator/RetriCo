"""Base retriever with shared helpers for all retrieval strategies."""

from typing import Any, Dict, List, Optional
import logging

from ..core.base import BaseProcessor
from ..models.entity import Entity, EntityMention
from ..models.relation import Relation
from ..models.graph import Subgraph

logger = logging.getLogger(__name__)


class BaseRetriever(BaseProcessor):
    """Base class for retriever processors.

    Provides lazy initialization of graph store, embedding model, and vector
    store, plus shared helpers for entity lookup and subgraph conversion.

    All subclasses must output ``{"subgraph": Subgraph}``.
    """

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self._store = None
        self._embedding_model = None
        self._vector_store = None

    def _ensure_store(self):
        """Lazily create the graph store via ``create_store(config)``."""
        if self._store is None:
            from ..store import create_store
            self._store = create_store(self.config_dict)

    def _ensure_embedding_model(self):
        """Lazily create the embedding model via ``create_embedding_model(config)``."""
        if self._embedding_model is None:
            from ..modeling.embeddings import create_embedding_model
            self._embedding_model = create_embedding_model(self.config_dict)

    def _ensure_vector_store(self):
        """Lazily create the vector store via ``create_vector_store(config)``."""
        if self._vector_store is None:
            from ..store.vector import create_vector_store
            self._vector_store = create_vector_store(self.config_dict)

    def _lookup_entity(self, mention: EntityMention) -> Optional[Dict[str, Any]]:
        """Look up an entity by linked_entity_id first, then by label."""
        self._ensure_store()
        if mention.linked_entity_id:
            record = self._store.get_entity_by_id(mention.linked_entity_id)
        else:
            record = self._store.get_entity_by_label(mention.text)
        return record

    def _raw_to_subgraph(self, raw: Dict[str, Any]) -> Subgraph:
        """Convert a raw store subgraph dict to a Subgraph model.

        Args:
            raw: Dict with ``entities`` (list of entity dicts) and
                ``relations`` (list of relation dicts with head/tail/type/score).
        """
        sg_entities = []
        for ent_dict in raw.get("entities", []):
            if ent_dict is None:
                continue
            sg_entities.append(Entity(
                id=ent_dict.get("id", ""),
                label=ent_dict.get("label", ""),
                entity_type=ent_dict.get("entity_type", ""),
            ))

        sg_relations = []
        for rel_dict in raw.get("relations", []):
            if rel_dict is None or rel_dict.get("type") is None:
                continue
            sg_relations.append(Relation(
                head_text=rel_dict.get("head", ""),
                tail_text=rel_dict.get("tail", ""),
                relation_type=rel_dict.get("type", ""),
                score=rel_dict.get("score", 0.0) or 0.0,
            ))

        return Subgraph(entities=sg_entities, relations=sg_relations)
