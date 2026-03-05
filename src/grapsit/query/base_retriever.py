"""Base retriever with shared helpers for all retrieval strategies."""

from typing import Any, Dict, List, Optional
import logging

from ..core.base import BaseProcessor
from ..models.entity import Entity, EntityMention
from ..models.relation import Relation
from ..models.document import Chunk
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
        """Lazily create the graph store (shared from pool if available)."""
        if self._store is None:
            from ..store.pool import resolve_from_pool_or_create
            self._store = resolve_from_pool_or_create(self.config_dict, "graph")

    def _ensure_embedding_model(self):
        """Lazily create the embedding model via ``create_embedding_model(config)``."""
        if self._embedding_model is None:
            from ..modeling.embeddings import create_embedding_model
            self._embedding_model = create_embedding_model(self.config_dict)

    def _ensure_vector_store(self):
        """Lazily create the vector store (shared from pool if available)."""
        if self._vector_store is None:
            from ..store.pool import resolve_from_pool_or_create
            self._vector_store = resolve_from_pool_or_create(self.config_dict, "vector")

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

        # Build ID → label map so relations show entity names, not IDs
        id_to_label = {e.id: e.label for e in sg_entities}

        sg_relations = []
        for rel_dict in raw.get("relations", []):
            if rel_dict is None or rel_dict.get("type") is None:
                continue
            head_id = rel_dict.get("head", "")
            tail_id = rel_dict.get("tail", "")
            kwargs: Dict[str, Any] = {
                "head_text": id_to_label.get(head_id, head_id),
                "tail_text": id_to_label.get(tail_id, tail_id),
                "relation_type": rel_dict.get("type", ""),
                "score": rel_dict.get("score", 0.0) or 0.0,
            }
            # Pass through chunk_id (already a list from store)
            if "chunk_id" in rel_dict:
                cid = rel_dict["chunk_id"]
                if isinstance(cid, list):
                    kwargs["chunk_id"] = cid
                elif isinstance(cid, str):
                    kwargs["chunk_id"] = [cid] if cid else []
            if rel_dict.get("start_date") is not None:
                kwargs["start_date"] = rel_dict["start_date"]
            if rel_dict.get("end_date") is not None:
                kwargs["end_date"] = rel_dict["end_date"]
            sg_relations.append(Relation(**kwargs))

        return Subgraph(entities=sg_entities, relations=sg_relations)

    def _get_chunks_from_relations(self, relations: List[Relation]) -> List[Chunk]:
        """Collect all chunk IDs from relations and batch-fetch chunks.

        Args:
            relations: List of Relation objects with chunk_id lists.

        Returns:
            List of Chunk objects fetched from the store.
        """
        self._ensure_store()
        all_chunk_ids: List[str] = []
        seen: set = set()
        for rel in relations:
            for cid in rel.chunk_id:
                if cid and cid not in seen:
                    seen.add(cid)
                    all_chunk_ids.append(cid)

        if not all_chunk_ids:
            return []

        raw_chunks = self._store.get_chunks_by_ids(all_chunk_ids)
        chunks = []
        for raw in raw_chunks:
            chunks.append(Chunk(
                id=raw.get("id", ""),
                document_id=raw.get("document_id", ""),
                text=raw.get("text", ""),
                index=int(raw.get("index", 0)),
                start_char=int(raw.get("start_char", 0)),
                end_char=int(raw.get("end_char", 0)),
            ))
        return chunks
