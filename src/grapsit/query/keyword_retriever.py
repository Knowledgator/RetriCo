"""Keyword retriever — find relevant chunks via full-text search."""

from typing import Any, Dict
import logging

from ..core.base import BaseProcessor
from ..core.registry import processor_registry
from ..models.document import Chunk
from ..models.entity import Entity
from ..models.relation import Relation
from ..models.graph import Subgraph

logger = logging.getLogger(__name__)


class KeywordRetrieverProcessor(BaseProcessor):
    """Retrieve chunks by full-text searching a relational store.

    Two modes controlled by ``expand_entities`` (default: ``False``):

    **Chunks-only** (``expand_entities=False``):
        query -> relational_store.search() -> Subgraph(chunks=[...])
        Only needs a relational store — no graph store, parser, or embeddings.

    **Entity expansion** (``expand_entities=True``):
        query -> relational_store.search() -> chunks
              -> graph_store.get_entities_for_chunk() -> entity_ids
              -> graph_store.get_subgraph() -> Subgraph(entities, relations, chunks)
        Additionally requires a graph store connection.

    Config keys:
        top_k: int — number of chunks to retrieve (default: 10)
        chunk_table: str — table name for chunk records (default: "chunks")
        expand_entities: bool — look up entities in matched chunks and build
            a full subgraph via the graph store (default: False)
        max_hops: int — subgraph expansion depth when expand_entities=True
            (default: 1)
        relational_store_type, sqlite_path, etc. — passed to create_relational_store
        store_type, neo4j_uri, etc. — passed to create_store (graph, only
            when expand_entities=True)
    """

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.top_k: int = config_dict.get("top_k", 10)
        self.chunk_table: str = config_dict.get("chunk_table", "chunks")
        self.expand_entities: bool = config_dict.get("expand_entities", False)
        self.max_hops: int = config_dict.get("max_hops", 1)
        self._relational_store = None
        self._store = None

    def _ensure_relational_store(self):
        """Lazily create the relational store (shared from pool if available)."""
        if self._relational_store is None:
            from ..store.pool import resolve_from_pool_or_create
            self._relational_store = resolve_from_pool_or_create(
                self.config_dict, "relational"
            )

    def _ensure_store(self):
        """Lazily create the graph store (shared from pool if available)."""
        if self._store is None:
            from ..store.pool import resolve_from_pool_or_create
            self._store = resolve_from_pool_or_create(self.config_dict, "graph")

    @staticmethod
    def _raw_to_subgraph(raw: Dict[str, Any], chunks: list) -> Subgraph:
        """Convert a raw store subgraph dict to a Subgraph model."""
        sg_entities = []
        for ent_dict in raw.get("entities", []):
            if ent_dict is None:
                continue
            sg_entities.append(Entity(
                id=ent_dict.get("id", ""),
                label=ent_dict.get("label", ""),
                entity_type=ent_dict.get("entity_type", ""),
            ))

        id_to_label = {e.id: e.label for e in sg_entities}

        sg_relations = []
        for rel_dict in raw.get("relations", []):
            if rel_dict is None or rel_dict.get("type") is None:
                continue
            head_id = rel_dict.get("head", "")
            tail_id = rel_dict.get("tail", "")
            sg_relations.append(Relation(
                head_text=id_to_label.get(head_id, head_id),
                tail_text=id_to_label.get(tail_id, tail_id),
                relation_type=rel_dict.get("type", ""),
                score=rel_dict.get("score", 0.0) or 0.0,
            ))

        return Subgraph(entities=sg_entities, relations=sg_relations, chunks=chunks)

    def __call__(self, *, query: str, **kwargs) -> Dict[str, Any]:
        """Find relevant chunks via full-text search.

        Returns:
            {"subgraph": Subgraph} — with chunks always populated; entities
            and relations populated only when ``expand_entities=True``.
        """
        self._ensure_relational_store()

        # Full-text search for matching chunks
        results = self._relational_store.search(
            self.chunk_table, query, top_k=self.top_k
        )

        if not results:
            return {"subgraph": Subgraph()}

        # Build Chunk objects from search results
        chunks = []
        for record in results:
            chunk_id = record.get("id")
            if not chunk_id:
                continue
            chunks.append(Chunk(
                id=chunk_id,
                document_id=record.get("document_id", ""),
                text=record.get("text", ""),
                index=record.get("index", 0),
                start_char=record.get("start_char", 0),
                end_char=record.get("end_char", 0),
            ))

        if not self.expand_entities:
            return {"subgraph": Subgraph(chunks=chunks)}

        # Expand: look up entities for each chunk via graph store
        self._ensure_store()

        all_entity_ids = []
        for chunk in chunks:
            entities = self._store.get_entities_for_chunk(chunk.id)
            for ent in entities:
                eid = ent.get("id")
                if eid and eid not in all_entity_ids:
                    all_entity_ids.append(eid)

        if not all_entity_ids:
            return {"subgraph": Subgraph(chunks=chunks)}

        raw = self._store.get_subgraph(all_entity_ids, max_hops=self.max_hops)
        return {"subgraph": self._raw_to_subgraph(raw, chunks)}


@processor_registry.register("keyword_retriever")
def create_keyword_retriever(config_dict: dict, pipeline=None):
    return KeywordRetrieverProcessor(config_dict, pipeline)
