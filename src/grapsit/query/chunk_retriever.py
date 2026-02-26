"""Chunk retriever — get source text chunks for subgraph entities."""

from typing import Any, Dict, List
import logging

from ..core.base import BaseProcessor
from ..core.registry import processor_registry
from ..models.document import Chunk
from ..models.graph import Subgraph

logger = logging.getLogger(__name__)


class ChunkRetrieverProcessor(BaseProcessor):
    """Retrieve source text chunks for entities in a subgraph.

    Config keys:
        neo4j_uri: str — bolt URI (default: "bolt://localhost:7687")
        neo4j_user: str — (default: "neo4j")
        neo4j_password: str — (default: "password")
        neo4j_database: str — (default: "neo4j")
        max_chunks: int — optional limit on chunks returned (default: None, no limit)
        chunk_entity_source: str — which entities to get chunks for:
            "all" (default) — all entities in subgraph
            "head" — only head entities from scored triples
            "tail" — only tail entities from scored triples
            "both" — explicit alias for "all"
    """

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.max_chunks: int = config_dict.get("max_chunks", 0)
        self.chunk_entity_source: str = config_dict.get("chunk_entity_source", "all")
        self._store = None

    def _ensure_store(self):
        if self._store is None:
            from ..store import create_store
            self._store = create_store(self.config_dict)

    def _select_entities(self, subgraph: Subgraph, scored_triples: Any = None):
        """Select which entities to retrieve chunks for based on chunk_entity_source."""
        if self.chunk_entity_source in ("all", "both") or not scored_triples:
            return subgraph.entities

        # Build set of labels to filter by
        target_labels = set()
        for triple in scored_triples:
            if isinstance(triple, dict):
                if self.chunk_entity_source in ("head", "both"):
                    head = triple.get("head", "")
                    if head:
                        target_labels.add(head.lower())
                if self.chunk_entity_source in ("tail", "both"):
                    tail = triple.get("tail", "")
                    if tail:
                        target_labels.add(tail.lower())

        if not target_labels:
            return subgraph.entities

        return [e for e in subgraph.entities if e.label.lower() in target_labels]

    def __call__(self, *, subgraph: Any, **kwargs) -> Dict[str, Any]:
        """Retrieve chunks for entities in the subgraph.

        Args:
            subgraph: Subgraph or dict with entities/relations.
            scored_triples: Optional list of scored triple dicts (for
                chunk_entity_source filtering).

        Returns:
            {"subgraph": Subgraph} with chunks populated
        """
        self._ensure_store()

        # Accept Subgraph or dict
        if isinstance(subgraph, dict):
            subgraph = Subgraph(**subgraph)
        elif not isinstance(subgraph, Subgraph):
            subgraph = Subgraph()

        scored_triples = kwargs.get("scored_triples")
        entities_to_query = self._select_entities(subgraph, scored_triples)

        seen_ids = set()
        chunks = []

        for entity in entities_to_query:
            raw_chunks = self._store.get_chunks_for_entity(entity.id)
            for chunk_dict in raw_chunks:
                if chunk_dict is None:
                    continue
                chunk_id = chunk_dict.get("id", "")
                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)
                chunks.append(Chunk(
                    id=chunk_id,
                    document_id=chunk_dict.get("document_id", ""),
                    text=chunk_dict.get("text", ""),
                    index=chunk_dict.get("index", 0),
                    start_char=chunk_dict.get("start_char", 0),
                    end_char=chunk_dict.get("end_char", 0),
                ))

        if self.max_chunks > 0:
            chunks = chunks[:self.max_chunks]

        return {
            "subgraph": Subgraph(
                entities=subgraph.entities,
                relations=subgraph.relations,
                chunks=chunks,
            )
        }


@processor_registry.register("chunk_retriever")
def create_chunk_retriever(config_dict: dict, pipeline=None):
    return ChunkRetrieverProcessor(config_dict, pipeline)
