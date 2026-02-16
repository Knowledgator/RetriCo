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
    """

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.max_chunks: int = config_dict.get("max_chunks", 0)
        self._store = None

    def _ensure_store(self):
        if self._store is None:
            from ..store.neo4j_store import Neo4jGraphStore
            self._store = Neo4jGraphStore(
                uri=self.config_dict.get("neo4j_uri", "bolt://localhost:7687"),
                user=self.config_dict.get("neo4j_user", "neo4j"),
                password=self.config_dict.get("neo4j_password", "password"),
                database=self.config_dict.get("neo4j_database", "neo4j"),
            )

    def __call__(self, *, subgraph: Any, **kwargs) -> Dict[str, Any]:
        """Retrieve chunks for all entities in the subgraph.

        Returns:
            {"subgraph": Subgraph} with chunks populated
        """
        self._ensure_store()

        # Accept Subgraph or dict
        if isinstance(subgraph, dict):
            subgraph = Subgraph(**subgraph)
        elif not isinstance(subgraph, Subgraph):
            subgraph = Subgraph()

        seen_ids = set()
        chunks = []

        for entity in subgraph.entities:
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
