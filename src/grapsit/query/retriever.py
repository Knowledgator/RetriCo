"""Entity retriever — lookup entities in Neo4j, k-hop subgraph expansion."""

from typing import Any, Dict, List
import logging

from ..core.base import BaseProcessor
from ..core.registry import processor_registry
from ..models.entity import Entity, EntityMention
from ..models.relation import Relation
from ..models.graph import Subgraph

logger = logging.getLogger(__name__)


class RetrieverProcessor(BaseProcessor):
    """Look up entities in Neo4j and expand to a k-hop subgraph.

    Config keys:
        neo4j_uri: str — bolt URI (default: "bolt://localhost:7687")
        neo4j_user: str — (default: "neo4j")
        neo4j_password: str — (default: "password")
        neo4j_database: str — (default: "neo4j")
        max_hops: int — subgraph expansion depth (default: 2)
    """

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.max_hops: int = config_dict.get("max_hops", 2)
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

    def __call__(self, *, entities: List[EntityMention], **kwargs) -> Dict[str, Any]:
        """Look up entities and build a subgraph.

        Returns:
            {"subgraph": Subgraph}
        """
        self._ensure_store()

        # Look up each entity mention — by linked_entity_id first, then by label
        matched_ids = []
        for mention in entities:
            if mention.linked_entity_id:
                record = self._store.get_entity_by_id(mention.linked_entity_id)
            else:
                record = self._store.get_entity_by_label(mention.text)
            if record is not None:
                matched_ids.append(record["id"])

        if not matched_ids:
            return {"subgraph": Subgraph()}

        # Get subgraph around matched entities
        raw = self._store.get_subgraph(matched_ids, max_hops=self.max_hops)

        # Convert raw dicts to Pydantic models
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

        return {"subgraph": Subgraph(entities=sg_entities, relations=sg_relations)}


@processor_registry.register("retriever")
def create_retriever(config_dict: dict, pipeline=None):
    return RetrieverProcessor(config_dict, pipeline)
