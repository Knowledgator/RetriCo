"""Entity retriever — lookup entities in Neo4j, k-hop subgraph expansion."""

from typing import Any, Dict, List
import logging

from ..core.registry import query_registry
from ..models.entity import EntityMention
from ..models.graph import Subgraph
from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class RetrieverProcessor(BaseRetriever):
    """Look up entities in Neo4j and expand to a k-hop subgraph.

    Config keys:
        neo4j_uri: str — bolt URI (default: "bolt://localhost:7687")
        neo4j_user: str — (default: "neo4j")
        neo4j_password: str — (default: "password")
        neo4j_database: str — (default: "neo4j")
        max_hops: int — subgraph expansion depth (default: 2)
    """

    default_inputs = {"entities": "parser_result.entities"}
    default_output = "retriever_result"

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.max_hops: int = config_dict.get("max_hops", 2)
        self.active_after: str | None = config_dict.get("active_after")
        self.active_before: str | None = config_dict.get("active_before")

    def __call__(self, *, entities: List[EntityMention], **kwargs) -> Dict[str, Any]:
        """Look up entities and build a subgraph.

        Returns:
            {"subgraph": Subgraph}
        """
        self._ensure_store()

        # Look up each entity mention — by linked_entity_id first, then by label
        matched_ids = []
        for mention in entities:
            record = self._lookup_entity(mention)
            if record is not None:
                matched_ids.append(record["id"])

        if not matched_ids:
            return {"subgraph": Subgraph()}

        # Get subgraph around matched entities
        raw = self._store.get_subgraph(
            matched_ids, max_hops=self.max_hops,
            active_after=self.active_after, active_before=self.active_before,
        )

        return {"subgraph": self._raw_to_subgraph(raw)}


@query_registry.register("retriever")
def create_retriever(config_dict: dict, pipeline=None):
    return RetrieverProcessor(config_dict, pipeline)
