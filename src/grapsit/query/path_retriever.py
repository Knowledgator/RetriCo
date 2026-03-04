"""Path retriever — find shortest paths between parsed entities."""

from typing import Any, Dict, List
import logging

from ..core.registry import query_registry
from ..models.document import Chunk
from ..models.entity import Entity, EntityMention
from ..models.relation import Relation
from ..models.graph import Subgraph
from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class PathRetrieverProcessor(BaseRetriever):
    """Retrieve subgraph by finding shortest paths between parsed entities.

    Flow: lookup entities -> get_top_shortest_paths (DB-level top_k) ->
    collect path nodes and edges into a Subgraph.

    Config keys:
        max_path_length: int — maximum path length for shortest path queries (default: 5)
        top_k: int — maximum number of paths to return (default: 3)
        store_type, neo4j_uri, etc. — passed to create_store
    """

    default_inputs = {"entities": "parser_result.entities"}
    default_output = "path_retriever_result"

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.max_path_length: int = config_dict.get("max_path_length", 5)
        self.top_k: int = config_dict.get("top_k", 3)

    def __call__(self, *, entities: List[EntityMention], **kwargs) -> Dict[str, Any]:
        """Find shortest paths between entity pairs and build a subgraph.

        Returns:
            {"subgraph": Subgraph}
        """
        if not entities:
            return {"subgraph": Subgraph()}

        self._ensure_store()

        # Look up each entity mention
        matched = []
        for mention in entities:
            record = self._lookup_entity(mention)
            if record is not None:
                matched.append(record)
                logger.info(f"Entity matched: {mention.text!r} -> {record.get('label', '?')!r} (id={record.get('id', '?')[:8]}...)")
            else:
                logger.info(f"Entity NOT found: {mention.text!r} ({mention.label})")

        if len(matched) < 2:
            # Need at least 2 entities for paths; fall back to single entity subgraph
            if matched:
                raw = self._store.get_subgraph([matched[0]["id"]], max_hops=1)
                return {"subgraph": self._raw_to_subgraph(raw)}
            return {"subgraph": Subgraph()}

        # Single DB call: find top_k shortest paths among all matched entities
        entity_ids = [m["id"] for m in matched]
        try:
            top_paths = self._store.get_top_shortest_paths(
                entity_ids,
                max_length=self.max_path_length,
                top_k=self.top_k,
            )
        except NotImplementedError:
            logger.warning("Store does not support get_top_shortest_paths")
            return {"subgraph": Subgraph()}

        logger.info(f"Path retriever: got {len(top_paths)} paths (top_k={self.top_k})")

        # Build subgraph from selected paths
        all_entities: Dict[str, Entity] = {}
        all_relations: List[Relation] = []

        for path in top_paths:
            nodes_list = path.get("nodes", [])
            rels_list = path.get("rels", [])

            for node in nodes_list:
                if node is None:
                    continue
                nid = node.get("id", "")
                if nid and nid not in all_entities:
                    all_entities[nid] = Entity(
                        id=nid,
                        label=node.get("label", ""),
                        entity_type=node.get("entity_type", ""),
                    )

            for i, rel in enumerate(rels_list):
                if rel is None:
                    continue
                head_node = (nodes_list[i] or {}) if i < len(nodes_list) else {}
                tail_node = (nodes_list[i + 1] or {}) if i + 1 < len(nodes_list) else {}
                all_relations.append(Relation(
                    head_text=head_node.get("label", "") or head_node.get("id", ""),
                    tail_text=tail_node.get("label", "") or tail_node.get("id", ""),
                    relation_type=rel.get("type", ""),
                    score=rel.get("score", 0.0) or 0.0,
                ))

        return {
            "subgraph": Subgraph(
                entities=list(all_entities.values()),
                relations=all_relations,
            )
        }


@query_registry.register("path_retriever")
def create_path_retriever(config_dict: dict, pipeline=None):
    return PathRetrieverProcessor(config_dict, pipeline)
