"""Path retriever — find shortest paths between parsed entities."""

from typing import Any, Dict, List
import itertools
import logging

from ..core.registry import query_registry
from ..models.entity import Entity, EntityMention
from ..models.relation import Relation
from ..models.graph import Subgraph
from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class PathRetrieverProcessor(BaseRetriever):
    """Retrieve subgraph by finding shortest paths between parsed entities.

    Flow: lookup entities -> generate pairs -> get_shortest_paths per pair ->
    collect all path nodes and edges into a Subgraph.

    Config keys:
        max_path_length: int — maximum path length for shortest path queries (default: 5)
        max_pairs: int — maximum number of entity pairs to query (default: 10)
        store_type, neo4j_uri, etc. — passed to create_store
    """

    default_inputs = {"entities": "parser_result.entities"}
    default_output = "path_retriever_result"

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.max_path_length: int = config_dict.get("max_path_length", 5)
        self.max_pairs: int = config_dict.get("max_pairs", 10)

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

        # Generate entity pairs and find shortest paths
        pairs = list(itertools.combinations(matched, 2))[:self.max_pairs]

        all_entities: Dict[str, Entity] = {}
        all_relations: List[Relation] = []

        for source, target in pairs:
            try:
                paths = self._store.get_shortest_paths(
                    source["id"], target["id"],
                    max_length=self.max_path_length,
                )
            except NotImplementedError:
                logger.warning("Store does not support get_shortest_paths")
                continue

            for path in paths:
                # Collect nodes
                for node in path.get("nodes", []):
                    if node is None:
                        continue
                    nid = node.get("id", "")
                    if nid and nid not in all_entities:
                        all_entities[nid] = Entity(
                            id=nid,
                            label=node.get("label", ""),
                            entity_type=node.get("entity_type", ""),
                        )

                # Collect path edges
                nodes_list = path.get("nodes", [])
                rels_list = path.get("rels", [])
                for i, rel in enumerate(rels_list):
                    if rel is None:
                        continue
                    head_text = ""
                    tail_text = ""
                    if i < len(nodes_list):
                        head_text = (nodes_list[i] or {}).get("id", "")
                    if i + 1 < len(nodes_list):
                        tail_text = (nodes_list[i + 1] or {}).get("id", "")
                    all_relations.append(Relation(
                        head_text=head_text,
                        tail_text=tail_text,
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
