"""Subgraph fusion — merge multiple retriever results into one Subgraph."""

from typing import Any, Dict, List

import logging

from ..core.base import BaseProcessor
from ..core.registry import query_registry
from ..models.graph import Subgraph
from ..models.entity import Entity
from ..models.relation import Relation
from ..models.document import Chunk

logger = logging.getLogger(__name__)


class SubgraphFusionProcessor(BaseProcessor):
    """Merge multiple retriever Subgraphs into a single Subgraph.

    Accepts ``subgraph_0``, ``subgraph_1``, ... kwargs, each a :class:`Subgraph`.

    Config keys:
        strategy: str — "union", "rrf", "weighted", or "intersection" (default: "union")
        top_k: int — max entities to keep (0 = all, default: 0)
        weights: List[float] — per-retriever weights for "weighted" strategy
        min_sources: int — minimum retrievers an entity must appear in for "intersection" (default: 2)
        rrf_k: int — RRF constant (default: 60)
    """

    default_inputs: Dict[str, str] = {}
    default_output = "fusion_result"

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.strategy: str = config_dict.get("strategy", "union")
        self.top_k: int = config_dict.get("top_k", 0)
        self.weights: List[float] = config_dict.get("weights", [])
        self.min_sources: int = config_dict.get("min_sources", 2)
        self.rrf_k: int = config_dict.get("rrf_k", 60)

    def __call__(self, **kwargs) -> Dict[str, Any]:
        # Collect subgraphs from kwargs (subgraph_0, subgraph_1, ...)
        subgraphs: List[Subgraph] = []
        i = 0
        while True:
            key = f"subgraph_{i}"
            if key in kwargs:
                val = kwargs[key]
                if isinstance(val, Subgraph):
                    subgraphs.append(val)
                elif isinstance(val, dict):
                    subgraphs.append(Subgraph(**val))
                else:
                    subgraphs.append(val)
                i += 1
            else:
                break

        if not subgraphs:
            return {"subgraph": Subgraph()}

        if len(subgraphs) == 1:
            return {"subgraph": subgraphs[0]}

        strategy = self.strategy
        if strategy == "union":
            merged = self._union(subgraphs)
        elif strategy == "rrf":
            merged = self._rrf(subgraphs)
        elif strategy == "weighted":
            merged = self._weighted(subgraphs)
        elif strategy == "intersection":
            merged = self._intersection(subgraphs)
        else:
            raise ValueError(
                f"Unknown fusion strategy: {strategy!r}. "
                "Expected 'union', 'rrf', 'weighted', or 'intersection'."
            )

        logger.debug(
            "Fusion (%s): %d entities, %d relations, %d chunks",
            strategy, len(merged.entities), len(merged.relations), len(merged.chunks),
        )
        return {"subgraph": merged}

    # -- strategies ----------------------------------------------------------

    def _union(self, subgraphs: List[Subgraph]) -> Subgraph:
        entities = _dedup_entities(
            [e for sg in subgraphs for e in sg.entities]
        )
        chunks = _dedup_chunks(
            [c for sg in subgraphs for c in sg.chunks]
        )
        entity_labels = {e.label for e in entities}
        relations = _filter_relations(
            [r for sg in subgraphs for r in sg.relations],
            entity_labels,
        )
        return Subgraph(entities=entities, relations=relations, chunks=chunks)

    def _rrf(self, subgraphs: List[Subgraph]) -> Subgraph:
        scores: Dict[str, float] = {}
        entity_map: Dict[str, Entity] = {}

        for sg in subgraphs:
            for rank, entity in enumerate(sg.entities):
                eid = entity.id or entity.label
                scores[eid] = scores.get(eid, 0.0) + 1.0 / (self.rrf_k + rank)
                if eid not in entity_map:
                    entity_map[eid] = entity

        sorted_ids = sorted(scores, key=lambda k: scores[k], reverse=True)
        if self.top_k > 0:
            sorted_ids = sorted_ids[: self.top_k]

        entities = [entity_map[eid] for eid in sorted_ids]
        entity_labels = {e.label for e in entities}
        relations = _filter_relations(
            [r for sg in subgraphs for r in sg.relations],
            entity_labels,
        )
        chunks = _dedup_chunks(
            [c for sg in subgraphs for c in sg.chunks]
        )
        return Subgraph(entities=entities, relations=relations, chunks=chunks)

    def _weighted(self, subgraphs: List[Subgraph]) -> Subgraph:
        weights = self.weights if self.weights else [1.0] * len(subgraphs)
        if len(weights) < len(subgraphs):
            weights = list(weights) + [1.0] * (len(subgraphs) - len(weights))

        scores: Dict[str, float] = {}
        entity_map: Dict[str, Entity] = {}

        for idx, sg in enumerate(subgraphs):
            w = weights[idx]
            for rank, entity in enumerate(sg.entities):
                eid = entity.id or entity.label
                scores[eid] = scores.get(eid, 0.0) + w / (rank + 1)
                if eid not in entity_map:
                    entity_map[eid] = entity

        sorted_ids = sorted(scores, key=lambda k: scores[k], reverse=True)
        if self.top_k > 0:
            sorted_ids = sorted_ids[: self.top_k]

        entities = [entity_map[eid] for eid in sorted_ids]
        entity_labels = {e.label for e in entities}
        relations = _filter_relations(
            [r for sg in subgraphs for r in sg.relations],
            entity_labels,
        )
        chunks = _dedup_chunks(
            [c for sg in subgraphs for c in sg.chunks]
        )
        return Subgraph(entities=entities, relations=relations, chunks=chunks)

    def _intersection(self, subgraphs: List[Subgraph]) -> Subgraph:
        # Count how many retrievers each entity appears in
        source_count: Dict[str, int] = {}
        entity_map: Dict[str, Entity] = {}

        for sg in subgraphs:
            seen_in_this = set()
            for entity in sg.entities:
                eid = entity.id or entity.label
                if eid not in seen_in_this:
                    source_count[eid] = source_count.get(eid, 0) + 1
                    seen_in_this.add(eid)
                if eid not in entity_map:
                    entity_map[eid] = entity

        entities = [
            entity_map[eid]
            for eid, count in source_count.items()
            if count >= self.min_sources
        ]
        entity_labels = {e.label for e in entities}
        relations = _filter_relations(
            [r for sg in subgraphs for r in sg.relations],
            entity_labels,
        )
        chunks = _dedup_chunks(
            [c for sg in subgraphs for c in sg.chunks]
        )
        return Subgraph(entities=entities, relations=relations, chunks=chunks)


# -- helpers -----------------------------------------------------------------

def _dedup_entities(entities: List[Entity]) -> List[Entity]:
    """Deduplicate entities by id (or label if no id)."""
    seen = set()
    result = []
    for e in entities:
        key = e.id or e.label
        if key not in seen:
            seen.add(key)
            result.append(e)
    return result


def _dedup_chunks(chunks: List[Chunk]) -> List[Chunk]:
    """Deduplicate chunks by id."""
    seen = set()
    result = []
    for c in chunks:
        if c.id not in seen:
            seen.add(c.id)
            result.append(c)
    return result


def _filter_relations(
    relations: List[Relation], entity_labels: set
) -> List[Relation]:
    """Keep relations where both head and tail are in the entity labels set.

    Also deduplicates by (head_text, tail_text, relation_type).
    """
    seen = set()
    result = []
    for r in relations:
        key = (r.head_text, r.tail_text, r.relation_type)
        if key in seen:
            continue
        if r.head_text in entity_labels and r.tail_text in entity_labels:
            seen.add(key)
            result.append(r)
    return result


@query_registry.register("fusion")
def _create_fusion(config_dict: Dict[str, Any], pipeline=None):
    return SubgraphFusionProcessor(config_dict, pipeline)
