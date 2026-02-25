"""Data ingest processor — converts raw structured data for graph_writer."""

from typing import Any, Dict, List
import logging

from ..core.base import BaseProcessor
from ..core.registry import processor_registry
from ..models.entity import EntityMention
from ..models.relation import Relation

logger = logging.getLogger(__name__)


class DataIngestProcessor(BaseProcessor):
    """Convert flat JSON entities/relations into graph_writer format.

    Expected input (via ``$input``)::

        {
            "entities": [
                {"text": "Einstein", "label": "person", "properties": {"birth_year": 1879}},
                {"text": "Ulm", "label": "location"},
            ],
            "relations": [
                {
                    "head": "Einstein",
                    "tail": "Ulm",
                    "type": "born_in",
                    "score": 1.0,
                    "properties": {"year": 1879},
                },
            ],
        }

    Entity fields:
        - ``text`` (required): entity display name
        - ``label`` (required): entity type (e.g. "person", "organization")
        - ``id`` (optional): explicit entity ID for linking; used as ``linked_entity_id``
        - ``properties`` (optional): arbitrary metadata dict stored on the Entity node

    Relation fields:
        - ``head`` (required): text of the head entity (must match an entity ``text``)
        - ``tail`` (required): text of the tail entity
        - ``type`` (required): relation type (e.g. "born_in", "works_at")
        - ``score`` (optional): confidence score, default 1.0
        - ``properties`` (optional): arbitrary metadata dict

    Output matches graph_writer input format::

        {
            "entities": [[EntityMention, ...]],
            "relations": [[Relation, ...]],
            "chunks": [],
            "documents": [],
        }
    """

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)

    def __call__(
        self,
        *,
        entities: List[Dict[str, Any]] = None,
        relations: List[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if entities is None:
            entities = []
        if relations is None:
            relations = []

        mentions = []
        for ent in entities:
            mention = EntityMention(
                text=ent["text"],
                label=ent["label"],
                score=ent.get("score", 1.0),
            )
            if "id" in ent:
                mention.linked_entity_id = ent["id"]
            mentions.append(mention)

        rels = []
        for rel in relations:
            r = Relation(
                head_text=rel["head"],
                tail_text=rel["tail"],
                relation_type=rel["type"],
                score=rel.get("score", 1.0),
                head_label=rel.get("head_label", ""),
                tail_label=rel.get("tail_label", ""),
                properties=rel.get("properties", {}),
            )
            rels.append(r)

        logger.info(
            f"Data ingest: {len(mentions)} entities, {len(rels)} relations"
        )

        return {
            "entities": [mentions],
            "relations": [rels],
            "chunks": [],
            "documents": [],
        }


@processor_registry.register("data_ingest")
def create_data_ingest(config_dict: dict, pipeline=None):
    return DataIngestProcessor(config_dict, pipeline)
