"""Data ingest processor — converts raw structured data for graph_writer.

Accepts a **list of dictionaries**, each grouping entities, relations, and
an optional source text together.  When ``text`` is provided in an item,
the processor delegates to :class:`ChunkerProcessor` to produce real
:class:`Chunk` / :class:`Document` objects that are written to the graph
and can later be embedded.  Entities and relations from that item are
automatically linked to the resulting chunks.
"""

from typing import Any, Dict, List
import logging

from ..core.base import BaseProcessor
from ..core.registry import construct_registry
from ..models.document import Document
from ..models.entity import EntityMention
from ..models.relation import Relation

logger = logging.getLogger(__name__)


class DataIngestProcessor(BaseProcessor):
    """Convert structured data items into graph_writer format.

    Expected input (via ``$input``)::

        {
            "data": [
                {
                    "entities": [
                        {"text": "Einstein", "label": "person"},
                        {"text": "Ulm", "label": "location"},
                    ],
                    "relations": [
                        {"head": "Einstein", "tail": "Ulm", "type": "born_in"},
                    ],
                    "text": "Einstein was born in Ulm.",  # optional
                },
                {
                    "entities": [
                        {"text": "Berlin", "label": "location"},
                    ],
                    "relations": [],
                },
            ]
        }

    Each item in the list contains:

    ``entities`` (required):
        List of entity dicts with keys:
        - ``text`` (required): entity display name
        - ``label`` (required): entity type (e.g. "person", "organization")
        - ``id`` (optional): explicit entity ID for linking; used as ``linked_entity_id``
        - ``properties`` (optional): arbitrary metadata dict stored on the Entity node
        - ``score`` (optional): confidence score, default 1.0

    ``relations`` (optional):
        List of relation dicts with keys:
        - ``head`` (required): text of the head entity (must match an entity ``text``)
        - ``tail`` (required): text of the tail entity
        - ``type`` (required): relation type (e.g. "born_in", "works_at")
        - ``score`` (optional): confidence score, default 1.0
        - ``properties`` (optional): arbitrary metadata dict
        - ``head_label`` / ``tail_label`` (optional): entity type hints

    ``text`` (optional):
        Source text for this item.  When provided, the text is chunked
        (using the chunker config) to produce :class:`Chunk` and
        :class:`Document` objects.  Entities and relations from the same
        item are linked to the resulting chunks via ``chunk_id``.

    ``metadata`` (optional):
        Arbitrary metadata dict stored on the :class:`Document` node
        created from ``text``.  Ignored when ``text`` is not provided.

    Config keys (chunking, only used when text is supplied):
        - ``chunk_method``: "sentence" | "fixed" | "paragraph" (default: "sentence")
        - ``chunk_size``: int (default: 512)
        - ``chunk_overlap``: int (default: 50)

    Output matches graph_writer input format::

        {
            "entities": List[List[EntityMention]],  # one inner list per item
            "relations": List[List[Relation]],       # one inner list per item
            "chunks": List[Chunk],
            "documents": List[Document],
        }
    """

    default_inputs = {"data": "$input.data"}
    default_output = "ingest_result"

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self._chunker = None

    def _ensure_chunker(self):
        if self._chunker is None:
            from .chunker import ChunkerProcessor

            chunker_config = {
                "method": self.config_dict.get("chunk_method", "sentence"),
                "chunk_size": self.config_dict.get("chunk_size", 512),
                "overlap": self.config_dict.get("chunk_overlap", 50),
            }
            self._chunker = ChunkerProcessor(chunker_config)

    def __call__(
        self,
        *,
        data: List[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if data is None:
            data = []

        all_entities: List[List[EntityMention]] = []
        all_relations: List[List[Relation]] = []
        all_chunks = []
        all_documents = []

        # Collect texts that need chunking along with their item indices
        docs_to_chunk: List[Document] = []
        text_item_indices: List[int] = []
        for i, item in enumerate(data):
            text = item.get("text")
            if text:
                doc = Document(text=text, metadata=item.get("metadata", {}))
                docs_to_chunk.append(doc)
                text_item_indices.append(i)

        # Chunk all texts in one batch if any
        # item_index -> list of chunk IDs
        item_to_chunk_ids: Dict[int, List[str]] = {}

        if docs_to_chunk:
            self._ensure_chunker()
            chunker_result = self._chunker(documents=docs_to_chunk)
            all_chunks = chunker_result["chunks"]
            all_documents = chunker_result["documents"]

            # Build item_index -> chunk_ids mapping via document_id
            doc_to_text_pos = {}
            for pos, doc in enumerate(all_documents):
                doc_to_text_pos[doc.id] = pos
            for chunk in all_chunks:
                text_pos = doc_to_text_pos.get(chunk.document_id)
                if text_pos is not None:
                    item_idx = text_item_indices[text_pos]
                    item_to_chunk_ids.setdefault(item_idx, []).append(chunk.id)

        total_ents = 0
        total_rels = 0
        linked_ents = 0
        linked_rels = 0

        for i, item in enumerate(data):
            entities = item.get("entities", [])
            relations = item.get("relations", [])
            chunk_ids = item_to_chunk_ids.get(i, [])

            # Build entity mentions for this item
            item_mentions: List[EntityMention] = []
            for ent in entities:
                mention_kwargs: Dict[str, Any] = {
                    "text": ent["text"],
                    "label": ent["label"],
                    "score": ent.get("score", 1.0),
                }
                if "id" in ent:
                    mention_kwargs["linked_entity_id"] = ent["id"]
                if "properties" in ent:
                    mention_kwargs["properties"] = ent["properties"]
                if chunk_ids:
                    for cid in chunk_ids:
                        mention = EntityMention(**mention_kwargs, chunk_id=cid)
                        item_mentions.append(mention)
                    linked_ents += 1
                else:
                    item_mentions.append(EntityMention(**mention_kwargs))
                total_ents += 1
            all_entities.append(item_mentions)

            # Build relations for this item
            item_relations: List[Relation] = []
            for rel in relations:
                cid = chunk_ids[0] if chunk_ids else ""
                r = Relation(
                    head_text=rel["head"],
                    tail_text=rel["tail"],
                    relation_type=rel["type"],
                    score=rel.get("score", 1.0),
                    chunk_id=cid,
                    head_label=rel.get("head_label", ""),
                    tail_label=rel.get("tail_label", ""),
                    properties=rel.get("properties", {}),
                )
                item_relations.append(r)
                total_rels += 1
                if cid:  # non-empty string = linked
                    linked_rels += 1
            all_relations.append(item_relations)

        # Ensure at least one group even for empty input
        if not all_entities:
            all_entities = [[]]
        if not all_relations:
            all_relations = [[]]

        if docs_to_chunk:
            logger.info(
                f"Data ingest: {total_ents} entities ({linked_ents} linked), "
                f"{total_rels} relations ({linked_rels} linked), "
                f"{len(all_chunks)} chunks from {len(docs_to_chunk)} texts"
            )
        else:
            logger.info(
                f"Data ingest: {total_ents} entities, {total_rels} relations"
            )

        return {
            "entities": all_entities,
            "relations": all_relations,
            "chunks": all_chunks,
            "documents": all_documents,
        }


@construct_registry.register("data_ingest")
def create_data_ingest(config_dict: dict, pipeline=None):
    return DataIngestProcessor(config_dict, pipeline)
