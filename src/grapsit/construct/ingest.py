"""Data ingest processor — converts raw structured data for graph_writer.

Optionally accepts ``texts`` so that chunks and documents are created
alongside structured entities/relations.  When texts are provided, the
processor delegates to :class:`ChunkerProcessor` to produce real
:class:`Chunk` / :class:`Document` objects that are written to the graph
and can later be embedded.

Entities and relations can reference their source text via an optional
``text_index`` field.  This creates ``MENTIONED_IN`` edges (entities) and
``chunk_id`` properties (relations) so that chunk retrieval works for
ingested data.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional
import logging

from ..core.base import BaseProcessor
from ..core.registry import construct_registry
from ..models.entity import EntityMention
from ..models.relation import Relation

logger = logging.getLogger(__name__)


class DataIngestProcessor(BaseProcessor):
    """Convert flat JSON entities/relations into graph_writer format.

    Expected input (via ``$input``)::

        {
            "entities": [
                {"text": "Einstein", "label": "person", "text_index": 0},
                {"text": "Ulm", "label": "location", "text_index": 0},
                {"text": "Berlin", "label": "location"},
            ],
            "relations": [
                {"head": "Einstein", "tail": "Ulm", "type": "born_in", "text_index": 0},
            ],
            "texts": ["Einstein was born in Ulm.", "He later lived in Berlin."],
        }

    Entity fields:
        - ``text`` (required): entity display name
        - ``label`` (required): entity type (e.g. "person", "organization")
        - ``id`` (optional): explicit entity ID for linking; used as ``linked_entity_id``
        - ``properties`` (optional): arbitrary metadata dict stored on the Entity node
        - ``text_index`` (optional): 0-based index into ``texts`` list.  When
          texts are provided, this links the entity to the chunks produced
          from that text via ``MENTIONED_IN`` edges.

    Relation fields:
        - ``head`` (required): text of the head entity (must match an entity ``text``)
        - ``tail`` (required): text of the tail entity
        - ``type`` (required): relation type (e.g. "born_in", "works_at")
        - ``score`` (optional): confidence score, default 1.0
        - ``properties`` (optional): arbitrary metadata dict
        - ``text_index`` (optional): 0-based index into ``texts`` list.
          Sets ``chunk_id`` on the relation edge in the graph.

    Texts (optional):
        When ``texts`` are provided, they are chunked (using the chunker
        config in ``config_dict``) to produce :class:`Chunk` and
        :class:`Document` objects.  Entities/relations with ``text_index``
        are linked to the corresponding chunks.

    Config keys (chunking, only used when texts are supplied):
        - ``chunk_method``: "sentence" | "fixed" | "paragraph" (default: "sentence")
        - ``chunk_size``: int (default: 512)
        - ``chunk_overlap``: int (default: 50)

    Output matches graph_writer input format::

        {
            "entities": List[List[EntityMention]],  # per-chunk when texts provided
            "relations": List[List[Relation]],       # per-chunk when texts provided
            "chunks": List[Chunk],
            "documents": List[Document],
        }
    """

    default_inputs = {"entities": "$input.entities", "relations": "$input.relations"}
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
        entities: List[Dict[str, Any]] = None,
        relations: List[Dict[str, Any]] = None,
        texts: List[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if entities is None:
            entities = []
        if relations is None:
            relations = []

        # Chunk texts if provided
        chunks = []
        documents = []
        # Map: text_index -> list of chunk IDs produced from that text
        text_to_chunk_ids: Dict[int, List[str]] = {}

        if texts:
            self._ensure_chunker()
            chunker_result = self._chunker(texts=texts)
            chunks = chunker_result["chunks"]
            documents = chunker_result["documents"]

            # Build text_index -> chunk_ids mapping via document_id
            doc_to_text_idx = {}
            for i, doc in enumerate(documents):
                doc_to_text_idx[doc.id] = i
            for chunk in chunks:
                tidx = doc_to_text_idx.get(chunk.document_id)
                if tidx is not None:
                    text_to_chunk_ids.setdefault(tidx, []).append(chunk.id)

        # Build entity mentions, linking to chunks when text_index is present
        # Group by chunk for the list-of-lists output format
        chunk_id_to_idx: Dict[str, int] = {c.id: i for i, c in enumerate(chunks)}
        per_chunk_entities: List[List[EntityMention]] = [[] for _ in chunks]
        unlinked_entities: List[EntityMention] = []

        for ent in entities:
            text_index: Optional[int] = ent.get("text_index")
            chunk_ids = text_to_chunk_ids.get(text_index, []) if text_index is not None else []

            if chunk_ids:
                # Create a mention for each chunk from this text
                for cid in chunk_ids:
                    mention = EntityMention(
                        text=ent["text"],
                        label=ent["label"],
                        score=ent.get("score", 1.0),
                        chunk_id=cid,
                    )
                    if "id" in ent:
                        mention.linked_entity_id = ent["id"]
                    cidx = chunk_id_to_idx[cid]
                    per_chunk_entities[cidx].append(mention)
            else:
                # No text link — unlinked entity
                mention = EntityMention(
                    text=ent["text"],
                    label=ent["label"],
                    score=ent.get("score", 1.0),
                )
                if "id" in ent:
                    mention.linked_entity_id = ent["id"]
                unlinked_entities.append(mention)

        # Build relations, setting chunk_id when text_index is present
        per_chunk_relations: List[List[Relation]] = [[] for _ in chunks]
        unlinked_relations: List[Relation] = []

        for rel in relations:
            text_index: Optional[int] = rel.get("text_index")
            chunk_ids = text_to_chunk_ids.get(text_index, []) if text_index is not None else []

            if chunk_ids:
                # Link relation to the first chunk from its source text
                cid = chunk_ids[0]
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
                cidx = chunk_id_to_idx[cid]
                per_chunk_relations[cidx].append(r)
            else:
                r = Relation(
                    head_text=rel["head"],
                    tail_text=rel["tail"],
                    relation_type=rel["type"],
                    score=rel.get("score", 1.0),
                    head_label=rel.get("head_label", ""),
                    tail_label=rel.get("tail_label", ""),
                    properties=rel.get("properties", {}),
                )
                unlinked_relations.append(r)

        # Combine per-chunk and unlinked into the list-of-lists format
        if chunks:
            out_entities = per_chunk_entities
            out_relations = per_chunk_relations
            # Append unlinked items as an extra group
            if unlinked_entities:
                out_entities.append(unlinked_entities)
            if unlinked_relations:
                out_relations.append(unlinked_relations)
        else:
            # No texts — everything is unlinked, wrapped in one group
            out_entities = [unlinked_entities]
            out_relations = [unlinked_relations]

        linked_ents = sum(len(g) for g in per_chunk_entities)
        linked_rels = sum(len(g) for g in per_chunk_relations)
        total_ents = linked_ents + len(unlinked_entities)
        total_rels = linked_rels + len(unlinked_relations)

        if texts:
            logger.info(
                f"Data ingest: {total_ents} entities ({linked_ents} linked), "
                f"{total_rels} relations ({linked_rels} linked), "
                f"{len(chunks)} chunks from {len(texts)} texts"
            )
        else:
            logger.info(
                f"Data ingest: {total_ents} entities, {total_rels} relations"
            )

        return {
            "entities": out_entities,
            "relations": out_relations,
            "chunks": chunks,
            "documents": documents,
        }


@construct_registry.register("data_ingest")
def create_data_ingest(config_dict: dict, pipeline=None):
    return DataIngestProcessor(config_dict, pipeline)
