"""Graph writer processor — deduplicates and writes to Neo4j."""

from typing import Any, Dict, List
import json
import logging
from pathlib import Path

from ..core.base import BaseProcessor
from ..core.registry import construct_registry
from ..models.document import Chunk, Document
from ..models.entity import Entity, EntityMention
from ..models.relation import Relation
from ..store.pool import resolve_from_pool_or_create

logger = logging.getLogger(__name__)


class GraphWriterProcessor(BaseProcessor):
    """Deduplicate entities and write everything to the graph store.

    Config keys:
        store_type: str (default: "neo4j") — "neo4j" or "falkordb"
        neo4j_uri: str (default: "bolt://localhost:7687")
        neo4j_user: str (default: "neo4j")
        neo4j_password: str (default: "password")
        neo4j_database: str (default: "neo4j")
        falkordb_host: str (default: "localhost")
        falkordb_port: int (default: 6379)
        falkordb_graph: str (default: "retrico")
        setup_indexes: bool (default: True)
        json_output: str (default: None) — path to save extracted data as JSON
            in the ingest-ready format (compatible with ``ingest_data()``).
        write_reversed_relations: bool (default: False) — if True, for each
            relation ``A -[REL]-> B`` also write ``B -[REV_REL]-> A``.  This
            enables path-based retrievers to traverse the graph in both
            directions.  The reverse relation type is prefixed with ``REV_``
            (e.g. ``BORN_IN`` → ``REV_BORN_IN``).
    """

    default_inputs = {
        "entities": "relex_result.entities",
        "relations": "relex_result.relations",
        "chunks": "chunker_result.chunks",
        "documents": "chunker_result.documents",
    }
    default_output = "writer_result"

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.json_output = config_dict.get("json_output", None)
        self.write_reversed = config_dict.get("write_reversed_relations", False)
        self.chunk_table = config_dict.get("chunk_table", "chunks")
        self.document_table = config_dict.get("document_table", "documents")
        # Graph store is optional — a relational-only pipeline may skip it
        self.store = None
        if self._has_graph_config(config_dict):
            self.store = resolve_from_pool_or_create(config_dict, "graph")
            if config_dict.get("setup_indexes", True):
                try:
                    self.store.setup_indexes()
                except Exception as e:
                    logger.warning(f"Could not setup indexes: {e}")
        # Optional relational store for chunk/document storage
        self.relational_store = None
        if self._has_relational_config(config_dict):
            try:
                self.relational_store = resolve_from_pool_or_create(config_dict, "relational")
            except (ValueError, KeyError) as e:
                logger.debug(f"No relational store configured: {e}")

    @staticmethod
    def _has_graph_config(config_dict: dict) -> bool:
        """Check if config contains graph store configuration."""
        if config_dict.get("store_type"):
            return True
        pool = config_dict.get("__store_pool__")
        if pool is not None:
            name = config_dict.get("graph_store_name", "default")
            return pool.has_graph(name)
        return False

    @staticmethod
    def _has_relational_config(config_dict: dict) -> bool:
        """Check if config contains relational store configuration."""
        if config_dict.get("relational_store_type"):
            return True
        pool = config_dict.get("__store_pool__")
        if pool is not None:
            name = config_dict.get("relational_store_name", "default")
            return pool.has_relational(name)
        return False

    def __call__(
        self,
        *,
        chunks: List[Chunk] = None,
        documents: List[Document] = None,
        entities: List[List[EntityMention]] = None,
        relations: List[List[Relation]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Write the full graph to Neo4j.

        Args:
            chunks: All chunks.
            documents: Source documents.
            entities: Per-chunk entity mentions (list of lists).
            relations: Per-chunk relations (list of lists).

        Returns:
            {"entity_count": int, "relation_count": int, "chunk_count": int,
             "entity_map": Dict[str, Entity]}
        """
        if chunks is None:
            chunks = []
        if documents is None:
            documents = []
        if entities is None:
            entities = []
        if relations is None:
            relations = []

        # 1. Write documents to graph store (if configured)
        if self.store is not None:
            for doc in documents:
                self.store.write_document(doc)

        # 2. Write chunks + link to documents in graph store (if configured)
        if self.store is not None:
            for chunk in chunks:
                self.store.write_chunk(chunk)
                if chunk.document_id:
                    self.store.write_chunk_document_link(chunk.id, chunk.document_id)

        # 3. Deduplicate entities by canonical name (or linked_entity_id if available)
        entity_map: Dict[str, Entity] = {}  # dedup_key -> Entity
        for chunk_mentions in entities:
            if not isinstance(chunk_mentions, list):
                chunk_mentions = [chunk_mentions]
            for mention in chunk_mentions:
                if isinstance(mention, dict):
                    mention = EntityMention(**mention)
                # Use linked_entity_id as dedup key if available, else canonical name
                if mention.linked_entity_id:
                    key = mention.linked_entity_id
                else:
                    key = mention.text.strip().lower()
                if key not in entity_map:
                    entity_kwargs = {
                        "label": mention.text.strip(),
                        "entity_type": mention.label,
                    }
                    if mention.linked_entity_id:
                        entity_kwargs["id"] = mention.linked_entity_id
                    if mention.properties:
                        entity_kwargs["properties"] = dict(mention.properties)
                    entity_map[key] = Entity(**entity_kwargs)
                elif mention.properties:
                    # Merge properties from subsequent mentions
                    entity_map[key].properties.update(mention.properties)
                entity_map[key].mentions.append(mention)

        # 4. Write entities + mention links to graph store (if configured)
        if self.store is not None:
            for entity in entity_map.values():
                self.store.write_entity(entity)
                for mention in entity.mentions:
                    if mention.chunk_id:
                        self.store.write_mention_link(entity.id, mention.chunk_id, mention)

        # Build a text-based lookup for relation resolution
        # (entity_map may be keyed by linked_entity_id instead of canonical name)
        text_to_entity: Dict[str, Entity] = {}
        for entity in entity_map.values():
            text_to_entity[entity.label.strip().lower()] = entity
            for mention in entity.mentions:
                text_to_entity[mention.text.strip().lower()] = entity

        # 5. Deduplicate relations and accumulate chunk_ids
        # Key: (head_key, tail_key, rel_type) -> Relation with merged chunk_ids
        deduped_relations: Dict[tuple, tuple] = {}  # key -> (rel, head_entity, tail_entity)
        for chunk_relations in relations:
            if not isinstance(chunk_relations, list):
                chunk_relations = [chunk_relations]
            for rel in chunk_relations:
                if isinstance(rel, dict):
                    rel = Relation(**rel)
                head_key = rel.head_text.strip().lower()
                tail_key = rel.tail_text.strip().lower()
                head_entity = text_to_entity.get(head_key)
                tail_entity = text_to_entity.get(tail_key)
                if head_entity and tail_entity:
                    rel_type_key = rel.relation_type.strip().upper().replace(" ", "_")
                    dedup_key = (head_key, tail_key, rel_type_key)
                    if dedup_key in deduped_relations:
                        existing_rel = deduped_relations[dedup_key][0]
                        # Accumulate chunk_ids from this relation
                        for cid in rel.chunk_id:
                            if cid and cid not in existing_rel.chunk_id:
                                existing_rel.chunk_id.append(cid)
                    else:
                        deduped_relations[dedup_key] = (rel, head_entity, tail_entity)
                else:
                    logger.debug(
                        f"Skipping relation {rel.head_text} -> {rel.tail_text}: "
                        f"head_found={head_entity is not None}, tail_found={tail_entity is not None}"
                    )

        # Write deduplicated relations
        rel_count = 0
        for rel, head_entity, tail_entity in deduped_relations.values():
            if self.store is not None:
                self.store.write_relation(rel, head_entity.id, tail_entity.id)
                if self.write_reversed:
                    rev_type = "REV_" + rel.relation_type.strip().upper().replace(" ", "_")
                    rev_rel = Relation(
                        head_text=rel.tail_text,
                        tail_text=rel.head_text,
                        relation_type=rev_type,
                        score=rel.score,
                        chunk_id=list(rel.chunk_id),
                        head_label=rel.tail_label,
                        tail_label=rel.head_label,
                        start_date=rel.start_date,
                        end_date=rel.end_date,
                        properties=rel.properties,
                    )
                    self.store.write_relation(rev_rel, tail_entity.id, head_entity.id)
            rel_count += 1

        # 6. Write chunks/documents to relational store (if configured)
        if self.relational_store is not None:
            self._write_to_relational(documents, chunks)

        result = {
            "entity_count": len(entity_map),
            "relation_count": rel_count,
            "chunk_count": len(chunks),
            "entity_map": entity_map,
        }
        logger.info(
            f"Graph written: {result['entity_count']} entities, "
            f"{result['relation_count']} relations, {result['chunk_count']} chunks"
        )

        # Export to JSON in ingest-ready format if configured
        if self.json_output:
            self._save_json(entity_map, text_to_entity, relations, chunks, documents)

        return result

    def _write_to_relational(
        self,
        documents: List[Document],
        chunks: List[Chunk],
    ) -> None:
        """Write documents and chunks to the relational store."""
        if documents:
            doc_records = []
            for doc in documents:
                record: Dict[str, Any] = {"id": doc.id, "source": doc.source}
                if doc.metadata:
                    record["metadata"] = json.dumps(doc.metadata)
                doc_records.append(record)
            self.relational_store.write_records(self.document_table, doc_records)

        if chunks:
            chunk_records = []
            for chunk in chunks:
                chunk_records.append({
                    "id": chunk.id,
                    "document_id": chunk.document_id or "",
                    "text": chunk.text,
                    "index": chunk.index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                })
            self.relational_store.write_records(self.chunk_table, chunk_records)

        logger.debug(
            f"Relational store: wrote {len(documents)} documents to "
            f"'{self.document_table}', {len(chunks)} chunks to '{self.chunk_table}'"
        )

    def _save_json(
        self,
        entity_map: Dict[str, Entity],
        text_to_entity: Dict[str, Entity],
        relations: List[List],
        chunks: List[Chunk],
        documents: List[Document],
    ) -> None:
        """Save data in ingest-ready format (compatible with ``ingest_data()``).

        Groups entities and relations by document.  Each document becomes one
        item in the output list with its source text, metadata, entities, and
        relations.  Entities/relations not linked to any chunk are collected
        into a separate item without ``text``.
        """
        # Map chunk_id -> document_id, and document_id -> Document
        chunk_to_doc: Dict[str, str] = {}
        doc_by_id: Dict[str, Document] = {}
        for chunk in chunks:
            if chunk.document_id:
                chunk_to_doc[chunk.id] = chunk.document_id
        for doc in documents:
            doc_by_id[doc.id] = doc

        # Collect entities per document (and unlinked)
        doc_entities: Dict[str, List[Dict[str, Any]]] = {}  # doc_id -> [entity dicts]
        unlinked_entities: List[Dict[str, Any]] = []
        seen_entity_docs: Dict[str, set] = {}  # entity key -> set of doc_ids already added

        for entity in entity_map.values():
            entry: Dict[str, Any] = {
                "text": entity.label,
                "label": entity.entity_type,
            }
            if entity.id:
                entry["id"] = entity.id
            if entity.properties:
                entry["properties"] = entity.properties

            # Find which documents this entity belongs to via its mentions
            entity_key = entity.label.strip().lower()
            seen_entity_docs[entity_key] = set()
            placed = False
            for mention in entity.mentions:
                if mention.chunk_id and mention.chunk_id in chunk_to_doc:
                    doc_id = chunk_to_doc[mention.chunk_id]
                    if doc_id not in seen_entity_docs[entity_key]:
                        seen_entity_docs[entity_key].add(doc_id)
                        doc_entities.setdefault(doc_id, []).append(entry)
                        placed = True
            if not placed:
                unlinked_entities.append(entry)

        # Collect relations per document (and unlinked)
        doc_relations: Dict[str, List[Dict[str, Any]]] = {}  # doc_id -> [relation dicts]
        unlinked_relations: List[Dict[str, Any]] = []

        for chunk_relations in relations:
            if not isinstance(chunk_relations, list):
                chunk_relations = [chunk_relations]
            for rel in chunk_relations:
                if isinstance(rel, dict):
                    rel = Relation(**rel)
                head_key = rel.head_text.strip().lower()
                tail_key = rel.tail_text.strip().lower()
                head_entity = text_to_entity.get(head_key)
                tail_entity = text_to_entity.get(tail_key)
                if not (head_entity and tail_entity):
                    continue
                entry: Dict[str, Any] = {
                    "head": head_entity.label,
                    "tail": tail_entity.label,
                    "type": rel.relation_type,
                }
                if rel.score != 1.0:
                    entry["score"] = rel.score
                if rel.head_label:
                    entry["head_label"] = rel.head_label
                if rel.tail_label:
                    entry["tail_label"] = rel.tail_label
                if rel.start_date is not None:
                    entry["start_date"] = rel.start_date
                if rel.end_date is not None:
                    entry["end_date"] = rel.end_date
                if rel.properties:
                    entry["properties"] = rel.properties

                # Place relation in its document group (use first matching chunk_id)
                placed = False
                for cid in rel.chunk_id:
                    doc_id = chunk_to_doc.get(cid, "")
                    if doc_id and doc_id in doc_by_id:
                        doc_relations.setdefault(doc_id, []).append(entry)
                        placed = True
                        break
                if not placed:
                    unlinked_relations.append(entry)

        # Build the output list — one item per document
        output: List[Dict[str, Any]] = []
        all_doc_ids = sorted(
            set(list(doc_entities.keys()) + list(doc_relations.keys())),
            key=lambda did: next(
                (c.index for c in chunks if c.document_id == did), 0
            ),
        )
        for doc_id in all_doc_ids:
            item: Dict[str, Any] = {
                "entities": doc_entities.get(doc_id, []),
                "relations": doc_relations.get(doc_id, []),
            }
            doc = doc_by_id.get(doc_id)
            if doc and doc.text:
                item["text"] = doc.text
            if doc and doc.metadata:
                item["metadata"] = doc.metadata
            output.append(item)

        # Add unlinked entities/relations as a separate item (if any)
        if unlinked_entities or unlinked_relations:
            output.append({
                "entities": unlinked_entities,
                "relations": unlinked_relations,
            })

        filepath = Path(self.json_output)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        total_ents = sum(len(item.get("entities", [])) for item in output)
        total_rels = sum(len(item.get("relations", [])) for item in output)
        logger.info(
            f"Saved {len(output)} items ({total_ents} entities, "
            f"{total_rels} relations) to {self.json_output}"
        )


@construct_registry.register("graph_writer")
def create_graph_writer(config_dict: dict, pipeline=None):
    return GraphWriterProcessor(config_dict, pipeline)
