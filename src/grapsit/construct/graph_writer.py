"""Graph writer processor — deduplicates and writes to Neo4j."""

from typing import Any, Dict, List
import logging

from ..core.base import BaseProcessor
from ..core.registry import processor_registry
from ..models.document import Chunk, Document
from ..models.entity import Entity, EntityMention
from ..models.relation import Relation
from ..store.neo4j_store import Neo4jGraphStore

logger = logging.getLogger(__name__)


class GraphWriterProcessor(BaseProcessor):
    """Deduplicate entities and write everything to Neo4j.

    Config keys:
        neo4j_uri: str (default: "bolt://localhost:7687")
        neo4j_user: str (default: "neo4j")
        neo4j_password: str (default: "password")
        neo4j_database: str (default: "neo4j")
        setup_indexes: bool (default: True)
    """

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.store = Neo4jGraphStore(
            uri=config_dict.get("neo4j_uri", "bolt://localhost:7687"),
            user=config_dict.get("neo4j_user", "neo4j"),
            password=config_dict.get("neo4j_password", "password"),
            database=config_dict.get("neo4j_database", "neo4j"),
        )
        if config_dict.get("setup_indexes", True):
            try:
                self.store.setup_indexes()
            except Exception as e:
                logger.warning(f"Could not setup indexes: {e}")

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

        # 1. Write documents
        for doc in documents:
            self.store.write_document(doc)

        # 2. Write chunks + link to documents
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
                    entity_map[key] = Entity(**entity_kwargs)
                entity_map[key].mentions.append(mention)

        # 4. Write entities + mention links
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

        # 5. Write relations
        rel_count = 0
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
                    self.store.write_relation(rel, head_entity.id, tail_entity.id)
                    rel_count += 1
                else:
                    logger.debug(
                        f"Skipping relation {rel.head_text} -> {rel.tail_text}: "
                        f"head_found={head_entity is not None}, tail_found={tail_entity is not None}"
                    )

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
        return result


@processor_registry.register("graph_writer")
def create_graph_writer(config_dict: dict, pipeline=None):
    return GraphWriterProcessor(config_dict, pipeline)
