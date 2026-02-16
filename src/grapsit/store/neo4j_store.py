"""Neo4j graph store for knowledge graph persistence."""

from typing import Any, Dict, List, Optional
import logging

from neo4j import GraphDatabase

from ..models.document import Chunk, Document
from ..models.entity import Entity, EntityMention
from ..models.relation import Relation

logger = logging.getLogger(__name__)


class Neo4jGraphStore:
    """CRUD operations for a knowledge graph in Neo4j.

    Schema::

        (:Entity:TypeName {id, label, entity_type, properties})
        (:Chunk {id, document_id, text, index, start_char, end_char})
        (:Document {id, source, metadata})

        (entity)-[:MENTIONED_IN {start, end, score}]->(chunk)
        (entity)-[:REL_TYPE {score, chunk_id}]->(entity)
        (chunk)-[:PART_OF]->(document)
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver = None

    @property
    def driver(self):
        if self._driver is None:
            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self._driver

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None

    def _run(self, query: str, parameters: dict = None) -> list:
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    # -- Setup ---------------------------------------------------------------

    def setup_indexes(self):
        """Create indexes and constraints."""
        queries = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE INDEX entity_label IF NOT EXISTS FOR (e:Entity) ON (e.label)",
            "CREATE INDEX chunk_doc IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
        ]
        for q in queries:
            try:
                self._run(q)
            except Exception as e:
                logger.debug(f"Index/constraint already exists or error: {e}")

    # -- Documents -----------------------------------------------------------

    def write_document(self, doc: Document):
        self._run(
            """
            MERGE (d:Document {id: $id})
            SET d.source = $source, d.metadata = $metadata
            """,
            {"id": doc.id, "source": doc.source, "metadata": str(doc.metadata)},
        )

    # -- Chunks --------------------------------------------------------------

    def write_chunk(self, chunk: Chunk):
        self._run(
            """
            MERGE (c:Chunk {id: $id})
            SET c.document_id = $document_id,
                c.text = $text,
                c.index = $index,
                c.start_char = $start_char,
                c.end_char = $end_char
            """,
            {
                "id": chunk.id,
                "document_id": chunk.document_id,
                "text": chunk.text,
                "index": chunk.index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
            },
        )

    def write_chunk_document_link(self, chunk_id: str, document_id: str):
        self._run(
            """
            MATCH (c:Chunk {id: $chunk_id})
            MATCH (d:Document {id: $document_id})
            MERGE (c)-[:PART_OF]->(d)
            """,
            {"chunk_id": chunk_id, "document_id": document_id},
        )

    # -- Entities ------------------------------------------------------------

    def write_entity(self, entity: Entity):
        """Create or merge an Entity node with an additional type label."""
        type_label = _sanitize_label(entity.entity_type) if entity.entity_type else None
        set_clause = "SET e.label = $label, e.entity_type = $entity_type, e.properties = $properties"

        if type_label:
            query = f"""
            MERGE (e:Entity {{id: $id}})
            {set_clause}
            WITH e
            CALL apoc.create.addLabels(e, [$type_label]) YIELD node
            RETURN node
            """
            params = {
                "id": entity.id,
                "label": entity.label,
                "entity_type": entity.entity_type,
                "properties": str(entity.properties),
                "type_label": type_label,
            }
        else:
            query = f"""
            MERGE (e:Entity {{id: $id}})
            {set_clause}
            """
            params = {
                "id": entity.id,
                "label": entity.label,
                "entity_type": entity.entity_type,
                "properties": str(entity.properties),
            }

        try:
            self._run(query, params)
        except Exception:
            # Fallback without APOC (addLabels requires APOC plugin)
            self._run(
                f"MERGE (e:Entity {{id: $id}}) {set_clause}",
                {
                    "id": entity.id,
                    "label": entity.label,
                    "entity_type": entity.entity_type,
                    "properties": str(entity.properties),
                },
            )

    def write_mention_link(self, entity_id: str, chunk_id: str, mention: EntityMention):
        self._run(
            """
            MATCH (e:Entity {id: $entity_id})
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (e)-[r:MENTIONED_IN]->(c)
            SET r.start = $start, r.end = $end, r.score = $score, r.text = $text
            """,
            {
                "entity_id": entity_id,
                "chunk_id": chunk_id,
                "start": mention.start,
                "end": mention.end,
                "score": mention.score,
                "text": mention.text,
            },
        )

    # -- Relations -----------------------------------------------------------

    def write_relation(self, relation: Relation, head_entity_id: str, tail_entity_id: str):
        rel_type = _sanitize_label(relation.relation_type)
        self._run(
            f"""
            MATCH (h:Entity {{id: $head_id}})
            MATCH (t:Entity {{id: $tail_id}})
            MERGE (h)-[r:`{rel_type}`]->(t)
            SET r.score = $score, r.chunk_id = $chunk_id, r.id = $id
            """,
            {
                "head_id": head_entity_id,
                "tail_id": tail_entity_id,
                "score": relation.score,
                "chunk_id": relation.chunk_id,
                "id": relation.id,
            },
        )

    # -- Reads ---------------------------------------------------------------

    def get_entity_by_label(self, label: str) -> Optional[Dict[str, Any]]:
        records = self._run(
            "MATCH (e:Entity) WHERE toLower(e.label) = toLower($label) RETURN e",
            {"label": label},
        )
        if records:
            return records[0]["e"]
        return None

    def get_entity_neighbors(self, entity_id: str, max_hops: int = 1) -> List[Dict[str, Any]]:
        records = self._run(
            f"""
            MATCH path = (e:Entity {{id: $id}})-[*1..{max_hops}]-(neighbor:Entity)
            RETURN DISTINCT neighbor
            """,
            {"id": entity_id},
        )
        return [r["neighbor"] for r in records]

    def get_entity_relations(self, entity_id: str) -> List[Dict[str, Any]]:
        records = self._run(
            """
            MATCH (e:Entity {id: $id})-[r]->(t:Entity)
            RETURN type(r) as relation_type, r.score as score, t.id as target_id,
                   t.label as target_label, r.chunk_id as chunk_id
            UNION
            MATCH (s:Entity)-[r]->(e:Entity {id: $id})
            WHERE NOT type(r) = 'MENTIONED_IN'
            RETURN type(r) as relation_type, r.score as score, s.id as target_id,
                   s.label as target_label, r.chunk_id as chunk_id
            """,
            {"id": entity_id},
        )
        return records

    def get_chunks_for_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        records = self._run(
            """
            MATCH (e:Entity {id: $id})-[:MENTIONED_IN]->(c:Chunk)
            RETURN c
            """,
            {"id": entity_id},
        )
        return [r["c"] for r in records]

    def get_subgraph(self, entity_ids: List[str], max_hops: int = 1) -> Dict[str, Any]:
        """Retrieve a subgraph around a set of entity IDs."""
        records = self._run(
            f"""
            MATCH (e:Entity) WHERE e.id IN $ids
            OPTIONAL MATCH path = (e)-[*1..{max_hops}]-(neighbor:Entity)
            WITH collect(DISTINCT e) + collect(DISTINCT neighbor) AS all_nodes
            UNWIND all_nodes AS n
            WITH collect(DISTINCT n) AS nodes
            UNWIND nodes AS a
            UNWIND nodes AS b
            OPTIONAL MATCH (a)-[r]->(b) WHERE NOT type(r) = 'MENTIONED_IN'
            RETURN collect(DISTINCT properties(a)) AS entities,
                   collect(DISTINCT {{head: a.id, tail: b.id, type: type(r), score: r.score}}) AS relations
            """,
            {"ids": entity_ids},
        )
        if records:
            return records[0]
        return {"entities": [], "relations": []}

    def clear_all(self):
        """Delete all nodes and relationships."""
        self._run("MATCH (n) DETACH DELETE n")


def _sanitize_label(s: str) -> str:
    """Sanitize a string for use as a Neo4j label or relationship type."""
    s = s.strip().replace(" ", "_").replace("-", "_")
    s = "".join(c for c in s if c.isalnum() or c == "_")
    if s and s[0].isdigit():
        s = "_" + s
    return s.upper() if s else "UNKNOWN"
