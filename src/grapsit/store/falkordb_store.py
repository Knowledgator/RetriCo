"""FalkorDB graph store for knowledge graph persistence."""

from typing import Any, Dict, List, Optional
import logging

from .base import BaseGraphStore
from ..models.document import Chunk, Document
from ..models.entity import Entity, EntityMention
from ..models.relation import Relation

logger = logging.getLogger(__name__)


class FalkorDBGraphStore(BaseGraphStore):
    """CRUD operations for a knowledge graph in FalkorDB.

    FalkorDB is a Redis-based graph database supporting OpenCypher.

    Schema::

        (:Entity {id, label, entity_type, properties})
        (:Chunk {id, document_id, text, index, start_char, end_char})
        (:Document {id, source, metadata})

        (entity)-[:MENTIONED_IN {start, end, score, text}]->(chunk)
        (entity)-[:REL_TYPE {score, chunk_id, id}]->(entity)
        (chunk)-[:PART_OF]->(document)
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        graph: str = "grapsit",
    ):
        self.host = host
        self.port = port
        self.graph_name = graph
        self._db = None
        self._graph = None

    def _ensure_connection(self):
        if self._graph is None:
            from falkordb import FalkorDB
            self._db = FalkorDB(host=self.host, port=self.port)
            self._graph = self._db.select_graph(self.graph_name)

    def _run(self, query: str, parameters: dict = None) -> list:
        self._ensure_connection()
        params = parameters or {}
        result = self._graph.query(query, params)
        return result.result_set

    def close(self):
        # FalkorDB Python client uses Redis connection pooling;
        # no explicit close needed, but reset references.
        self._graph = None
        self._db = None

    # -- Setup ---------------------------------------------------------------

    def setup_indexes(self):
        """Create indexes (FalkorDB does not support IF NOT EXISTS, so wrap in try/except)."""
        index_queries = [
            "CREATE INDEX FOR (e:Entity) ON (e.id)",
            "CREATE INDEX FOR (e:Entity) ON (e.label)",
            "CREATE INDEX FOR (c:Chunk) ON (c.id)",
            "CREATE INDEX FOR (c:Chunk) ON (c.document_id)",
            "CREATE INDEX FOR (d:Document) ON (d.id)",
        ]
        for q in index_queries:
            try:
                self._run(q)
            except Exception as e:
                logger.debug(f"Index already exists or error: {e}")

    # -- Documents -----------------------------------------------------------

    def write_document(self, doc: Document):
        self._run(
            "MERGE (d:Document {id: $id}) SET d.source = $source, d.metadata = $metadata",
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
        """Create or merge an Entity node (no APOC — single Entity label only)."""
        self._run(
            """
            MERGE (e:Entity {id: $id})
            SET e.label = $label, e.entity_type = $entity_type, e.properties = $properties
            """,
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

    def get_entity_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        rows = self._run(
            "MATCH (e:Entity {id: $id}) RETURN e",
            {"id": entity_id},
        )
        if rows:
            return _node_to_dict(rows[0][0])
        return None

    def get_all_entities(self) -> List[Dict[str, Any]]:
        rows = self._run("MATCH (e:Entity) RETURN e")
        return [_node_to_dict(row[0]) for row in rows]

    def get_entity_by_label(self, label: str) -> Optional[Dict[str, Any]]:
        rows = self._run(
            "MATCH (e:Entity) WHERE toLower(e.label) = toLower($label) RETURN e",
            {"label": label},
        )
        if rows:
            return _node_to_dict(rows[0][0])
        return None

    def get_entity_neighbors(self, entity_id: str, max_hops: int = 1) -> List[Dict[str, Any]]:
        rows = self._run(
            f"""
            MATCH path = (e:Entity {{id: $id}})-[*1..{max_hops}]-(neighbor:Entity)
            RETURN DISTINCT neighbor
            """,
            {"id": entity_id},
        )
        return [_node_to_dict(row[0]) for row in rows]

    def get_entity_relations(self, entity_id: str) -> List[Dict[str, Any]]:
        # FalkorDB supports UNION
        rows = self._run(
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
        # FalkorDB result_set returns lists of values; convert to dicts
        columns = ["relation_type", "score", "target_id", "target_label", "chunk_id"]
        return [dict(zip(columns, row)) for row in rows]

    def get_chunks_for_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        rows = self._run(
            "MATCH (e:Entity {id: $id})-[:MENTIONED_IN]->(c:Chunk) RETURN c",
            {"id": entity_id},
        )
        return [_node_to_dict(row[0]) for row in rows]

    def get_subgraph(self, entity_ids: List[str], max_hops: int = 1) -> Dict[str, Any]:
        rows = self._run(
            f"""
            MATCH (e:Entity) WHERE e.id IN $ids
            OPTIONAL MATCH path = (e)-[*1..{max_hops}]-(neighbor:Entity)
            WITH collect(DISTINCT e) + collect(DISTINCT neighbor) AS all_nodes
            UNWIND all_nodes AS n
            WITH collect(DISTINCT n) AS nodes
            UNWIND nodes AS a
            UNWIND nodes AS b
            OPTIONAL MATCH (a)-[r]->(b) WHERE NOT type(r) = 'MENTIONED_IN'
            RETURN collect(DISTINCT a) AS entities,
                   collect(DISTINCT [a.id, b.id, type(r), r.score]) AS relations
            """,
            {"ids": entity_ids},
        )
        if not rows:
            return {"entities": [], "relations": []}

        row = rows[0]
        # Parse entities (Node objects)
        raw_entities = row[0] if row[0] else []
        entities = [_node_to_dict(n) for n in raw_entities if n is not None]

        # Parse relations (lists of [head_id, tail_id, type, score])
        raw_relations = row[1] if row[1] else []
        relations = []
        for rel_data in raw_relations:
            if rel_data is None or (isinstance(rel_data, list) and rel_data[2] is None):
                continue
            relations.append({
                "head": rel_data[0],
                "tail": rel_data[1],
                "type": rel_data[2],
                "score": rel_data[3],
            })

        return {"entities": entities, "relations": relations}

    def clear_all(self):
        self._run("MATCH (n) DETACH DELETE n")


def _node_to_dict(node) -> Dict[str, Any]:
    """Convert a FalkorDB Node object to a plain dict.

    FalkorDB Node objects expose properties via .properties dict.
    If the node is already a dict (e.g. in tests), return it as-is.
    """
    if isinstance(node, dict):
        return node
    if hasattr(node, "properties"):
        return dict(node.properties)
    return {}


def _sanitize_label(s: str) -> str:
    """Sanitize a string for use as a relationship type."""
    s = s.strip().replace(" ", "_").replace("-", "_")
    s = "".join(c for c in s if c.isalnum() or c == "_")
    if s and s[0].isdigit():
        s = "_" + s
    return s.upper() if s else "UNKNOWN"
