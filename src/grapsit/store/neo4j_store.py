"""Neo4j graph store for knowledge graph persistence."""

from typing import Any, Dict, List, Optional
import logging

from neo4j import GraphDatabase

from .base import BaseGraphStore
from ..models.document import Chunk, Document
from ..models.entity import Entity, EntityMention
from ..models.relation import Relation

logger = logging.getLogger(__name__)


class Neo4jGraphStore(BaseGraphStore):
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

    # -- Raw Cypher ----------------------------------------------------------

    def run_cypher(self, query: str, params: dict = None) -> list:
        """Execute a raw Cypher query and return results."""
        return self._run(query, params)

    # -- Setup ---------------------------------------------------------------

    def setup_indexes(self):
        """Create indexes and constraints."""
        queries = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT community_id IF NOT EXISTS FOR (co:Community) REQUIRE co.id IS UNIQUE",
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

    def get_entity_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Look up an entity by its ID."""
        records = self._run(
            "MATCH (e:Entity {id: $id}) RETURN e",
            {"id": entity_id},
        )
        if records:
            return records[0]["e"]
        return None

    def get_all_entities(self) -> List[Dict[str, Any]]:
        """Return all Entity nodes (for knowledge base loading)."""
        records = self._run("MATCH (e:Entity) RETURN e")
        return [r["e"] for r in records]

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

    # -- Chunk lookups -------------------------------------------------------

    def get_entities_for_chunk(self, chunk_id: str) -> list:
        records = self._run(
            "MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk {id: $id}) RETURN e",
            {"id": chunk_id},
        )
        return [r["e"] for r in records]

    def get_chunk_by_id(self, chunk_id: str):
        records = self._run(
            "MATCH (c:Chunk {id: $id}) RETURN c",
            {"id": chunk_id},
        )
        if records:
            return records[0]["c"]
        return None

    # -- Path queries --------------------------------------------------------

    def get_shortest_paths(self, source_id: str, target_id: str, max_length: int = 5) -> list:
        records = self._run(
            f"""
            MATCH (s:Entity {{id: $source}}), (t:Entity {{id: $target}}),
                  path = shortestPath((s)-[*1..{max_length}]-(t))
            WHERE ALL(r IN relationships(path) WHERE NOT type(r) = 'MENTIONED_IN')
            RETURN [n IN nodes(path) | properties(n)] AS nodes,
                   [r IN relationships(path) | {{type: type(r), score: r.score}}] AS rels
            """,
            {"source": source_id, "target": target_id},
        )
        return records

    # -- Community CRUD ------------------------------------------------------

    def write_community(self, community_id: str, level: int, title: str, summary: str):
        self._run(
            """
            MERGE (co:Community {id: $id})
            SET co.level = $level, co.title = $title, co.summary = $summary
            """,
            {"id": community_id, "level": level, "title": title, "summary": summary},
        )

    def write_community_membership(self, entity_id: str, community_id: str, level: int):
        self._run(
            """
            MATCH (e:Entity {id: $entity_id})
            MATCH (co:Community {id: $community_id})
            MERGE (e)-[r:MEMBER_OF]->(co)
            SET r.level = $level
            """,
            {"entity_id": entity_id, "community_id": community_id, "level": level},
        )

    def get_community_members(self, community_id: str) -> list:
        records = self._run(
            "MATCH (e:Entity)-[:MEMBER_OF]->(co:Community {id: $id}) RETURN e",
            {"id": community_id},
        )
        return [r["e"] for r in records]

    def get_all_communities(self) -> list:
        records = self._run("MATCH (co:Community) RETURN co")
        return [r["co"] for r in records]

    def detect_communities(self, method: str = "louvain", **params) -> dict:
        graph_name = params.get("graph_name", "community_graph")
        # Project entity-to-entity relationships (exclude MENTIONED_IN, MEMBER_OF, PART_OF)
        try:
            self._run(
                f"""
                CALL gds.graph.project(
                    '{graph_name}',
                    'Entity',
                    {{
                        ALL: {{
                            type: '*',
                            orientation: 'UNDIRECTED'
                        }}
                    }}
                )
                """
            )
        except Exception as e:
            # Graph may already exist; drop and retry
            logger.debug(f"Graph projection error (retrying): {e}")
            try:
                self._run(f"CALL gds.graph.drop('{graph_name}')")
            except Exception:
                pass
            self._run(
                f"""
                CALL gds.graph.project(
                    '{graph_name}',
                    'Entity',
                    {{
                        ALL: {{
                            type: '*',
                            orientation: 'UNDIRECTED'
                        }}
                    }}
                )
                """
            )

        algo = "gds.leiden.stream" if method == "leiden" else "gds.louvain.stream"
        records = self._run(f"CALL {algo}('{graph_name}') YIELD nodeId, communityId "
                            "RETURN gds.util.asNode(nodeId).id AS entity_id, "
                            "toString(communityId) AS community_id")

        # Clean up
        try:
            self._run(f"CALL gds.graph.drop('{graph_name}')")
        except Exception:
            pass

        return {r["entity_id"]: r["community_id"] for r in records}

    def write_community_hierarchy(self, child_id: str, parent_id: str):
        self._run(
            """
            MATCH (child:Community {id: $child_id})
            MATCH (parent:Community {id: $parent_id})
            MERGE (child)-[:CHILD_OF]->(parent)
            """,
            {"child_id": child_id, "parent_id": parent_id},
        )

    def get_top_entities_by_degree(self, entity_ids=None, top_k=10):
        if entity_ids:
            records = self._run(
                """
                MATCH (e:Entity) WHERE e.id IN $ids
                OPTIONAL MATCH (e)-[r]-()
                WHERE NOT type(r) IN ['MENTIONED_IN', 'MEMBER_OF', 'PART_OF']
                RETURN e, count(r) AS degree
                ORDER BY degree DESC
                LIMIT $top_k
                """,
                {"ids": entity_ids, "top_k": top_k},
            )
        else:
            records = self._run(
                """
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[r]-()
                WHERE NOT type(r) IN ['MENTIONED_IN', 'MEMBER_OF', 'PART_OF']
                RETURN e, count(r) AS degree
                ORDER BY degree DESC
                LIMIT $top_k
                """,
                {"top_k": top_k},
            )
        results = []
        for rec in records:
            entity_dict = dict(rec["e"]) if not isinstance(rec["e"], dict) else rec["e"]
            entity_dict["degree"] = rec["degree"]
            results.append(entity_dict)
        return results

    def update_community_embedding(self, community_id: str, embedding):
        self._run(
            "MATCH (co:Community {id: $id}) SET co.embedding = $embedding",
            {"id": community_id, "embedding": embedding},
        )

    def update_entity_embedding(self, entity_id: str, embedding):
        self._run(
            "MATCH (e:Entity {id: $id}) SET e.embedding = $embedding",
            {"id": entity_id, "embedding": embedding},
        )

    def update_chunk_embedding(self, chunk_id: str, embedding):
        self._run(
            "MATCH (c:Chunk {id: $id}) SET c.embedding = $embedding",
            {"id": chunk_id, "embedding": embedding},
        )

    def get_all_triples(self) -> list:
        """Return all (head_label, relation_type, tail_label) triples."""
        records = self._run(
            """
            MATCH (h:Entity)-[r]->(t:Entity)
            WHERE NOT type(r) IN ['MENTIONED_IN', 'MEMBER_OF', 'PART_OF', 'CHILD_OF']
            RETURN h.label AS head, type(r) AS rel, t.label AS tail
            """
        )
        return [(r["head"], r["rel"], r["tail"]) for r in records]

    def get_inter_community_edges(self, community_memberships):
        # Get all entity-entity relationships and aggregate by community pair
        records = self._run(
            """
            MATCH (h:Entity)-[r]->(t:Entity)
            WHERE NOT type(r) IN ['MENTIONED_IN', 'MEMBER_OF', 'PART_OF', 'CHILD_OF']
            RETURN h.id AS head_id, t.id AS tail_id
            """
        )
        from collections import Counter
        edge_counts = Counter()
        for rec in records:
            head_comm = community_memberships.get(rec["head_id"])
            tail_comm = community_memberships.get(rec["tail_id"])
            if head_comm and tail_comm and head_comm != tail_comm:
                pair = tuple(sorted([head_comm, tail_comm]))
                edge_counts[pair] += 1
        return [(a, b, w) for (a, b), w in edge_counts.items()]

    # -- Danger zone ---------------------------------------------------------

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
