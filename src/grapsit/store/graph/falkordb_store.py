"""FalkorDB graph store for knowledge graph persistence."""

from typing import Any, Dict, List, Optional
import logging

from .base import BaseGraphStore
from ...models.document import Chunk, Document
from ...models.entity import Entity, EntityMention
from ...models.relation import Relation

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
        query_timeout: int = 0,
    ):
        self.host = host
        self.port = port
        self.graph_name = graph
        self.query_timeout = query_timeout  # milliseconds, 0 = server default
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
        kwargs = {}
        if self.query_timeout > 0:
            kwargs["timeout"] = self.query_timeout
        result = self._graph.query(query, params, **kwargs)
        return result.result_set

    def close(self):
        # FalkorDB Python client uses Redis connection pooling;
        # no explicit close needed, but reset references.
        self._graph = None
        self._db = None

    # -- Raw Cypher ----------------------------------------------------------

    def run_cypher(self, query: str, params: dict = None) -> list:
        """Execute a raw Cypher query and return results."""
        return self._run(query, params)

    # -- Setup ---------------------------------------------------------------

    def setup_indexes(self):
        """Create indexes (FalkorDB does not support IF NOT EXISTS, so wrap in try/except)."""
        index_queries = [
            "CREATE INDEX FOR (e:Entity) ON (e.id)",
            "CREATE INDEX FOR (e:Entity) ON (e.label)",
            "CREATE INDEX FOR (c:Chunk) ON (c.id)",
            "CREATE INDEX FOR (c:Chunk) ON (c.document_id)",
            "CREATE INDEX FOR (d:Document) ON (d.id)",
            "CREATE INDEX FOR (co:Community) ON (co.id)",
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
        # Exact match (case-insensitive)
        rows = self._run(
            "MATCH (e:Entity) WHERE toLower(e.label) = toLower($label) RETURN e",
            {"label": label},
        )
        if rows:
            return _node_to_dict(rows[0][0])
        # Fallback: index-friendly STARTS WITH on raw label (no toLower to avoid full scan)
        rows = self._run(
            "MATCH (e:Entity) WHERE e.label STARTS WITH $label RETURN e LIMIT 1",
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

    # -- Chunk lookups -------------------------------------------------------

    def get_entities_for_chunk(self, chunk_id: str) -> list:
        rows = self._run(
            "MATCH (e:Entity)-[:MENTIONED_IN]->(c:Chunk {id: $id}) RETURN e",
            {"id": chunk_id},
        )
        return [_node_to_dict(row[0]) for row in rows]

    def get_chunk_by_id(self, chunk_id: str):
        rows = self._run(
            "MATCH (c:Chunk {id: $id}) RETURN c",
            {"id": chunk_id},
        )
        if rows:
            return _node_to_dict(rows[0][0])
        return None

    # -- Path queries --------------------------------------------------------

    def get_shortest_paths(self, source_id: str, target_id: str, max_length: int = 5) -> list:
        # FalkorDB doesn't support shortestPath or undirected traversals.
        # Try directed path in both directions, then fall back to a
        # meet-in-the-middle approach via get_subgraph.
        for src, tgt in [(source_id, target_id), (target_id, source_id)]:
            rows = self._run(
                f"""
                MATCH path = (s:Entity {{id: $source}})-[*1..{max_length}]->(t:Entity {{id: $target}})
                WHERE ALL(r IN relationships(path) WHERE NOT type(r) = 'MENTIONED_IN')
                RETURN nodes(path) AS nodes, relationships(path) AS rels
                ORDER BY length(path)
                LIMIT 1
                """,
                {"source": src, "target": tgt},
            )
            if rows:
                results = []
                for row in rows:
                    nodes = [_node_to_dict(n) for n in (row[0] or [])]
                    rels = []
                    for edge in (row[1] or []):
                        if hasattr(edge, "relation"):
                            rel_dict = {"type": edge.relation}
                            if hasattr(edge, "properties"):
                                rel_dict["score"] = edge.properties.get("score")
                            rels.append(rel_dict)
                        elif isinstance(edge, dict):
                            rels.append(edge)
                    results.append({"nodes": nodes, "rels": rels})
                return results

        # Fallback: get overlapping neighborhoods (meet-in-the-middle).
        # Query outgoing neighbours of each entity and find shared nodes.
        hops = min(max_length // 2 + 1, max_length)
        raw = self.get_subgraph([source_id, target_id], max_hops=hops)
        if not raw or (not raw.get("entities") and not raw.get("nodes")):
            return []
        # Convert subgraph to a single "path" result
        nodes_list = raw.get("entities") or raw.get("nodes") or []
        rels_list = raw.get("relations") or raw.get("rels") or []
        path_nodes = [n if isinstance(n, dict) else _node_to_dict(n) for n in nodes_list]
        path_rels = []
        for r in rels_list:
            if isinstance(r, dict):
                path_rels.append(r)
            elif hasattr(r, "relation"):
                rel_dict = {"type": r.relation}
                if hasattr(r, "properties"):
                    rel_dict["score"] = r.properties.get("score")
                path_rels.append(rel_dict)
        return [{"nodes": path_nodes, "rels": path_rels}]

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
        rows = self._run(
            "MATCH (e:Entity)-[:MEMBER_OF]->(co:Community {id: $id}) RETURN e",
            {"id": community_id},
        )
        return [_node_to_dict(row[0]) for row in rows]

    def get_all_communities(self) -> list:
        rows = self._run("MATCH (co:Community) RETURN co")
        return [_node_to_dict(row[0]) for row in rows]

    def detect_communities(self, method: str = "louvain", **params) -> dict:
        """Run community detection using FalkorDB's CDLP algorithm.

        FalkorDB provides ``algo.labelPropagation()`` (Community Detection
        via Label Propagation).  Louvain/Leiden are not natively supported;
        a warning is logged and CDLP is used instead.
        """
        if method in ("louvain", "leiden"):
            logger.warning(
                "FalkorDB does not support %s; using CDLP (label propagation) instead.",
                method,
            )

        max_iterations = params.get("max_iterations", 10)
        rows = self._run(
            """
            CALL algo.labelPropagation({nodeLabels: ['Entity'], maxIterations: $max_iter})
            YIELD node, communityId
            RETURN node.id AS entity_id, toString(communityId) AS community_id
            """,
            {"max_iter": max_iterations},
        )
        columns = ["entity_id", "community_id"]
        return {
            row[0]: row[1]
            for row in rows
        }

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
            rows = self._run(
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
            rows = self._run(
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
        for row in rows:
            entity_dict = _node_to_dict(row[0])
            entity_dict["degree"] = row[1]
            results.append(entity_dict)
        return results

    def update_community_embedding(self, community_id: str, embedding):
        self._run(
            "MATCH (co:Community {id: $id}) SET co.embedding = vecf32($embedding)",
            {"id": community_id, "embedding": list(embedding)},
        )

    def update_entity_embedding(self, entity_id: str, embedding):
        self._run(
            "MATCH (e:Entity {id: $id}) SET e.embedding = vecf32($embedding)",
            {"id": entity_id, "embedding": list(embedding)},
        )

    def update_chunk_embedding(self, chunk_id: str, embedding):
        self._run(
            "MATCH (c:Chunk {id: $id}) SET c.embedding = vecf32($embedding)",
            {"id": chunk_id, "embedding": list(embedding)},
        )

    def get_all_triples(self) -> list:
        """Return all (head_label, relation_type, tail_label) triples."""
        rows = self._run(
            """
            MATCH (h:Entity)-[r]->(t:Entity)
            WHERE NOT type(r) IN ['MENTIONED_IN', 'MEMBER_OF', 'PART_OF', 'CHILD_OF']
            RETURN h.label AS head, type(r) AS rel, t.label AS tail
            """
        )
        return [(row[0], row[1], row[2]) for row in rows]

    def get_inter_community_edges(self, community_memberships):
        rows = self._run(
            """
            MATCH (h:Entity)-[r]->(t:Entity)
            WHERE NOT type(r) IN ['MENTIONED_IN', 'MEMBER_OF', 'PART_OF', 'CHILD_OF']
            RETURN h.id AS head_id, t.id AS tail_id
            """
        )
        from collections import Counter
        edge_counts = Counter()
        columns = ["head_id", "tail_id"]
        for row in rows:
            rec = dict(zip(columns, row))
            head_comm = community_memberships.get(rec["head_id"])
            tail_comm = community_memberships.get(rec["tail_id"])
            if head_comm and tail_comm and head_comm != tail_comm:
                pair = tuple(sorted([head_comm, tail_comm]))
                edge_counts[pair] += 1
        return [(a, b, w) for (a, b), w in edge_counts.items()]

    # -- Danger zone ---------------------------------------------------------

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
