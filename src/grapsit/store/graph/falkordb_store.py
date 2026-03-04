"""FalkorDB graph store for knowledge graph persistence."""

from typing import Any, Dict, List, Optional
import ast
import logging
import uuid

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
            "CALL db.idx.fulltext.createNodeIndex('Chunk', 'text')",
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

    def fulltext_search_chunks(self, query: str, top_k: int = 10, index_name: str = "chunk_text_idx"):
        rows = self._run(
            "CALL db.idx.fulltext.queryNodes('Chunk', $query) YIELD node "
            "RETURN node LIMIT $top_k",
            {"query": query, "top_k": top_k},
        )
        return [_node_to_dict(row[0]) for row in rows]

    # -- Path queries --------------------------------------------------------

    def _parse_path_rows(self, rows: list) -> list:
        """Convert raw Cypher path result rows into path dicts."""
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

    def get_shortest_paths(self, source_id: str, target_id: str, max_length: int = 5) -> list:
        # FalkorDB doesn't support shortestPath() or complex path functions
        # like relationships(path) reliably. Use explicit node chain matching.

        # Try directed paths in both directions.
        for src, tgt in [(source_id, target_id), (target_id, source_id)]:
            rows = self._run(
                f"""
                MATCH (s:Entity {{id: $source}})-[*1..{max_length}]->(t:Entity {{id: $target}})
                RETURN s, t
                LIMIT 1
                """,
                {"source": src, "target": tgt},
            )
            if rows:
                # Found a directed connection — now get the actual path nodes
                # by matching intermediate Entity nodes step by step.
                return self._get_entity_chain(src, tgt, max_length)

        # Fallback: find a shared intermediate Entity connected to both.
        rows = self._run(
            """
            MATCH (s:Entity {id: $source})-[r1]->(mid:Entity)<-[r2]-(t:Entity {id: $target})
            WHERE NOT type(r1) = 'MENTIONED_IN' AND NOT type(r2) = 'MENTIONED_IN'
            RETURN s, mid, t, type(r1) AS t1, r1.score AS s1, type(r2) AS t2, r2.score AS s2
            LIMIT 3
            """,
            {"source": source_id, "target": target_id},
        )
        if rows:
            results = []
            for row in rows:
                nodes = [_node_to_dict(row[0]), _node_to_dict(row[1]), _node_to_dict(row[2])]
                rels = [
                    {"type": row[3], "score": row[4]},
                    {"type": row[5], "score": row[6]},
                ]
                results.append({"nodes": nodes, "rels": rels})
            return results

        # Try the other direction for intermediate: s<-mid->t
        rows = self._run(
            """
            MATCH (s:Entity {id: $source})<-[r1]-(mid:Entity)-[r2]->(t:Entity {id: $target})
            WHERE NOT type(r1) = 'MENTIONED_IN' AND NOT type(r2) = 'MENTIONED_IN'
            RETURN s, mid, t, type(r1) AS t1, r1.score AS s1, type(r2) AS t2, r2.score AS s2
            LIMIT 3
            """,
            {"source": source_id, "target": target_id},
        )
        if rows:
            results = []
            for row in rows:
                nodes = [_node_to_dict(row[0]), _node_to_dict(row[1]), _node_to_dict(row[2])]
                rels = [
                    {"type": row[3], "score": row[4]},
                    {"type": row[5], "score": row[6]},
                ]
                results.append({"nodes": nodes, "rels": rels})
            return results

        return []

    def _get_entity_chain(self, source_id: str, target_id: str, max_length: int) -> list:
        """Get a directed entity chain from source to target, step by step."""
        # Try lengths 1, 2, 3... up to max_length to find the shortest
        for length in range(1, min(max_length, 4) + 1):
            if length == 1:
                rows = self._run(
                    """
                    MATCH (s:Entity {id: $source})-[r]->(t:Entity {id: $target})
                    WHERE NOT type(r) = 'MENTIONED_IN'
                    RETURN s, t, type(r) AS rtype, r.score AS rscore
                    LIMIT 1
                    """,
                    {"source": source_id, "target": target_id},
                )
                if rows:
                    row = rows[0]
                    return [{"nodes": [_node_to_dict(row[0]), _node_to_dict(row[1])],
                             "rels": [{"type": row[2], "score": row[3]}]}]
            elif length == 2:
                rows = self._run(
                    """
                    MATCH (s:Entity {id: $source})-[r1]->(m:Entity)-[r2]->(t:Entity {id: $target})
                    WHERE NOT type(r1) = 'MENTIONED_IN' AND NOT type(r2) = 'MENTIONED_IN'
                    RETURN s, m, t, type(r1) AS t1, r1.score AS s1, type(r2) AS t2, r2.score AS s2
                    LIMIT 1
                    """,
                    {"source": source_id, "target": target_id},
                )
                if rows:
                    row = rows[0]
                    return [{"nodes": [_node_to_dict(row[0]), _node_to_dict(row[1]), _node_to_dict(row[2])],
                             "rels": [{"type": row[3], "score": row[4]}, {"type": row[5], "score": row[6]}]}]
            elif length == 3:
                rows = self._run(
                    """
                    MATCH (s:Entity {id: $source})-[r1]->(m1:Entity)-[r2]->(m2:Entity)-[r3]->(t:Entity {id: $target})
                    WHERE NOT type(r1) = 'MENTIONED_IN' AND NOT type(r2) = 'MENTIONED_IN'
                      AND NOT type(r3) = 'MENTIONED_IN'
                    RETURN s, m1, m2, t,
                           type(r1) AS t1, r1.score AS s1,
                           type(r2) AS t2, r2.score AS s2,
                           type(r3) AS t3, r3.score AS s3
                    LIMIT 1
                    """,
                    {"source": source_id, "target": target_id},
                )
                if rows:
                    row = rows[0]
                    return [{"nodes": [_node_to_dict(row[0]), _node_to_dict(row[1]),
                                       _node_to_dict(row[2]), _node_to_dict(row[3])],
                             "rels": [{"type": row[4], "score": row[5]},
                                      {"type": row[6], "score": row[7]},
                                      {"type": row[8], "score": row[9]}]}]
            else:
                # length 4: use variable-length but only return endpoint + count
                rows = self._run(
                    f"""
                    MATCH (s:Entity {{id: $source}})-[*{length}]->(t:Entity {{id: $target}})
                    RETURN s, t
                    LIMIT 1
                    """,
                    {"source": source_id, "target": target_id},
                )
                if rows:
                    row = rows[0]
                    return [{"nodes": [_node_to_dict(row[0]), _node_to_dict(row[1])], "rels": []}]
        return []

    def get_top_shortest_paths(
        self, entity_ids: list, max_length: int = 5, top_k: int = 3,
    ) -> list:
        """Find top-k shortest paths among a set of entities (directed, both directions)."""
        if len(entity_ids) < 2:
            return []
        import itertools
        all_paths: list = []
        for src, tgt in itertools.combinations(entity_ids, 2):
            paths = self.get_shortest_paths(src, tgt, max_length)
            for p in paths:
                length = len(p.get("rels", []))
                all_paths.append((length, p))
        all_paths.sort(key=lambda x: x[0])
        return [p for _, p in all_paths[:top_k]]

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

    # -- Mutations -----------------------------------------------------------

    def delete_entity(self, entity_id: str) -> bool:
        if self.get_entity_by_id(entity_id) is None:
            return False
        self._run("MATCH (e:Entity {id: $id}) DETACH DELETE e", {"id": entity_id})
        return True

    def delete_relation(self, relation_id: str) -> bool:
        rows = self._run(
            """
            MATCH ()-[r]->()
            WHERE r.id = $id
            AND NOT type(r) IN ['MENTIONED_IN', 'MEMBER_OF', 'PART_OF', 'CHILD_OF']
            DELETE r
            RETURN count(r) AS cnt
            """,
            {"id": relation_id},
        )
        if rows and len(rows) > 0:
            cnt = rows[0][0] if isinstance(rows[0], (list, tuple)) else rows[0]
            return cnt > 0
        return False

    def delete_chunk(self, chunk_id: str) -> bool:
        if self.get_chunk_by_id(chunk_id) is None:
            return False
        self._run("MATCH (c:Chunk {id: $id}) DETACH DELETE c", {"id": chunk_id})
        return True

    def update_entity(
        self,
        entity_id: str,
        *,
        label: Optional[str] = None,
        entity_type: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        existing = self.get_entity_by_id(entity_id)
        if existing is None:
            return False

        set_parts = []
        params: Dict[str, Any] = {"id": entity_id}

        if label is not None:
            set_parts.append("e.label = $label")
            params["label"] = label
        if entity_type is not None:
            set_parts.append("e.entity_type = $entity_type")
            params["entity_type"] = entity_type
        if properties is not None:
            existing_props: Dict[str, Any] = {}
            raw = existing.get("properties", "{}")
            if isinstance(raw, str):
                try:
                    existing_props = ast.literal_eval(raw)
                except (ValueError, SyntaxError):
                    existing_props = {}
            elif isinstance(raw, dict):
                existing_props = raw
            existing_props.update(properties)
            set_parts.append("e.properties = $properties")
            params["properties"] = str(existing_props)

        if not set_parts:
            return True

        query = f"MATCH (e:Entity {{id: $id}}) SET {', '.join(set_parts)}"
        self._run(query, params)
        return True

    def add_entity(
        self,
        label: str,
        entity_type: str = "",
        *,
        properties: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
    ) -> str:
        entity_id = id or str(uuid.uuid4())
        self._run(
            """
            CREATE (e:Entity {id: $id, label: $label,
                              entity_type: $entity_type,
                              properties: $properties})
            """,
            {
                "id": entity_id,
                "label": label,
                "entity_type": entity_type,
                "properties": str(properties or {}),
            },
        )
        return entity_id

    def add_relation(
        self,
        head_id: str,
        tail_id: str,
        relation_type: str,
        *,
        properties: Optional[Dict[str, Any]] = None,
        id: Optional[str] = None,
    ) -> str:
        if self.get_entity_by_id(head_id) is None:
            raise ValueError(f"Head entity '{head_id}' does not exist")
        if self.get_entity_by_id(tail_id) is None:
            raise ValueError(f"Tail entity '{tail_id}' does not exist")

        rel_id = id or str(uuid.uuid4())
        rel_type = _sanitize_label(relation_type)
        props = properties or {}
        self._run(
            f"""
            MATCH (h:Entity {{id: $head_id}})
            MATCH (t:Entity {{id: $tail_id}})
            CREATE (h)-[r:`{rel_type}` {{id: $id, properties: $properties}}]->(t)
            """,
            {
                "head_id": head_id,
                "tail_id": tail_id,
                "id": rel_id,
                "properties": str(props),
            },
        )
        return rel_id

    def merge_entities(self, source_id: str, target_id: str) -> bool:
        if source_id == target_id:
            return True

        source = self.get_entity_by_id(source_id)
        if source is None:
            return False
        target = self.get_entity_by_id(target_id)
        if target is None:
            return False

        # 1. Move MENTIONED_IN edges
        self._run(
            """
            MATCH (s:Entity {id: $src})-[r:MENTIONED_IN]->(c:Chunk)
            MATCH (t:Entity {id: $tgt})
            MERGE (t)-[r2:MENTIONED_IN]->(c)
            SET r2.start = r.start, r2.end = r.end, r2.score = r.score, r2.text = r.text
            DELETE r
            """,
            {"src": source_id, "tgt": target_id},
        )

        # 2. Move MEMBER_OF edges
        self._run(
            """
            MATCH (s:Entity {id: $src})-[r:MEMBER_OF]->(co:Community)
            MATCH (t:Entity {id: $tgt})
            MERGE (t)-[r2:MEMBER_OF]->(co)
            SET r2.level = r.level
            DELETE r
            """,
            {"src": source_id, "tgt": target_id},
        )

        # 3. Move outgoing entity-entity relations
        out_rows = self._run(
            """
            MATCH (s:Entity {id: $src})-[r]->(other:Entity)
            WHERE NOT type(r) IN ['MENTIONED_IN', 'MEMBER_OF', 'PART_OF', 'CHILD_OF']
            RETURN type(r) AS rel_type, r.id AS id, r.score AS score,
                   r.chunk_id AS chunk_id, other.id AS other_id
            """,
            {"src": source_id},
        )
        columns = ["rel_type", "id", "score", "chunk_id", "other_id"]
        for row in out_rows:
            rec = dict(zip(columns, row)) if isinstance(row, (list, tuple)) else row
            rt = rec["rel_type"]
            other = rec["other_id"]
            if other == source_id:
                other = target_id
            self._run(
                f"""
                MATCH (h:Entity {{id: $head}}), (t:Entity {{id: $tail}})
                CREATE (h)-[r:`{rt}` {{id: $id, score: $score, chunk_id: $chunk_id}}]->(t)
                """,
                {
                    "head": target_id,
                    "tail": other,
                    "id": rec.get("id") or str(uuid.uuid4()),
                    "score": rec.get("score"),
                    "chunk_id": rec.get("chunk_id"),
                },
            )

        # 4. Move incoming entity-entity relations
        in_rows = self._run(
            """
            MATCH (other:Entity)-[r]->(s:Entity {id: $src})
            WHERE NOT type(r) IN ['MENTIONED_IN', 'MEMBER_OF', 'PART_OF', 'CHILD_OF']
            RETURN type(r) AS rel_type, r.id AS id, r.score AS score,
                   r.chunk_id AS chunk_id, other.id AS other_id
            """,
            {"src": source_id},
        )
        for row in in_rows:
            rec = dict(zip(columns, row)) if isinstance(row, (list, tuple)) else row
            rt = rec["rel_type"]
            other = rec["other_id"]
            if other == source_id:
                other = target_id
            self._run(
                f"""
                MATCH (h:Entity {{id: $head}}), (t:Entity {{id: $tail}})
                CREATE (h)-[r:`{rt}` {{id: $id, score: $score, chunk_id: $chunk_id}}]->(t)
                """,
                {
                    "head": other,
                    "tail": target_id,
                    "id": rec.get("id") or str(uuid.uuid4()),
                    "score": rec.get("score"),
                    "chunk_id": rec.get("chunk_id"),
                },
            )

        # 5. Merge properties (target wins)
        source_props: Dict[str, Any] = {}
        raw = source.get("properties", "{}")
        if isinstance(raw, str):
            try:
                source_props = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                pass
        elif isinstance(raw, dict):
            source_props = raw

        target_props: Dict[str, Any] = {}
        raw = target.get("properties", "{}")
        if isinstance(raw, str):
            try:
                target_props = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                pass
        elif isinstance(raw, dict):
            target_props = raw

        merged = {**source_props, **target_props}
        self._run(
            "MATCH (e:Entity {id: $id}) SET e.properties = $properties",
            {"id": target_id, "properties": str(merged)},
        )

        # 6. Delete source
        self._run("MATCH (e:Entity {id: $id}) DETACH DELETE e", {"id": source_id})
        return True

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
