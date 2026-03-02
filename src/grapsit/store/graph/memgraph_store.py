"""Memgraph graph store — thin subclass of Neo4jGraphStore.

Memgraph uses the same Neo4j Python driver (Bolt protocol) and OpenCypher,
so only the defaults, index syntax, and entity writing (no APOC) differ.
"""

import logging
import uuid

from .neo4j_store import Neo4jGraphStore, _sanitize_label
from ...models.entity import Entity

logger = logging.getLogger(__name__)


class MemgraphGraphStore(Neo4jGraphStore):
    """Knowledge graph store backed by Memgraph.

    Inherits all CRUD operations from :class:`Neo4jGraphStore`.
    Overrides only what differs in Memgraph's OpenCypher dialect.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "",
        password: str = "",
        database: str = "memgraph",
    ):
        super().__init__(uri=uri, user=user, password=password, database=database)

    # -- Setup ---------------------------------------------------------------

    def setup_indexes(self):
        """Create indexes using Memgraph's ``CREATE INDEX ON`` syntax."""
        index_queries = [
            "CREATE INDEX ON :Entity(id)",
            "CREATE INDEX ON :Entity(label)",
            "CREATE INDEX ON :Chunk(id)",
            "CREATE INDEX ON :Chunk(document_id)",
            "CREATE INDEX ON :Document(id)",
            "CREATE INDEX ON :Community(id)",
            "CREATE TEXT INDEX chunk_text_idx ON :Chunk",
        ]
        for q in index_queries:
            try:
                self._run(q)
            except Exception as e:
                logger.debug(f"Index already exists or error: {e}")

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

    # -- Community detection (MAGE) ------------------------------------------

    def fulltext_search_chunks(self, query: str, top_k: int = 10, index_name: str = "chunk_text_idx"):
        # Memgraph text search uses text_search.search with the index name
        escaped = query.replace("\\", "\\\\").replace("'", "\\'")
        records = self._run(
            f"""
            CALL text_search.search('{index_name}', '{escaped}', 'ALL')
            YIELD node
            RETURN node LIMIT $top_k
            """,
            {"top_k": top_k},
        )
        return [r["node"] for r in records]

    def detect_communities(self, method: str = "louvain", **params) -> dict:
        """Run community detection using Memgraph MAGE.

        Uses ``community_detection.get()`` which implements Louvain.
        Leiden is not supported by MAGE; a warning is logged if requested.
        """
        if method == "leiden":
            logger.warning(
                "Memgraph MAGE does not support Leiden; falling back to Louvain."
            )

        records = self._run(
            """
            CALL community_detection.get()
            YIELD node, community_id
            WHERE 'Entity' IN labels(node)
            RETURN node.id AS entity_id, toString(community_id) AS community_id
            """
        )
        return {r["entity_id"]: r["community_id"] for r in records}
