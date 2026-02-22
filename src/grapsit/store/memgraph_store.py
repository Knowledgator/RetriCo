"""Memgraph graph store — thin subclass of Neo4jGraphStore.

Memgraph uses the same Neo4j Python driver (Bolt protocol) and OpenCypher,
so only the defaults, index syntax, and entity writing (no APOC) differ.
"""

import logging

from .neo4j_store import Neo4jGraphStore, _sanitize_label
from ..models.entity import Entity

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
