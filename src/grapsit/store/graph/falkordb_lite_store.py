"""FalkorDBLite graph store — embedded, zero-config graph database.

FalkorDBLite runs an embedded Redis+FalkorDB server inside the Python process,
so no external database setup is required. It shares the same Cypher query
interface as regular FalkorDB, so all query logic is inherited.

Install: ``pip install falkordblite``
"""

import logging
import os
from typing import Optional

from .falkordb_store import FalkorDBGraphStore

logger = logging.getLogger(__name__)


class FalkorDBLiteGraphStore(FalkorDBGraphStore):
    """Embedded FalkorDB graph store — no external server required.

    Uses FalkorDBLite (``falkordblite``) to run an embedded Redis+FalkorDB
    instance backed by a local file. All Cypher query methods are inherited
    from :class:`FalkorDBGraphStore`.

    Args:
        db_path: Path to the database file. Defaults to ``"grapsit.db"``
            in the current working directory.
        graph: Graph name within the database.
        query_timeout: Query timeout in milliseconds (0 = no limit).
    """

    def __init__(
        self,
        db_path: str = "grapsit.db",
        graph: str = "grapsit",
        query_timeout: int = 0,
    ):
        # Skip FalkorDBGraphStore.__init__ — we don't need host/port
        self.db_path = db_path
        self.graph_name = graph
        self.query_timeout = query_timeout
        self._db = None
        self._graph = None

    def _ensure_connection(self):
        if self._graph is None:
            from redislite.falkordb_client import FalkorDB
            self._db = FalkorDB(self.db_path)
            self._graph = self._db.select_graph(self.graph_name)

    def close(self):
        self._graph = None
        if self._db is not None:
            # FalkorDBLite may expose a shutdown method; clean up gracefully
            if hasattr(self._db, 'shutdown'):
                try:
                    self._db.shutdown()
                except Exception:
                    pass
            self._db = None
