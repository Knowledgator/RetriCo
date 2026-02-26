"""Graph-DB-backed vector store using native vector indexes.

Supports Neo4j, FalkorDB, and Memgraph.  The backend is selected via the
``store_type`` key in *store_config* (defaults to ``"neo4j"``).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .vector_base import BaseVectorStore

logger = logging.getLogger(__name__)

_SUPPORTED_STORE_TYPES = {"neo4j", "falkordb", "memgraph"}

# Map well-known index names to (node_label, property_name)
_INDEX_MAP: Dict[str, Tuple[str, str]] = {
    "entity_embeddings": ("Entity", "embedding"),
    "chunk_embeddings": ("Chunk", "embedding"),
    "community_embeddings": ("Community", "embedding"),
}

# Map index names to graph store update methods
_UPDATE_METHOD_MAP: Dict[str, str] = {
    "entity_embeddings": "update_entity_embedding",
    "chunk_embeddings": "update_chunk_embedding",
    "community_embeddings": "update_community_embedding",
}

# Default initial capacity for Memgraph vector indexes
_MEMGRAPH_DEFAULT_CAPACITY = 10000


class GraphDBVectorStore(BaseVectorStore):
    """Vector store backed by a graph database's native vector index.

    Supports Neo4j (``db.index.vector.queryNodes``), FalkorDB
    (``db.idx.vector.queryNodes``), and Memgraph
    (``vector_search.search``).  The backend is selected via the
    ``store_type`` key in *store_config* (defaults to ``"neo4j"``).

    Index names are mapped to node labels automatically:

    - ``"entity_embeddings"``  → ``(:Entity {embedding})``
    - ``"chunk_embeddings"``   → ``(:Chunk {embedding})``
    - ``"community_embeddings"`` → ``(:Community {embedding})``

    Custom index names can be registered via *extra_index_map*.

    Args:
        store_config: Dict forwarded to ``create_store()`` (must contain
            connection details such as ``neo4j_uri`` or ``falkordb_host``).
        extra_index_map: Optional mapping of additional index names to
            ``(node_label, property_name)`` tuples.
    """

    def __init__(
        self,
        store_config: Dict[str, Any],
        extra_index_map: Optional[Dict[str, Tuple[str, str]]] = None,
    ):
        self._store_config = dict(store_config)
        self._store_type = self._store_config.get("store_type", "neo4j")
        if self._store_type not in _SUPPORTED_STORE_TYPES:
            raise ValueError(
                f"GraphDBVectorStore does not support store_type={self._store_type!r}. "
                f"Supported: {sorted(_SUPPORTED_STORE_TYPES)}. "
                f"Use vector_store_type='in_memory', 'faiss', or 'qdrant' instead."
            )
        self._graph_store = None
        self._index_map: Dict[str, Tuple[str, str]] = dict(_INDEX_MAP)
        if extra_index_map:
            self._index_map.update(extra_index_map)
        # Track created indexes so create_index is idempotent
        self._created_indexes: Dict[str, int] = {}

    # -- lazy init -------------------------------------------------------------

    def _ensure_store(self):
        if self._graph_store is None:
            from . import create_store

            self._graph_store = create_store(self._store_config)

    def _resolve_index(self, name: str) -> Tuple[str, str]:
        """Return ``(node_label, property_name)`` for an index name."""
        if name in self._index_map:
            return self._index_map[name]
        raise KeyError(
            f"Unknown vector index name {name!r}. "
            f"Known indexes: {list(self._index_map)}. "
            f"Register custom names via extra_index_map."
        )

    # -- BaseVectorStore interface --------------------------------------------

    def create_index(self, name: str, dimension: int):
        if name in self._created_indexes:
            if self._created_indexes[name] != dimension:
                raise ValueError(
                    f"Index {name!r} already created with dimension "
                    f"{self._created_indexes[name]}, cannot recreate with {dimension}."
                )
            return

        self._ensure_store()
        node_label, prop = self._resolve_index(name)

        try:
            if self._store_type == "falkordb":
                cypher = (
                    f"CREATE VECTOR INDEX FOR (n:{node_label}) ON (n.{prop}) "
                    f"OPTIONS {{dimension: {dimension}, similarityFunction: 'cosine'}}"
                )
            elif self._store_type == "memgraph":
                capacity = self._store_config.get(
                    "vector_capacity", _MEMGRAPH_DEFAULT_CAPACITY
                )
                cypher = (
                    f'CREATE VECTOR INDEX {name} ON :{node_label}({prop}) '
                    f'WITH CONFIG {{"dimension": {dimension}, '
                    f'"capacity": {capacity}, "metric": "cos"}}'
                )
            else:
                # Neo4j syntax
                cypher = (
                    f"CREATE VECTOR INDEX {name} IF NOT EXISTS "
                    f"FOR (n:{node_label}) ON (n.{prop}) "
                    f"OPTIONS {{indexConfig: {{"
                    f"`vector.dimensions`: {dimension}, "
                    f"`vector.similarity_function`: 'cosine'"
                    f"}}}}"
                )
            self._graph_store.run_cypher(cypher)
            logger.info(f"Created vector index {name!r} on :{node_label}.{prop} (dim={dimension})")
        except Exception as e:
            # Index may already exist — log and continue
            logger.debug(f"Vector index creation note for {name!r}: {e}")

        self._created_indexes[name] = dimension

    def store_embeddings(self, index_name: str, items: List[Tuple[str, List[float]]]):
        if not items:
            return

        self._ensure_store()
        node_label, prop = self._resolve_index(index_name)

        # Use dedicated update methods when available
        update_method_name = _UPDATE_METHOD_MAP.get(index_name)
        if update_method_name and hasattr(self._graph_store, update_method_name):
            update_fn = getattr(self._graph_store, update_method_name)
            for item_id, embedding in items:
                emb = list(embedding) if not isinstance(embedding, list) else embedding
                try:
                    update_fn(item_id, emb)
                except Exception as e:
                    logger.debug(f"Could not store embedding for {item_id}: {e}")
        else:
            # Fallback: raw Cypher SET
            use_vecf32 = self._store_type == "falkordb"
            for item_id, embedding in items:
                emb = list(embedding) if not isinstance(embedding, list) else embedding
                try:
                    if use_vecf32:
                        self._graph_store.run_cypher(
                            f"MATCH (n:{node_label} {{id: $id}}) SET n.{prop} = vecf32($embedding)",
                            {"id": item_id, "embedding": emb},
                        )
                    else:
                        self._graph_store.run_cypher(
                            f"MATCH (n:{node_label} {{id: $id}}) SET n.{prop} = $embedding",
                            {"id": item_id, "embedding": emb},
                        )
                except Exception as e:
                    logger.debug(f"Could not store embedding for {item_id}: {e}")

    def search_similar(
        self, index_name: str, query_vector: List[float], top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        self._ensure_store()
        node_label, prop = self._resolve_index(index_name)
        vec = list(query_vector) if not isinstance(query_vector, list) else query_vector

        if self._store_type == "falkordb":
            cypher = (
                f"CALL db.idx.vector.queryNodes('{node_label}', '{prop}', "
                f"$top_k, vecf32($vec)) YIELD node, score "
                f"RETURN node.id AS id, score"
            )
            params = {"top_k": top_k, "vec": vec}
        elif self._store_type == "memgraph":
            cypher = (
                f'CALL vector_search.search("{index_name}", $top_k, $vec) '
                f"YIELD node, similarity "
                f"RETURN node.id AS id, similarity AS score"
            )
            params = {"top_k": top_k, "vec": vec}
        else:
            # Neo4j
            cypher = (
                f"CALL db.index.vector.queryNodes('{index_name}', $top_k, $vec) "
                f"YIELD node, score "
                f"RETURN node.id AS id, score"
            )
            params = {"top_k": top_k, "vec": vec}

        try:
            records = self._graph_store.run_cypher(cypher, params)
        except Exception as e:
            logger.warning(f"Vector search failed for {index_name!r}: {e}")
            return []

        results = []
        for rec in records:
            if isinstance(rec, dict):
                results.append((rec["id"], float(rec["score"])))
            elif isinstance(rec, (list, tuple)):
                # FalkorDB / Memgraph may return list of lists
                results.append((rec[0], float(rec[1])))
        return results

    def get_embedding(self, index_name: str, item_id: str) -> Optional[List[float]]:
        self._ensure_store()
        node_label, prop = self._resolve_index(index_name)

        try:
            records = self._graph_store.run_cypher(
                f"MATCH (n:{node_label} {{id: $id}}) RETURN n.{prop} AS embedding",
                {"id": item_id},
            )
        except Exception as e:
            logger.debug(f"get_embedding failed for {item_id}: {e}")
            return None

        if not records:
            return None

        rec = records[0]
        emb = rec["embedding"] if isinstance(rec, dict) else rec[0]
        if emb is None:
            return None
        return list(emb)

    def delete_index(self, name: str):
        self._ensure_store()

        try:
            if self._store_type == "falkordb":
                node_label, prop = self._resolve_index(name)
                self._graph_store.run_cypher(
                    f"DROP INDEX ON :{node_label}({prop})"
                )
            elif self._store_type == "memgraph":
                self._graph_store.run_cypher(f"DROP VECTOR INDEX {name}")
            else:
                self._graph_store.run_cypher(f"DROP INDEX {name} IF EXISTS")
        except Exception as e:
            logger.debug(f"delete_index note for {name!r}: {e}")

        self._created_indexes.pop(name, None)
        logger.info(f"Deleted vector index {name!r}")

    def close(self):
        if self._graph_store is not None:
            try:
                self._graph_store.close()
            except Exception:
                pass
            self._graph_store = None
        self._created_indexes.clear()
