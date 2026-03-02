"""Keyword retriever — find relevant chunks via full-text search."""

from typing import Any, Dict
import logging

from ..core.base import BaseProcessor
from ..core.registry import query_registry
from ..models.document import Chunk
from ..models.entity import Entity
from ..models.relation import Relation
from ..models.graph import Subgraph

logger = logging.getLogger(__name__)


class KeywordRetrieverProcessor(BaseProcessor):
    """Retrieve chunks by full-text search from a relational store or graph DB.

    Two search sources controlled by ``search_source``:

    **Relational** (``search_source="relational"``, default):
        Uses a relational store (SQLite FTS5, PostgreSQL tsvector, Elasticsearch).
        Requires a relational store connection.

    **Graph** (``search_source="graph"``):
        Uses the graph database's native full-text index (Neo4j Lucene,
        FalkorDB FTS, Memgraph Tantivy).  The FTS index is created
        automatically during ``setup_indexes()`` — no extra setup needed.

    Two entity modes controlled by ``expand_entities``:

    **Chunks-only** (``expand_entities=False``, default for relational):
        query -> search -> Subgraph(chunks=[...])
        Only needs the search store — no graph store, parser, or embeddings.

    **Entity expansion** (``expand_entities=True``, default for graph):
        query -> search -> chunks
              -> graph_store.get_entities_for_chunk() -> entity_ids
              -> graph_store.get_subgraph() -> Subgraph(entities, relations, chunks)
        Additionally requires a graph store connection (for relational source).

    Config keys:
        search_source: str — ``"relational"`` (default) or ``"graph"``
        top_k: int — number of chunks to retrieve (default: 10)
        chunk_table: str — table name for chunk records (default: "chunks",
            relational only)
        expand_entities: bool — look up entities in matched chunks and build
            a full subgraph via the graph store (default: False for relational,
            True for graph)
        max_hops: int — subgraph expansion depth when expand_entities=True
            (default: 1)
        fulltext_index: str — name of the FTS index (default: "chunk_text_idx",
            graph only)
        relational_store_type, sqlite_path, etc. — passed to create_relational_store
        store_type, neo4j_uri, etc. — passed to create_store (graph)
    """

    default_inputs = {"query": "$input.query"}
    default_output = "keyword_retriever_result"

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.search_source: str = config_dict.get("search_source", "relational")
        self.top_k: int = config_dict.get("top_k", 10)
        self.chunk_table: str = config_dict.get("chunk_table", "chunks")
        self.max_hops: int = config_dict.get("max_hops", 1)
        self.fulltext_index: str = config_dict.get("fulltext_index", "chunk_text_idx")
        # Default expand_entities depends on search source
        if "expand_entities" in config_dict:
            self.expand_entities: bool = config_dict["expand_entities"]
        else:
            self.expand_entities = self.search_source == "graph"
        self._relational_store = None
        self._store = None

    def _ensure_relational_store(self):
        """Lazily create the relational store (shared from pool if available)."""
        if self._relational_store is None:
            from ..store.pool import resolve_from_pool_or_create
            self._relational_store = resolve_from_pool_or_create(
                self.config_dict, "relational"
            )

    def _ensure_store(self):
        """Lazily create the graph store (shared from pool if available)."""
        if self._store is None:
            from ..store.pool import resolve_from_pool_or_create
            self._store = resolve_from_pool_or_create(self.config_dict, "graph")

    @staticmethod
    def _raw_to_subgraph(raw: Dict[str, Any], chunks: list) -> Subgraph:
        """Convert a raw store subgraph dict to a Subgraph model."""
        sg_entities = []
        for ent_dict in raw.get("entities", []):
            if ent_dict is None:
                continue
            sg_entities.append(Entity(
                id=ent_dict.get("id", ""),
                label=ent_dict.get("label", ""),
                entity_type=ent_dict.get("entity_type", ""),
            ))

        id_to_label = {e.id: e.label for e in sg_entities}

        sg_relations = []
        for rel_dict in raw.get("relations", []):
            if rel_dict is None or rel_dict.get("type") is None:
                continue
            head_id = rel_dict.get("head", "")
            tail_id = rel_dict.get("tail", "")
            sg_relations.append(Relation(
                head_text=id_to_label.get(head_id, head_id),
                tail_text=id_to_label.get(tail_id, tail_id),
                relation_type=rel_dict.get("type", ""),
                score=rel_dict.get("score", 0.0) or 0.0,
            ))

        return Subgraph(entities=sg_entities, relations=sg_relations, chunks=chunks)

    def _search_relational(self, query: str):
        """Search via relational store full-text search."""
        self._ensure_relational_store()
        return self._relational_store.search(
            self.chunk_table, query, top_k=self.top_k
        )

    def _search_graph(self, query: str):
        """Search via graph DB native full-text index."""
        self._ensure_store()
        return self._store.fulltext_search_chunks(
            query, top_k=self.top_k, index_name=self.fulltext_index
        )

    def __call__(self, *, query: str, **kwargs) -> Dict[str, Any]:
        """Find relevant chunks via full-text search.

        Returns:
            {"subgraph": Subgraph} — with chunks always populated; entities
            and relations populated only when ``expand_entities=True``.
        """
        if self.search_source == "graph":
            results = self._search_graph(query)
        else:
            results = self._search_relational(query)

        if not results:
            return {"subgraph": Subgraph()}

        # Build Chunk objects from search results
        chunks = []
        for record in results:
            chunk_id = record.get("id")
            if not chunk_id:
                continue
            chunks.append(Chunk(
                id=chunk_id,
                document_id=record.get("document_id", ""),
                text=record.get("text", ""),
                index=record.get("index", 0),
                start_char=record.get("start_char", 0),
                end_char=record.get("end_char", 0),
            ))

        if not self.expand_entities:
            return {"subgraph": Subgraph(chunks=chunks)}

        # Expand: look up entities for each chunk via graph store
        self._ensure_store()

        all_entity_ids = []
        for chunk in chunks:
            entities = self._store.get_entities_for_chunk(chunk.id)
            for ent in entities:
                eid = ent.get("id")
                if eid and eid not in all_entity_ids:
                    all_entity_ids.append(eid)

        if not all_entity_ids:
            return {"subgraph": Subgraph(chunks=chunks)}

        raw = self._store.get_subgraph(all_entity_ids, max_hops=self.max_hops)
        return {"subgraph": self._raw_to_subgraph(raw, chunks)}


@query_registry.register("keyword_retriever")
def create_keyword_retriever(config_dict: dict, pipeline=None):
    return KeywordRetrieverProcessor(config_dict, pipeline)
