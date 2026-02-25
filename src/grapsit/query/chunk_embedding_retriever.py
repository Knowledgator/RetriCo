"""Chunk embedding retriever — find relevant chunks via vector search."""

from typing import Any, Dict
import logging

from ..core.registry import processor_registry
from ..models.graph import Subgraph
from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class ChunkEmbeddingRetrieverProcessor(BaseRetriever):
    """Retrieve subgraph by embedding the query and searching chunk embeddings.

    Flow: embed query -> vector search chunk_embeddings -> get entities for
    each chunk -> get_subgraph around those entities.

    Assumes chunk embeddings are pre-populated in the vector store.

    Config keys:
        top_k: int — number of chunks to retrieve (default: 5)
        max_hops: int — subgraph expansion depth (default: 1)
        vector_index_name: str — vector index for chunk embeddings
            (default: "chunk_embeddings")
        embedding_method, model_name, etc. — passed to create_embedding_model
        vector_store_type, etc. — passed to create_vector_store
        store_type, neo4j_uri, etc. — passed to create_store
    """

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.top_k: int = config_dict.get("top_k", 5)
        self.max_hops: int = config_dict.get("max_hops", 1)
        self.vector_index_name: str = config_dict.get(
            "vector_index_name", "chunk_embeddings"
        )

    def __call__(self, *, query: str, **kwargs) -> Dict[str, Any]:
        """Find relevant chunks and build a subgraph from their entities.

        Returns:
            {"subgraph": Subgraph}
        """
        self._ensure_store()
        self._ensure_embedding_model()
        self._ensure_vector_store()

        # Embed the query
        query_embedding = self._embedding_model.encode([query])[0]

        # Search for similar chunk embeddings
        results = self._vector_store.search_similar(
            self.vector_index_name, query_embedding, top_k=self.top_k
        )

        if not results:
            return {"subgraph": Subgraph()}

        # Get entities mentioned in matched chunks
        all_entity_ids = []
        for chunk_id, _score in results:
            entities = self._store.get_entities_for_chunk(chunk_id)
            for ent in entities:
                eid = ent.get("id")
                if eid and eid not in all_entity_ids:
                    all_entity_ids.append(eid)

        if not all_entity_ids:
            return {"subgraph": Subgraph()}

        raw = self._store.get_subgraph(all_entity_ids, max_hops=self.max_hops)
        return {"subgraph": self._raw_to_subgraph(raw)}


@processor_registry.register("chunk_embedding_retriever")
def create_chunk_embedding_retriever(config_dict: dict, pipeline=None):
    return ChunkEmbeddingRetrieverProcessor(config_dict, pipeline)
