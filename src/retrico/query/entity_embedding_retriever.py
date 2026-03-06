"""Entity embedding retriever — find similar entities via vector search."""

from typing import Any, Dict, List
import logging

from ..core.registry import query_registry
from ..models.entity import EntityMention
from ..models.graph import Subgraph
from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class EntityEmbeddingRetrieverProcessor(BaseRetriever):
    """Retrieve subgraph by embedding parsed entity mentions and searching
    entity embeddings for similar entities.

    Flow: embed entity texts -> vector search entity_embeddings per entity ->
    collect matched entity IDs -> get_subgraph.

    Assumes entity embeddings are pre-populated in the vector store.

    Config keys:
        top_k: int — number of similar entities per mention (default: 5)
        max_hops: int — subgraph expansion depth (default: 2)
        vector_index_name: str — vector index for entity embeddings
            (default: "entity_embeddings")
        embedding_method, model_name, etc. — passed to create_embedding_model
        vector_store_type, etc. — passed to create_vector_store
        store_type, neo4j_uri, etc. — passed to create_store
    """

    default_inputs = {"entities": "parser_result.entities"}
    default_output = "entity_embedding_retriever_result"

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.top_k: int = config_dict.get("top_k", 5)
        self.max_hops: int = config_dict.get("max_hops", 2)
        self.vector_index_name: str = config_dict.get(
            "vector_index_name", "entity_embeddings"
        )

    def __call__(self, *, entities: List[EntityMention], **kwargs) -> Dict[str, Any]:
        """Find similar entities via embeddings and build a subgraph.

        Returns:
            {"subgraph": Subgraph}
        """
        if not entities:
            return {"subgraph": Subgraph()}

        self._ensure_store()
        self._ensure_embedding_model()
        self._ensure_vector_store()

        # Embed all entity mention texts
        texts = [m.text for m in entities]
        embeddings = self._embedding_model.encode(texts)

        # Search for similar entities for each mention
        all_entity_ids = []
        for emb in embeddings:
            results = self._vector_store.search_similar(
                self.vector_index_name, emb, top_k=self.top_k
            )
            for entity_id, _score in results:
                if entity_id not in all_entity_ids:
                    all_entity_ids.append(entity_id)

        if not all_entity_ids:
            return {"subgraph": Subgraph()}

        raw = self._store.get_subgraph(all_entity_ids, max_hops=self.max_hops)
        return {"subgraph": self._raw_to_subgraph(raw)}


@query_registry.register("entity_embedding_retriever")
def create_entity_embedding_retriever(config_dict: dict, pipeline=None):
    return EntityEmbeddingRetrieverProcessor(config_dict, pipeline)
