"""Community retriever — find relevant communities via vector search."""

from typing import Any, Dict
import logging

from ..core.registry import query_registry
from ..models.graph import Subgraph
from .base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


class CommunityRetrieverProcessor(BaseRetriever):
    """Retrieve subgraph by embedding the query and searching community embeddings.

    Flow: embed query -> vector search community_embeddings -> get community
    members -> get_subgraph around members.

    Config keys:
        top_k: int — number of communities to retrieve (default: 3)
        max_hops: int — subgraph expansion depth around community members (default: 1)
        vector_index_name: str — vector index name for community embeddings
            (default: "community_embeddings")
        embedding_method, model_name, etc. — passed to create_embedding_model
        vector_store_type, etc. — passed to create_vector_store
        store_type, neo4j_uri, etc. — passed to create_store
    """

    default_inputs = {"query": "$input.query"}
    default_output = "community_retriever_result"

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.top_k: int = config_dict.get("top_k", 3)
        self.max_hops: int = config_dict.get("max_hops", 1)
        self.vector_index_name: str = config_dict.get(
            "vector_index_name", "community_embeddings"
        )

    def __call__(self, *, query: str, **kwargs) -> Dict[str, Any]:
        """Find relevant communities and build a subgraph from their members.

        Returns:
            {"subgraph": Subgraph}
        """
        self._ensure_store()
        self._ensure_embedding_model()
        self._ensure_vector_store()

        # Embed the query
        query_embedding = self._embedding_model.encode([query])[0]

        # Search for similar community embeddings
        results = self._vector_store.search_similar(
            self.vector_index_name, query_embedding, top_k=self.top_k
        )

        if not results:
            return {"subgraph": Subgraph()}

        # Get all entity members from matched communities
        all_entity_ids = []
        for community_id, _score in results:
            members = self._store.get_community_members(community_id)
            for member in members:
                eid = member.get("id")
                if eid and eid not in all_entity_ids:
                    all_entity_ids.append(eid)

        if not all_entity_ids:
            return {"subgraph": Subgraph()}

        raw = self._store.get_subgraph(all_entity_ids, max_hops=self.max_hops)
        return {"subgraph": self._raw_to_subgraph(raw)}


@query_registry.register("community_retriever")
def create_community_retriever(config_dict: dict, pipeline=None):
    return CommunityRetrieverProcessor(config_dict, pipeline)
