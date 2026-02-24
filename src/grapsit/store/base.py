"""Abstract base class for graph stores."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..models.document import Chunk, Document
from ..models.entity import Entity, EntityMention
from ..models.relation import Relation


class BaseGraphStore(ABC):
    """Abstract interface for knowledge graph persistence.

    Schema::

        (:Entity {id, label, entity_type, properties})
        (:Chunk {id, document_id, text, index, start_char, end_char})
        (:Document {id, source, metadata})

        (entity)-[:MENTIONED_IN {start, end, score, text}]->(chunk)
        (entity)-[:REL_TYPE {score, chunk_id, id}]->(entity)
        (chunk)-[:PART_OF]->(document)
    """

    # -- Setup ---------------------------------------------------------------

    @abstractmethod
    def setup_indexes(self):
        """Create indexes and constraints."""

    @abstractmethod
    def close(self):
        """Close the connection."""

    # -- Write ---------------------------------------------------------------

    @abstractmethod
    def write_document(self, doc: Document):
        """Create or merge a Document node."""

    @abstractmethod
    def write_chunk(self, chunk: Chunk):
        """Create or merge a Chunk node."""

    @abstractmethod
    def write_chunk_document_link(self, chunk_id: str, document_id: str):
        """Create a PART_OF relationship between a Chunk and a Document."""

    @abstractmethod
    def write_entity(self, entity: Entity):
        """Create or merge an Entity node."""

    @abstractmethod
    def write_mention_link(self, entity_id: str, chunk_id: str, mention: EntityMention):
        """Create a MENTIONED_IN relationship between an Entity and a Chunk."""

    @abstractmethod
    def write_relation(self, relation: Relation, head_entity_id: str, tail_entity_id: str):
        """Create a typed relationship between two Entity nodes."""

    # -- Read ----------------------------------------------------------------

    @abstractmethod
    def get_entity_by_id(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Look up an entity by its ID."""

    @abstractmethod
    def get_all_entities(self) -> List[Dict[str, Any]]:
        """Return all Entity nodes."""

    @abstractmethod
    def get_entity_by_label(self, label: str) -> Optional[Dict[str, Any]]:
        """Look up an entity by its label (case-insensitive)."""

    @abstractmethod
    def get_entity_neighbors(self, entity_id: str, max_hops: int = 1) -> List[Dict[str, Any]]:
        """Get neighboring entities within max_hops."""

    @abstractmethod
    def get_entity_relations(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all relations involving an entity."""

    @abstractmethod
    def get_chunks_for_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get chunks where an entity is mentioned."""

    @abstractmethod
    def get_subgraph(self, entity_ids: List[str], max_hops: int = 1) -> Dict[str, Any]:
        """Retrieve a subgraph around a set of entity IDs."""

    @abstractmethod
    def clear_all(self):
        """Delete all nodes and relationships."""

    # -- Chunk lookups -------------------------------------------------------

    def get_entities_for_chunk(self, chunk_id: str) -> List[Dict[str, Any]]:
        """Get entities mentioned in a chunk (reverse MENTIONED_IN)."""
        raise NotImplementedError

    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Look up a chunk by its ID."""
        raise NotImplementedError

    # -- Path queries --------------------------------------------------------

    def get_shortest_paths(
        self, source_id: str, target_id: str, max_length: int = 5,
    ) -> List[Dict[str, Any]]:
        """Find shortest paths between two entities (entity relations only)."""
        raise NotImplementedError

    # -- Community CRUD ------------------------------------------------------

    def write_community(
        self, community_id: str, level: int, title: str, summary: str,
    ):
        """Create or merge a Community node."""
        raise NotImplementedError

    def write_community_membership(
        self, entity_id: str, community_id: str, level: int,
    ):
        """Create a MEMBER_OF relationship between an Entity and a Community."""
        raise NotImplementedError

    def get_community_members(self, community_id: str) -> List[Dict[str, Any]]:
        """Get all entities that are members of a community."""
        raise NotImplementedError

    def get_all_communities(self) -> List[Dict[str, Any]]:
        """Return all Community nodes."""
        raise NotImplementedError

    def detect_communities(self, method: str = "louvain", **params) -> Dict[str, str]:
        """Run community detection and return entity_id → community_id mapping."""
        raise NotImplementedError

    def write_community_hierarchy(self, child_id: str, parent_id: str):
        """Create CHILD_OF relationship between communities."""
        raise NotImplementedError

    def get_top_entities_by_degree(
        self, entity_ids: List[str] = None, top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get top-k entities by degree centrality (relationship count).

        Args:
            entity_ids: If provided, only consider these entities.
            top_k: Number of top entities to return.

        Returns:
            List of dicts with entity properties plus ``degree`` count.
        """
        raise NotImplementedError

    def update_community_embedding(self, community_id: str, embedding: List[float]):
        """Store embedding vector on a Community node."""
        raise NotImplementedError

    def get_inter_community_edges(
        self, community_memberships: Dict[str, str],
    ) -> List[Any]:
        """Get weighted edges between communities based on member relationships.

        Args:
            community_memberships: Mapping of entity_id → community_id.

        Returns:
            List of (community_a, community_b, weight) tuples.
        """
        raise NotImplementedError
