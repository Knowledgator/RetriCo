"""Entity models."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import uuid


class EntityMention(BaseModel):
    """A mention of an entity in text (from NER)."""
    text: str
    label: str
    start: int = 0
    end: int = 0
    score: float = 0.0
    chunk_id: str = ""
    linked_entity_id: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)


class Entity(BaseModel):
    """A deduplicated entity node in the knowledge graph."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    label: str
    entity_type: str = ""
    properties: Dict[str, Any] = Field(default_factory=dict)
    mentions: List[EntityMention] = Field(default_factory=list)

    @property
    def canonical_name(self) -> str:
        """Normalized entity name for deduplication."""
        return self.label.strip().lower()
