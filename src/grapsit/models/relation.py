"""Relation model."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
import uuid


class Relation(BaseModel):
    """A relation (edge) between two entities."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    head_text: str
    tail_text: str
    relation_type: str
    score: float = 0.0
    chunk_id: str = ""
    head_label: str = ""
    tail_label: str = ""
    properties: Dict[str, Any] = Field(default_factory=dict)
