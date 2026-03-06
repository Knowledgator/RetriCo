"""Relation model."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator
import uuid


class Relation(BaseModel):
    """A relation (edge) between two entities."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    head_text: str
    tail_text: str
    relation_type: str
    score: float = 0.0
    chunk_id: List[str] = Field(default_factory=list)
    head_label: str = ""
    tail_label: str = ""
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _migrate_chunk_id(cls, values):
        """Migrate old str chunk_id to List[str] for backward compat."""
        if isinstance(values, dict):
            cid = values.get("chunk_id")
            if isinstance(cid, str):
                values["chunk_id"] = [cid] if cid else []
        return values
