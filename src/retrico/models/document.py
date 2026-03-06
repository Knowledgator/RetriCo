"""Document and Chunk models."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
import uuid


class Document(BaseModel):
    """Source document."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""
    text: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Text chunk from a document."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    text: str = ""
    index: int = 0
    start_char: int = 0
    end_char: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
