"""Graph-level models."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .entity import Entity
from .relation import Relation
from .document import Chunk


class KGTriple(BaseModel):
    """A single knowledge graph triple."""
    head: str
    relation: str
    tail: str
    score: float = 0.0
    chunk_id: str = ""


class Subgraph(BaseModel):
    """A retrieved subgraph."""
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)
    chunks: List[Chunk] = Field(default_factory=list)


class QueryResult(BaseModel):
    """Result of a knowledge graph query."""
    query: str = ""
    subgraph: Subgraph = Field(default_factory=Subgraph)
    answer: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
