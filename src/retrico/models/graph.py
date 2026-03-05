"""Graph-level models."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator

from .entity import Entity
from .relation import Relation
from .document import Chunk


class KGTriple(BaseModel):
    """A single knowledge graph triple."""
    head: str
    relation: str
    tail: str
    score: float = 0.0
    chunk_id: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _migrate_chunk_id(cls, values):
        """Migrate old str chunk_id to List[str] for backward compat."""
        if isinstance(values, dict):
            cid = values.get("chunk_id")
            if isinstance(cid, str):
                values["chunk_id"] = [cid] if cid else []
        return values


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
