"""Reasoner — LLM-based multi-hop reasoning over a subgraph."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import json
import re
import logging

from ..core.base import BaseProcessor
from ..core.registry import processor_registry
from ..models.entity import Entity
from ..models.relation import Relation
from ..models.document import Chunk
from ..models.graph import Subgraph, QueryResult

logger = logging.getLogger(__name__)


class BaseReasoner(ABC):
    """Abstract base for reasoning strategies."""

    @abstractmethod
    def reason(self, query: str, subgraph: Subgraph) -> QueryResult:
        ...


class LLMReasoner(BaseReasoner):
    """LLM-based multi-hop reasoning.

    Given a query and subgraph (entities, relations, chunks), the LLM:
    1. Identifies implicit/inferred relations
    2. Selects the most relevant chunks
    3. Produces a concise answer
    """

    def __init__(self, config_dict: Dict[str, Any]):
        self._client = None
        self._client_kwargs = {
            "api_key": config_dict.get("api_key"),
            "base_url": config_dict.get("base_url"),
            "model": config_dict.get("model", "gpt-4o-mini"),
            "temperature": config_dict.get("temperature", 0.1),
            "max_completion_tokens": config_dict.get("max_completion_tokens", 4096),
            "timeout": config_dict.get("timeout", 60.0),
        }

    def _ensure_client(self):
        if self._client is None:
            from ..llm.openai_client import OpenAIClient
            self._client = OpenAIClient(**self._client_kwargs)

    def _format_subgraph_context(self, subgraph: Subgraph) -> str:
        """Format subgraph into a structured text context for the LLM."""
        parts = []

        if subgraph.entities:
            parts.append("Entities:")
            for e in subgraph.entities:
                etype = f" ({e.entity_type})" if e.entity_type else ""
                parts.append(f"  - {e.label}{etype} [id: {e.id}]")

        if subgraph.relations:
            parts.append("\nRelations:")
            for r in subgraph.relations:
                parts.append(f"  - {r.head_text} --{r.relation_type}--> {r.tail_text} (score: {r.score})")

        if subgraph.chunks:
            parts.append("\nSource chunks:")
            for c in subgraph.chunks:
                parts.append(f"  [chunk {c.id}]: {c.text}")

        return "\n".join(parts) if parts else "(empty subgraph)"

    def reason(self, query: str, subgraph: Subgraph) -> QueryResult:
        self._ensure_client()

        context = self._format_subgraph_context(subgraph)

        system_msg = (
            "You are a knowledge graph reasoning system. "
            "Given a query and a subgraph of entities, relations, and source text chunks, "
            "you analyze the information to answer the query."
        )

        user_msg = (
            f"Query: {query}\n\n"
            f"Knowledge Graph Context:\n{context}\n\n"
            f"Based on the subgraph above, respond with a JSON object:\n"
            f'{{\n'
            f'  "inferred_relations": [{{"head": "entity1", "tail": "entity2", "relation": "relation_type"}}],\n'
            f'  "relevant_chunk_ids": ["chunk_id_1", ...],\n'
            f'  "answer": "Your concise answer to the query"\n'
            f'}}\n\n'
            f"- inferred_relations: new relations you can infer that are not explicitly stated\n"
            f"- relevant_chunk_ids: IDs of the most relevant source chunks for answering\n"
            f"- answer: a concise answer based on the available evidence\n\n"
            f"Return ONLY the JSON object."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            raw = self._client.complete(
                messages, response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.warning(f"LLM reasoning failed: {e}")
            return QueryResult(
                query=query,
                subgraph=subgraph,
                answer=None,
            )

        return self._parse_response(raw, query, subgraph)

    def _parse_response(
        self, raw: str, query: str, subgraph: Subgraph
    ) -> QueryResult:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
            text = text.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM reasoning response as JSON")
            return QueryResult(query=query, subgraph=subgraph, answer=None)

        # Build inferred relations
        inferred = []
        for rel in parsed.get("inferred_relations", []):
            head = rel.get("head", "")
            tail = rel.get("tail", "")
            rtype = rel.get("relation", "")
            if head and tail and rtype:
                inferred.append(Relation(
                    head_text=head,
                    tail_text=tail,
                    relation_type=rtype,
                    score=0.0,
                    properties={"inferred": True},
                ))

        # Filter chunks by relevant IDs
        relevant_ids = set(parsed.get("relevant_chunk_ids", []))
        if relevant_ids:
            filtered_chunks = [c for c in subgraph.chunks if c.id in relevant_ids]
        else:
            filtered_chunks = list(subgraph.chunks)

        enriched = Subgraph(
            entities=subgraph.entities,
            relations=list(subgraph.relations) + inferred,
            chunks=filtered_chunks,
        )

        answer = parsed.get("answer", None)

        return QueryResult(
            query=query,
            subgraph=enriched,
            answer=answer,
            metadata={"inferred_relation_count": len(inferred)},
        )


class ReasonerProcessor(BaseProcessor):
    """Reasoner processor for the DAG pipeline.

    Config keys:
        method: str — reasoning strategy (default: "llm")
        api_key, model, base_url, temperature, max_completion_tokens, timeout — LLM config
    """

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.method: str = config_dict.get("method", "llm")
        self._reasoner: Optional[BaseReasoner] = None

    def _ensure_reasoner(self):
        if self._reasoner is None:
            if self.method == "llm":
                self._reasoner = LLMReasoner(self.config_dict)
            else:
                raise ValueError(f"Unknown reasoner method: {self.method}")

    def __call__(self, *, query: str, subgraph: Any, **kwargs) -> Dict[str, Any]:
        """Run reasoning over the subgraph.

        Returns:
            {"result": QueryResult}
        """
        self._ensure_reasoner()

        if isinstance(subgraph, dict):
            subgraph = Subgraph(**subgraph)
        elif not isinstance(subgraph, Subgraph):
            subgraph = Subgraph()

        result = self._reasoner.reason(query, subgraph)
        return {"result": result}


@processor_registry.register("reasoner")
def create_reasoner(config_dict: dict, pipeline=None):
    return ReasonerProcessor(config_dict, pipeline)
