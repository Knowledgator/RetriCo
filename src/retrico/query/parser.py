"""Query parser — NER on query text (GLiNER or LLM)."""

from typing import Any, Dict, List, Optional
import logging

from ..core.base import BaseProcessor
from ..core.registry import query_registry
from ..extraction.gliner_engine import GLiNEREngine
from ..extraction.llm_engine import LLMExtractionEngine
from ..llm.prompts import (
    QUERY_PARSER_SYSTEM_PROMPT,
    QUERY_PARSER_USER_PROMPT_TEMPLATE,
    QUERY_PARSER_TOOL_SYSTEM_PROMPT,
)
from ..models.entity import EntityMention

logger = logging.getLogger(__name__)


class QueryParserProcessor(BaseProcessor):
    """Extract entities from a query string using GLiNER or LLM.

    Config keys:
        method: str — "gliner" (default) or "llm"
        labels: List[str] — entity type labels to detect

    GLiNER-specific:
        model: str — GLiNER pretrained model name
        threshold: float — confidence threshold (default: 0.3)
        device: str — "cpu" or "cuda" (default: "cpu")

    LLM-specific:
        api_key: str — LLM API key
        base_url: str — API base URL
        model: str — model name (default: "gpt-4o-mini")
        temperature: float — sampling temperature (default: 0.1)
        system_prompt: str — override default query parser system prompt.
        user_prompt_template: str — override default query parser user prompt.
            Available placeholders: {labels_instruction}, {label_constraint}, {query}.

    Tool-calling specific:
        tool_system_prompt: str — override default tool-calling system prompt.
            Available placeholder: {schema_info} (appended if present).
    """

    default_inputs = {"query": "$input.query"}
    default_output = "parser_result"

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.method: str = config_dict.get("method", "gliner")
        self.labels: List[str] = config_dict.get("labels", [])
        self._gliner_engine: Optional[GLiNEREngine] = None
        self._llm_engine: Optional[LLMExtractionEngine] = None
        self._llm_client = None  # For tool-calling mode (uses raw client)
        self._tool_system_prompt = config_dict.get("tool_system_prompt", QUERY_PARSER_TOOL_SYSTEM_PROMPT)

    def _get_gliner_engine(self) -> GLiNEREngine:
        if self._gliner_engine is None:
            self._gliner_engine = GLiNEREngine(
                model=self.config_dict.get("model", "urchade/gliner_multi-v2.1"),
                labels=self.labels,
                threshold=self.config_dict.get("threshold", 0.3),
                device=self.config_dict.get("device", "cpu"),
                flat_ner=self.config_dict.get("flat_ner", True),
            )
        return self._gliner_engine

    def _get_llm_engine(self) -> LLMExtractionEngine:
        if self._llm_engine is None:
            engine_kwargs = {
                "api_key": self.config_dict.get("api_key"),
                "base_url": self.config_dict.get("base_url"),
                "model": self.config_dict.get("model", "gpt-4o-mini"),
                "labels": self.labels,
                "temperature": self.config_dict.get("temperature", 0.1),
                "max_completion_tokens": self.config_dict.get("max_completion_tokens", 2048),
                "timeout": self.config_dict.get("timeout", 30.0),
            }
            if "system_prompt" in self.config_dict:
                engine_kwargs["query_system_prompt"] = self.config_dict["system_prompt"]
            if "user_prompt_template" in self.config_dict:
                engine_kwargs["query_user_prompt_template"] = self.config_dict["user_prompt_template"]
            self._llm_engine = LLMExtractionEngine(**engine_kwargs)
        return self._llm_engine

    def _ensure_llm_client(self):
        """Ensure raw LLM client for tool-calling mode."""
        if self._llm_client is None:
            from ..llm.openai_client import OpenAIClient
            self._llm_client = OpenAIClient(
                api_key=self.config_dict.get("api_key"),
                base_url=self.config_dict.get("base_url"),
                model=self.config_dict.get("model", "gpt-4o-mini"),
                temperature=self.config_dict.get("temperature", 0.1),
                max_completion_tokens=self.config_dict.get("max_completion_tokens", 2048),
                timeout=self.config_dict.get("timeout", 30.0),
            )

    def _parse_gliner(self, query: str) -> List[EntityMention]:
        engine = self._get_gliner_engine()
        result = engine.extract_single(query)
        return result.entities[0] if result.entities else []

    def _parse_llm(self, query: str) -> List[EntityMention]:
        engine = self._get_llm_engine()
        return engine.extract_from_query(query)

    def _parse_tool(self, query: str) -> Dict[str, Any]:
        """Parse query using LLM tool calling with search_triples.

        Returns:
            {"query": str, "entities": List[EntityMention],
             "triple_queries": [{"head": str|None, "relation": str|None, "tail": str|None}]}
        """
        self._ensure_llm_client()
        from ..llm.tools import TRIPLE_QUERY_TOOLS

        labels = self.labels
        relation_labels = self.config_dict.get("relation_labels", [])

        schema_lines = []
        if labels:
            schema_lines.append(f"Entity types: {', '.join(labels)}")
        if relation_labels:
            schema_lines.append(f"Relation types: {', '.join(relation_labels)}")
        schema_info = "\n".join(schema_lines) if schema_lines else ""

        system_prompt = self._tool_system_prompt
        if schema_info:
            system_prompt += f"\nKnowledge Graph Schema:\n{schema_info}\n"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        try:
            result = self._llm_client.complete_with_tools(messages, tools=TRIPLE_QUERY_TOOLS)
        except Exception as e:
            logger.warning(f"Tool-calling query parse failed: {e}")
            return {"query": query, "entities": [], "triple_queries": []}

        triple_queries = []
        entity_texts = set()

        for tc in result.get("tool_calls", []):
            if tc["name"] != "search_triples":
                continue
            args = tc["arguments"] if isinstance(tc["arguments"], dict) else {}
            head = args.get("head")
            relation = args.get("relation")
            tail = args.get("tail")
            triple_queries.append({
                "head": head,
                "relation": relation,
                "tail": tail,
            })
            if head:
                entity_texts.add(head)
            if tail:
                entity_texts.add(tail)

        # Build EntityMention list from head/tail values for backward compat
        mentions = []
        for text in entity_texts:
            idx = query.find(text)
            start = idx if idx >= 0 else 0
            end = start + len(text) if idx >= 0 else 0
            mentions.append(EntityMention(
                text=text,
                label="",
                start=start,
                end=end,
                score=1.0,
            ))

        return {"query": query, "entities": mentions, "triple_queries": triple_queries}

    def __call__(self, *, query: str, **kwargs) -> Dict[str, Any]:
        """Parse a query string to extract entities.

        Returns:
            {"query": str, "entities": List[EntityMention]}
            For method="tool", also includes "triple_queries".
        """
        if self.method == "tool":
            return self._parse_tool(query)
        elif self.method == "llm":
            entities = self._parse_llm(query)
        else:
            entities = self._parse_gliner(query)
        return {"query": query, "entities": entities}


@query_registry.register("query_parser")
def create_query_parser(config_dict: dict, pipeline=None):
    return QueryParserProcessor(config_dict, pipeline)
