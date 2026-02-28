"""Query parser — NER on query text (GLiNER or LLM)."""

from typing import Any, Dict, List, Optional
import logging

from ..core.base import BaseProcessor
from ..core.registry import query_registry
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
        api_key: str — OpenAI API key
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
        self._model = None  # GLiNER model
        self._client = None  # OpenAI client
        self._system_prompt = config_dict.get("system_prompt", QUERY_PARSER_SYSTEM_PROMPT)
        self._user_prompt_template = config_dict.get("user_prompt_template", QUERY_PARSER_USER_PROMPT_TEMPLATE)
        self._tool_system_prompt = config_dict.get("tool_system_prompt", QUERY_PARSER_TOOL_SYSTEM_PROMPT)

    def _load_gliner(self):
        if self._model is None:
            from gliner import GLiNER
            model_name = self.config_dict.get("model", "urchade/gliner_multi-v2.1")
            logger.info(f"Loading GLiNER model for query parsing: {model_name}")
            self._model = GLiNER.from_pretrained(model_name)
            device = self.config_dict.get("device", "cpu")
            if device != "cpu":
                self._model = self._model.to(device)

    def _ensure_llm_client(self):
        if self._client is None:
            from ..llm.openai_client import OpenAIClient
            self._client = OpenAIClient(
                api_key=self.config_dict.get("api_key"),
                base_url=self.config_dict.get("base_url"),
                model=self.config_dict.get("model", "gpt-4o-mini"),
                temperature=self.config_dict.get("temperature", 0.1),
                max_completion_tokens=self.config_dict.get("max_completion_tokens", 2048),
                timeout=self.config_dict.get("timeout", 30.0),
            )

    def _parse_gliner(self, query: str) -> List[EntityMention]:
        self._load_gliner()
        threshold = self.config_dict.get("threshold", 0.3)
        flat_ner = self.config_dict.get("flat_ner", True)
        # Use inference() for consistent output across GLiNER and GLiNER-relex models
        entities_batch = self._model.inference(
            texts=[query], labels=self.labels,
            threshold=threshold, flat_ner=flat_ner,
        )
        if isinstance(entities_batch, tuple):
            entities_batch = entities_batch[0]

        raw = entities_batch[0] if entities_batch else []
        
        mentions = []
        for ent in raw:
            entity_label = ent.get("label", "")
            generated_label = ent.get("generated_label", None)
            if generated_label is not None:
                entity_label = generated_label[0]
            mentions.append(EntityMention(
                text=ent["text"],
                label=entity_label,
                start=ent.get("start", 0),
                end=ent.get("end", 0),
                score=ent.get("score", 0.0),
            ))
        return mentions

    def _parse_llm(self, query: str) -> List[EntityMention]:
        import json
        import re

        self._ensure_llm_client()

        if self.labels:
            labels_instruction = f"Entity labels: {', '.join(self.labels)}"
            label_constraint = " (one of the labels above)"
        else:
            labels_instruction = "Identify all named entities and assign an appropriate type."
            label_constraint = ""

        prompt = self._user_prompt_template.format(
            labels_instruction=labels_instruction,
            label_constraint=label_constraint,
            query=query,
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = self._client.complete(
                messages, response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.warning(f"LLM query parsing failed: {e}")
            return []

        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*\n?", "", text)
            text = re.sub(r"\n?```\s*$", "", text)
            text = text.strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM query parser response as JSON")
            return []

        if isinstance(parsed, dict) and "entities" in parsed:
            parsed = parsed["entities"]
        if not isinstance(parsed, list):
            return []

        mentions = []
        for ent in parsed:
            ent_text = ent.get("text", "")
            if not ent_text:
                continue
            label = ent.get("label", "")
            if self.labels and label not in self.labels:
                continue
            idx = query.find(ent_text)
            start = idx if idx >= 0 else 0
            end = start + len(ent_text) if idx >= 0 else 0
            mentions.append(EntityMention(
                text=ent_text,
                label=label,
                start=start,
                end=end,
                score=1.0,
            ))
        return mentions

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
            result = self._client.complete_with_tools(messages, tools=TRIPLE_QUERY_TOOLS)
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
