"""Query parser — NER on query text (GLiNER or LLM)."""

from typing import Any, Dict, List, Optional
import logging

from ..core.base import BaseProcessor
from ..core.registry import processor_registry
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
    """

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.method: str = config_dict.get("method", "gliner")
        self.labels: List[str] = config_dict.get("labels", [])
        self._model = None  # GLiNER model
        self._client = None  # OpenAI client

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

        prompt = (
            f"Extract all named entities from this query.\n\n"
            f"{labels_instruction}\n\n"
            f"Query: \"{query}\"\n\n"
            f"Return a JSON array of objects with:\n"
            f'- "text": the exact entity text as it appears\n'
            f'- "label": the entity type{label_constraint}\n\n'
            f"Return ONLY the JSON array, no other text."
        )
        messages = [
            {"role": "system", "content": "You are a named entity recognition system."},
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

    def __call__(self, *, query: str, **kwargs) -> Dict[str, Any]:
        """Parse a query string to extract entities.

        Returns:
            {"query": str, "entities": List[EntityMention]}
        """
        if self.method == "llm":
            entities = self._parse_llm(query)
        else:
            entities = self._parse_gliner(query)
        return {"query": query, "entities": entities}


@processor_registry.register("query_parser")
def create_query_parser(config_dict: dict, pipeline=None):
    return QueryParserProcessor(config_dict, pipeline)
