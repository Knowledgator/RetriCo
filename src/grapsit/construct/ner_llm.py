"""LLM-based named entity recognition processor."""

from typing import Any, Dict, List, Optional
import json
import re
import logging

from ..core.base import BaseProcessor
from ..core.registry import processor_registry
from ..llm.openai_client import OpenAIClient
from ..models.entity import EntityMention
from ..models.document import Chunk

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a named entity recognition system. "
    "Extract entities from the given text and return them as JSON."
)

_USER_PROMPT_TEMPLATE = """\
Extract all named entities from the text below.

{labels_instruction}

Text:
\"\"\"{text}\"\"\"

Return a JSON array of objects with these fields:
- "text": the exact entity text as it appears
- "label": the entity type{label_constraint}

Return ONLY the JSON array, no other text."""

# JSON schema for structured output mode.
_NER_SCHEMA = {
    "name": "ner_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "label": {"type": "string"},
                    },
                    "required": ["text", "label"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["entities"],
        "additionalProperties": False,
    },
}


def _parse_entities_json(raw: str) -> List[Dict[str, Any]]:
    """Parse LLM response into a list of entity dicts.

    Handles:
    - Plain JSON arrays
    - {"entities": [...]} wrapper objects
    - Markdown code blocks
    """
    text = raw.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM NER response as JSON")
        return []

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and "entities" in parsed:
        return parsed["entities"]
    return []


def _find_entity_offsets(text: str, entity_text: str, start_hint: int = 0) -> tuple:
    """Find start/end offsets for an entity in text.

    Uses the hint first, then falls back to string search.
    """
    if start_hint >= 0 and text[start_hint:start_hint + len(entity_text)] == entity_text:
        return start_hint, start_hint + len(entity_text)

    idx = text.find(entity_text)
    if idx >= 0:
        return idx, idx + len(entity_text)

    # Case-insensitive fallback
    idx = text.lower().find(entity_text.lower())
    if idx >= 0:
        return idx, idx + len(entity_text)

    return 0, 0


class NERLLMProcessor(BaseProcessor):
    """Extract entities from chunks using an LLM.

    Config keys:
        api_key: str — OpenAI API key
        base_url: str — API base URL (for vLLM/local servers)
        model: str — model name (default: "gpt-4o-mini")
        labels: List[str] — entity type labels to detect
        temperature: float — sampling temperature (default: 0.1)
        max_completion_tokens: int — max tokens (default: 4096)
        timeout: float — request timeout seconds (default: 60.0)
        structured_output: bool — use JSON schema constraints (default: True).
            Eliminates JSON parse errors on models that support it (OpenAI, vLLM).
            Falls back to json_object mode automatically if unsupported.
    """

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.labels: List[str] = config_dict.get("labels", [])
        self.structured_output: bool = config_dict.get("structured_output", True)
        self._structured_failed: bool = False
        self._client: Optional[OpenAIClient] = None
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
            self._client = OpenAIClient(**self._client_kwargs)

    def _use_structured(self) -> bool:
        """Whether to attempt structured output."""
        return self.structured_output and not self._structured_failed

    def _complete_with_fallback(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM, using structured output with json_object fallback."""
        if self._use_structured():
            try:
                return self._client.complete(
                    messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": _NER_SCHEMA,
                    },
                )
            except Exception as e:
                logger.info(
                    f"Structured output not supported, falling back to json_object: {e}"
                )
                self._structured_failed = True

        return self._client.complete(
            messages, response_format={"type": "json_object"}
        )

    def _extract_chunk(self, text: str, chunk_id: str) -> List[EntityMention]:
        """Extract entities from a single chunk."""
        if self.labels:
            labels_instruction = f"Entity labels: {', '.join(self.labels)}"
            label_constraint = " (one of the labels above)"
        else:
            labels_instruction = (
                "No specific entity labels are provided. "
                "Identify all named entities and assign an appropriate type to each."
            )
            label_constraint = ""

        prompt = _USER_PROMPT_TEMPLATE.format(
            labels_instruction=labels_instruction,
            label_constraint=label_constraint,
            text=text,
        )
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = self._complete_with_fallback(messages)
        except Exception as e:
            logger.warning(f"LLM NER request failed for chunk {chunk_id}: {e}")
            return []

        raw_entities = _parse_entities_json(raw)
        mentions = []
        for ent in raw_entities:
            ent_text = ent.get("text", "")
            if not ent_text:
                continue
            label = ent.get("label", "")
            # Only filter by label when labels are specified
            if self.labels and label not in self.labels:
                continue
            start, end = _find_entity_offsets(text, ent_text)
            mentions.append(EntityMention(
                text=ent_text,
                label=label,
                start=start,
                end=end,
                score=1.0,
                chunk_id=chunk_id,
            ))
        return mentions

    def __call__(self, *, chunks: List[Chunk] = None, texts: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Run LLM-based NER on chunks.

        Returns:
            {"entities": List[List[EntityMention]], "chunks": chunks}
        """
        self._ensure_client()

        if chunks is None:
            chunks = []
        if texts is None:
            texts = [c.text for c in chunks]

        all_mentions: List[List[EntityMention]] = []
        for i, text in enumerate(texts):
            chunk_id = chunks[i].id if i < len(chunks) else ""
            mentions = self._extract_chunk(text, chunk_id)
            all_mentions.append(mentions)

        return {"entities": all_mentions, "chunks": chunks}


@processor_registry.register("ner_llm")
def create_ner_llm(config_dict: dict, pipeline=None):
    return NERLLMProcessor(config_dict, pipeline)
