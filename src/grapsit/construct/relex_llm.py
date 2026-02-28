"""LLM-based relation extraction processor."""

from typing import Any, Dict, List, Optional
import json
import re
import logging

from ..core.base import BaseProcessor
from ..core.registry import construct_registry
from ..llm.openai_client import OpenAIClient
from ..llm.prompts import (
    RELEX_SYSTEM_PROMPT,
    RELEX_STANDALONE_PROMPT_TEMPLATE,
    RELEX_WITH_ENTITIES_PROMPT_TEMPLATE,
)
from ..models.entity import EntityMention
from ..models.relation import Relation
from ..models.document import Chunk

logger = logging.getLogger(__name__)

# JSON schemas for structured output mode.
_RELATION_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "head": {"type": "string"},
        "tail": {"type": "string"},
        "relation": {"type": "string"},
    },
    "required": ["head", "tail", "relation"],
    "additionalProperties": False,
}

_STANDALONE_SCHEMA = {
    "name": "relex_standalone_response",
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
            "relations": {
                "type": "array",
                "items": _RELATION_ITEM_SCHEMA,
            },
        },
        "required": ["entities", "relations"],
        "additionalProperties": False,
    },
}

_WITH_ENTITIES_SCHEMA = {
    "name": "relex_with_entities_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "relations": {
                "type": "array",
                "items": _RELATION_ITEM_SCHEMA,
            },
        },
        "required": ["relations"],
        "additionalProperties": False,
    },
}


def _format_entities_list(mentions: List[EntityMention]) -> str:
    """Format entity mentions for inclusion in a prompt."""
    lines = []
    for m in mentions:
        lines.append(f'- "{m.text}" ({m.label})')
    return "\n".join(lines) if lines else "(none)"


def _parse_standalone_json(raw: str) -> tuple:
    """Parse standalone mode response: {"entities": [...], "relations": [...]}."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM relex standalone response as JSON")
        return [], []

    if isinstance(parsed, dict):
        entities = parsed.get("entities", [])
        relations = parsed.get("relations", [])
        return entities, relations

    return [], []


def _parse_relations_json(raw: str) -> List[Dict[str, Any]]:
    """Parse with-entities mode response: [...] or {"relations": [...]}."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM relex response as JSON")
        return []

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict) and "relations" in parsed:
        return parsed["relations"]
    return []


def _find_entity_offsets(text: str, entity_text: str, start_hint: int = 0) -> tuple:
    """Find start/end offsets for an entity in text."""
    if start_hint >= 0 and text[start_hint:start_hint + len(entity_text)] == entity_text:
        return start_hint, start_hint + len(entity_text)

    idx = text.find(entity_text)
    if idx >= 0:
        return idx, idx + len(entity_text)

    idx = text.lower().find(entity_text.lower())
    if idx >= 0:
        return idx, idx + len(entity_text)

    return 0, 0


class RelexLLMProcessor(BaseProcessor):
    """Extract relations using an LLM.

    Operates in two modes:

    1. **Standalone** (no ``entities`` input) — LLM extracts both entities
       and relations in a single call.
    2. **With pre-extracted entities** — entities from a separate NER step
       are listed in the prompt, and the LLM only extracts relations.

    Config keys:
        api_key: str — OpenAI API key
        base_url: str — API base URL (for vLLM/local servers)
        model: str — model name (default: "gpt-4o-mini")
        entity_labels: List[str] — entity type labels
        relation_labels: List[str] — relation type labels
        temperature: float — sampling temperature (default: 0.1)
        max_completion_tokens: int — max tokens (default: 4096)
        timeout: float — request timeout seconds (default: 60.0)
        structured_output: bool — use JSON schema constraints (default: True).
            Eliminates JSON parse errors on models that support it (OpenAI, vLLM).
            Falls back to json_object mode automatically if unsupported.
        system_prompt: str — override default relex system prompt.
        standalone_prompt_template: str — override default standalone user prompt.
            Available placeholders: {entity_labels_instruction}, {relation_labels_instruction}, {text}.
        with_entities_prompt_template: str — override default with-entities user prompt.
            Available placeholders: {relation_labels_instruction}, {entities_list}, {text}.
    """

    default_inputs = {"chunks": "ner_result.chunks", "entities": "ner_result.entities"}
    default_output = "relex_result"

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.entity_labels: List[str] = config_dict.get("entity_labels", [])
        self.relation_labels: List[str] = config_dict.get("relation_labels", [])
        self._system_prompt = config_dict.get("system_prompt", RELEX_SYSTEM_PROMPT)
        self._standalone_prompt_template = config_dict.get(
            "standalone_prompt_template", RELEX_STANDALONE_PROMPT_TEMPLATE
        )
        self._with_entities_prompt_template = config_dict.get(
            "with_entities_prompt_template", RELEX_WITH_ENTITIES_PROMPT_TEMPLATE
        )
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

    def _complete_with_fallback(
        self, messages: List[Dict[str, str]], schema: Dict[str, Any]
    ) -> str:
        """Call the LLM, using structured output with json_object fallback."""
        if self._use_structured():
            try:
                return self._client.complete(
                    messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": schema,
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

    def _label_instructions(self) -> tuple:
        """Build label instruction strings for prompts."""
        if self.entity_labels:
            ent_instr = f"Entity labels: {', '.join(self.entity_labels)}"
        else:
            ent_instr = (
                "No specific entity labels are provided. "
                "Identify all named entities and assign an appropriate type to each."
            )
        if self.relation_labels:
            rel_instr = f"Relation labels: {', '.join(self.relation_labels)}"
        else:
            rel_instr = (
                "No specific relation labels are provided. "
                "Identify all meaningful relationships and assign an appropriate type to each."
            )
        return ent_instr, rel_instr

    def _extract_standalone(
        self, text: str, chunk_id: str
    ) -> tuple:
        """Extract entities + relations in one LLM call."""
        ent_instr, rel_instr = self._label_instructions()
        prompt = self._standalone_prompt_template.format(
            entity_labels_instruction=ent_instr,
            relation_labels_instruction=rel_instr,
            text=text,
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = self._complete_with_fallback(messages, _STANDALONE_SCHEMA)
        except Exception as e:
            logger.warning(f"LLM relex standalone failed for chunk {chunk_id}: {e}")
            return [], []

        raw_entities, raw_relations = _parse_standalone_json(raw)

        mentions = []
        entity_label_map: Dict[str, str] = {}
        for ent in raw_entities:
            ent_text = ent.get("text", "")
            if not ent_text:
                continue
            label = ent.get("label", "")
            entity_label_map[ent_text.strip().lower()] = label
            start, end = _find_entity_offsets(text, ent_text)
            mentions.append(EntityMention(
                text=ent_text,
                label=label,
                start=start,
                end=end,
                score=1.0,
                chunk_id=chunk_id,
            ))

        relations = []
        for rel in raw_relations:
            head_text = rel.get("head", "")
            tail_text = rel.get("tail", "")
            relations.append(Relation(
                head_text=head_text,
                tail_text=tail_text,
                relation_type=rel.get("relation", ""),
                head_label=entity_label_map.get(head_text.strip().lower(), ""),
                tail_label=entity_label_map.get(tail_text.strip().lower(), ""),
                score=1.0,
                chunk_id=chunk_id,
            ))

        return mentions, relations

    def _extract_with_entities(
        self, text: str, chunk_id: str, entities: List[EntityMention]
    ) -> List[Relation]:
        """Extract only relations given pre-extracted entities."""
        _, rel_instr = self._label_instructions()
        prompt = self._with_entities_prompt_template.format(
            relation_labels_instruction=rel_instr,
            entities_list=_format_entities_list(entities),
            text=text,
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = self._complete_with_fallback(messages, _WITH_ENTITIES_SCHEMA)
        except Exception as e:
            logger.warning(f"LLM relex with-entities failed for chunk {chunk_id}: {e}")
            return []

        raw_relations = _parse_relations_json(raw)

        # Build lookup: entity text (lowered) -> label from pre-extracted entities
        entity_label_map = {m.text.strip().lower(): m.label for m in entities}

        relations = []
        for rel in raw_relations:
            head_text = rel.get("head", "")
            tail_text = rel.get("tail", "")
            relations.append(Relation(
                head_text=head_text,
                tail_text=tail_text,
                relation_type=rel.get("relation", ""),
                head_label=entity_label_map.get(head_text.strip().lower(), ""),
                tail_label=entity_label_map.get(tail_text.strip().lower(), ""),
                score=1.0,
                chunk_id=chunk_id,
            ))
        return relations

    def __call__(
        self,
        *,
        chunks: List[Chunk] = None,
        texts: List[str] = None,
        entities: Optional[List[List[Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run LLM-based relation extraction.

        Args:
            chunks: Text chunks to process.
            texts: Raw texts (used if chunks not provided).
            entities: Optional pre-extracted entities per chunk.

        Returns:
            {"relations": List[List[Relation]],
             "entities": List[List[EntityMention]],
             "chunks": List[Chunk]}
        """
        self._ensure_client()

        if chunks is None:
            chunks = []
        if texts is None:
            texts = [c.text for c in chunks]

        has_entities = entities is not None

        all_relations: List[List[Relation]] = []
        all_entities: List[List[EntityMention]] = []

        for i, text in enumerate(texts):
            chunk_id = chunks[i].id if i < len(chunks) else ""

            if has_entities:
                chunk_entities = entities[i] if i < len(entities) else []
                # Convert dicts/objects to EntityMention for prompt formatting
                normalized = _normalize_mentions(chunk_entities, chunk_id)
                rels = self._extract_with_entities(text, chunk_id, normalized)
                all_entities.append(normalized)
                all_relations.append(rels)
            else:
                mentions, rels = self._extract_standalone(text, chunk_id)
                all_entities.append(mentions)
                all_relations.append(rels)

        return {"relations": all_relations, "entities": all_entities, "chunks": chunks}


def _normalize_mentions(items: List[Any], chunk_id: str) -> List[EntityMention]:
    """Convert a list of entity mentions/dicts to EntityMention objects."""
    result = []
    for m in items:
        if isinstance(m, EntityMention):
            result.append(m)
        elif isinstance(m, dict):
            result.append(EntityMention(
                text=m.get("text", ""),
                label=m.get("label", ""),
                start=m.get("start", 0),
                end=m.get("end", 0),
                score=m.get("score", 1.0),
                chunk_id=m.get("chunk_id", chunk_id),
            ))
        else:
            result.append(EntityMention(
                text=getattr(m, "text", ""),
                label=getattr(m, "label", ""),
                start=getattr(m, "start", 0),
                end=getattr(m, "end", 0),
                score=getattr(m, "score", 1.0),
                chunk_id=getattr(m, "chunk_id", chunk_id),
            ))
    return result


@construct_registry.register("relex_llm")
def create_relex_llm(config_dict: dict, pipeline=None):
    return RelexLLMProcessor(config_dict, pipeline)
