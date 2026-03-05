"""Unified LLM extraction engine for NER, relex, or both."""

from typing import Any, Dict, List, Optional, Tuple
import logging

from ..models.entity import EntityMention
from ..models.relation import Relation
from ..llm.prompts import (
    NER_SYSTEM_PROMPT,
    NER_USER_PROMPT_TEMPLATE,
    RELEX_SYSTEM_PROMPT,
    RELEX_STANDALONE_PROMPT_TEMPLATE,
    RELEX_WITH_ENTITIES_PROMPT_TEMPLATE,
    QUERY_PARSER_SYSTEM_PROMPT,
    QUERY_PARSER_USER_PROMPT_TEMPLATE,
)
from .gliner_engine import ExtractionResult
from .utils import (
    parse_entities_json,
    parse_relations_json,
    parse_standalone_json,
    find_entity_offsets,
    build_labels_instruction,
    build_relation_labels_instruction,
    normalize_mentions,
    format_entities_list,
    strip_markdown_fences,
)

logger = logging.getLogger(__name__)

# JSON schemas for structured output mode.
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

_RELATION_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "head": {"type": "string"},
        "tail": {"type": "string"},
        "relation": {"type": "string"},
        "start_date": {"type": ["string", "null"]},
        "end_date": {"type": ["string", "null"]},
    },
    "required": ["head", "tail", "relation", "start_date", "end_date"],
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


class LLMExtractionEngine:
    """Unified LLM engine for NER, relex, or both.

    - NER-only: provide only ``labels``
    - NER + relex: provide ``labels`` + ``relation_labels``
    - Relex with pre-extracted entities: call extract() with entities arg
    """

    def __init__(
        self,
        *,
        api_key: str = None,
        base_url: str = None,
        model: str = "gpt-4o-mini",
        labels: List[str] = None,
        relation_labels: List[str] = None,
        temperature: float = 0.1,
        max_completion_tokens: int = 4096,
        timeout: float = 60.0,
        structured_output: bool = True,
        # Prompt overrides
        ner_system_prompt: str = NER_SYSTEM_PROMPT,
        ner_user_prompt_template: str = NER_USER_PROMPT_TEMPLATE,
        relex_system_prompt: str = RELEX_SYSTEM_PROMPT,
        relex_standalone_prompt_template: str = RELEX_STANDALONE_PROMPT_TEMPLATE,
        relex_with_entities_prompt_template: str = RELEX_WITH_ENTITIES_PROMPT_TEMPLATE,
        # Query-specific prompts
        query_system_prompt: str = QUERY_PARSER_SYSTEM_PROMPT,
        query_user_prompt_template: str = QUERY_PARSER_USER_PROMPT_TEMPLATE,
        # Force standalone (relex) mode even when relation_labels is empty
        force_relex: bool = False,
    ):
        self.labels = labels or []
        self.relation_labels = relation_labels or []
        self._force_relex = force_relex
        self.structured_output = structured_output
        self._structured_failed = False
        self._client = None
        self._client_kwargs = {
            "api_key": api_key,
            "base_url": base_url,
            "model": model,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "timeout": timeout,
        }
        # Prompts
        self._ner_system_prompt = ner_system_prompt
        self._ner_user_prompt_template = ner_user_prompt_template
        self._relex_system_prompt = relex_system_prompt
        self._relex_standalone_prompt_template = relex_standalone_prompt_template
        self._relex_with_entities_prompt_template = relex_with_entities_prompt_template
        self._query_system_prompt = query_system_prompt
        self._query_user_prompt_template = query_user_prompt_template

    @property
    def has_relex(self) -> bool:
        return self._force_relex or len(self.relation_labels) > 0

    def _ensure_client(self):
        if self._client is None:
            from ..llm.openai_client import OpenAIClient
            self._client = OpenAIClient(**self._client_kwargs)

    def _use_structured(self) -> bool:
        return self.structured_output and not self._structured_failed

    def _complete_with_fallback(
        self, messages: List[Dict[str, str]], schema: Optional[Dict] = None
    ) -> str:
        """Call the LLM, using structured output with json_object fallback."""
        if self._use_structured() and schema is not None:
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

    # -- NER extraction --------------------------------------------------------

    def _extract_ner(self, text: str) -> List[EntityMention]:
        """Extract entities from a single text using NER prompts."""
        labels_instruction, label_constraint = build_labels_instruction(self.labels)

        prompt = self._ner_user_prompt_template.format(
            labels_instruction=labels_instruction,
            label_constraint=label_constraint,
            text=text,
        )
        messages = [
            {"role": "system", "content": self._ner_system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = self._complete_with_fallback(messages, _NER_SCHEMA)
        except Exception as e:
            logger.warning(f"LLM NER request failed: {e}")
            return []

        raw_entities = parse_entities_json(raw)
        mentions = []
        for ent in raw_entities:
            ent_text = ent.get("text", "")
            if not ent_text:
                continue
            label = ent.get("label", "")
            if self.labels and label not in self.labels:
                continue
            start, end = find_entity_offsets(text, ent_text)
            mentions.append(EntityMention(
                text=ent_text,
                label=label,
                start=start,
                end=end,
                score=1.0,
            ))
        return mentions

    # -- Standalone relex extraction -------------------------------------------

    def _extract_standalone(self, text: str) -> Tuple[List[EntityMention], List[Relation]]:
        """Extract entities + relations in one LLM call."""
        labels_instruction, _ = build_labels_instruction(self.labels)
        rel_instruction = build_relation_labels_instruction(self.relation_labels)

        prompt = self._relex_standalone_prompt_template.format(
            entity_labels_instruction=labels_instruction,
            relation_labels_instruction=rel_instruction,
            text=text,
        )
        messages = [
            {"role": "system", "content": self._relex_system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = self._complete_with_fallback(messages, _STANDALONE_SCHEMA)
        except Exception as e:
            logger.warning(f"LLM relex standalone failed: {e}")
            return [], []

        raw_entities, raw_relations = parse_standalone_json(raw)

        mentions = []
        entity_label_map: Dict[str, str] = {}
        for ent in raw_entities:
            ent_text = ent.get("text", "")
            if not ent_text:
                continue
            label = ent.get("label", "")
            entity_label_map[ent_text.strip().lower()] = label
            start, end = find_entity_offsets(text, ent_text)
            mentions.append(EntityMention(
                text=ent_text,
                label=label,
                start=start,
                end=end,
                score=1.0,
            ))

        relations = []
        for rel in raw_relations:
            head_text = rel.get("head", "")
            tail_text = rel.get("tail", "")
            rel_kwargs: Dict[str, Any] = {
                "head_text": head_text,
                "tail_text": tail_text,
                "relation_type": rel.get("relation", ""),
                "head_label": entity_label_map.get(head_text.strip().lower(), ""),
                "tail_label": entity_label_map.get(tail_text.strip().lower(), ""),
                "score": 1.0,
            }
            if rel.get("start_date"):
                rel_kwargs["start_date"] = rel["start_date"]
            if rel.get("end_date"):
                rel_kwargs["end_date"] = rel["end_date"]
            relations.append(Relation(**rel_kwargs))

        return mentions, relations

    # -- With-entities relex extraction ----------------------------------------

    def _extract_with_entities(
        self, text: str, entities: List[EntityMention]
    ) -> List[Relation]:
        """Extract only relations given pre-extracted entities."""
        rel_instruction = build_relation_labels_instruction(self.relation_labels)

        prompt = self._relex_with_entities_prompt_template.format(
            relation_labels_instruction=rel_instruction,
            entities_list=format_entities_list(entities),
            text=text,
        )
        messages = [
            {"role": "system", "content": self._relex_system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = self._complete_with_fallback(messages, _WITH_ENTITIES_SCHEMA)
        except Exception as e:
            logger.warning(f"LLM relex with-entities failed: {e}")
            return []

        raw_relations = parse_relations_json(raw)

        entity_label_map = {m.text.strip().lower(): m.label for m in entities}

        relations = []
        for rel in raw_relations:
            head_text = rel.get("head", "")
            tail_text = rel.get("tail", "")
            rel_kwargs: Dict[str, Any] = {
                "head_text": head_text,
                "tail_text": tail_text,
                "relation_type": rel.get("relation", ""),
                "head_label": entity_label_map.get(head_text.strip().lower(), ""),
                "tail_label": entity_label_map.get(tail_text.strip().lower(), ""),
                "score": 1.0,
            }
            if rel.get("start_date"):
                rel_kwargs["start_date"] = rel["start_date"]
            if rel.get("end_date"):
                rel_kwargs["end_date"] = rel["end_date"]
            relations.append(Relation(**rel_kwargs))
        return relations

    # -- Query mode extraction -------------------------------------------------

    def extract_from_query(self, query: str) -> List[EntityMention]:
        """Extract entities from a query string (query parser mode).

        Uses json_object mode only (no structured output) and query-specific prompts.
        """
        import json
        self._ensure_client()

        labels_instruction, label_constraint = build_labels_instruction(self.labels)

        prompt = self._query_user_prompt_template.format(
            labels_instruction=labels_instruction,
            label_constraint=label_constraint,
            query=query,
        )
        messages = [
            {"role": "system", "content": self._query_system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            raw = self._client.complete(
                messages, response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.warning(f"LLM query parsing failed: {e}")
            return []

        text = strip_markdown_fences(raw)

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

    # -- Batch extraction ------------------------------------------------------

    def extract_from_text(
        self, text: str, entities: Optional[List[EntityMention]] = None
    ) -> Tuple[List[EntityMention], List[Relation]]:
        """Extract from a single text (construct mode).

        Returns (entities, relations) tuple.
        """
        self._ensure_client()

        if self.has_relex:
            if entities is not None:
                rels = self._extract_with_entities(text, entities)
                return list(entities), rels
            else:
                return self._extract_standalone(text)
        else:
            mentions = self._extract_ner(text)
            return mentions, []

    def extract(
        self, texts: List[str], entities: Optional[List[List[Any]]] = None
    ) -> ExtractionResult:
        """Batch extraction over multiple texts.

        Args:
            texts: List of text strings.
            entities: Optional pre-extracted entities per text (list-of-lists).

        Returns:
            ExtractionResult with entities and relations.
        """
        self._ensure_client()

        has_entities = entities is not None
        all_entities: List[List[EntityMention]] = []
        all_relations: List[List[Relation]] = []

        for i, text in enumerate(texts):
            if has_entities:
                chunk_ents = entities[i] if i < len(entities) else []
                normalized = normalize_mentions(chunk_ents)
                rels = self._extract_with_entities(text, normalized)
                all_entities.append(normalized)
                all_relations.append(rels)
            elif self.has_relex:
                mentions, rels = self._extract_standalone(text)
                all_entities.append(mentions)
                all_relations.append(rels)
            else:
                mentions = self._extract_ner(text)
                all_entities.append(mentions)
                all_relations.append([])

        return ExtractionResult(entities=all_entities, relations=all_relations)
