"""Shared parsing and conversion utilities for extraction engines."""

from typing import Any, Dict, List, Tuple
import json
import re
import logging

from ..models.entity import EntityMention

logger = logging.getLogger(__name__)


def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()
    return text


def parse_entities_json(raw: str) -> List[Dict[str, Any]]:
    """Parse LLM response into a list of entity dicts.

    Handles plain JSON arrays, {"entities": [...]} wrappers, and markdown fences.
    """
    text = strip_markdown_fences(raw)

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


def parse_relations_json(raw: str) -> List[Dict[str, Any]]:
    """Parse with-entities mode response: [...] or {"relations": [...]}."""
    text = strip_markdown_fences(raw)

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


def parse_standalone_json(raw: str) -> Tuple[List, List]:
    """Parse standalone mode response: {"entities": [...], "relations": [...]}."""
    text = strip_markdown_fences(raw)

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


def find_entity_offsets(text: str, entity_text: str, start_hint: int = 0) -> Tuple[int, int]:
    """Find start/end offsets for an entity in text.

    Uses the hint first, then falls back to string search, then case-insensitive.
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


def build_labels_instruction(labels: List[str]) -> Tuple[str, str]:
    """Build labels_instruction and label_constraint strings for NER prompts."""
    if labels:
        labels_instruction = f"Entity labels: {', '.join(labels)}"
        label_constraint = " (one of the labels above)"
    else:
        labels_instruction = (
            "No specific entity labels are provided. "
            "Identify all named entities and assign an appropriate type to each."
        )
        label_constraint = ""
    return labels_instruction, label_constraint


def build_relation_labels_instruction(relation_labels: List[str]) -> str:
    """Build relation label instruction string for relex prompts."""
    if relation_labels:
        return f"Relation labels: {', '.join(relation_labels)}"
    return (
        "No specific relation labels are provided. "
        "Identify all meaningful relationships and assign an appropriate type to each."
    )


def normalize_mentions(items: List[Any], chunk_id: str = "") -> List[EntityMention]:
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


def format_entities_list(mentions: List[EntityMention]) -> str:
    """Format entity mentions for inclusion in a prompt."""
    lines = []
    for m in mentions:
        lines.append(f'- "{m.text}" ({m.label})')
    return "\n".join(lines) if lines else "(none)"


def mentions_to_gliner_spans(
    entities_per_chunk: List[List[Any]],
) -> List[List[Dict[str, Any]]]:
    """Convert per-chunk entity mentions to GLiNER input_spans format.

    Accepts:
      - List[List[EntityMention]]
      - List[List[dict]] (already in GLiNER format)

    Returns:
        List[List[dict]] with keys: start, end, text, label, score
    """
    result: List[List[Dict[str, Any]]] = []
    for chunk_mentions in entities_per_chunk:
        spans: List[Dict[str, Any]] = []
        for m in chunk_mentions:
            if isinstance(m, dict):
                spans.append({
                    "start": m.get("start", 0),
                    "end": m.get("end", 0),
                    "text": m.get("text", ""),
                    "label": m.get("label", ""),
                    "score": m.get("score", 0.0),
                })
            elif isinstance(m, EntityMention):
                spans.append({
                    "start": m.start,
                    "end": m.end,
                    "text": m.text,
                    "label": m.label,
                    "score": m.score,
                })
            else:
                # Generic object with attributes
                spans.append({
                    "start": getattr(m, "start", 0),
                    "end": getattr(m, "end", 0),
                    "text": getattr(m, "text", ""),
                    "label": getattr(m, "label", ""),
                    "score": getattr(m, "score", 0.0),
                })
        result.append(spans)
    return result
