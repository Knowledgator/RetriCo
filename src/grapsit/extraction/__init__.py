"""Extraction engines — standalone NER, relex, and entity linking.

Provides unified engines that can be used:
1. Inside pipeline processors (construct and query stages)
2. Standalone by users for extraction without a database
"""

from .gliner_engine import GLiNEREngine, ExtractionResult
from .llm_engine import LLMExtractionEngine
from .linker import EntityLinkerEngine
from .utils import (
    strip_markdown_fences,
    parse_entities_json,
    parse_relations_json,
    parse_standalone_json,
    find_entity_offsets,
    build_labels_instruction,
    build_relation_labels_instruction,
    normalize_mentions,
    format_entities_list,
    mentions_to_gliner_spans,
)

__all__ = [
    "GLiNEREngine",
    "LLMExtractionEngine",
    "EntityLinkerEngine",
    "ExtractionResult",
    # Utilities
    "strip_markdown_fences",
    "parse_entities_json",
    "parse_relations_json",
    "parse_standalone_json",
    "find_entity_offsets",
    "build_labels_instruction",
    "build_relation_labels_instruction",
    "normalize_mentions",
    "format_entities_list",
    "mentions_to_gliner_spans",
]
