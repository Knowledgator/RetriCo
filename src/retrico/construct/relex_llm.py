"""LLM-based relation extraction processor."""

from typing import Any, Dict, List, Optional
import logging

from ..core.base import BaseProcessor
from ..core.registry import construct_registry
from ..extraction.llm_engine import LLMExtractionEngine
from ..extraction.utils import (
    parse_standalone_json as _parse_standalone_json,
    parse_relations_json as _parse_relations_json,
    format_entities_list as _format_entities_list,
    normalize_mentions as _normalize_mentions,
)
from ..models.entity import EntityMention
from ..models.relation import Relation
from ..models.document import Chunk

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
_parse_standalone_json = _parse_standalone_json
_parse_relations_json = _parse_relations_json
_format_entities_list = _format_entities_list
_normalize_mentions = _normalize_mentions


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
        engine_kwargs = {
            "api_key": config_dict.get("api_key"),
            "base_url": config_dict.get("base_url"),
            "model": config_dict.get("model", "gpt-4o-mini"),
            "labels": config_dict.get("entity_labels", []),
            "relation_labels": config_dict.get("relation_labels", []),
            "temperature": config_dict.get("temperature", 0.1),
            "max_completion_tokens": config_dict.get("max_completion_tokens", 4096),
            "timeout": config_dict.get("timeout", 60.0),
            "structured_output": config_dict.get("structured_output", True),
            "force_relex": True,
        }
        if "system_prompt" in config_dict:
            engine_kwargs["relex_system_prompt"] = config_dict["system_prompt"]
        if "standalone_prompt_template" in config_dict:
            engine_kwargs["relex_standalone_prompt_template"] = config_dict["standalone_prompt_template"]
        if "with_entities_prompt_template" in config_dict:
            engine_kwargs["relex_with_entities_prompt_template"] = config_dict["with_entities_prompt_template"]
        self._engine = LLMExtractionEngine(**engine_kwargs)

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
        if chunks is None:
            chunks = []
        if texts is None:
            texts = [c.text for c in chunks]

        result = self._engine.extract(texts, entities=entities)

        # Assign chunk_id to entities and relations
        for i in range(len(texts)):
            chunk_id = chunks[i].id if i < len(chunks) else ""
            if i < len(result.entities):
                for m in result.entities[i]:
                    m.chunk_id = chunk_id
            if i < len(result.relations):
                for r in result.relations[i]:
                    r.chunk_id = [chunk_id] if chunk_id else []

        return {"relations": result.relations, "entities": result.entities, "chunks": chunks}


@construct_registry.register("relex_llm")
def create_relex_llm(config_dict: dict, pipeline=None):
    return RelexLLMProcessor(config_dict, pipeline)
