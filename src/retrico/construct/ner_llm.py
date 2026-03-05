"""LLM-based named entity recognition processor."""

from typing import Any, Dict, List
import logging

from ..core.base import BaseProcessor
from ..core.registry import construct_registry
from ..extraction.llm_engine import LLMExtractionEngine
from ..extraction.utils import (
    parse_entities_json as _parse_entities_json,
    find_entity_offsets as _find_entity_offsets,
)
from ..models.document import Chunk

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
_parse_entities_json = _parse_entities_json
_find_entity_offsets = _find_entity_offsets


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
        system_prompt: str — override default NER system prompt.
        user_prompt_template: str — override default NER user prompt template.
            Available placeholders: {labels_instruction}, {label_constraint}, {text}.
    """

    default_inputs = {"chunks": "chunker_result.chunks"}
    default_output = "ner_result"

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        engine_kwargs = {
            "api_key": config_dict.get("api_key"),
            "base_url": config_dict.get("base_url"),
            "model": config_dict.get("model", "gpt-4o-mini"),
            "labels": config_dict.get("labels", []),
            "temperature": config_dict.get("temperature", 0.1),
            "max_completion_tokens": config_dict.get("max_completion_tokens", 4096),
            "timeout": config_dict.get("timeout", 60.0),
            "structured_output": config_dict.get("structured_output", True),
        }
        if "system_prompt" in config_dict:
            engine_kwargs["ner_system_prompt"] = config_dict["system_prompt"]
        if "user_prompt_template" in config_dict:
            engine_kwargs["ner_user_prompt_template"] = config_dict["user_prompt_template"]
        self._engine = LLMExtractionEngine(**engine_kwargs)

    def __call__(self, *, chunks: List[Chunk] = None, texts: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Run LLM-based NER on chunks.

        Returns:
            {"entities": List[List[EntityMention]], "chunks": chunks}
        """
        if chunks is None:
            chunks = []
        if texts is None:
            texts = [c.text for c in chunks]

        result = self._engine.extract(texts)

        # Assign chunk_id to each mention
        for i, mentions in enumerate(result.entities):
            chunk_id = chunks[i].id if i < len(chunks) else ""
            for m in mentions:
                m.chunk_id = chunk_id

        return {"entities": result.entities, "chunks": chunks}


@construct_registry.register("ner_llm")
def create_ner_llm(config_dict: dict, pipeline=None):
    return NERLLMProcessor(config_dict, pipeline)
