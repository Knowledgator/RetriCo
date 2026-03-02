"""GLiNER-based named entity recognition processor."""

from typing import Any, Dict, List
import logging

from ..core.base import BaseProcessor
from ..core.registry import construct_registry
from ..extraction.gliner_engine import GLiNEREngine
from ..models.entity import EntityMention
from ..models.document import Chunk

logger = logging.getLogger(__name__)


class NERGLiNERProcessor(BaseProcessor):
    """Extract entities from chunks using GLiNER.

    Config keys:
        model: str — GLiNER model name (default: "urchade/gliner_multi-v2.1")
        labels: List[str] — entity type labels to detect
        threshold: float — confidence threshold (default: 0.3)
        batch_size: int — inference batch size (default: 8)
        device: str — "cpu" or "cuda" (default: "cpu")
        flat_ner: bool — flatten nested entities (default: True)
    """

    default_inputs = {"chunks": "chunker_result.chunks"}
    default_output = "ner_result"

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self._engine = GLiNEREngine(
            model=config_dict.get("model", "urchade/gliner_multi-v2.1"),
            labels=config_dict.get("labels", []),
            threshold=config_dict.get("threshold", 0.3),
            batch_size=config_dict.get("batch_size", 8),
            device=config_dict.get("device", "cpu"),
            flat_ner=config_dict.get("flat_ner", True),
        )

    def __call__(self, *, chunks: List[Chunk] = None, texts: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Run NER on chunks using batched inference.

        Returns:
            {"entities": List[List[EntityMention]], "chunks": chunks}

        entities is a list-of-lists aligned with chunks — one list of mentions per chunk.
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


@construct_registry.register("ner_gliner")
def create_ner_gliner(config_dict: dict, pipeline=None):
    return NERGLiNERProcessor(config_dict, pipeline)
