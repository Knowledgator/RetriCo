"""GLiNER-relex relation extraction processor.

GLiNER-relex can perform both entity extraction and relation extraction.
When pre-extracted entities are provided (from a separate NER step), they
are passed as ``input_spans`` so the model skips its own NER and only
resolves relations between the given spans.

The ``input_spans`` format expected by GLiNER-relex is::

    List[List[Dict]]   # one list of span dicts per text

where each span dict is::

    {
        "start": int,   # start character offset
        "end": int,     # end character offset
        "text": str,    # extracted text
        "label": str,   # entity type
        "score": float  # confidence (0-1)
    }

Any NER output (including ``EntityMention`` objects) is automatically
converted to this format before being forwarded.
"""

from typing import Any, Dict, List, Optional
import logging

from ..core.base import BaseProcessor
from ..core.registry import construct_registry
from ..extraction.gliner_engine import GLiNEREngine
from ..extraction.utils import mentions_to_gliner_spans as _mentions_to_gliner_spans
from ..models.relation import Relation
from ..models.entity import EntityMention
from ..models.document import Chunk

logger = logging.getLogger(__name__)

# Re-export for backward compatibility
_mentions_to_gliner_spans = _mentions_to_gliner_spans


class RelexGLiNERProcessor(BaseProcessor):
    """Extract relations using GLiNER-relex model.

    Operates in two modes:

    1. **Standalone** (no ``entities`` input) — the model performs its own
       entity extraction alongside relation extraction.
    2. **With pre-extracted entities** — entities from a separate NER step
       are converted to ``input_spans`` and forwarded to the model so it
       only resolves relations between the given spans.

    Config keys:
        model: str — model name (default: "knowledgator/gliner-relex-large-v0.5")
        entity_labels: List[str] — entity type labels
        relation_labels: List[str] — relation type labels
        threshold: float — entity threshold (default: 0.5)
        relation_threshold: float — relation threshold (default: 0.5)
        adjacency_threshold: float — adjacency threshold (default: 0.55)
        batch_size: int — inference batch size (default: 8)
        device: str — "cpu" or "cuda"
    """

    default_inputs = {"chunks": "ner_result.chunks", "entities": "ner_result.entities"}
    default_output = "relex_result"

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self._engine = GLiNEREngine(
            model=config_dict.get("model", "knowledgator/gliner-relex-large-v0.5"),
            labels=config_dict.get("entity_labels", []),
            relation_labels=config_dict.get("relation_labels", []),
            threshold=config_dict.get("threshold", 0.5),
            relation_threshold=config_dict.get("relation_threshold", 0.5),
            adjacency_threshold=config_dict.get("adjacency_threshold", 0.55),
            batch_size=config_dict.get("batch_size", 8),
            device=config_dict.get("device", "cpu"),
            flat_ner=config_dict.get("flat_ner", False),
        )

    def __call__(
        self,
        *,
        chunks: List[Chunk] = None,
        texts: List[str] = None,
        entities: Optional[List[List[Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run relation extraction on chunks.

        Args:
            chunks: Text chunks to process.
            texts: Raw texts (used if chunks not provided).
            entities: Optional pre-extracted entities per chunk
                (List[List[EntityMention]] or List[List[dict]]).
                When provided, these are passed as ``input_spans`` to
                GLiNER-relex so it skips its own NER.

        Returns:
            {"relations": List[List[Relation]], "entities": ..., "chunks": chunks}

            ``relations`` is a list-of-lists aligned with chunks.
            ``entities`` contains the per-chunk entities — either the ones
            extracted by the model (standalone mode) or the ones passed in.
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
                    r.chunk_id = chunk_id

        return {"relations": result.relations, "entities": result.entities, "chunks": chunks}


@construct_registry.register("relex_gliner")
def create_relex_gliner(config_dict: dict, pipeline=None):
    return RelexGLiNERProcessor(config_dict, pipeline)
