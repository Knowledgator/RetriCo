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
from ..core.registry import processor_registry
from ..models.relation import Relation
from ..models.entity import EntityMention
from ..models.document import Chunk

logger = logging.getLogger(__name__)


def _mentions_to_gliner_spans(
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

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.model_name: str = config_dict.get("model", "knowledgator/gliner-relex-large-v0.5")
        self.entity_labels: List[str] = config_dict.get("entity_labels", [])
        self.relation_labels: List[str] = config_dict.get("relation_labels", [])
        self.threshold: float = config_dict.get("threshold", 0.5)
        self.relation_threshold: float = config_dict.get("relation_threshold", 0.5)
        self.adjacency_threshold: float = config_dict.get("adjacency_threshold", 0.55)
        self.batch_size: int = config_dict.get("batch_size", 8)
        self.device: str = config_dict.get("device", "cpu")
        self.flat_ner: bool = config_dict.get("flat_ner", False)
        self._model = None

    def _load_model(self):
        if self._model is None:
            from gliner import GLiNER
            logger.info(f"Loading GLiNER-relex model: {self.model_name}")
            self._model = GLiNER.from_pretrained(self.model_name)
            if self.device != "cpu":
                self._model = self._model.to(self.device)

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
        self._load_model()

        if chunks is None:
            chunks = []
        if texts is None:
            texts = [c.text for c in chunks]

        # Convert pre-extracted entities to GLiNER span format
        input_spans: Optional[List[List[Dict[str, Any]]]] = None
        if entities is not None:
            input_spans = _mentions_to_gliner_spans(entities)

        print('Input spans for GLiNER-relex:', input_spans)
        
        # Batched inference
        inference_kwargs: Dict[str, Any] = {
            "texts": texts,
            "labels": self.entity_labels,
            "relations": self.relation_labels,
            "threshold": self.threshold,
            "adjacency_threshold": self.adjacency_threshold,
            "relation_threshold": self.relation_threshold,
            "return_relations": True,
            "flat_ner": self.flat_ner,
            "batch_size": self.batch_size,
        }
        if input_spans is not None:
            inference_kwargs["input_spans"] = input_spans

        try:
            entities_batch, relations_batch = self._model.inference(**inference_kwargs)
        except Exception as e:
            logger.warning(f"Relation extraction failed: {e}")
            entities_batch = [[] for _ in texts]
            relations_batch = [[] for _ in texts]

        # Convert results per chunk
        all_relations: List[List[Relation]] = []
        all_entities: List[List[EntityMention]] = []

        for i in range(len(texts)):
            chunk_id = chunks[i].id if i < len(chunks) else ""
            raw_entities = entities_batch[i] if i < len(entities_batch) else []
            raw_relations = relations_batch[i] if i < len(relations_batch) else []

            # Convert entities
            mentions = []
            for ent in raw_entities:
                mentions.append(EntityMention(
                    text=ent["text"],
                    label=ent["label"],
                    start=ent.get("start", 0),
                    end=ent.get("end", 0),
                    score=ent.get("score", 0.0),
                    chunk_id=chunk_id,
                ))
            all_entities.append(mentions)

            # Convert relations
            # GLiNER-relex returns head/tail as {"entity_idx": int} references
            # into the entities list, not inline text/label dicts.
            relations = []
            for rel in raw_relations:
                head_ref = rel["head"]
                tail_ref = rel["tail"]
                head_ent = raw_entities[head_ref["entity_idx"]] if "entity_idx" in head_ref else head_ref
                tail_ent = raw_entities[tail_ref["entity_idx"]] if "entity_idx" in tail_ref else tail_ref
                relations.append(Relation(
                    head_text=head_ent.get("text", ""),
                    tail_text=tail_ent.get("text", ""),
                    relation_type=rel["relation"],
                    score=rel.get("score", 0.0),
                    chunk_id=chunk_id,
                    head_label=head_ent.get("label", ""),
                    tail_label=tail_ent.get("label", ""),
                ))
            all_relations.append(relations)

        return {"relations": all_relations, "entities": all_entities, "chunks": chunks}


@processor_registry.register("relex_gliner")
def create_relex_gliner(config_dict: dict, pipeline=None):
    return RelexGLiNERProcessor(config_dict, pipeline)
