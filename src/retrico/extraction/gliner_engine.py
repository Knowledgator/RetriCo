"""Unified GLiNER engine for NER, relex, or both."""

from typing import Any, Dict, List, Optional
import logging

from ..models.entity import EntityMention
from ..models.relation import Relation
from .utils import mentions_to_gliner_spans

logger = logging.getLogger(__name__)


class ExtractionResult:
    """Result of an extraction operation.

    Attributes:
        entities: Per-text entity mentions (list-of-lists aligned with input texts).
        relations: Per-text relations (list-of-lists, empty inner lists if NER-only).
    """

    __slots__ = ("entities", "relations")

    def __init__(
        self,
        entities: List[List[EntityMention]],
        relations: List[List[Relation]],
    ):
        self.entities = entities
        self.relations = relations


class GLiNEREngine:
    """Unified GLiNER engine for NER, relex, or both.

    - NER-only: provide only ``labels``, uses standard GLiNER model
    - NER + relex: provide ``labels`` + ``relation_labels``, uses GLiNER-relex model
    - Relex with pre-extracted entities: call extract() with entities arg
    """

    def __init__(
        self,
        *,
        model: str = "gliner-community/gliner_small-v2.5",
        labels: List[str] = None,
        relation_labels: List[str] = None,
        threshold: float = 0.3,
        relation_threshold: float = 0.5,
        adjacency_threshold: float = 0.55,
        batch_size: int = 8,
        device: str = "cpu",
        flat_ner: bool = True,
    ):
        self.model_name = model
        self.labels = labels or []
        self.relation_labels = relation_labels or []
        self.threshold = threshold
        self.relation_threshold = relation_threshold
        self.adjacency_threshold = adjacency_threshold
        self.batch_size = batch_size
        self.device = device
        # Auto-set flat_ner: False for relex mode unless explicitly overridden
        self.flat_ner = flat_ner if not self.relation_labels else False
        self._model = None

    @property
    def has_relex(self) -> bool:
        return len(self.relation_labels) > 0

    def _load_model(self):
        if self._model is None:
            from gliner import GLiNER
            logger.info(f"Loading GLiNER model: {self.model_name}")
            self._model = GLiNER.from_pretrained(self.model_name)
            if self.device != "cpu":
                self._model = self._model.to(self.device)

    def extract(
        self,
        texts: List[str],
        entities: Optional[List[List[Any]]] = None,
    ) -> ExtractionResult:
        """Extract entities and/or relations from texts.

        Args:
            texts: List of text strings to process.
            entities: Optional pre-extracted entities per text (list-of-lists).
                When provided with relex mode, entities are passed as input_spans
                so the model skips its own NER.

        Returns:
            ExtractionResult with entities and relations.
        """
        self._load_model()

        if not texts:
            return ExtractionResult(entities=[], relations=[])

        if self.has_relex:
            return self._run_ner_and_relex(texts, entities)
        else:
            return self._run_ner_only(texts)

    def extract_single(
        self, text: str, entities: Optional[List[Any]] = None,
    ) -> ExtractionResult:
        """Single-text convenience wrapper."""
        ents = [entities] if entities is not None else None
        return self.extract([text], entities=ents)

    @staticmethod
    def _convert_raw_entity(ent: Dict) -> Dict:
        """Handle generated_label[0] fallback for entity label."""
        label = ent.get("label", "")
        generated_label = ent.get("generated_label", None)
        if generated_label is not None:
            label = generated_label[0]
        return {
            "text": ent["text"],
            "label": label,
            "start": ent.get("start", 0),
            "end": ent.get("end", 0),
            "score": ent.get("score", 0.0),
        }

    def _run_ner_only(self, texts: List[str]) -> ExtractionResult:
        """NER-only mode using GLiNER inference."""
        entities_batch = self._model.inference(
            texts=texts,
            labels=self.labels,
            threshold=self.threshold,
            flat_ner=self.flat_ner,
            batch_size=self.batch_size,
        )

        all_mentions: List[List[EntityMention]] = []
        for raw_entities in entities_batch:
            mentions = []
            for ent in raw_entities:
                converted = self._convert_raw_entity(ent)
                mentions.append(EntityMention(
                    text=converted["text"],
                    label=converted["label"],
                    start=converted["start"],
                    end=converted["end"],
                    score=converted["score"],
                ))
            all_mentions.append(mentions)

        empty_relations = [[] for _ in texts]
        return ExtractionResult(entities=all_mentions, relations=empty_relations)

    def _run_ner_and_relex(
        self,
        texts: List[str],
        entities: Optional[List[List[Any]]] = None,
    ) -> ExtractionResult:
        """NER + relex mode (or relex-only with pre-extracted entities)."""
        input_spans: Optional[List[List[Dict[str, Any]]]] = None
        if entities is not None:
            input_spans = mentions_to_gliner_spans(entities)

        inference_kwargs: Dict[str, Any] = {
            "texts": texts,
            "labels": self.labels,
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

        all_entities: List[List[EntityMention]] = []
        all_relations: List[List[Relation]] = []

        for i in range(len(texts)):
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
                ))
            all_entities.append(mentions)

            # Convert relations — GLiNER-relex uses entity_idx references
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
                    head_label=head_ent.get("label", ""),
                    tail_label=tail_ent.get("label", ""),
                ))
            all_relations.append(relations)

        return ExtractionResult(entities=all_entities, relations=all_relations)
