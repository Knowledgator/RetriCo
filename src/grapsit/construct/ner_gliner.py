"""GLiNER-based named entity recognition processor."""

from typing import Any, Dict, List
import logging

from ..core.base import BaseProcessor
from ..core.registry import processor_registry
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

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.model_name: str = config_dict.get("model", "urchade/gliner_multi-v2.1")
        self.labels: List[str] = config_dict.get("labels", [])
        self.threshold: float = config_dict.get("threshold", 0.3)
        self.batch_size: int = config_dict.get("batch_size", 8)
        self.device: str = config_dict.get("device", "cpu")
        self.flat_ner: bool = config_dict.get("flat_ner", True)
        self._model = None

    def _load_model(self):
        if self._model is None:
            from gliner import GLiNER
            logger.info(f"Loading GLiNER model: {self.model_name}")
            self._model = GLiNER.from_pretrained(self.model_name)
            if self.device != "cpu":
                self._model = self._model.to(self.device)

    def __call__(self, *, chunks: List[Chunk] = None, texts: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Run NER on chunks using batched inference.

        Returns:
            {"entities": List[List[EntityMention]], "chunks": chunks}

        entities is a list-of-lists aligned with chunks — one list of mentions per chunk.
        """
        self._load_model()

        if chunks is None:
            chunks = []
        if texts is None:
            texts = [c.text for c in chunks]

        # Batched inference — returns List[List[Dict]] aligned with texts
        entities_batch = self._model.inference(
            texts=texts,
            labels=self.labels,
            threshold=self.threshold,
            flat_ner=self.flat_ner,
            batch_size=self.batch_size,
        )

        all_mentions: List[List[EntityMention]] = []
        for i, raw_entities in enumerate(entities_batch):
            chunk_id = chunks[i].id if i < len(chunks) else ""
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
            all_mentions.append(mentions)

        return {"entities": all_mentions, "chunks": chunks}


@processor_registry.register("ner_gliner")
def create_ner_gliner(config_dict: dict, pipeline=None):
    return NERGLiNERProcessor(config_dict, pipeline)
