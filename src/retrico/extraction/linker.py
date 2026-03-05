"""Entity linker engine — wraps GLinker for standalone entity linking."""

from typing import Any, Dict, List, Optional
import logging

from ..models.entity import EntityMention

logger = logging.getLogger(__name__)


class EntityLinkerEngine:
    """Standalone entity linking via GLinker.

    Links entity mentions to a reference knowledge base using GLinker.

    Args:
        model: GLinker model name.
        threshold: Linking threshold.
        executor: Pre-built GLinker DAGExecutor (takes priority).
        entities: KB entities — file path (str) or list of dicts.
    """

    def __init__(
        self,
        *,
        model: str = "knowledgator/gliner-linker-large-v1.0",
        threshold: float = 0.5,
        executor: Any = None,
        entities: Any = None,
    ):
        self.model = model
        self.threshold = threshold
        self._executor = executor
        self._kb_loaded = executor is not None
        self._kb_entities = entities

    def _ensure_executor(self, external_entities: bool = False):
        """Lazily create or configure the GLinker executor."""
        if self._executor is not None:
            if not self._kb_loaded:
                self._load_kb_entities()
                self._kb_loaded = True
            return

        from glinker import ProcessorFactory as GLinkerFactory

        kwargs = {
            "model_name": self.model,
            "threshold": self.threshold,
        }
        if external_entities:
            kwargs["external_entities"] = True

        self._executor = GLinkerFactory.create_simple(**kwargs)
        self._load_kb_entities()
        self._kb_loaded = True

    def _load_kb_entities(self):
        """Load knowledge base entities into the executor."""
        if self._kb_entities is not None:
            self._executor.load_entities(self._kb_entities)

    def link(
        self,
        texts: List[str],
        entities: Optional[List[List[Any]]] = None,
    ) -> List[List[EntityMention]]:
        """Link entities in texts to the knowledge base.

        Args:
            texts: Texts to process.
            entities: Optional pre-extracted entities (list-of-lists).

        Returns:
            List-of-lists of EntityMention with linked_entity_id set.
        """
        if not texts:
            return []

        glinker_input = {"texts": texts}

        if entities is not None:
            self._ensure_executor(external_entities=True)
            glinker_input["entities"] = [
                _mentions_to_glinker_spans(chunk_ents)
                for chunk_ents in entities
            ]
        else:
            self._ensure_executor(external_entities=False)

        result = self._executor.execute(glinker_input)
        l0_result = result.get("l0_result")

        if entities is not None:
            linked_map = _build_linked_map(l0_result)
            return [_apply_links(chunk_ents, linked_map) for chunk_ents in entities]
        else:
            return _create_entities_from_result(l0_result, texts)

    def link_single(
        self, text: str, entities: Optional[List[Any]] = None,
    ) -> List[EntityMention]:
        """Single-text convenience wrapper."""
        ents = [entities] if entities is not None else None
        result = self.link([text], entities=ents)
        return result[0] if result else []


def _mentions_to_glinker_spans(mentions) -> List[Dict[str, Any]]:
    """Convert EntityMention objects to GLinker span format."""
    spans = []
    for m in mentions:
        if isinstance(m, EntityMention):
            spans.append({
                "text": m.text,
                "label": m.label,
                "start": m.start,
                "end": m.end,
            })
        elif isinstance(m, dict):
            spans.append({
                "text": m.get("text", ""),
                "label": m.get("label", ""),
                "start": m.get("start", 0),
                "end": m.get("end", 0),
            })
        else:
            spans.append({
                "text": getattr(m, "text", ""),
                "label": getattr(m, "label", ""),
                "start": getattr(m, "start", 0),
                "end": getattr(m, "end", 0),
            })
    return spans


def _flatten_l0_entities(l0_result) -> list:
    """Flatten l0_result.entities into a flat list."""
    raw = l0_result.entities if hasattr(l0_result, "entities") else []
    if not raw:
        return []
    flat = []
    for item in raw:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def _build_linked_map(l0_result) -> Dict[str, str]:
    """Build mention_text -> entity_id map from l0_result."""
    if l0_result is None:
        return {}
    linked_map = {}
    for ent in _flatten_l0_entities(l0_result):
        mention_text = getattr(ent, "mention_text", None) or getattr(ent, "text", "")
        linked_ent = getattr(ent, "linked_entity", None)
        if linked_ent is not None:
            entity_id = getattr(linked_ent, "entity_id", None)
            if entity_id:
                linked_map[mention_text] = entity_id
                continue
        candidates = getattr(ent, "candidates", None)
        if candidates and len(candidates) > 0:
            entity_id = getattr(candidates[0], "entity_id", None)
            if entity_id:
                linked_map[mention_text] = entity_id
    return linked_map


def _apply_links(mentions, linked_map: Dict[str, str]):
    """Apply linked_entity_id to a list of EntityMentions."""
    result = []
    for m in mentions:
        if isinstance(m, EntityMention):
            entity_id = linked_map.get(m.text)
            if entity_id:
                m = m.model_copy(update={"linked_entity_id": entity_id})
            result.append(m)
        elif isinstance(m, dict):
            entity_id = linked_map.get(m.get("text", ""))
            if entity_id:
                m = {**m, "linked_entity_id": entity_id}
            result.append(EntityMention(**m) if isinstance(m, dict) else m)
        else:
            result.append(m)
    return result


def _create_entities_from_result(
    l0_result, texts: List[str]
) -> List[List[EntityMention]]:
    """Create EntityMention objects from GLinker end-to-end results."""
    if l0_result is None:
        return [[] for _ in texts]

    raw_entities = l0_result.entities if hasattr(l0_result, "entities") else []

    if raw_entities and isinstance(raw_entities[0], list):
        all_entities = []
        for text_ents in raw_entities:
            chunk_mentions = []
            for ent in text_ents:
                chunk_mentions.append(_l0_entity_to_mention(ent))
            all_entities.append(chunk_mentions)
        while len(all_entities) < len(texts):
            all_entities.append([])
        return all_entities

    # Fallback: flat list, group by text_idx
    glinker_entities = _flatten_l0_entities(l0_result)
    per_text: Dict[int, List[EntityMention]] = {i: [] for i in range(len(texts))}

    for ent in glinker_entities:
        text_idx = getattr(ent, "text_idx", 0) if hasattr(ent, "text_idx") else 0
        if text_idx >= len(texts):
            text_idx = 0
        per_text[text_idx].append(_l0_entity_to_mention(ent))

    result = []
    for i in range(len(texts)):
        result.append(per_text.get(i, []))
    return result


def _l0_entity_to_mention(ent) -> EntityMention:
    """Convert a single GLinker L0Entity to EntityMention."""
    mention_text = getattr(ent, "mention_text", "") or getattr(ent, "text", "")
    label = getattr(ent, "label", "") or getattr(ent, "type", "")
    start = getattr(ent, "mention_start", 0) or getattr(ent, "start", 0)
    end = getattr(ent, "mention_end", 0) or getattr(ent, "end", 0)
    score = 0.0

    linked_id = None
    linked_ent = getattr(ent, "linked_entity", None)
    if linked_ent is not None:
        linked_id = getattr(linked_ent, "entity_id", None)
        score = getattr(linked_ent, "confidence", 0.0) or 0.0
    elif getattr(ent, "candidates", None):
        linked_id = getattr(ent.candidates[0], "entity_id", None)

    return EntityMention(
        text=mention_text,
        label=label,
        start=start,
        end=end,
        score=score,
        linked_entity_id=linked_id,
    )
