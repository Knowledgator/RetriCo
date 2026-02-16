"""Entity linker processor — links entity mentions to a knowledge base via GLinker."""

from typing import Any, Dict, List, Optional
import logging

from ..core.base import BaseProcessor
from ..core.registry import processor_registry
from ..models.document import Chunk
from ..models.entity import EntityMention

logger = logging.getLogger(__name__)


class EntityLinkerProcessor(BaseProcessor):
    """Link entity mentions to a reference knowledge base using GLinker.

    Supports three usage modes:
    1. Pre-built executor: user passes their own GLinker DAGExecutor
    2. From parameters: grapsit initializes GLinker from config
    3. End-to-end: GLinker does its own NER + linking when no upstream entities

    Config keys:
        executor: pre-built GLinker DAGExecutor (takes priority)
        model: GLinker model name (default: "knowledgator/gliner-linker-large-v1.0")
        threshold: linking threshold (default: 0.5)
        entities: KB entities — file path (str) or list of dicts
        neo4j_uri/user/password/database: load KB from Neo4j
    """

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.model = config_dict.get("model", "knowledgator/gliner-linker-large-v1.0")
        self.threshold = config_dict.get("threshold", 0.5)
        self._executor = config_dict.get("executor", None)
        self._kb_loaded = self._executor is not None

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
        entities_cfg = self.config_dict.get("entities")
        if entities_cfg is not None:
            # File path or list of dicts
            self._executor.load_entities(entities_cfg)
            return

        # Load from Neo4j if configured
        neo4j_uri = self.config_dict.get("neo4j_uri")
        if neo4j_uri:
            from ..store.neo4j_store import Neo4jGraphStore
            store = Neo4jGraphStore(
                uri=neo4j_uri,
                user=self.config_dict.get("neo4j_user", "neo4j"),
                password=self.config_dict.get("neo4j_password", "password"),
                database=self.config_dict.get("neo4j_database", "neo4j"),
            )
            raw_entities = store.get_all_entities()
            store.close()

            kb_entities = []
            for ent in raw_entities:
                kb_entities.append({
                    "entity_id": ent.get("id", ""),
                    "label": ent.get("label", ""),
                    "description": ent.get("entity_type", ""),
                })
            if kb_entities:
                self._executor.load_entities(kb_entities)

    def __call__(
        self,
        *,
        entities=None,
        chunks: List[Chunk] = None,
        texts: List[str] = None,
        query: str = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Link entities to the knowledge base.

        Three input modes:
        A. entities is list-of-lists (build pipeline with upstream NER)
        B. entities is flat list (query pipeline with upstream parser)
        C. No entities — GLinker does end-to-end NER + linking
        """
        # Detect input mode
        is_query_mode = query is not None
        has_entities = entities is not None and len(entities) > 0

        if has_entities:
            is_flat = _is_flat_entity_list(entities)
        else:
            is_flat = False

        # Determine texts to process
        if is_query_mode and not has_entities:
            # Query mode, end-to-end
            process_texts = [query]
        elif is_query_mode and has_entities:
            process_texts = [query]
        elif chunks:
            process_texts = [c.text if isinstance(c, Chunk) else c for c in chunks]
        elif texts:
            process_texts = texts
        else:
            process_texts = []

        if not process_texts:
            if is_query_mode:
                return {"query": query or "", "entities": entities or []}
            return {"entities": entities or [], "chunks": chunks or []}

        # Build GLinker input
        glinker_input = {"texts": process_texts}

        if has_entities:
            self._ensure_executor(external_entities=True)
            if is_flat:
                # Flat list from query parser — wrap as single list
                glinker_input["entities"] = [_mentions_to_glinker_spans(entities)]
            else:
                # List-of-lists from NER
                glinker_input["entities"] = [
                    _mentions_to_glinker_spans(chunk_ents) for chunk_ents in entities
                ]
        else:
            self._ensure_executor(external_entities=False)

        # Execute GLinker
        result = self._executor.execute(glinker_input)
        l0_result = result.get("l0_result")

        # Parse results and update entity mentions
        if has_entities:
            linked_entities = self._update_existing_entities(
                entities, l0_result, is_flat
            )
        else:
            linked_entities = self._create_entities_from_result(
                l0_result, process_texts, chunks
            )

        if is_query_mode:
            if is_flat or not has_entities:
                flat_result = linked_entities if isinstance(linked_entities, list) and not any(
                    isinstance(e, list) for e in linked_entities
                ) else (linked_entities[0] if linked_entities else [])
            else:
                flat_result = linked_entities[0] if linked_entities else []
            return {"query": query, "entities": flat_result}
        else:
            return {"entities": linked_entities, "chunks": chunks or []}

    def _update_existing_entities(
        self, entities, l0_result, is_flat: bool
    ):
        """Update existing EntityMentions with linked_entity_id from GLinker results."""
        if l0_result is None:
            return entities

        # Build a lookup from mention text -> entity_id
        linked_map = _build_linked_map(l0_result)

        if is_flat:
            return _apply_links(entities, linked_map)
        else:
            return [_apply_links(chunk_ents, linked_map) for chunk_ents in entities]

    def _create_entities_from_result(
        self, l0_result, texts: List[str], chunks: Optional[List[Chunk]]
    ) -> List[List[EntityMention]]:
        """Create EntityMention objects from GLinker end-to-end results."""
        if l0_result is None:
            return [[] for _ in texts]

        all_entities = []
        raw_entities = l0_result.entities if hasattr(l0_result, "entities") else []

        # l0_result.entities is List[List[L0Entity]] — one inner list per text
        # If already grouped by text, use that directly
        if raw_entities and isinstance(raw_entities[0], list):
            for text_idx, text_ents in enumerate(raw_entities):
                chunk_mentions = []
                for ent in text_ents:
                    chunk_mentions.append(self._l0_entity_to_mention(
                        ent, text_idx, chunks
                    ))
                all_entities.append(chunk_mentions)
            # Pad if needed
            while len(all_entities) < len(texts):
                all_entities.append([])
            return all_entities

        # Fallback: flat list, group by text_idx
        glinker_entities = _flatten_l0_entities(l0_result)

        # Group entities by text index
        per_text: Dict[int, List[EntityMention]] = {i: [] for i in range(len(texts))}

        for ent in glinker_entities:
            text_idx = getattr(ent, "text_idx", 0) if hasattr(ent, "text_idx") else 0
            if text_idx >= len(texts):
                text_idx = 0

            mention_text = getattr(ent, "mention_text", "") or getattr(ent, "text", "")
            label = getattr(ent, "label", "") or getattr(ent, "type", "")
            start = getattr(ent, "start", 0)
            end = getattr(ent, "end", 0)
            score = getattr(ent, "score", 0.0)

            linked_id = None
            linked_ent = getattr(ent, "linked_entity", None)
            if linked_ent is not None:
                linked_id = getattr(linked_ent, "entity_id", None)
                if score == 0.0:
                    score = getattr(linked_ent, "score", 0.0)

            chunk_id = ""
            if chunks and text_idx < len(chunks):
                chunk_id = chunks[text_idx].id if isinstance(chunks[text_idx], Chunk) else ""

            mention = EntityMention(
                text=mention_text,
                label=label,
                start=start,
                end=end,
                score=score,
                chunk_id=chunk_id,
                linked_entity_id=linked_id,
            )
            per_text[text_idx].append(mention)

        for i in range(len(texts)):
            all_entities.append(per_text.get(i, []))

        return all_entities


    def _l0_entity_to_mention(self, ent, text_idx: int, chunks) -> EntityMention:
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

        chunk_id = ""
        if chunks and text_idx < len(chunks):
            chunk_id = chunks[text_idx].id if isinstance(chunks[text_idx], Chunk) else ""

        return EntityMention(
            text=mention_text,
            label=label,
            start=start,
            end=end,
            score=score,
            chunk_id=chunk_id,
            linked_entity_id=linked_id,
        )


def _is_flat_entity_list(entities) -> bool:
    """Check if entities is a flat list (query mode) vs list-of-lists (build mode)."""
    if not entities:
        return True
    first = entities[0]
    return not isinstance(first, list)


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
    """Flatten l0_result.entities which is a list-of-lists into a flat list."""
    raw = l0_result.entities if hasattr(l0_result, "entities") else []
    if not raw:
        return []
    # l0_result.entities is List[List[L0Entity]] — one inner list per input text
    flat = []
    for item in raw:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def _build_linked_map(l0_result) -> Dict[str, str]:
    """Build mention_text -> entity_id map from l0_result."""
    linked_map = {}
    for ent in _flatten_l0_entities(l0_result):
        mention_text = getattr(ent, "mention_text", None) or getattr(ent, "text", "")
        # Try linked_entity first (l3_linked stage)
        linked_ent = getattr(ent, "linked_entity", None)
        if linked_ent is not None:
            entity_id = getattr(linked_ent, "entity_id", None)
            if entity_id:
                linked_map[mention_text] = entity_id
                continue
        # Fallback: use top candidate if available (l2_found stage)
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


@processor_registry.register("entity_linker")
def create_entity_linker(config_dict: dict, pipeline=None):
    return EntityLinkerProcessor(config_dict, pipeline)
