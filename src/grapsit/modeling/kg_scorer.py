"""KG scorer — score existing triples and predict missing links at query time."""

from typing import Any, Dict, List, Optional
import json
import logging
import os

from ..core.base import BaseProcessor
from ..core.registry import processor_registry

logger = logging.getLogger(__name__)


class KGScorerProcessor(BaseProcessor):
    """Score triples and predict missing links using a trained KG embedding model.

    Can load a model from disk (model_path) or receive one from the pipeline context.
    Scores existing triples in a subgraph and optionally predicts new links.

    When ``triple_queries`` are provided (from a tool-calling parser), acts as a
    universal retriever: looks up entities/relations in the graph store, scores
    candidate triples with KGE, and builds a Subgraph.

    Config keys:
        model_path: str — directory with saved model weights + mappings.
        model_name: str — PyKEEN model class name (default: "RotatE").
        embedding_dim: int — embedding dimension (default: 128).
        top_k: int — number of top predictions per entity (default: 10).
        score_threshold: float — minimum score for predictions (default: None).
        predict_tails: bool — predict tail entities (default: True).
        predict_heads: bool — predict head entities (default: False).
        device: str — "cpu" or "cuda" (default: "cpu").

    Store config (for triple_queries mode):
        neo4j_uri, neo4j_user, neo4j_password, neo4j_database, store_type,
        falkordb_host, falkordb_port, falkordb_graph, etc.
    """

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self._model = None
        self._entity_to_id = None
        self._relation_to_id = None
        self._store = None
        self.model_path = config_dict.get("model_path")
        self.model_name = config_dict.get("model_name", "RotatE")
        self.embedding_dim = config_dict.get("embedding_dim", 128)
        self.top_k = config_dict.get("top_k", 10)
        self.score_threshold = config_dict.get("score_threshold")
        self.predict_tails = config_dict.get("predict_tails", True)
        self.predict_heads = config_dict.get("predict_heads", False)
        self.device = config_dict.get("device", "cpu")

    def _ensure_store(self):
        """Lazily create the graph store via ``create_store(config)``."""
        if self._store is None:
            from ..store import create_store
            self._store = create_store(self.config_dict)

    def _load_from_disk(self):
        """Load model, entity_to_id, and relation_to_id from disk."""
        try:
            import torch
            from pykeen.models import get_model_cls
        except ImportError:
            raise ImportError(
                "pykeen and torch are required for KG scoring. "
                "Install with: pip install pykeen"
            )

        if not self.model_path or not os.path.isdir(self.model_path):
            raise ValueError(f"Invalid model_path: {self.model_path}")

        with open(os.path.join(self.model_path, "entity_to_id.json")) as f:
            self._entity_to_id = json.load(f)
        with open(os.path.join(self.model_path, "relation_to_id.json")) as f:
            self._relation_to_id = json.load(f)

        # Load metadata if available
        metadata_path = os.path.join(self.model_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = json.load(f)
            model_name = metadata.get("model_name", self.model_name)
            embedding_dim = metadata.get("entity_dim", self.embedding_dim)
        else:
            model_name = self.model_name
            embedding_dim = self.embedding_dim

        model_cls = get_model_cls(model_name)

        from pykeen.triples import TriplesFactory
        import numpy as np

        # Create a dummy TriplesFactory for model initialization
        dummy_triples = np.array(
            [[list(self._entity_to_id.keys())[0],
              list(self._relation_to_id.keys())[0],
              list(self._entity_to_id.keys())[0]]],
            dtype=str,
        )
        tf = TriplesFactory.from_labeled_triples(
            dummy_triples,
            entity_to_id=self._entity_to_id,
            relation_to_id=self._relation_to_id,
        )

        self._model = model_cls(
            triples_factory=tf,
            embedding_dim=embedding_dim,
        ).to(self.device)

        state_dict = torch.load(
            os.path.join(self.model_path, "model.pt"),
            map_location=self.device,
        )
        self._model.load_state_dict(state_dict)
        self._model.eval()

    def _resolve_triple_queries(self, triple_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve triple queries against the graph store and score with KGE.

        For each triple query (head?, relation?, tail?), look up matching
        entities and relations in the store, build candidate triples, and
        score them with the KG embedding model if available.

        Returns:
            {"scored_triples": [...], "predictions": [...], "subgraph": Subgraph}
        """
        from ..models.graph import Subgraph
        from ..models.entity import Entity
        from ..models.relation import Relation

        self._ensure_store()

        try:
            import torch
        except ImportError:
            torch = None

        all_entities = {}  # id -> entity dict
        all_relations = []
        scored_triples = []

        for tq in triple_queries:
            head_label = tq.get("head")
            relation_type = tq.get("relation")
            tail_label = tq.get("tail")

            # Look up head entity
            head_record = None
            if head_label:
                head_record = self._store.get_entity_by_label(head_label)
                if head_record:
                    all_entities[head_record["id"]] = head_record

            # Look up tail entity
            tail_record = None
            if tail_label:
                tail_record = self._store.get_entity_by_label(tail_label)
                if tail_record:
                    all_entities[tail_record["id"]] = tail_record

            # Find matching relations
            candidate_rels = []
            if head_record:
                rels = self._store.get_entity_relations(head_record["id"])
                for rel in rels:
                    matches = True
                    if relation_type:
                        rel_type_stored = rel.get("type", "")
                        rel_type_query = relation_type.upper().replace(" ", "_")
                        if rel_type_stored.upper() != rel_type_query:
                            matches = False
                    if tail_label and matches:
                        rel_tail = rel.get("tail", "")
                        if rel_tail.lower() != tail_label.lower():
                            matches = False
                    if matches:
                        candidate_rels.append(rel)
                        tail_id = rel.get("tail_id")
                        if tail_id and tail_id not in all_entities:
                            tail_ent = self._store.get_entity_by_id(tail_id)
                            if tail_ent:
                                all_entities[tail_id] = tail_ent

            elif tail_record:
                rels = self._store.get_entity_relations(tail_record["id"])
                for rel in rels:
                    matches = True
                    if relation_type:
                        rel_type_stored = rel.get("type", "")
                        rel_type_query = relation_type.upper().replace(" ", "_")
                        if rel_type_stored.upper() != rel_type_query:
                            matches = False
                    if head_label and matches:
                        rel_head = rel.get("head", "")
                        if rel_head.lower() != head_label.lower():
                            matches = False
                    if matches:
                        candidate_rels.append(rel)
                        head_id = rel.get("head_id")
                        if head_id and head_id not in all_entities:
                            head_ent = self._store.get_entity_by_id(head_id)
                            if head_ent:
                                all_entities[head_id] = head_ent

            # Score candidates with KGE if model is available
            for rel in candidate_rels:
                head_l = rel.get("head", "")
                tail_l = rel.get("tail", "")
                rel_t = rel.get("type", "")
                score = rel.get("score", 0.0) or 0.0

                if self._model is not None and self._entity_to_id and torch is not None:
                    h_id = self._entity_to_id.get(head_l)
                    r_id = self._relation_to_id.get(rel_t)
                    t_id = self._entity_to_id.get(tail_l)
                    if h_id is not None and r_id is not None and t_id is not None:
                        hrt = torch.tensor([[h_id, r_id, t_id]], dtype=torch.long).to(self.device)
                        with torch.no_grad():
                            score = self._model.score_hrt(hrt).item()

                if self.score_threshold is not None and score < self.score_threshold:
                    continue

                scored_triples.append({
                    "head": head_l,
                    "relation": rel_t,
                    "tail": tail_l,
                    "score": score,
                })
                all_relations.append(rel)

        # Sort by score descending, limit to top_k
        scored_triples.sort(key=lambda x: x["score"], reverse=True)
        scored_triples = scored_triples[:self.top_k]

        # Build Subgraph
        sg_entities = []
        for ent_dict in all_entities.values():
            sg_entities.append(Entity(
                id=ent_dict.get("id", ""),
                label=ent_dict.get("label", ""),
                entity_type=ent_dict.get("entity_type", ""),
            ))

        sg_relations = []
        for rel in all_relations:
            sg_relations.append(Relation(
                head_text=rel.get("head", ""),
                tail_text=rel.get("tail", ""),
                relation_type=rel.get("type", ""),
                score=rel.get("score", 0.0) or 0.0,
            ))

        subgraph = Subgraph(entities=sg_entities, relations=sg_relations)

        # Predict missing links if configured
        predictions = []
        if (self.predict_tails or self.predict_heads) and self._model is not None and torch is not None:
            entity_labels = [e.get("label", "") for e in all_entities.values()]
            entity_labels = [l for l in entity_labels if l]
            id_to_entity = {v: k for k, v in self._entity_to_id.items()}
            id_to_relation = {v: k for k, v in self._relation_to_id.items()}
            predictions = self._predict_links(
                entity_labels, id_to_entity, id_to_relation, torch,
            )
            if predictions:
                subgraph = self._enrich_subgraph(subgraph, predictions)

        return {
            "scored_triples": scored_triples,
            "predictions": predictions,
            "subgraph": subgraph,
        }

    def __call__(self, **kwargs) -> Dict[str, Any]:
        try:
            import torch
        except ImportError:
            raise ImportError(
                "torch is required for KG scoring. "
                "Install with: pip install torch"
            )

        # If triple_queries are provided, use store-based resolution
        triple_queries = kwargs.get("triple_queries")
        if triple_queries:
            # Load KGE model if available
            model = kwargs.get("model")
            entity_to_id = kwargs.get("entity_to_id")
            relation_to_id = kwargs.get("relation_to_id")
            if model is not None:
                self._model = model
                self._entity_to_id = entity_to_id or {}
                self._relation_to_id = relation_to_id or {}
            elif self._model is None and self.model_path:
                try:
                    self._load_from_disk()
                except Exception:
                    logger.info("No KGE model loaded, scoring with graph store scores only")
            return self._resolve_triple_queries(triple_queries)

        # Get model from context or load from disk
        model = kwargs.get("model")
        entity_to_id = kwargs.get("entity_to_id")
        relation_to_id = kwargs.get("relation_to_id")

        if model is not None:
            self._model = model
            self._entity_to_id = entity_to_id or {}
            self._relation_to_id = relation_to_id or {}
        elif self._model is None:
            self._load_from_disk()

        entities = kwargs.get("entities", [])
        subgraph = kwargs.get("subgraph")

        id_to_entity = {v: k for k, v in self._entity_to_id.items()}
        id_to_relation = {v: k for k, v in self._relation_to_id.items()}

        # Score existing triples in subgraph
        scored_triples = []
        if subgraph is not None:
            relations = []
            if hasattr(subgraph, "relations"):
                relations = subgraph.relations or []
            elif isinstance(subgraph, dict):
                relations = subgraph.get("relations", [])

            for rel in relations:
                head_label = (
                    _get_field(rel, "head_label")
                    or _get_field(rel, "head_text")
                    or _get_field(rel, "head")
                )
                tail_label = (
                    _get_field(rel, "tail_label")
                    or _get_field(rel, "tail_text")
                    or _get_field(rel, "tail")
                )
                rel_type = _get_field(rel, "relation_type") or _get_field(rel, "type")

                if not all([head_label, tail_label, rel_type]):
                    continue

                h_id = self._entity_to_id.get(head_label)
                r_id = self._relation_to_id.get(rel_type)
                t_id = self._entity_to_id.get(tail_label)

                if h_id is None or r_id is None or t_id is None:
                    continue

                hrt = torch.tensor([[h_id, r_id, t_id]], dtype=torch.long).to(self.device)
                with torch.no_grad():
                    score = self._model.score_hrt(hrt).item()

                scored_triples.append({
                    "head": head_label,
                    "relation": rel_type,
                    "tail": tail_label,
                    "score": score,
                })

        # Predict missing links
        predictions = []
        if self.predict_tails or self.predict_heads:
            entity_labels = _extract_entity_labels(entities)
            predictions = self._predict_links(
                entity_labels, id_to_entity, id_to_relation, torch,
            )

        # Enrich subgraph with predictions
        if subgraph is not None and predictions:
            subgraph = self._enrich_subgraph(subgraph, predictions)

        logger.info(
            f"Scored {len(scored_triples)} triples, "
            f"predicted {len(predictions)} links"
        )

        return {
            "scored_triples": scored_triples,
            "predictions": predictions,
            "subgraph": subgraph,
        }

    def _predict_links(
        self,
        entity_labels: List[str],
        id_to_entity: Dict[int, str],
        id_to_relation: Dict[int, str],
        torch_module,
    ) -> List[Dict[str, Any]]:
        """Predict missing links for given entities."""
        predictions = []
        num_entities = len(self._entity_to_id)
        num_relations = len(self._relation_to_id)

        for label in entity_labels:
            e_id = self._entity_to_id.get(label)
            if e_id is None:
                continue

            for r_id in range(num_relations):
                if self.predict_tails:
                    # Score (entity, relation, ?) for all possible tails
                    heads = torch_module.full((num_entities, 1), e_id, dtype=torch_module.long)
                    rels = torch_module.full((num_entities, 1), r_id, dtype=torch_module.long)
                    tails = torch_module.arange(num_entities).unsqueeze(1)
                    hrt = torch_module.cat([heads, rels, tails], dim=1).to(self.device)

                    with torch_module.no_grad():
                        scores = self._model.score_hrt(hrt).squeeze()

                    top_k_indices = scores.topk(min(self.top_k, num_entities)).indices
                    for idx in top_k_indices:
                        score = scores[idx.item()].item()
                        if self.score_threshold is not None and score < self.score_threshold:
                            continue
                        tail_label = id_to_entity.get(idx.item(), str(idx.item()))
                        if tail_label == label:
                            continue
                        predictions.append({
                            "head": label,
                            "relation": id_to_relation.get(r_id, str(r_id)),
                            "tail": tail_label,
                            "score": score,
                            "type": "predicted_tail",
                        })

                if self.predict_heads:
                    # Score (?, relation, entity) for all possible heads
                    heads = torch_module.arange(num_entities).unsqueeze(1)
                    rels = torch_module.full((num_entities, 1), r_id, dtype=torch_module.long)
                    tails = torch_module.full((num_entities, 1), e_id, dtype=torch_module.long)
                    hrt = torch_module.cat([heads, rels, tails], dim=1).to(self.device)

                    with torch_module.no_grad():
                        scores = self._model.score_hrt(hrt).squeeze()

                    top_k_indices = scores.topk(min(self.top_k, num_entities)).indices
                    for idx in top_k_indices:
                        score = scores[idx.item()].item()
                        if self.score_threshold is not None and score < self.score_threshold:
                            continue
                        head_label = id_to_entity.get(idx.item(), str(idx.item()))
                        if head_label == label:
                            continue
                        predictions.append({
                            "head": head_label,
                            "relation": id_to_relation.get(r_id, str(r_id)),
                            "tail": label,
                            "score": score,
                            "type": "predicted_head",
                        })

        # Sort by score descending and limit
        predictions.sort(key=lambda x: x["score"], reverse=True)
        return predictions[:self.top_k]

    def _enrich_subgraph(self, subgraph, predictions):
        """Add predicted relations to the subgraph."""
        from ..models.graph import Subgraph
        from ..models.relation import Relation

        if isinstance(subgraph, Subgraph):
            for pred in predictions:
                subgraph.relations.append(
                    Relation(
                        head_text=pred["head"],
                        tail_text=pred["tail"],
                        relation_type=pred["relation"],
                        score=pred["score"],
                    )
                )
        return subgraph


def _get_field(obj, field):
    """Get a field from a dict or object."""
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)


def _extract_entity_labels(entities) -> List[str]:
    """Extract entity labels from various input formats."""
    labels = []
    for ent in entities:
        if isinstance(ent, str):
            labels.append(ent)
        elif isinstance(ent, dict):
            labels.append(ent.get("text") or ent.get("label", ""))
        elif hasattr(ent, "text"):
            labels.append(ent.text)
        elif hasattr(ent, "label"):
            labels.append(ent.label)
    return [l for l in labels if l]


@processor_registry.register("kg_scorer")
def create_kg_scorer(config_dict: dict, pipeline=None):
    return KGScorerProcessor(config_dict, pipeline)
