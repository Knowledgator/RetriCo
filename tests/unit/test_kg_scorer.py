"""Tests for KG scorer processor."""

import json
import os
import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch

from grapsit.modeling.kg_scorer import KGScorerProcessor, _extract_entity_labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeModel:
    """Fake PyKEEN model for testing scoring."""

    def __init__(self):
        self.entity_representations = [MagicMock()]
        self.relation_representations = [MagicMock()]

    def score_hrt(self, hrt_batch):
        batch_size = hrt_batch.shape[0]
        return torch.randn(batch_size, 1)

    def eval(self):
        pass

    def to(self, device):
        return self

    def load_state_dict(self, state_dict):
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def scorer_config():
    return {
        "top_k": 5,
        "predict_tails": True,
        "predict_heads": False,
        "device": "cpu",
    }


@pytest.fixture
def entity_to_id():
    return {"Einstein": 0, "Ulm": 1, "Physics": 2}


@pytest.fixture
def relation_to_id():
    return {"BORN_IN": 0, "FIELD": 1}


@pytest.fixture
def fake_model():
    return FakeModel()


@pytest.fixture
def simple_subgraph():
    """A dict-based subgraph with relations."""
    return {
        "entities": [
            {"label": "Einstein", "id": "e1"},
            {"label": "Ulm", "id": "e2"},
        ],
        "relations": [
            {"head_label": "Einstein", "relation_type": "BORN_IN", "tail_label": "Ulm", "score": 0.9},
        ],
    }


@pytest.fixture
def pydantic_subgraph():
    """A Pydantic Subgraph object."""
    from grapsit.models.graph import Subgraph
    from grapsit.models.entity import Entity
    from grapsit.models.relation import Relation

    return Subgraph(
        entities=[
            Entity(id="e1", label="Einstein", entity_type="person"),
            Entity(id="e2", label="Ulm", entity_type="location"),
        ],
        relations=[
            Relation(
                head_text="Einstein", tail_text="Ulm",
                relation_type="BORN_IN", score=0.9,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKGScorer:
    def test_score_existing_triples(self, scorer_config, fake_model, entity_to_id, relation_to_id, simple_subgraph):
        proc = KGScorerProcessor(scorer_config)

        result = proc(
            model=fake_model,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            subgraph=simple_subgraph,
            entities=[],
        )

        assert "scored_triples" in result
        assert len(result["scored_triples"]) == 1
        assert result["scored_triples"][0]["head"] == "Einstein"
        assert result["scored_triples"][0]["relation"] == "BORN_IN"
        assert result["scored_triples"][0]["tail"] == "Ulm"
        assert "score" in result["scored_triples"][0]

    def test_predict_tails(self, scorer_config, fake_model, entity_to_id, relation_to_id):
        proc = KGScorerProcessor(scorer_config)

        result = proc(
            model=fake_model,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            entities=[{"text": "Einstein"}],
        )

        assert "predictions" in result
        assert isinstance(result["predictions"], list)

    def test_predict_heads(self, entity_to_id, relation_to_id, fake_model):
        config = {"top_k": 3, "predict_tails": False, "predict_heads": True, "device": "cpu"}
        proc = KGScorerProcessor(config)

        result = proc(
            model=fake_model,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            entities=[{"text": "Ulm"}],
        )

        assert "predictions" in result
        for pred in result["predictions"]:
            assert pred["type"] == "predicted_head"

    def test_no_subgraph(self, scorer_config, fake_model, entity_to_id, relation_to_id):
        proc = KGScorerProcessor(scorer_config)

        result = proc(
            model=fake_model,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            entities=[],
        )

        assert result["scored_triples"] == []
        assert result["subgraph"] is None

    def test_score_with_pydantic_subgraph(self, scorer_config, fake_model, entity_to_id, relation_to_id, pydantic_subgraph):
        proc = KGScorerProcessor(scorer_config)

        result = proc(
            model=fake_model,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            subgraph=pydantic_subgraph,
            entities=[],
        )

        assert len(result["scored_triples"]) == 1

    def test_enrich_pydantic_subgraph(self, fake_model, entity_to_id, relation_to_id, pydantic_subgraph):
        config = {"top_k": 3, "predict_tails": True, "predict_heads": False, "device": "cpu"}
        proc = KGScorerProcessor(config)

        result = proc(
            model=fake_model,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            subgraph=pydantic_subgraph,
            entities=[{"text": "Einstein"}],
        )

        assert result["subgraph"] is not None
        # Original relation + predicted ones
        assert len(result["subgraph"].relations) >= 1

    def test_unknown_entity_skipped(self, scorer_config, fake_model, entity_to_id, relation_to_id):
        proc = KGScorerProcessor(scorer_config)

        result = proc(
            model=fake_model,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            entities=[{"text": "UnknownEntity"}],
        )

        assert result["predictions"] == []

    def test_score_threshold(self, fake_model, entity_to_id, relation_to_id):
        config = {
            "top_k": 5,
            "predict_tails": True,
            "predict_heads": False,
            "score_threshold": 999.0,  # Very high threshold
            "device": "cpu",
        }
        proc = KGScorerProcessor(config)

        result = proc(
            model=fake_model,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id,
            entities=[{"text": "Einstein"}],
        )

        assert result["predictions"] == []

    def test_entity_label_extraction(self):
        # String inputs
        assert _extract_entity_labels(["Einstein", "Bohr"]) == ["Einstein", "Bohr"]

        # Dict inputs
        assert _extract_entity_labels([{"text": "Einstein"}, {"label": "Bohr"}]) == ["Einstein", "Bohr"]

        # Object inputs
        obj = MagicMock()
        obj.text = "Einstein"
        assert _extract_entity_labels([obj]) == ["Einstein"]

        # Empty handling
        assert _extract_entity_labels([{"text": ""}, {"label": ""}]) == []

    def test_default_config(self):
        proc = KGScorerProcessor({})
        assert proc.model_name == "RotatE"
        assert proc.embedding_dim == 128
        assert proc.top_k == 10
        assert proc.predict_tails is True
        assert proc.predict_heads is False
        assert proc.device == "cpu"

    def test_processor_registration(self):
        from grapsit.core.registry import processor_registry
        assert "kg_scorer" in processor_registry._factories


class TestKGScorerLoadFromDisk:
    def test_load_mappings_from_disk(self, tmp_path):
        """Test that entity_to_id and relation_to_id are loaded from disk."""
        entity_to_id = {"Einstein": 0, "Ulm": 1, "Physics": 2}
        relation_to_id = {"BORN_IN": 0, "FIELD": 1}

        model_dir = str(tmp_path / "kg_model")
        os.makedirs(model_dir)

        with open(os.path.join(model_dir, "entity_to_id.json"), "w") as f:
            json.dump(entity_to_id, f)
        with open(os.path.join(model_dir, "relation_to_id.json"), "w") as f:
            json.dump(relation_to_id, f)
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump({
                "model_name": "RotatE",
                "entity_dim": 64,
                "relation_dim": 64,
                "num_entities": 3,
                "num_relations": 2,
            }, f)
        # Save a dummy state dict
        torch.save({"dummy": torch.zeros(1)}, os.path.join(model_dir, "model.pt"))

        # Mock pykeen imports inside _load_from_disk
        fake = FakeModel()
        mock_model_cls = MagicMock(return_value=fake)
        mock_get_cls = MagicMock(return_value=mock_model_cls)
        mock_tf = MagicMock()
        mock_tf.from_labeled_triples.return_value = MagicMock()

        with patch.dict("sys.modules", {
            "pykeen": MagicMock(),
            "pykeen.models": MagicMock(get_model_cls=mock_get_cls),
            "pykeen.triples": MagicMock(TriplesFactory=mock_tf),
        }):
            config = {"model_path": model_dir, "device": "cpu", "top_k": 3}
            proc = KGScorerProcessor(config)
            proc._load_from_disk()

            assert proc._entity_to_id == entity_to_id
            assert proc._relation_to_id == relation_to_id

    def test_invalid_model_path_raises(self):
        config = {"model_path": "/nonexistent/path", "device": "cpu"}
        proc = KGScorerProcessor(config)

        # Mock pykeen imports so ImportError doesn't fire
        with patch.dict("sys.modules", {
            "pykeen": MagicMock(),
            "pykeen.models": MagicMock(),
            "pykeen.triples": MagicMock(),
        }):
            with pytest.raises(ValueError, match="Invalid model_path"):
                proc._load_from_disk()

    def test_no_model_path_raises(self):
        config = {"device": "cpu"}
        proc = KGScorerProcessor(config)

        with patch.dict("sys.modules", {
            "pykeen": MagicMock(),
            "pykeen.models": MagicMock(),
            "pykeen.triples": MagicMock(),
        }):
            with pytest.raises(ValueError, match="Invalid model_path"):
                proc._load_from_disk()


class TestKGScorerTripleQueries:
    """Tests for triple_queries mode (store-based resolution)."""

    def _make_store(self):
        """Create a mock graph store with test data."""
        store = MagicMock()
        store.get_entity_by_label.side_effect = lambda label: {
            "einstein": {"id": "e1", "label": "Einstein", "entity_type": "person"},
            "ulm": {"id": "e2", "label": "Ulm", "entity_type": "location"},
            "physics": {"id": "e3", "label": "Physics", "entity_type": "field"},
        }.get(label.lower())

        store.get_entity_by_id.side_effect = lambda eid: {
            "e1": {"id": "e1", "label": "Einstein", "entity_type": "person"},
            "e2": {"id": "e2", "label": "Ulm", "entity_type": "location"},
            "e3": {"id": "e3", "label": "Physics", "entity_type": "field"},
        }.get(eid)

        store.get_entity_relations.side_effect = lambda eid: {
            "e1": [
                {"head": "Einstein", "tail": "Ulm", "type": "BORN_IN", "score": 0.9,
                 "head_id": "e1", "tail_id": "e2"},
                {"head": "Einstein", "tail": "Physics", "type": "FIELD", "score": 0.8,
                 "head_id": "e1", "tail_id": "e3"},
            ],
            "e2": [
                {"head": "Einstein", "tail": "Ulm", "type": "BORN_IN", "score": 0.9,
                 "head_id": "e1", "tail_id": "e2"},
            ],
            "e3": [
                {"head": "Einstein", "tail": "Physics", "type": "FIELD", "score": 0.8,
                 "head_id": "e1", "tail_id": "e3"},
            ],
        }.get(eid, [])
        return store

    def test_triple_query_head_known(self):
        config = {"top_k": 10, "predict_tails": False, "predict_heads": False, "device": "cpu"}
        proc = KGScorerProcessor(config)
        proc._store = self._make_store()

        result = proc(triple_queries=[
            {"head": "Einstein", "relation": "BORN_IN", "tail": None},
        ])

        assert "scored_triples" in result
        assert "subgraph" in result
        assert len(result["scored_triples"]) == 1
        assert result["scored_triples"][0]["head"] == "Einstein"
        assert result["scored_triples"][0]["tail"] == "Ulm"
        assert result["subgraph"] is not None
        assert len(result["subgraph"].entities) >= 1

    def test_triple_query_tail_known(self):
        config = {"top_k": 10, "predict_tails": False, "predict_heads": False, "device": "cpu"}
        proc = KGScorerProcessor(config)
        proc._store = self._make_store()

        result = proc(triple_queries=[
            {"head": None, "relation": "BORN_IN", "tail": "Ulm"},
        ])

        assert len(result["scored_triples"]) == 1
        assert result["scored_triples"][0]["head"] == "Einstein"
        assert result["scored_triples"][0]["tail"] == "Ulm"

    def test_triple_query_no_relation_filter(self):
        config = {"top_k": 10, "predict_tails": False, "predict_heads": False, "device": "cpu"}
        proc = KGScorerProcessor(config)
        proc._store = self._make_store()

        result = proc(triple_queries=[
            {"head": "Einstein", "relation": None, "tail": None},
        ])

        # Should return all relations for Einstein
        assert len(result["scored_triples"]) == 2

    def test_triple_query_head_and_tail(self):
        config = {"top_k": 10, "predict_tails": False, "predict_heads": False, "device": "cpu"}
        proc = KGScorerProcessor(config)
        proc._store = self._make_store()

        result = proc(triple_queries=[
            {"head": "Einstein", "relation": None, "tail": "Ulm"},
        ])

        assert len(result["scored_triples"]) == 1
        assert result["scored_triples"][0]["relation"] == "BORN_IN"

    def test_triple_query_unknown_entity(self):
        config = {"top_k": 10, "predict_tails": False, "predict_heads": False, "device": "cpu"}
        proc = KGScorerProcessor(config)
        proc._store = self._make_store()

        result = proc(triple_queries=[
            {"head": "UnknownPerson", "relation": "BORN_IN", "tail": None},
        ])

        assert result["scored_triples"] == []
        assert len(result["subgraph"].entities) == 0

    def test_multiple_triple_queries(self):
        config = {"top_k": 10, "predict_tails": False, "predict_heads": False, "device": "cpu"}
        proc = KGScorerProcessor(config)
        proc._store = self._make_store()

        result = proc(triple_queries=[
            {"head": "Einstein", "relation": "BORN_IN", "tail": None},
            {"head": "Einstein", "relation": "FIELD", "tail": None},
        ])

        assert len(result["scored_triples"]) == 2

    def test_triple_query_score_threshold(self):
        config = {"top_k": 10, "predict_tails": False, "predict_heads": False,
                  "device": "cpu", "score_threshold": 0.85}
        proc = KGScorerProcessor(config)
        proc._store = self._make_store()

        result = proc(triple_queries=[
            {"head": "Einstein", "relation": None, "tail": None},
        ])

        # BORN_IN has score 0.9 (passes), FIELD has score 0.8 (filtered)
        assert len(result["scored_triples"]) == 1
        assert result["scored_triples"][0]["relation"] == "BORN_IN"

    def test_triple_query_with_kge_scoring(self):
        config = {"top_k": 10, "predict_tails": False, "predict_heads": False, "device": "cpu"}
        proc = KGScorerProcessor(config)
        proc._store = self._make_store()
        proc._model = FakeModel()
        proc._entity_to_id = {"Einstein": 0, "Ulm": 1, "Physics": 2}
        proc._relation_to_id = {"BORN_IN": 0, "FIELD": 1}

        result = proc(triple_queries=[
            {"head": "Einstein", "relation": "BORN_IN", "tail": None},
        ])

        assert len(result["scored_triples"]) == 1
        # Score should come from KGE model, not store
        assert "score" in result["scored_triples"][0]

    def test_triple_query_subgraph_has_entities_and_relations(self):
        config = {"top_k": 10, "predict_tails": False, "predict_heads": False, "device": "cpu"}
        proc = KGScorerProcessor(config)
        proc._store = self._make_store()

        result = proc(triple_queries=[
            {"head": "Einstein", "relation": "BORN_IN", "tail": None},
        ])

        subgraph = result["subgraph"]
        entity_labels = {e.label for e in subgraph.entities}
        assert "Einstein" in entity_labels
        assert "Ulm" in entity_labels
        assert len(subgraph.relations) >= 1

    def test_backward_compat_existing_subgraph(self):
        """Existing subgraph mode still works when no triple_queries."""
        config = {"top_k": 5, "predict_tails": True, "predict_heads": False, "device": "cpu"}
        proc = KGScorerProcessor(config)
        fake = FakeModel()

        result = proc(
            model=fake,
            entity_to_id={"Einstein": 0, "Ulm": 1, "Physics": 2},
            relation_to_id={"BORN_IN": 0, "FIELD": 1},
            subgraph={
                "entities": [{"label": "Einstein", "id": "e1"}],
                "relations": [{"head_label": "Einstein", "relation_type": "BORN_IN", "tail_label": "Ulm"}],
            },
            entities=[],
        )

        assert len(result["scored_triples"]) == 1
