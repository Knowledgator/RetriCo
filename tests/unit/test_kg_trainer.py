"""Tests for KG trainer processor."""

import sys
import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Mock pykeen before importing
# ---------------------------------------------------------------------------

mock_pykeen = MagicMock()
mock_pipeline_module = MagicMock()


class FakeMetricResults:
    def to_dict(self):
        return {"hits_at_10": 0.85, "mean_rank": 12.5}


class FakePipelineResult:
    def __init__(self):
        self.model = MagicMock()
        self.metric_results = FakeMetricResults()


def fake_pipeline(**kwargs):
    return FakePipelineResult()


mock_pipeline_module.pipeline = fake_pipeline
mock_pykeen.pipeline = mock_pipeline_module

sys.modules.setdefault("pykeen", mock_pykeen)
sys.modules.setdefault("pykeen.pipeline", mock_pipeline_module)

from retrico.modeling.kg_trainer import KGTrainerProcessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def trainer_config():
    return {
        "model": "RotatE",
        "embedding_dim": 64,
        "epochs": 10,
        "batch_size": 128,
        "lr": 0.01,
        "negative_sampler": "basic",
        "num_negatives": 2,
        "device": "cpu",
        "use_tqdm": False,
    }


@pytest.fixture
def mock_training_factory():
    return MagicMock()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKGTrainer:
    def test_basic_training(self, trainer_config, mock_training_factory):
        proc = KGTrainerProcessor(trainer_config)

        result = proc(training=mock_training_factory)

        assert "model" in result
        assert "result" in result
        assert "metrics" in result
        assert result["metrics"]["hits_at_10"] == 0.85

    def test_training_with_validation(self, trainer_config, mock_training_factory):
        proc = KGTrainerProcessor(trainer_config)
        val_factory = MagicMock()
        test_factory = MagicMock()

        result = proc(
            training=mock_training_factory,
            validation=val_factory,
            testing=test_factory,
        )

        assert "model" in result
        assert "metrics" in result

    def test_missing_training_raises(self, trainer_config):
        proc = KGTrainerProcessor(trainer_config)

        with pytest.raises(ValueError, match="training"):
            proc()

    def test_default_config(self):
        proc = KGTrainerProcessor({})
        assert proc.model_name == "RotatE"
        assert proc.embedding_dim == 128
        assert proc.epochs == 100
        assert proc.batch_size == 256
        assert proc.lr == 0.001
        assert proc.device == "cpu"

    def test_custom_config(self, trainer_config):
        proc = KGTrainerProcessor(trainer_config)
        assert proc.model_name == "RotatE"
        assert proc.embedding_dim == 64
        assert proc.epochs == 10
        assert proc.batch_size == 128
        assert proc.lr == 0.01
        assert proc.num_negatives == 2

    def test_metrics_fallback(self, trainer_config, mock_training_factory):
        """Test metrics extraction when to_dict() fails."""
        class BadMetrics:
            def to_dict(self):
                raise Exception("no dict")
            def __str__(self):
                return "metrics_string"

        original_pipeline = mock_pipeline_module.pipeline

        def pipeline_with_bad_metrics(**kwargs):
            result = FakePipelineResult()
            result.metric_results = BadMetrics()
            return result

        mock_pipeline_module.pipeline = pipeline_with_bad_metrics
        try:
            proc = KGTrainerProcessor(trainer_config)
            result = proc(training=mock_training_factory)
            assert "raw" in result["metrics"]
        finally:
            mock_pipeline_module.pipeline = original_pipeline

    def test_processor_registration(self):
        from retrico.core.registry import processor_registry
        assert "kg_trainer" in processor_registry._factories
