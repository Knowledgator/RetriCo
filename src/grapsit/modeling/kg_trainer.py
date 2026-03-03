"""KG trainer — train PyKEEN KG embedding model."""

from typing import Any, Dict
import logging

from ..core.base import BaseProcessor
from ..core.registry import modeling_registry

logger = logging.getLogger(__name__)


class KGTrainerProcessor(BaseProcessor):
    """Train a PyKEEN knowledge graph embedding model.

    Config keys:
        model: str — PyKEEN model name (default: "RotatE").
        embedding_dim: int — embedding dimension (default: 128).
        epochs: int — training epochs (default: 100).
        batch_size: int — training batch size (default: 256).
        lr: float — learning rate (default: 0.001).
        negative_sampler: str — negative sampling strategy (default: "basic").
        num_negatives: int — negatives per positive (default: 1).
        device: str — "cpu" or "cuda" (default: "cpu").
        use_tqdm: bool — show progress bar (default: True).
    """

    default_inputs = {"training": "triple_reader_result.training"}
    default_output = "kg_trainer_result"

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.model_name = config_dict.get("model", "RotatE")
        self.embedding_dim = config_dict.get("embedding_dim", 128)
        self.epochs = config_dict.get("epochs", 100)
        self.batch_size = config_dict.get("batch_size", 256)
        self.lr = config_dict.get("lr", 0.001)
        self.negative_sampler = config_dict.get("negative_sampler", "basic")
        self.num_negatives = config_dict.get("num_negatives", 1)
        self.device = config_dict.get("device", "cpu")
        self.use_tqdm = config_dict.get("use_tqdm", True)

    def __call__(self, **kwargs) -> Dict[str, Any]:
        try:
            from pykeen.pipeline import pipeline
        except ImportError:
            raise ImportError(
                "pykeen is required for KG modeling. "
                "Install with: pip install pykeen"
            )

        training = kwargs.get("training")
        if training is None:
            raise ValueError("'training' TriplesFactory is required.")

        validation = kwargs.get("validation")
        testing = kwargs.get("testing")

        pipeline_kwargs = dict(
            training=training,
            model=self.model_name,
            model_kwargs={"embedding_dim": self.embedding_dim},
            epochs=self.epochs,
            training_kwargs={
                "batch_size": self.batch_size,
                "use_tqdm": self.use_tqdm,
            },
            optimizer_kwargs={"lr": self.lr},
            negative_sampler=self.negative_sampler,
            negative_sampler_kwargs={"num_negs_per_pos": self.num_negatives},
            device=self.device,
        )

        if validation is not None:
            pipeline_kwargs["validation"] = validation
        if testing is not None:
            pipeline_kwargs["testing"] = testing

        result = pipeline(**pipeline_kwargs)

        metrics = {}
        if result.metric_results is not None:
            try:
                metrics = result.metric_results.to_dict()
            except Exception:
                metrics = {"raw": str(result.metric_results)}

        logger.info(
            f"Training complete: model={self.model_name}, "
            f"epochs={self.epochs}, dim={self.embedding_dim}"
        )

        return {
            "model": result.model,
            "result": result,
            "metrics": metrics,
        }


@modeling_registry.register("kg_trainer")
def create_kg_trainer(config_dict: dict, pipeline=None):
    return KGTrainerProcessor(config_dict, pipeline)
