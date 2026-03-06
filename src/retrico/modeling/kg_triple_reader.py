"""KG triple reader — reads triples from graph store or TSV for PyKEEN training."""

from typing import Any, Dict
import logging

from ..core.base import BaseProcessor
from ..core.registry import modeling_registry
from ..store.pool import resolve_from_pool_or_create

logger = logging.getLogger(__name__)


class KGTripleReaderProcessor(BaseProcessor):
    """Read triples from a graph store or TSV file and create PyKEEN TriplesFactory.

    Config keys:
        source: str — "graph_store" or "tsv" (default: "graph_store").
        tsv_path: str — path to TSV file (required if source="tsv").
        train_ratio: float — fraction for training (default: 0.8).
        val_ratio: float — fraction for validation (default: 0.1).
        test_ratio: float — fraction for test (default: 0.1).
        random_seed: int — random seed for splitting (default: 42).
        Store params: store_type, neo4j_uri, etc.
    """

    default_inputs = {}
    default_output = "triple_reader_result"

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self._store = None
        self.source = config_dict.get("source", "graph_store")
        self.tsv_path = config_dict.get("tsv_path")
        self.train_ratio = config_dict.get("train_ratio", 0.8)
        self.val_ratio = config_dict.get("val_ratio", 0.1)
        self.test_ratio = config_dict.get("test_ratio", 0.1)
        self.random_seed = config_dict.get("random_seed", 42)

    def _ensure_store(self):
        if self._store is None:
            self._store = resolve_from_pool_or_create(self.config_dict, "graph")

    def __call__(self, **kwargs) -> Dict[str, Any]:
        try:
            from pykeen.triples import TriplesFactory
        except ImportError:
            raise ImportError(
                "pykeen is required for KG modeling. "
                "Install with: pip install pykeen"
            )

        if self.source == "tsv":
            if not self.tsv_path:
                raise ValueError("tsv_path required when source='tsv'")
            triples_factory = TriplesFactory.from_path(self.tsv_path)
        else:
            self._ensure_store()
            raw_triples = self._store.get_all_triples()
            if not raw_triples:
                raise ValueError("No triples found in graph store.")
            triples_factory = TriplesFactory.from_labeled_triples(
                triples=_triples_to_array(raw_triples),
            )

        training, validation, testing = triples_factory.split(
            [self.train_ratio, self.val_ratio, self.test_ratio],
            random_state=self.random_seed,
        )

        logger.info(
            f"Loaded {triples_factory.num_triples} triples: "
            f"train={training.num_triples}, val={validation.num_triples}, "
            f"test={testing.num_triples}"
        )

        return {
            "triples_factory": triples_factory,
            "training": training,
            "validation": validation,
            "testing": testing,
            "entity_to_id": triples_factory.entity_to_id,
            "relation_to_id": triples_factory.relation_to_id,
            "triple_count": triples_factory.num_triples,
        }


def _triples_to_array(triples):
    """Convert list of (head, rel, tail) tuples to numpy string array."""
    import numpy as np
    return np.array(triples, dtype=str)


@modeling_registry.register("kg_triple_reader")
def create_kg_triple_reader(config_dict: dict, pipeline=None):
    return KGTripleReaderProcessor(config_dict, pipeline)
