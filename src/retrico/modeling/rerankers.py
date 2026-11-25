"""Reranker model abstractions with lazy loading.

A reranker scores ``(query, passage)`` pairs. Backends only produce scores;
ordering, ``top_k`` and thresholding are handled by the pipeline processor
(:mod:`retrico.query.chunk_reranker`), so switching backends never changes
how results are post-processed.

Built-in backends (selected via the ``method`` config key):

- ``"cross_encoder"``: sentence-transformers ``CrossEncoder``
- ``"gliclass"``: GLiClass zero-shot classification pipeline

Custom backends can be added with :func:`register_reranker`.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type, Union
import inspect
import logging
import math

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract base for query–passage rerankers."""

    @abstractmethod
    def score(self, query: str, passages: List[str]) -> List[float]:
        """Score each passage for relevance to the query.

        Args:
            query: The query text.
            passages: Passage texts to score.

        Returns:
            One float per passage (higher = more relevant), in input order.
        """
        ...


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


class CrossEncoderReranker(BaseReranker):
    """sentence-transformers ``CrossEncoder`` reranker with lazy loading.

    Args:
        model_name: HuggingFace model name or path.
        device: Device to load the model on (e.g. "cpu", "cuda").
        batch_size: Batch size for scoring.
        max_length: Max token length of a (query, passage) pair.
        activation: ``None`` to return the model's scores as-is, or
            ``"sigmoid"`` to map raw logits into [0, 1] (useful for MS MARCO
            style models so that thresholds are comparable across backends).
    """

    def __init__(
        self,
        *,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: Optional[int] = None,
        activation: Optional[str] = None,
    ):
        if activation not in (None, "sigmoid"):
            raise ValueError(
                f"Unknown activation {activation!r}. Expected None or 'sigmoid'."
            )
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.activation = activation
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError:
                raise ImportError(
                    "sentence-transformers package required for cross-encoder "
                    "reranking. Install with: pip install sentence-transformers"
                )
            kwargs: Dict[str, Any] = {}
            if self.device is not None:
                kwargs["device"] = self.device
            if self.max_length is not None:
                kwargs["max_length"] = self.max_length
            self._model = CrossEncoder(self.model_name, **kwargs)
            logger.info(f"CrossEncoder reranker loaded: {self.model_name}")

    def score(self, query: str, passages: List[str]) -> List[float]:
        if not passages:
            return []
        self._ensure_model()
        pairs = [(query, p) for p in passages]
        raw = self._model.predict(
            pairs, batch_size=self.batch_size, show_progress_bar=False
        )
        scores = [float(s) for s in raw]
        if self.activation == "sigmoid":
            scores = [_sigmoid(s) for s in scores]
        return scores


class GLiClassReranker(BaseReranker):
    """GLiClass zero-shot classification reranker with lazy loading.

    Each passage is classified against a single label — the query — and the
    label's probability is used as the relevance score.

    Args:
        model_name: HuggingFace model name or path.
        device: Device to run on. Defaults to ``"cuda:0"`` when available,
            otherwise ``"cpu"``.
        batch_size: Batch size for scoring.
        max_length: Max token length of the model input.
        prompt: Task description passed to the GLiClass pipeline.
    """

    DEFAULT_PROMPT = "Does the passage answer the query?"

    def __init__(
        self,
        *,
        model_name: str = "knowledgator/gliclass-v4-edge",
        device: Optional[str] = None,
        batch_size: int = 8,
        max_length: int = 2048,
        prompt: Optional[str] = DEFAULT_PROMPT,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.prompt = prompt
        self._pipeline = None

    def _resolve_device(self) -> str:
        if self.device is not None:
            return self.device
        try:
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _ensure_model(self):
        if self._pipeline is None:
            try:
                from gliclass import GLiClassModel, ZeroShotClassificationPipeline
            except ImportError:
                raise ImportError(
                    "gliclass package required for GLiClass reranking. "
                    "Install with: pip install gliclass"
                )
            device = self._resolve_device()
            model = GLiClassModel.from_pretrained(self.model_name)
            self._pipeline = ZeroShotClassificationPipeline(
                model,
                self.model_name,  # loads the tokenizer
                classification_type="multi-label",
                max_length=self.max_length,
                device=device,
            )
            logger.info(
                f"GLiClass reranker loaded: {self.model_name} (device={device})"
            )

    def score(self, query: str, passages: List[str]) -> List[float]:
        if not passages:
            return []
        self._ensure_model()
        kwargs: Dict[str, Any] = {"threshold": 0.0, "batch_size": self.batch_size}
        if self.prompt:
            kwargs["prompt"] = self.prompt
        results = self._pipeline(passages, [query], **kwargs)

        scores: List[float] = []
        for preds in results:
            # With a single label and threshold=0.0, each passage yields
            # [{"label": query, "score": p}]; guard against empty output.
            if preds:
                scores.append(float(preds[0]["score"]))
            else:
                scores.append(0.0)
        return scores


# -- registry / factory ------------------------------------------------------

_RERANKERS: Dict[str, Callable[..., BaseReranker]] = {
    "cross_encoder": CrossEncoderReranker,
    "gliclass": GLiClassReranker,
}


def register_reranker(
    name: str, factory: Union[Type[BaseReranker], Callable[..., BaseReranker]]
) -> None:
    """Register a custom reranker backend.

    ``factory`` is called with the keyword arguments from the reranker config
    that match its signature (unless it accepts ``**kwargs``, in which case it
    receives all non-pipeline keys). It must return a :class:`BaseReranker`.

    Example::

        class MyReranker(BaseReranker):
            def __init__(self, *, model_name="my-model"):
                ...
            def score(self, query, passages):
                ...

        register_reranker("mine", MyReranker)
        builder.chunk_reranker(method="mine", model_name="my-model")
    """
    _RERANKERS[name] = factory


def list_rerankers() -> List[str]:
    """Return the names of all registered reranker backends."""
    return list(_RERANKERS.keys())


# Keys consumed by the processor, never forwarded to a backend.
_PROCESSOR_KEYS = {"method", "top_k", "score_threshold"}


def _filter_kwargs(factory: Callable, config: Dict[str, Any]) -> Dict[str, Any]:
    target = factory.__init__ if inspect.isclass(factory) else factory
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return dict(config)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(config)
    accepted = {n for n in params if n != "self"}
    return {k: v for k, v in config.items() if k in accepted}


def create_reranker(config: Dict[str, Any]) -> BaseReranker:
    """Create a reranker from a config dict.

    The ``method`` key selects the backend (default ``"cross_encoder"``).
    Only keys accepted by the backend's constructor are forwarded; pipeline
    keys (``top_k``, ``score_threshold``) and unknown keys are ignored.
    ``None`` values are dropped so that backend defaults apply.
    """
    config = dict(config)
    method = config.get("method", "cross_encoder")
    if method not in _RERANKERS:
        raise ValueError(
            f"Unknown reranker method {method!r}. "
            f"Available: {list_rerankers()}"
        )
    factory = _RERANKERS[method]
    backend_cfg = {
        k: v for k, v in config.items()
        if k not in _PROCESSOR_KEYS and v is not None
    }
    reranker = factory(**_filter_kwargs(factory, backend_cfg))
    if not isinstance(reranker, BaseReranker):
        raise TypeError(
            f"Reranker factory for {method!r} returned {type(reranker).__name__}, "
            "expected a BaseReranker instance."
        )
    return reranker
