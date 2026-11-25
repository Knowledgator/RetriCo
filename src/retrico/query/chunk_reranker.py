"""Chunk reranker — reorder retrieved chunks by relevance to the query."""

from typing import Any, Dict, List, Optional
import logging

from ..core.base import BaseProcessor
from ..core.registry import query_registry
from ..models.document import Chunk
from ..models.graph import Subgraph

logger = logging.getLogger(__name__)


class ChunkRerankerProcessor(BaseProcessor):
    """Rerank the chunks of a subgraph against the query.

    Scores every chunk with a reranker backend (see
    :mod:`retrico.modeling.rerankers`), stores the score in
    ``chunk.metadata["rerank_score"]``, sorts chunks by score (descending,
    stable — ties keep their retrieval order), then applies
    ``score_threshold`` and ``top_k``. Entities and relations pass through
    unchanged.

    Config keys:
        method: str — reranker backend: "cross_encoder" (default), "gliclass",
            or any name added with ``register_reranker``.
        model_name: str — backend model (backend default if omitted).
        top_k: int — max chunks to keep (0 = keep all, default: 0).
        score_threshold: float — drop chunks scoring below this
            (default: None). Score scale depends on the backend.
        device, batch_size, max_length — forwarded to the backend.
        prompt: str — GLiClass task description.
        activation: str — cross-encoder score activation (None or "sigmoid").
    """

    default_inputs = {
        "query": "$input.query",
        "subgraph": "chunk_result.subgraph",
    }
    default_output = "rerank_result"

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.top_k: int = config_dict.get("top_k", 0) or 0
        self.score_threshold: Optional[float] = config_dict.get("score_threshold")
        self._reranker = None

    def _ensure_reranker(self):
        if self._reranker is None:
            from ..modeling.rerankers import create_reranker
            self._reranker = create_reranker(self.config_dict)

    def __call__(self, *, query: str, subgraph: Any, **kwargs) -> Dict[str, Any]:
        """Rerank chunks of the subgraph.

        Returns:
            {"subgraph": Subgraph} with reranked chunks.
        """
        if isinstance(subgraph, dict):
            subgraph = Subgraph(**subgraph)
        elif not isinstance(subgraph, Subgraph):
            subgraph = Subgraph()

        chunks: List[Chunk] = list(subgraph.chunks)
        if not chunks:
            return {"subgraph": subgraph}

        self._ensure_reranker()
        scores = self._reranker.score(query, [c.text for c in chunks])
        if len(scores) != len(chunks):
            raise ValueError(
                f"Reranker returned {len(scores)} scores for {len(chunks)} chunks."
            )

        scored = []
        for chunk, score in zip(chunks, scores):
            score = float(score)
            new_chunk = chunk.model_copy(
                update={"metadata": {**chunk.metadata, "rerank_score": score}}
            )
            scored.append((score, new_chunk))

        # sorted() is stable, so equal scores keep retrieval order.
        scored.sort(key=lambda pair: -pair[0])

        if self.score_threshold is not None:
            scored = [p for p in scored if p[0] >= self.score_threshold]
        if self.top_k > 0:
            scored = scored[:self.top_k]

        logger.debug(
            f"Reranked {len(chunks)} chunks -> kept {len(scored)}"
        )

        return {
            "subgraph": Subgraph(
                entities=subgraph.entities,
                relations=subgraph.relations,
                chunks=[c for _, c in scored],
            )
        }


@query_registry.register("chunk_reranker")
def create_chunk_reranker(config_dict: dict, pipeline=None):
    return ChunkRerankerProcessor(config_dict, pipeline)
