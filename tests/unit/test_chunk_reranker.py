"""Tests for chunk reranking: backends, processor, builder wiring, query_graph."""

import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from retrico.core.builders import RetriCoSearch, RetriCoFusedSearch
from retrico.core.registry import query_registry
from retrico.models.document import Chunk
from retrico.models.entity import Entity
from retrico.models.graph import Subgraph
from retrico.modeling import rerankers as rr
from retrico.modeling.rerankers import (
    BaseReranker,
    CrossEncoderReranker,
    GLiClassReranker,
    create_reranker,
    list_rerankers,
    register_reranker,
)
from retrico.query.chunk_reranker import ChunkRerankerProcessor


# -- helpers -----------------------------------------------------------------


class KeywordReranker(BaseReranker):
    """Deterministic fake: score = count of query words found in passage."""

    instances = 0

    def __init__(self, *, model_name: str = "fake", scale: float = 1.0):
        KeywordReranker.instances += 1
        self.model_name = model_name
        self.scale = scale
        self.calls = []

    def score(self, query, passages):
        self.calls.append((query, list(passages)))
        words = query.lower().split()
        return [self.scale * sum(w in p.lower() for w in words) for p in passages]


@pytest.fixture(autouse=True)
def _fake_backend():
    saved = dict(rr._RERANKERS)
    register_reranker("keyword", KeywordReranker)
    KeywordReranker.instances = 0
    yield
    rr._RERANKERS.clear()
    rr._RERANKERS.update(saved)


def _subgraph(*texts):
    return Subgraph(
        entities=[Entity(id="e1", label="router")],
        chunks=[Chunk(id=f"c{i}", text=t, metadata={"src": i}) for i, t in enumerate(texts)],
    )


# -- factory / registry -------------------------------------------------------


class TestFactory:
    def test_builtin_backends_listed(self):
        names = list_rerankers()
        assert "cross_encoder" in names and "gliclass" in names

    def test_default_method_is_cross_encoder(self):
        assert isinstance(create_reranker({}), CrossEncoderReranker)

    def test_gliclass_selected(self):
        r = create_reranker({"method": "gliclass", "model_name": "m", "prompt": "p"})
        assert isinstance(r, GLiClassReranker)
        assert r.model_name == "m" and r.prompt == "p"

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown reranker method"):
            create_reranker({"method": "nope"})

    def test_filters_processor_and_unknown_keys(self):
        r = create_reranker({
            "method": "cross_encoder", "model_name": "m", "top_k": 3,
            "score_threshold": 0.1, "prompt": "ignored", "neo4j_uri": "x",
        })
        assert r.model_name == "m"

    def test_none_values_use_backend_defaults(self):
        r = create_reranker({"method": "gliclass", "model_name": None, "max_length": None})
        assert r.model_name == "knowledgator/gliclass-v4-edge"
        assert r.max_length == 2048

    def test_custom_backend(self):
        r = create_reranker({"method": "keyword", "model_name": "x", "scale": 2.0})
        assert isinstance(r, KeywordReranker) and r.scale == 2.0

    def test_factory_must_return_base_reranker(self):
        register_reranker("bad", lambda **kw: object())
        with pytest.raises(TypeError, match="expected a BaseReranker"):
            create_reranker({"method": "bad"})

    def test_backends_load_lazily(self):
        # Construction must not import heavy packages or load models.
        r = create_reranker({"method": "cross_encoder"})
        assert r._model is None
        g = create_reranker({"method": "gliclass"})
        assert g._pipeline is None


# -- cross-encoder adapter ----------------------------------------------------


class TestCrossEncoderReranker:
    def _install_fake(self, monkeypatch, outputs):
        model = MagicMock()
        model.predict.return_value = outputs
        ctor = MagicMock(return_value=model)
        monkeypatch.setitem(
            sys.modules, "sentence_transformers",
            types.SimpleNamespace(CrossEncoder=ctor),
        )
        return ctor, model

    def test_scores_pairs_in_order(self, monkeypatch):
        ctor, model = self._install_fake(monkeypatch, [0.5, -1.0])
        r = CrossEncoderReranker(model_name="ce", device="cpu", max_length=256, batch_size=4)
        assert r.score("q", ["a", "b"]) == [0.5, -1.0]
        ctor.assert_called_once_with("ce", device="cpu", max_length=256)
        args, kwargs = model.predict.call_args
        assert args[0] == [("q", "a"), ("q", "b")]
        assert kwargs["batch_size"] == 4

    def test_sigmoid_activation(self, monkeypatch):
        self._install_fake(monkeypatch, [0.0, 100.0, -100.0])
        r = CrossEncoderReranker(activation="sigmoid")
        scores = r.score("q", ["a", "b", "c"])
        assert scores[0] == pytest.approx(0.5)
        assert scores[1] == pytest.approx(1.0)
        assert scores[2] == pytest.approx(0.0)

    def test_invalid_activation(self):
        with pytest.raises(ValueError, match="activation"):
            CrossEncoderReranker(activation="softmax")

    def test_empty_passages_skip_model(self, monkeypatch):
        ctor, _ = self._install_fake(monkeypatch, [])
        assert CrossEncoderReranker().score("q", []) == []
        ctor.assert_not_called()

    def test_missing_package_message(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        with pytest.raises(ImportError, match="pip install sentence-transformers"):
            CrossEncoderReranker().score("q", ["a"])


# -- GLiClass adapter ---------------------------------------------------------


class TestGLiClassReranker:
    def _install_fake(self, monkeypatch, results):
        pipe = MagicMock(return_value=results)
        pipe_ctor = MagicMock(return_value=pipe)
        model = object()
        model_cls = MagicMock()
        model_cls.from_pretrained.return_value = model
        monkeypatch.setitem(
            sys.modules, "gliclass",
            types.SimpleNamespace(
                GLiClassModel=model_cls,
                ZeroShotClassificationPipeline=pipe_ctor,
            ),
        )
        return model_cls, pipe_ctor, pipe, model

    def test_pipeline_init_and_call(self, monkeypatch):
        results = [[{"label": "q", "score": 0.9}], [{"label": "q", "score": 0.1}]]
        model_cls, pipe_ctor, pipe, model = self._install_fake(monkeypatch, results)
        r = GLiClassReranker(model_name="gc", device="cpu", max_length=512, batch_size=2, prompt="P?")

        assert r.score("q", ["a", "b"]) == [0.9, 0.1]

        model_cls.from_pretrained.assert_called_once_with("gc")
        pipe_ctor.assert_called_once_with(
            model, "gc", classification_type="multi-label", max_length=512, device="cpu",
        )
        args, kwargs = pipe.call_args
        assert args == (["a", "b"], ["q"])
        assert kwargs == {"threshold": 0.0, "batch_size": 2, "prompt": "P?"}

    def test_default_prompt(self, monkeypatch):
        _, _, pipe, _ = self._install_fake(monkeypatch, [[{"label": "q", "score": 0.3}]])
        GLiClassReranker(device="cpu").score("q", ["a"])
        assert pipe.call_args.kwargs["prompt"] == "Does the passage answer the query?"

    def test_no_prompt(self, monkeypatch):
        _, _, pipe, _ = self._install_fake(monkeypatch, [[{"label": "q", "score": 0.3}]])
        GLiClassReranker(device="cpu", prompt=None).score("q", ["a"])
        assert "prompt" not in pipe.call_args.kwargs

    def test_empty_prediction_scores_zero(self, monkeypatch):
        self._install_fake(monkeypatch, [[], [{"label": "q", "score": 0.4}]])
        assert GLiClassReranker(device="cpu").score("q", ["a", "b"]) == [0.0, 0.4]

    def test_model_loaded_once(self, monkeypatch):
        model_cls, _, _, _ = self._install_fake(monkeypatch, [[{"label": "q", "score": 0.3}]])
        r = GLiClassReranker(device="cpu")
        r.score("q", ["a"])
        r.score("q", ["a"])
        assert model_cls.from_pretrained.call_count == 1

    def test_missing_package_message(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "gliclass", None)
        with pytest.raises(ImportError, match="pip install gliclass"):
            GLiClassReranker(device="cpu").score("q", ["a"])


# -- processor ----------------------------------------------------------------


class TestChunkRerankerProcessor:
    def _proc(self, **cfg):
        return ChunkRerankerProcessor({"method": "keyword", **cfg})

    def test_registered(self):
        assert "chunk_reranker" in query_registry

    def test_sorts_by_score_and_records_it(self):
        sg = _subgraph("wifi 7 support", "reset router password", "router ssid")
        out = self._proc()(query="reset router password", subgraph=sg)["subgraph"]
        assert [c.id for c in out.chunks] == ["c1", "c2", "c0"]
        assert [c.metadata["rerank_score"] for c in out.chunks] == [3.0, 1.0, 0.0]
        # Existing metadata is preserved and input chunks are not mutated.
        assert out.chunks[0].metadata["src"] == 1
        assert "rerank_score" not in sg.chunks[0].metadata

    def test_ties_keep_retrieval_order(self):
        sg = _subgraph("router a", "nothing", "router b", "router c")
        out = self._proc()(query="router", subgraph=sg)["subgraph"]
        assert [c.id for c in out.chunks] == ["c0", "c2", "c3", "c1"]

    def test_top_k(self):
        sg = _subgraph("a", "router", "router password")
        out = self._proc(top_k=2)(query="router password", subgraph=sg)["subgraph"]
        assert [c.id for c in out.chunks] == ["c2", "c1"]

    def test_score_threshold(self):
        sg = _subgraph("a", "router", "router password")
        out = self._proc(score_threshold=1.0)(query="router password", subgraph=sg)["subgraph"]
        assert [c.id for c in out.chunks] == ["c2", "c1"]

    def test_threshold_then_top_k(self):
        sg = _subgraph("a", "router", "router password")
        out = self._proc(score_threshold=1.0, top_k=1)(query="router password", subgraph=sg)["subgraph"]
        assert [c.id for c in out.chunks] == ["c2"]

    def test_entities_and_relations_pass_through(self):
        sg = _subgraph("router")
        out = self._proc()(query="router", subgraph=sg)["subgraph"]
        assert out.entities == sg.entities and out.relations == sg.relations

    def test_empty_chunks_do_not_load_backend(self):
        proc = self._proc()
        out = proc(query="q", subgraph=Subgraph(entities=[Entity(id="e", label="x")]))
        assert out["subgraph"].chunks == []
        assert KeywordReranker.instances == 0

    def test_accepts_dict_subgraph(self):
        sg = _subgraph("router").model_dump()
        out = self._proc()(query="router", subgraph=sg)["subgraph"]
        assert out.chunks[0].metadata["rerank_score"] == 1.0

    def test_backend_reused_across_calls(self):
        proc = self._proc()
        proc(query="a", subgraph=_subgraph("a"))
        proc(query="b", subgraph=_subgraph("b"))
        assert KeywordReranker.instances == 1

    def test_score_count_mismatch_raises(self):
        register_reranker("broken", lambda **kw: _Broken())
        proc = ChunkRerankerProcessor({"method": "broken"})
        with pytest.raises(ValueError, match="1 scores for 2 chunks"):
            proc(query="q", subgraph=_subgraph("a", "b"))


class _Broken(BaseReranker):
    def score(self, query, passages):
        return [1.0]


# -- builder wiring -----------------------------------------------------------


def _nodes(config):
    return {n["id"]: n for n in config["nodes"]}


class TestBuilderWiring:
    def test_chunk_reranker_config(self):
        b = RetriCoSearch()
        b.chunk_reranker(method="gliclass", model_name="m", top_k=5, prompt="P?")
        assert b._reranker_config == {
            "method": "gliclass", "top_k": 5, "model_name": "m", "prompt": "P?",
        }

    def test_no_reranker_dag_unchanged(self):
        b = RetriCoSearch()
        b.community_retriever()
        b.reasoner(api_key="sk")
        nodes = _nodes(b.get_config())
        assert "chunk_reranker" not in nodes
        assert nodes["reasoner"]["requires"] == ["chunk_retriever"]
        assert nodes["reasoner"]["inputs"]["subgraph"]["source"] == "chunk_result"

    def test_standard_with_reranker_and_reasoner(self):
        b = RetriCoSearch()
        b.community_retriever()
        b.chunk_reranker(method="cross_encoder", top_k=3)
        b.reasoner(api_key="sk")
        config = b.get_config()
        ids = [n["id"] for n in config["nodes"]]
        assert ids == ["retriever", "chunk_retriever", "chunk_reranker", "reasoner"]

        nodes = _nodes(config)
        rer = nodes["chunk_reranker"]
        assert rer["processor"] == "chunk_reranker"
        assert rer["requires"] == ["chunk_retriever"]
        assert rer["inputs"]["subgraph"] == {"source": "chunk_result", "fields": "subgraph"}
        assert rer["inputs"]["query"] == {"source": "$input", "fields": "query"}
        assert rer["output"] == {"key": "rerank_result"}
        assert rer["config"]["top_k"] == 3

        assert nodes["reasoner"]["requires"] == ["chunk_reranker"]
        assert nodes["reasoner"]["inputs"]["subgraph"]["source"] == "rerank_result"

    def test_standard_with_reranker_no_reasoner(self):
        b = RetriCoSearch()
        b.community_retriever()
        b.chunk_reranker()
        ids = [n["id"] for n in b.get_config()["nodes"]]
        assert ids[-2:] == ["chunk_retriever", "chunk_reranker"]

    def test_multi_retriever_fusion(self):
        b = RetriCoSearch()
        b.community_retriever()
        b.chunk_embedding_retriever()
        b.chunk_reranker()
        nodes = _nodes(b.get_config())
        assert nodes["chunk_retriever"]["requires"] == ["fusion"]
        assert nodes["chunk_reranker"]["requires"] == ["chunk_retriever"]

    def test_kg_scored_pipeline(self):
        b = RetriCoSearch()
        b.query_parser(method="tool", api_key="sk")
        b.kg_scorer(model_path="m")
        b.chunk_reranker(method="gliclass")
        b.reasoner(api_key="sk")
        ids = [n["id"] for n in b.get_config()["nodes"]]
        assert ids == ["query_parser", "kg_scorer", "chunk_retriever", "chunk_reranker", "reasoner"]

    def test_fused_search(self):
        b1 = RetriCoSearch(name="a")
        b1.community_retriever()
        b2 = RetriCoSearch(name="b")
        b2.chunk_embedding_retriever()
        fused = RetriCoFusedSearch(b1, b2, strategy="rrf")
        fused.chunk_reranker(method="gliclass", top_k=4)
        fused.reasoner(api_key="sk")
        nodes = _nodes(fused.get_config())
        assert nodes["chunk_reranker"]["requires"] == ["chunk_retriever"]
        assert nodes["chunk_reranker"]["config"]["method"] == "gliclass"
        assert nodes["reasoner"]["inputs"]["subgraph"]["source"] == "rerank_result"

    def test_switching_backend_only_changes_config(self):
        def build(method):
            b = RetriCoSearch()
            b.community_retriever()
            b.chunk_reranker(method=method)
            return b.get_config()["nodes"]

        a, g = build("cross_encoder"), build("gliclass")
        strip = lambda ns: [{k: v for k, v in n.items() if k != "config"} for n in ns]
        assert strip(a) == strip(g)


# -- query_graph --------------------------------------------------------------


def _mock_executor(keys, subgraph):
    ctx = MagicMock()
    ctx.has.side_effect = lambda key: key in keys
    ctx.get.side_effect = lambda key: {"subgraph": subgraph}
    executor = MagicMock()
    executor.run.return_value = ctx
    return executor


class TestQueryGraph:
    def test_single_strategy_reranker(self):
        import retrico

        captured = {}
        reranked = _subgraph("router")

        def fake_build(self, verbose=False):
            captured["config"] = self.get_config()
            return _mock_executor({"chunk_result", "rerank_result"}, reranked)

        with patch.object(RetriCoSearch, "build", fake_build):
            result = retrico.query_graph(
                "q",
                retrieval_strategy="community",
                reranker="gliclass",
                reranker_kwargs={"top_k": 2, "model_name": "m"},
            )

        nodes = _nodes(captured["config"])
        assert nodes["chunk_reranker"]["config"]["method"] == "gliclass"
        assert nodes["chunk_reranker"]["config"]["top_k"] == 2
        assert result.subgraph is reranked

    def test_multi_strategy_reranker(self):
        import retrico

        captured = {}

        def fake_build(self, verbose=False):
            captured["config"] = self.get_config()
            return _mock_executor({"chunk_result", "rerank_result"}, Subgraph())

        with patch.object(RetriCoFusedSearch, "build", fake_build):
            retrico.query_graph(
                "q",
                retrieval_strategy=["community", "chunk_embedding"],
                reranker="cross_encoder",
            )

        assert "chunk_reranker" in _nodes(captured["config"])

    def test_no_reranker_by_default(self):
        import retrico

        captured = {}

        def fake_build(self, verbose=False):
            captured["config"] = self.get_config()
            return _mock_executor({"chunk_result"}, Subgraph())

        with patch.object(RetriCoSearch, "build", fake_build):
            retrico.query_graph("q", retrieval_strategy="community")

        assert "chunk_reranker" not in _nodes(captured["config"])


# -- CLI ----------------------------------------------------------------------


class TestCLI:
    def _run(self, *extra):
        from click.testing import CliRunner
        import retrico
        from retrico.cli import cli

        with patch.object(retrico, "query_graph", return_value=retrico.QueryResult(query="q")) as qg:
            res = CliRunner().invoke(
                cli, ["query", "q", "--entity-labels", "person", "--strategy", "community", *extra],
            )
        assert res.exit_code == 0, res.output
        return qg.call_args.kwargs

    def test_reranker_flags(self):
        kw = self._run("--reranker", "gliclass", "--reranker-model", "m", "--rerank-top-k", "3")
        assert kw["reranker"] == "gliclass"
        assert kw["reranker_kwargs"] == {"model_name": "m", "top_k": 3}

    def test_reranker_without_options(self):
        kw = self._run("--reranker", "cross_encoder")
        assert kw["reranker"] == "cross_encoder"
        assert "reranker_kwargs" not in kw

    def test_no_reranker(self):
        kw = self._run()
        assert "reranker" not in kw
