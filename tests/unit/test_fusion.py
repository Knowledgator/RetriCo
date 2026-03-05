"""Tests for SubgraphFusionProcessor and multi-retriever builder integration."""

import pytest

from retrico.query.fusion import SubgraphFusionProcessor
from retrico.models.graph import Subgraph
from retrico.models.entity import Entity
from retrico.models.relation import Relation
from retrico.models.document import Chunk
from retrico.core.builders import RetriCoSearch, RetriCoFusedSearch
from retrico.core.registry import query_registry


# -- Helpers -----------------------------------------------------------------

def _entity(label, eid=None):
    return Entity(id=eid or label, label=label, entity_type="test")


def _relation(head, tail, rtype="RELATED_TO"):
    return Relation(head_text=head, tail_text=tail, relation_type=rtype)


def _chunk(cid, text="chunk text"):
    return Chunk(id=cid, text=text)


def _subgraph(entities=None, relations=None, chunks=None):
    return Subgraph(
        entities=entities or [],
        relations=relations or [],
        chunks=chunks or [],
    )


# -- Union strategy ----------------------------------------------------------

class TestUnionStrategy:
    def test_dedup_entities(self):
        sg1 = _subgraph(entities=[_entity("Alice"), _entity("Bob")])
        sg2 = _subgraph(entities=[_entity("Bob"), _entity("Charlie")])
        proc = SubgraphFusionProcessor({"strategy": "union"})
        result = proc(subgraph_0=sg1, subgraph_1=sg2)
        labels = [e.label for e in result["subgraph"].entities]
        assert labels == ["Alice", "Bob", "Charlie"]

    def test_dedup_chunks(self):
        sg1 = _subgraph(chunks=[_chunk("c1"), _chunk("c2")])
        sg2 = _subgraph(chunks=[_chunk("c2"), _chunk("c3")])
        proc = SubgraphFusionProcessor({"strategy": "union"})
        result = proc(subgraph_0=sg1, subgraph_1=sg2)
        chunk_ids = [c.id for c in result["subgraph"].chunks]
        assert chunk_ids == ["c1", "c2", "c3"]

    def test_preserves_valid_relations(self):
        sg1 = _subgraph(
            entities=[_entity("Alice"), _entity("Bob")],
            relations=[_relation("Alice", "Bob", "KNOWS")],
        )
        sg2 = _subgraph(
            entities=[_entity("Bob"), _entity("Charlie")],
            relations=[_relation("Bob", "Charlie", "WORKS_WITH")],
        )
        proc = SubgraphFusionProcessor({"strategy": "union"})
        result = proc(subgraph_0=sg1, subgraph_1=sg2)
        rtypes = [r.relation_type for r in result["subgraph"].relations]
        assert "KNOWS" in rtypes
        assert "WORKS_WITH" in rtypes

    def test_filters_dangling_relations(self):
        """Relations where one endpoint is not in entities should be dropped."""
        sg1 = _subgraph(
            entities=[_entity("Alice")],
            relations=[_relation("Alice", "Unknown", "RELATED")],
        )
        sg2 = _subgraph(entities=[_entity("Bob")])
        proc = SubgraphFusionProcessor({"strategy": "union"})
        result = proc(subgraph_0=sg1, subgraph_1=sg2)
        assert len(result["subgraph"].relations) == 0


# -- RRF strategy ------------------------------------------------------------

class TestRRFStrategy:
    def test_scoring_formula(self):
        """Entities in multiple retrievers should score higher."""
        sg1 = _subgraph(entities=[_entity("Alice"), _entity("Bob")])
        sg2 = _subgraph(entities=[_entity("Bob"), _entity("Charlie")])
        proc = SubgraphFusionProcessor({"strategy": "rrf", "rrf_k": 60})
        result = proc(subgraph_0=sg1, subgraph_1=sg2)
        labels = [e.label for e in result["subgraph"].entities]
        # Bob appears in both (rank 1 in sg1, rank 0 in sg2) → highest score
        assert labels[0] == "Bob"

    def test_top_k_filtering(self):
        sg1 = _subgraph(entities=[_entity("A"), _entity("B"), _entity("C")])
        sg2 = _subgraph(entities=[_entity("B"), _entity("D")])
        proc = SubgraphFusionProcessor({"strategy": "rrf", "top_k": 2, "rrf_k": 60})
        result = proc(subgraph_0=sg1, subgraph_1=sg2)
        assert len(result["subgraph"].entities) == 2

    def test_relation_pruning(self):
        """Relations with endpoints not in top_k entities should be dropped."""
        sg1 = _subgraph(
            entities=[_entity("A"), _entity("B"), _entity("C")],
            relations=[_relation("A", "C", "REL")],
        )
        sg2 = _subgraph(entities=[_entity("B")])
        # top_k=1 → only B survives
        proc = SubgraphFusionProcessor({"strategy": "rrf", "top_k": 1, "rrf_k": 60})
        result = proc(subgraph_0=sg1, subgraph_1=sg2)
        assert len(result["subgraph"].relations) == 0


# -- Weighted strategy -------------------------------------------------------

class TestWeightedStrategy:
    def test_equal_weights(self):
        sg1 = _subgraph(entities=[_entity("A"), _entity("B")])
        sg2 = _subgraph(entities=[_entity("B"), _entity("C")])
        proc = SubgraphFusionProcessor({"strategy": "weighted"})
        result = proc(subgraph_0=sg1, subgraph_1=sg2)
        labels = [e.label for e in result["subgraph"].entities]
        # B appears in both → highest score
        assert labels[0] == "B"

    def test_custom_weights(self):
        sg1 = _subgraph(entities=[_entity("A")])
        sg2 = _subgraph(entities=[_entity("B")])
        # Weight sg1 heavily
        proc = SubgraphFusionProcessor({
            "strategy": "weighted",
            "weights": [10.0, 1.0],
        })
        result = proc(subgraph_0=sg1, subgraph_1=sg2)
        labels = [e.label for e in result["subgraph"].entities]
        # A should rank higher due to weight
        assert labels[0] == "A"

    def test_top_k(self):
        sg1 = _subgraph(entities=[_entity("A"), _entity("B")])
        sg2 = _subgraph(entities=[_entity("C"), _entity("D")])
        proc = SubgraphFusionProcessor({"strategy": "weighted", "top_k": 2})
        result = proc(subgraph_0=sg1, subgraph_1=sg2)
        assert len(result["subgraph"].entities) == 2


# -- Intersection strategy ---------------------------------------------------

class TestIntersectionStrategy:
    def test_min_sources_filtering(self):
        sg1 = _subgraph(entities=[_entity("A"), _entity("B")])
        sg2 = _subgraph(entities=[_entity("B"), _entity("C")])
        sg3 = _subgraph(entities=[_entity("B"), _entity("D")])
        proc = SubgraphFusionProcessor({"strategy": "intersection", "min_sources": 2})
        result = proc(subgraph_0=sg1, subgraph_1=sg2, subgraph_2=sg3)
        labels = [e.label for e in result["subgraph"].entities]
        assert "B" in labels
        assert "A" not in labels
        assert "C" not in labels
        assert "D" not in labels

    def test_min_sources_3(self):
        sg1 = _subgraph(entities=[_entity("A"), _entity("B")])
        sg2 = _subgraph(entities=[_entity("B"), _entity("C")])
        sg3 = _subgraph(entities=[_entity("B"), _entity("A")])
        proc = SubgraphFusionProcessor({"strategy": "intersection", "min_sources": 3})
        result = proc(subgraph_0=sg1, subgraph_1=sg2, subgraph_2=sg3)
        labels = [e.label for e in result["subgraph"].entities]
        assert labels == ["B"]

    def test_relation_pruning(self):
        sg1 = _subgraph(
            entities=[_entity("A"), _entity("B")],
            relations=[_relation("A", "B", "REL")],
        )
        sg2 = _subgraph(entities=[_entity("B"), _entity("C")])
        proc = SubgraphFusionProcessor({"strategy": "intersection", "min_sources": 2})
        result = proc(subgraph_0=sg1, subgraph_1=sg2)
        # Only B survives → relation A-B dropped
        assert len(result["subgraph"].relations) == 0


# -- Edge cases --------------------------------------------------------------

class TestEdgeCases:
    def test_no_inputs(self):
        proc = SubgraphFusionProcessor({"strategy": "union"})
        result = proc()
        assert result["subgraph"].entities == []
        assert result["subgraph"].relations == []
        assert result["subgraph"].chunks == []

    def test_single_input_passthrough(self):
        sg = _subgraph(
            entities=[_entity("A")],
            relations=[_relation("A", "A", "SELF")],
            chunks=[_chunk("c1")],
        )
        proc = SubgraphFusionProcessor({"strategy": "union"})
        result = proc(subgraph_0=sg)
        assert result["subgraph"] is sg  # passthrough, no copy

    def test_unknown_strategy_raises(self):
        proc = SubgraphFusionProcessor({"strategy": "magic"})
        sg = _subgraph(entities=[_entity("A")])
        with pytest.raises(ValueError, match="Unknown fusion strategy"):
            proc(subgraph_0=sg, subgraph_1=sg)

    def test_dict_input_converted(self):
        proc = SubgraphFusionProcessor({"strategy": "union"})
        sg_dict = {"entities": [_entity("A")], "relations": [], "chunks": []}
        sg2 = _subgraph(entities=[_entity("B")])
        result = proc(subgraph_0=sg_dict, subgraph_1=sg2)
        labels = [e.label for e in result["subgraph"].entities]
        assert "A" in labels
        assert "B" in labels

    def test_dedup_relations(self):
        """Duplicate relations across retrievers should be deduplicated."""
        rel = _relation("A", "B", "KNOWS")
        sg1 = _subgraph(entities=[_entity("A"), _entity("B")], relations=[rel])
        sg2 = _subgraph(
            entities=[_entity("A"), _entity("B")],
            relations=[_relation("A", "B", "KNOWS")],
        )
        proc = SubgraphFusionProcessor({"strategy": "union"})
        result = proc(subgraph_0=sg1, subgraph_1=sg2)
        assert len(result["subgraph"].relations) == 1


# -- Registration ------------------------------------------------------------

class TestRegistration:
    def test_registered_as_fusion(self):
        factory = query_registry.get("fusion")
        assert factory is not None
        proc = factory({}, None)
        assert isinstance(proc, SubgraphFusionProcessor)


# -- Builder integration -----------------------------------------------------

class TestBuilderSingleRetriever:
    """Single retriever should produce the same DAG as before."""

    def test_backward_compat_single_retriever(self):
        builder = RetriCoSearch(name="test")
        builder.query_parser(labels=["person"])
        builder.retriever(max_hops=2)
        builder.chunk_retriever()
        config = builder.get_config()

        node_ids = [n["id"] for n in config["nodes"]]
        assert "retriever" in node_ids
        assert "fusion" not in node_ids
        # Output key is retriever_result (not retriever_result_0)
        retriever_node = [n for n in config["nodes"] if n["id"] == "retriever"][0]
        assert retriever_node["output"]["key"] == "retriever_result"
        # chunk_retriever reads from retriever_result
        chunk_node = [n for n in config["nodes"] if n["id"] == "chunk_retriever"][0]
        assert chunk_node["inputs"]["subgraph"]["source"] == "retriever_result"

    def test_backward_compat_query_based_retriever(self):
        builder = RetriCoSearch(name="test")
        builder.community_retriever()
        builder.chunk_retriever()
        config = builder.get_config()

        node_ids = [n["id"] for n in config["nodes"]]
        assert "retriever" in node_ids
        assert "fusion" not in node_ids
        retriever_node = [n for n in config["nodes"] if n["id"] == "retriever"][0]
        assert retriever_node["processor"] == "community_retriever"


class TestBuilderMultiRetriever:
    """Multiple retrievers should produce retriever_0..N + fusion node."""

    def test_multi_retriever_dag_structure(self):
        builder = RetriCoSearch(name="test")
        builder.query_parser(labels=["person"])
        builder.retriever(max_hops=2)
        builder.path_retriever()
        builder.chunk_retriever()
        config = builder.get_config()

        node_ids = [n["id"] for n in config["nodes"]]
        assert "retriever_0" in node_ids
        assert "retriever_1" in node_ids
        assert "fusion" in node_ids
        assert "retriever" not in node_ids  # no plain "retriever"

        # Check outputs
        r0 = [n for n in config["nodes"] if n["id"] == "retriever_0"][0]
        r1 = [n for n in config["nodes"] if n["id"] == "retriever_1"][0]
        assert r0["output"]["key"] == "retriever_result_0"
        assert r1["output"]["key"] == "retriever_result_1"

        # Fusion inputs
        fusion_node = [n for n in config["nodes"] if n["id"] == "fusion"][0]
        assert "subgraph_0" in fusion_node["inputs"]
        assert "subgraph_1" in fusion_node["inputs"]
        assert fusion_node["inputs"]["subgraph_0"]["source"] == "retriever_result_0"
        assert fusion_node["inputs"]["subgraph_1"]["source"] == "retriever_result_1"
        assert fusion_node["requires"] == ["retriever_0", "retriever_1"]

        # chunk_retriever reads from fusion_result
        chunk_node = [n for n in config["nodes"] if n["id"] == "chunk_retriever"][0]
        assert chunk_node["inputs"]["subgraph"]["source"] == "fusion_result"

    def test_auto_default_fusion(self):
        """When no explicit fusion(), default union is used."""
        builder = RetriCoSearch(name="test")
        builder.query_parser(labels=["person"])
        builder.retriever()
        builder.community_retriever()
        builder.chunk_retriever()
        config = builder.get_config()

        fusion_node = [n for n in config["nodes"] if n["id"] == "fusion"][0]
        assert fusion_node["config"]["strategy"] == "union"

    def test_explicit_fusion_config(self):
        builder = RetriCoSearch(name="test")
        builder.query_parser(labels=["person"])
        builder.retriever()
        builder.path_retriever()
        builder.fusion(strategy="rrf", top_k=10, rrf_k=30)
        builder.chunk_retriever()
        config = builder.get_config()

        fusion_node = [n for n in config["nodes"] if n["id"] == "fusion"][0]
        assert fusion_node["config"]["strategy"] == "rrf"
        assert fusion_node["config"]["top_k"] == 10
        assert fusion_node["config"]["rrf_k"] == 30

    def test_mixed_entity_and_query_strategies(self):
        """Entity-based + query-based retrievers wire inputs correctly."""
        builder = RetriCoSearch(name="test")
        builder.query_parser(labels=["person"])
        builder.retriever(max_hops=2)  # entity-based
        builder.community_retriever()  # query-based
        builder.chunk_retriever()
        config = builder.get_config()

        r0 = [n for n in config["nodes"] if n["id"] == "retriever_0"][0]
        r1 = [n for n in config["nodes"] if n["id"] == "retriever_1"][0]
        # Entity-based gets entities input
        assert "entities" in r0["inputs"]
        # Query-based gets query input
        assert "query" in r1["inputs"]

    def test_three_retrievers(self):
        builder = RetriCoSearch(name="test")
        builder.query_parser(labels=["person"])
        builder.retriever()
        builder.path_retriever()
        builder.community_retriever()
        builder.fusion(strategy="intersection", min_sources=2)
        builder.chunk_retriever()
        config = builder.get_config()

        node_ids = [n["id"] for n in config["nodes"]]
        assert "retriever_0" in node_ids
        assert "retriever_1" in node_ids
        assert "retriever_2" in node_ids
        assert "fusion" in node_ids

        fusion_node = [n for n in config["nodes"] if n["id"] == "fusion"][0]
        assert len(fusion_node["inputs"]) == 3
        assert fusion_node["config"]["strategy"] == "intersection"
        assert fusion_node["config"]["min_sources"] == 2

    def test_no_retriever_raises(self):
        builder = RetriCoSearch(name="test")
        builder.query_parser(labels=["person"])
        with pytest.raises(ValueError, match="Retriever config required"):
            builder.get_config()

    def test_reasoner_with_multi_retriever(self):
        builder = RetriCoSearch(name="test")
        builder.query_parser(labels=["person"])
        builder.retriever()
        builder.community_retriever()
        builder.chunk_retriever()
        builder.reasoner(api_key="test-key")
        config = builder.get_config()

        node_ids = [n["id"] for n in config["nodes"]]
        assert "reasoner" in node_ids
        reasoner = [n for n in config["nodes"] if n["id"] == "reasoner"][0]
        assert reasoner["requires"] == ["chunk_retriever"]


# -- RetriCoFusedSearch -------------------------------------------------------

class TestRetriCoFusedSearch:
    """Tests for RetriCoFusedSearch — wrapper combining multiple RetriCoSearchs."""

    def test_two_builders_dag_structure(self):
        """Two sub-builders produce retriever_0, retriever_1, fusion, chunk_retriever."""
        b1 = RetriCoSearch(name="entity")
        b1.query_parser(labels=["person"])
        b1.retriever(max_hops=2)

        b2 = RetriCoSearch(name="community")
        b2.community_retriever()

        fused = RetriCoFusedSearch(b1, b2)
        fused.chunk_retriever()
        config = fused.get_config()

        node_ids = [n["id"] for n in config["nodes"]]
        assert "retriever_0" in node_ids
        assert "retriever_1" in node_ids
        assert "fusion" in node_ids
        assert "chunk_retriever" in node_ids

        # Check fusion wiring
        fusion_node = [n for n in config["nodes"] if n["id"] == "fusion"][0]
        assert "subgraph_0" in fusion_node["inputs"]
        assert "subgraph_1" in fusion_node["inputs"]
        assert fusion_node["inputs"]["subgraph_0"]["source"] == "retriever_result_0"
        assert fusion_node["inputs"]["subgraph_1"]["source"] == "retriever_result_1"
        assert fusion_node["requires"] == ["retriever_0", "retriever_1"]

        # chunk_retriever reads from fusion_result
        chunk_node = [n for n in config["nodes"] if n["id"] == "chunk_retriever"][0]
        assert chunk_node["inputs"]["subgraph"]["source"] == "fusion_result"

    def test_parser_auto_inherited(self):
        """Parser is auto-inherited from first sub-builder that has one."""
        b1 = RetriCoSearch(name="entity")
        b1.query_parser(labels=["person", "location"])
        b1.retriever(max_hops=2)

        b2 = RetriCoSearch(name="community")
        b2.community_retriever()

        fused = RetriCoFusedSearch(b1, b2)
        fused.chunk_retriever()
        config = fused.get_config()

        node_ids = [n["id"] for n in config["nodes"]]
        assert "query_parser" in node_ids
        parser_node = [n for n in config["nodes"] if n["id"] == "query_parser"][0]
        assert parser_node["config"]["labels"] == ["person", "location"]

    def test_parser_inherited_from_second_builder(self):
        """If first builder has no parser, inherit from second."""
        b1 = RetriCoSearch(name="community")
        b1.community_retriever()

        b2 = RetriCoSearch(name="entity")
        b2.query_parser(labels=["org"])
        b2.retriever(max_hops=1)

        fused = RetriCoFusedSearch(b1, b2)
        fused.chunk_retriever()
        config = fused.get_config()

        parser_node = [n for n in config["nodes"] if n["id"] == "query_parser"][0]
        assert parser_node["config"]["labels"] == ["org"]

    def test_explicit_parser_overrides_sub_builders(self):
        """Explicit query_parser() on fused builder overrides sub-builder's."""
        b1 = RetriCoSearch(name="entity")
        b1.query_parser(labels=["person"])
        b1.retriever(max_hops=2)

        b2 = RetriCoSearch(name="community")
        b2.community_retriever()

        fused = RetriCoFusedSearch(b1, b2)
        fused.query_parser(labels=["city", "country"])
        fused.chunk_retriever()
        config = fused.get_config()

        parser_node = [n for n in config["nodes"] if n["id"] == "query_parser"][0]
        assert parser_node["config"]["labels"] == ["city", "country"]

    def test_store_auto_inherited(self):
        """Store config inherited from first sub-builder that has one."""
        from retrico.store.config import Neo4jConfig

        b1 = RetriCoSearch(name="entity")
        b1.store(Neo4jConfig(uri="bolt://myhost:7687", password="secret"))
        b1.query_parser(labels=["person"])
        b1.retriever(max_hops=2)

        b2 = RetriCoSearch(name="community")
        b2.community_retriever()

        fused = RetriCoFusedSearch(b1, b2)
        fused.chunk_retriever()
        config = fused.get_config()

        # Check stores section includes the graph store
        assert "stores" in config
        assert "graph" in config["stores"]

    def test_reasoner_on_fused_builder(self):
        """Reasoner configured on fused builder appears in DAG."""
        b1 = RetriCoSearch(name="entity")
        b1.query_parser(labels=["person"])
        b1.retriever()

        b2 = RetriCoSearch(name="community")
        b2.community_retriever()

        fused = RetriCoFusedSearch(b1, b2)
        fused.chunk_retriever()
        fused.reasoner(api_key="test-key", model="gpt-4o-mini")
        config = fused.get_config()

        node_ids = [n["id"] for n in config["nodes"]]
        assert "reasoner" in node_ids
        reasoner = [n for n in config["nodes"] if n["id"] == "reasoner"][0]
        assert reasoner["requires"] == ["chunk_retriever"]
        assert reasoner["config"]["api_key"] == "test-key"

    def test_mixed_entity_and_query_strategies(self):
        """Entity-based + query-based retrievers wire inputs correctly."""
        b1 = RetriCoSearch(name="entity")
        b1.query_parser(labels=["person"])
        b1.retriever(max_hops=2)

        b2 = RetriCoSearch(name="community")
        b2.community_retriever()

        fused = RetriCoFusedSearch(b1, b2)
        fused.chunk_retriever()
        config = fused.get_config()

        r0 = [n for n in config["nodes"] if n["id"] == "retriever_0"][0]
        r1 = [n for n in config["nodes"] if n["id"] == "retriever_1"][0]
        # Entity-based gets entities input
        assert "entities" in r0["inputs"]
        # Query-based gets query input
        assert "query" in r1["inputs"]

    def test_empty_builders_raises(self):
        """No retriever nodes → raises ValueError."""
        b1 = RetriCoSearch(name="empty1")
        b1.query_parser(labels=["person"])

        b2 = RetriCoSearch(name="empty2")

        fused = RetriCoFusedSearch(b1, b2)
        with pytest.raises(ValueError, match="No retriever nodes"):
            fused.get_config()

    def test_three_builders(self):
        """Three sub-builders produce 3 retrievers + fusion."""
        b1 = RetriCoSearch(name="entity")
        b1.query_parser(labels=["person"])
        b1.retriever()

        b2 = RetriCoSearch(name="path")
        b2.path_retriever()

        b3 = RetriCoSearch(name="community")
        b3.community_retriever()

        fused = RetriCoFusedSearch(b1, b2, b3, strategy="intersection", min_sources=2)
        fused.chunk_retriever()
        config = fused.get_config()

        node_ids = [n["id"] for n in config["nodes"]]
        assert "retriever_0" in node_ids
        assert "retriever_1" in node_ids
        assert "retriever_2" in node_ids
        assert "fusion" in node_ids

        fusion_node = [n for n in config["nodes"] if n["id"] == "fusion"][0]
        assert len(fusion_node["inputs"]) == 3
        assert fusion_node["config"]["strategy"] == "intersection"
        assert fusion_node["config"]["min_sources"] == 2

    def test_fusion_override_after_init(self):
        """fusion() on fused builder overrides init kwargs."""
        b1 = RetriCoSearch(name="entity")
        b1.query_parser(labels=["person"])
        b1.retriever()

        b2 = RetriCoSearch(name="community")
        b2.community_retriever()

        fused = RetriCoFusedSearch(b1, b2, strategy="union")
        fused.fusion(strategy="rrf", top_k=10, rrf_k=30)
        fused.chunk_retriever()
        config = fused.get_config()

        fusion_node = [n for n in config["nodes"] if n["id"] == "fusion"][0]
        assert fusion_node["config"]["strategy"] == "rrf"
        assert fusion_node["config"]["top_k"] == 10
        assert fusion_node["config"]["rrf_k"] == 30

    def test_single_sub_builder_still_emits_fusion(self):
        """Even with one sub-builder, RetriCoFusedSearch emits a fusion node."""
        b1 = RetriCoSearch(name="entity")
        b1.query_parser(labels=["person"])
        b1.retriever(max_hops=2)

        fused = RetriCoFusedSearch(b1)
        fused.chunk_retriever()
        config = fused.get_config()

        node_ids = [n["id"] for n in config["nodes"]]
        assert "retriever_0" in node_ids
        assert "fusion" in node_ids

    def test_default_chunk_retriever(self):
        """chunk_retriever defaults to {} if not explicitly called."""
        b1 = RetriCoSearch(name="entity")
        b1.query_parser(labels=["person"])
        b1.retriever()

        b2 = RetriCoSearch(name="community")
        b2.community_retriever()

        fused = RetriCoFusedSearch(b1, b2)
        # No explicit chunk_retriever() call
        config = fused.get_config()

        node_ids = [n["id"] for n in config["nodes"]]
        assert "chunk_retriever" in node_ids

    def test_stores_merged_from_sub_builders(self):
        """Named stores from all sub-builders are merged."""
        from retrico.store.config import Neo4jConfig

        b1 = RetriCoSearch(name="entity")
        b1.graph_store(Neo4jConfig(uri="bolt://host1:7687"), name="store1")
        b1.query_parser(labels=["person"])
        b1.retriever()

        b2 = RetriCoSearch(name="community")
        b2.graph_store(Neo4jConfig(uri="bolt://host2:7687"), name="store2")
        b2.community_retriever()

        fused = RetriCoFusedSearch(b1, b2)
        fused.chunk_retriever()
        config = fused.get_config()

        assert "stores" in config
        graph_stores = config["stores"]["graph"]
        assert "store1" in graph_stores
        assert "store2" in graph_stores
