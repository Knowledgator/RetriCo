"""Tests for reasoner processor."""

import json
import pytest
from unittest.mock import MagicMock

from grapsit.query.reasoner import ReasonerProcessor, LLMReasoner
from grapsit.models.entity import Entity
from grapsit.models.relation import Relation
from grapsit.models.document import Chunk
from grapsit.models.graph import Subgraph, QueryResult


def _make_subgraph():
    return Subgraph(
        entities=[
            Entity(id="e1", label="Washington", entity_type="location"),
            Entity(id="e2", label="USA", entity_type="location"),
        ],
        relations=[
            Relation(head_text="Washington", tail_text="USA", relation_type="LOCATED_IN", score=0.9),
        ],
        chunks=[
            Chunk(id="c1", document_id="d1", text="Washington is located in the USA."),
            Chunk(id="c2", document_id="d1", text="Washington DC is the capital."),
            Chunk(id="c3", document_id="d2", text="Unrelated text about cooking."),
        ],
    )


class TestReasonerProcessor:
    def test_basic_reasoning(self):
        proc = ReasonerProcessor({"method": "llm", "api_key": "test", "model": "test-model"})
        proc._reasoner = LLMReasoner(proc.config_dict)
        proc._reasoner._client = MagicMock()
        proc._reasoner._client.complete.return_value = json.dumps({
            "inferred_relations": [
                {"head": "Washington", "tail": "USA", "relation": "CAPITAL_OF"},
            ],
            "relevant_chunk_ids": ["c1", "c2"],
            "answer": "Washington is the capital of the USA.",
        })

        subgraph = _make_subgraph()
        result = proc(query="What is the capital of USA?", subgraph=subgraph)

        qr = result["result"]
        assert isinstance(qr, QueryResult)
        assert qr.answer == "Washington is the capital of the USA."
        assert qr.query == "What is the capital of USA?"
        # Original relation + inferred
        assert len(qr.subgraph.relations) == 2
        inferred = [r for r in qr.subgraph.relations if r.properties.get("inferred")]
        assert len(inferred) == 1
        assert inferred[0].relation_type == "CAPITAL_OF"
        # Only relevant chunks kept
        assert len(qr.subgraph.chunks) == 2
        assert all(c.id in ("c1", "c2") for c in qr.subgraph.chunks)

    def test_no_inferred_relations(self):
        proc = ReasonerProcessor({"method": "llm", "api_key": "test"})
        proc._reasoner = LLMReasoner(proc.config_dict)
        proc._reasoner._client = MagicMock()
        proc._reasoner._client.complete.return_value = json.dumps({
            "inferred_relations": [],
            "relevant_chunk_ids": [],
            "answer": "I don't know.",
        })

        subgraph = _make_subgraph()
        result = proc(query="test", subgraph=subgraph)

        qr = result["result"]
        assert qr.answer == "I don't know."
        assert len(qr.subgraph.relations) == 1  # only original
        # No relevant IDs → all chunks kept
        assert len(qr.subgraph.chunks) == 3

    def test_empty_subgraph(self):
        proc = ReasonerProcessor({"method": "llm", "api_key": "test"})
        proc._reasoner = LLMReasoner(proc.config_dict)
        proc._reasoner._client = MagicMock()
        proc._reasoner._client.complete.return_value = json.dumps({
            "inferred_relations": [],
            "relevant_chunk_ids": [],
            "answer": "No information available.",
        })

        result = proc(query="test", subgraph=Subgraph())
        qr = result["result"]
        assert qr.answer == "No information available."
        assert len(qr.subgraph.entities) == 0

    def test_llm_failure_graceful(self):
        proc = ReasonerProcessor({"method": "llm", "api_key": "test"})
        proc._reasoner = LLMReasoner(proc.config_dict)
        proc._reasoner._client = MagicMock()
        proc._reasoner._client.complete.side_effect = Exception("API error")

        subgraph = _make_subgraph()
        result = proc(query="test", subgraph=subgraph)

        qr = result["result"]
        assert qr.answer is None
        # Subgraph unchanged
        assert len(qr.subgraph.entities) == 2

    def test_accepts_dict_subgraph(self):
        proc = ReasonerProcessor({"method": "llm", "api_key": "test"})
        proc._reasoner = LLMReasoner(proc.config_dict)
        proc._reasoner._client = MagicMock()
        proc._reasoner._client.complete.return_value = json.dumps({
            "inferred_relations": [],
            "relevant_chunk_ids": [],
            "answer": "test answer",
        })

        subgraph_dict = {
            "entities": [{"id": "e1", "label": "X", "entity_type": ""}],
            "relations": [],
            "chunks": [],
        }
        result = proc(query="test", subgraph=subgraph_dict)
        assert result["result"].answer == "test answer"

    def test_metadata_inferred_count(self):
        proc = ReasonerProcessor({"method": "llm", "api_key": "test"})
        proc._reasoner = LLMReasoner(proc.config_dict)
        proc._reasoner._client = MagicMock()
        proc._reasoner._client.complete.return_value = json.dumps({
            "inferred_relations": [
                {"head": "A", "tail": "B", "relation": "R1"},
                {"head": "C", "tail": "D", "relation": "R2"},
            ],
            "relevant_chunk_ids": [],
            "answer": "test",
        })

        result = proc(query="test", subgraph=Subgraph())
        assert result["result"].metadata["inferred_relation_count"] == 2
