"""Tests for the chunker processor."""

import pytest
from grapsit.construct.chunker import ChunkerProcessor


class TestChunkerProcessor:
    def test_sentence_chunking(self, sample_texts):
        proc = ChunkerProcessor({"method": "sentence"})
        result = proc(texts=sample_texts)

        assert "chunks" in result
        assert "documents" in result
        assert len(result["chunks"]) >= 2
        assert len(result["documents"]) == 2

    def test_fixed_chunking(self):
        proc = ChunkerProcessor({"method": "fixed", "chunk_size": 20, "overlap": 5})
        result = proc(texts=["Hello world. This is a test sentence for chunking."])

        chunks = result["chunks"]
        assert len(chunks) >= 2
        # Verify overlap
        if len(chunks) >= 2:
            assert chunks[1].start_char < chunks[0].end_char

    def test_paragraph_chunking(self):
        proc = ChunkerProcessor({"method": "paragraph"})
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        result = proc(texts=[text])

        assert len(result["chunks"]) == 3

    def test_empty_text(self):
        proc = ChunkerProcessor({"method": "sentence"})
        result = proc(texts=[""])
        assert result["chunks"] == []

    def test_chunk_has_document_id(self):
        proc = ChunkerProcessor({"method": "sentence"})
        result = proc(texts=["One sentence."])
        chunk = result["chunks"][0]
        doc = result["documents"][0]
        assert chunk.document_id == doc.id
