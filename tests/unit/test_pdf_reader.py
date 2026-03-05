"""Tests for the PDF reader processor."""

import re
import sys
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from retrico.construct.pdf_reader import (
    PDFReaderProcessor,
    _normalize_text,
    _text_extraction,
    _convert_table,
)
from retrico.models.document import Chunk, Document


# -- Helper function tests ----------------------------------------------------

class TestNormalizeText:
    def test_empty_lines_become_newlines(self):
        result = _normalize_text(["hello", "", "world"])
        assert "\n" in result
        assert "hello" in result
        assert "world" in result

    def test_trailing_period_adds_newline(self):
        result = _normalize_text(["This is a sentence."])
        assert result.endswith("\n")

    def test_trailing_word_adds_space(self):
        result = _normalize_text(["This is a word"])
        assert result.endswith(" ")

    def test_whitespace_collapsed(self):
        result = _normalize_text(["  too   many   spaces  "])
        assert "  " not in result.strip()

    def test_trailing_comma_adds_space(self):
        result = _normalize_text(["item1,"])
        assert result.endswith(" ")

    def test_trailing_hyphen_adds_space(self):
        result = _normalize_text(["long-"])
        assert result.endswith(" ")


class TestConvertTable:
    def test_simple_table(self):
        table = [["Name", "Age"], ["Alice", "30"]]
        result = _convert_table(table)
        assert result == "|Name|Age|\n|Alice|30|"

    def test_none_cells(self):
        table = [["A", None, "C"]]
        result = _convert_table(table)
        assert result == "|A|None|C|"

    def test_newlines_in_cells_removed(self):
        table = [["line1\nline2", "ok"]]
        result = _convert_table(table)
        assert "\n" not in result.split("\n")[0]
        assert "line1 line2" in result

    def test_empty_table(self):
        result = _convert_table([])
        assert result == ""

    def test_single_cell(self):
        result = _convert_table([["only"]])
        assert result == "|only|"


class TestTextExtraction:
    def test_extracts_and_normalizes(self):
        element = MagicMock()
        element.get_text.return_value = "Hello world.\nThis is a test."
        result = _text_extraction(element)
        assert "Hello world." in result
        assert "This is a test." in result


# -- Processor tests ----------------------------------------------------------

class TestPDFReaderProcessor:
    def test_default_config(self):
        proc = PDFReaderProcessor({})
        assert proc.extract_text is True
        assert proc.extract_tables is True
        assert proc.page_ids is None

    def test_custom_config(self):
        proc = PDFReaderProcessor({
            "extract_text": False,
            "extract_tables": True,
            "page_ids": [0, 1, 2],
        })
        assert proc.extract_text is False
        assert proc.page_ids == [0, 1, 2]

    def test_empty_input(self):
        proc = PDFReaderProcessor({})
        result = proc(pdf_paths=[])
        assert result == {"chunks": [], "documents": []}

    def test_none_input(self):
        proc = PDFReaderProcessor({})
        result = proc(pdf_paths=None)
        assert result == {"chunks": [], "documents": []}

    def test_import_error_message(self):
        proc = PDFReaderProcessor({})
        with patch.dict("sys.modules", {"pdfplumber": None}):
            with pytest.raises(ImportError, match="pdfminer.six and pdfplumber"):
                proc(pdf_paths=["test.pdf"])

    def test_output_key(self):
        assert PDFReaderProcessor.default_output == "chunker_result"

    def test_processes_pdf(self):
        """Test that the processor creates Documents and page-level Chunks."""
        proc = PDFReaderProcessor({})

        # Mock pdfminer extracted pages
        mock_page_layout1 = MagicMock()
        mock_page_layout1.pageid = 1
        mock_page_layout1._objs = []

        mock_page_layout2 = MagicMock()
        mock_page_layout2.pageid = 2
        mock_page_layout2._objs = []

        # Mock pdfplumber pages
        mock_plumber_page1 = MagicMock()
        mock_plumber_page1.find_tables.return_value = []
        mock_plumber_page1.extract_tables.return_value = []
        mock_plumber_page1.bbox = [0, 0, 612, 792]

        mock_plumber_page2 = MagicMock()
        mock_plumber_page2.find_tables.return_value = []
        mock_plumber_page2.extract_tables.return_value = []
        mock_plumber_page2.bbox = [0, 0, 612, 792]

        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_plumber_page1, mock_plumber_page2]

        mock_pdfplumber = MagicMock()
        mock_pdfplumber.open.return_value = mock_pdf

        mock_extract_pages = MagicMock(return_value=[mock_page_layout1, mock_page_layout2])

        with patch.dict("sys.modules", {
            "pdfplumber": mock_pdfplumber,
            "pdfminer": MagicMock(),
            "pdfminer.high_level": MagicMock(extract_pages=mock_extract_pages),
            "pdfminer.layout": MagicMock(),
        }), patch("retrico.construct.pdf_reader._process_page") as mock_process:
            mock_process.side_effect = ["Page 1 content.", "Page 2 content."]
            result = proc(pdf_paths=["/tmp/test.pdf"])

        assert len(result["documents"]) == 1
        assert len(result["chunks"]) == 2

        doc = result["documents"][0]
        assert doc.source == "test.pdf"
        assert doc.metadata["source_pdf"] == "test.pdf"

        chunk0 = result["chunks"][0]
        assert chunk0.metadata["page_number"] == 1
        assert chunk0.metadata["source_pdf"] == "test.pdf"
        assert chunk0.document_id == doc.id
        assert chunk0.index == 0

        chunk1 = result["chunks"][1]
        assert chunk1.metadata["page_number"] == 2
        assert chunk1.index == 1

    def test_document_text_is_concatenated_pages(self):
        """Test that Document.text is all pages joined."""
        proc = PDFReaderProcessor({})

        mock_page1 = MagicMock()
        mock_page1.pageid = 1
        mock_page1._objs = []
        mock_page2 = MagicMock()
        mock_page2.pageid = 2
        mock_page2._objs = []

        mock_pdf = MagicMock()
        mock_pdf.pages = [MagicMock(), MagicMock()]
        mock_pdfplumber = MagicMock()
        mock_pdfplumber.open.return_value = mock_pdf

        with patch.dict("sys.modules", {
            "pdfplumber": mock_pdfplumber,
            "pdfminer": MagicMock(),
            "pdfminer.high_level": MagicMock(extract_pages=MagicMock(return_value=[mock_page1, mock_page2])),
            "pdfminer.layout": MagicMock(),
        }), patch("retrico.construct.pdf_reader._process_page") as mock_process:
            mock_process.side_effect = ["First page.", "Second page."]
            result = proc(pdf_paths=["/tmp/doc.pdf"])

        doc = result["documents"][0]
        assert "First page." in doc.text
        assert "Second page." in doc.text
        assert "\n\n" in doc.text

    def test_multiple_pdfs(self):
        """Test processing multiple PDF files."""
        proc = PDFReaderProcessor({})

        def make_page(pageid):
            p = MagicMock()
            p.pageid = pageid
            p._objs = []
            return p

        mock_pdfplumber = MagicMock()

        pdf1 = MagicMock()
        pdf1.pages = [MagicMock()]
        pdf2 = MagicMock()
        pdf2.pages = [MagicMock()]
        mock_pdfplumber.open.side_effect = [pdf1, pdf2]

        mock_extract = MagicMock(side_effect=[
            [make_page(1)],
            [make_page(1)],
        ])

        with patch.dict("sys.modules", {
            "pdfplumber": mock_pdfplumber,
            "pdfminer": MagicMock(),
            "pdfminer.high_level": MagicMock(extract_pages=mock_extract),
            "pdfminer.layout": MagicMock(),
        }), patch("retrico.construct.pdf_reader._process_page") as mock_process:
            mock_process.side_effect = ["Content A", "Content B"]
            result = proc(pdf_paths=["/tmp/a.pdf", "/tmp/b.pdf"])

        assert len(result["documents"]) == 2
        assert len(result["chunks"]) == 2
        assert result["documents"][0].source == "a.pdf"
        assert result["documents"][1].source == "b.pdf"


# -- Builder integration tests ------------------------------------------------

class TestPDFReaderBuilder:
    def test_builder_pdf_reader_config(self):
        from retrico.core.builders import RetriCoBuilder

        builder = RetriCoBuilder(name="test_pdf")
        builder.pdf_reader(extract_tables=True, page_ids=[0, 1])
        builder.ner_gliner(labels=["person"])
        builder.graph_writer()

        config = builder.get_config()
        nodes = config["nodes"]

        # Should have pdf_reader instead of chunker
        node_ids = [n["id"] for n in nodes]
        assert "pdf_reader" in node_ids
        assert "chunker" not in node_ids

        pdf_node = next(n for n in nodes if n["id"] == "pdf_reader")
        assert pdf_node["processor"] == "pdf_reader"
        assert pdf_node["config"]["extract_tables"] is True
        assert pdf_node["config"]["page_ids"] == [0, 1]
        assert pdf_node["output"]["key"] == "chunker_result"

        # NER should require pdf_reader
        ner_node = next(n for n in nodes if n["id"] == "ner")
        assert "pdf_reader" in ner_node["requires"]

    def test_builder_default_uses_chunker(self):
        from retrico.core.builders import RetriCoBuilder

        builder = RetriCoBuilder(name="test_default")
        builder.ner_gliner(labels=["person"])
        builder.graph_writer()

        config = builder.get_config()
        node_ids = [n["id"] for n in config["nodes"]]
        assert "chunker" in node_ids
        assert "pdf_reader" not in node_ids

    def test_builder_pdf_reader_with_relex(self):
        from retrico.core.builders import RetriCoBuilder

        builder = RetriCoBuilder(name="test_pdf_relex")
        builder.pdf_reader()
        builder.ner_gliner(labels=["person"])
        builder.relex_gliner(entity_labels=["person"], relation_labels=["knows"])
        builder.graph_writer()

        config = builder.get_config()
        nodes = config["nodes"]

        relex_node = next(n for n in nodes if n["id"] == "relex")
        assert "pdf_reader" in relex_node["requires"]

        writer_node = next(n for n in nodes if n["id"] == "graph_writer")
        assert "pdf_reader" in writer_node["requires"]

    def test_builder_pdf_reader_no_ner(self):
        """pdf_reader can be used without NER (chunks-only pipeline)."""
        from retrico.core.builders import RetriCoBuilder

        builder = RetriCoBuilder(name="test_pdf_no_ner")
        builder.pdf_reader()
        builder.graph_writer()

        config = builder.get_config()
        node_ids = [n["id"] for n in config["nodes"]]
        assert "pdf_reader" in node_ids
        assert "ner" not in node_ids

    def test_builder_pdf_reader_input_wiring(self):
        """pdf_reader reads pdf_paths from $input."""
        from retrico.core.builders import RetriCoBuilder

        builder = RetriCoBuilder(name="test_wiring")
        builder.pdf_reader()
        builder.graph_writer()

        config = builder.get_config()
        pdf_node = next(n for n in config["nodes"] if n["id"] == "pdf_reader")
        assert pdf_node["inputs"]["pdf_paths"]["source"] == "$input"
        assert pdf_node["inputs"]["pdf_paths"]["fields"] == "pdf_paths"


# -- Chunker page method tests -----------------------------------------------

class TestPageChunking:
    def test_page_chunking(self):
        from retrico.construct.chunker import ChunkerProcessor

        proc = ChunkerProcessor({"method": "page"})
        text = "Page 1 content\fPage 2 content\fPage 3 content"
        result = proc(texts=[text])

        assert len(result["chunks"]) == 3
        assert result["chunks"][0].text == "Page 1 content"
        assert result["chunks"][1].text == "Page 2 content"
        assert result["chunks"][2].text == "Page 3 content"
        assert result["chunks"][0].index == 0
        assert result["chunks"][1].index == 1

    def test_page_chunking_empty_pages_skipped(self):
        from retrico.construct.chunker import ChunkerProcessor

        proc = ChunkerProcessor({"method": "page"})
        text = "Page 1\f\fPage 3"
        result = proc(texts=[text])

        assert len(result["chunks"]) == 2
        assert result["chunks"][0].text == "Page 1"
        assert result["chunks"][1].text == "Page 3"

    def test_single_page(self):
        from retrico.construct.chunker import ChunkerProcessor

        proc = ChunkerProcessor({"method": "page"})
        text = "Just one page of content"
        result = proc(texts=[text])

        assert len(result["chunks"]) == 1
        assert result["chunks"][0].text == "Just one page of content"


# -- Registration test --------------------------------------------------------

class TestRegistration:
    def test_pdf_reader_registered(self):
        from retrico.core.registry import construct_registry
        assert "pdf_reader" in construct_registry._factories
