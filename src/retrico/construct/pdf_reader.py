"""PDF reader processor — extracts text and tables from PDF files.

Uses pdfminer.six for layout analysis and pdfplumber for table extraction.
Tables are converted to Markdown format. Each page becomes one Chunk.
"""

from typing import Any, Dict, List, Optional
import re
import logging
from pathlib import Path

from ..core.base import BaseProcessor
from ..core.registry import construct_registry
from ..models.document import Chunk, Document

logger = logging.getLogger(__name__)


def _normalize_text(line_texts: List[str]) -> str:
    """Normalize extracted text lines into a clean string."""
    norm_text = ""
    for line_text in line_texts:
        line_text = line_text.strip()
        if not line_text:
            line_text = "\n"
        else:
            line_text = re.sub(r"\s+", " ", line_text)
            if not re.search(r"[\w\d,\-]", line_text[-1]):
                line_text += "\n"
            else:
                line_text += " "
        norm_text += line_text
    return norm_text


def _text_extraction(element) -> str:
    """Extract and normalize text from a pdfminer text container."""
    line_texts = element.get_text().split("\n")
    return _normalize_text(line_texts)


def _convert_table(table: List[List[Optional[str]]]) -> str:
    """Convert a pdfplumber table to Markdown format."""
    table_string = ""
    for row in table:
        cleaned_row = [
            "None" if item is None else item.replace("\n", " ")
            for item in row
        ]
        table_string += f"|{'|'.join(cleaned_row)}|\n"
    return table_string.rstrip("\n")


def _process_page(page, extracted_page, text: bool = True, table: bool = True) -> str:
    """Process a single PDF page and extract text and tables.

    Args:
        page: pdfplumber page object.
        extracted_page: pdfminer extracted page layout.
        text: Whether to extract text.
        table: Whether to extract tables.

    Returns:
        Concatenated string of page content (text + markdown tables).
    """
    from pdfminer.layout import LTTextContainer, LTRect

    content = []
    tables = page.find_tables()
    extracted_tables = page.extract_tables()

    table_num = 0
    first_table_element = True
    table_extraction_process = False

    elements = list(extracted_page._objs)
    elements.sort(key=lambda a: a.y1, reverse=True)

    lower_side = 0
    upper_side = 0

    for i, element in enumerate(elements):
        if isinstance(element, LTTextContainer) and not table_extraction_process and text:
            line_text = _text_extraction(element)
            content.append(line_text)

        if isinstance(element, LTRect) and table:
            if first_table_element and table_num < len(tables):
                lower_side = page.bbox[3] - tables[table_num].bbox[3]
                upper_side = element.y1

                tbl = extracted_tables[table_num]
                table_string = _convert_table(tbl)
                content.append(table_string)
                table_extraction_process = True
                first_table_element = False

            if element.y0 >= lower_side and element.y1 <= upper_side:
                pass
            elif i + 1 >= len(elements):
                pass
            elif not isinstance(elements[i + 1], LTRect):
                table_extraction_process = False
                first_table_element = True
                table_num += 1

    return re.sub(r"\n+", "\n", "".join(content))


class PDFReaderProcessor(BaseProcessor):
    """Read PDF files and extract text + tables per page.

    Each page becomes one Chunk with metadata including ``page_number``
    and ``source_pdf``.  Tables are converted to Markdown format.

    Config keys:
        extract_text: bool (default: True) — extract regular text.
        extract_tables: bool (default: True) — extract tables as markdown.
        page_ids: List[int] (default: None) — specific pages to extract
            (0-indexed). None means all pages.
    """

    default_inputs = {"pdf_paths": "$input.pdf_paths"}
    default_output = "chunker_result"

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.extract_text = config_dict.get("extract_text", True)
        self.extract_tables = config_dict.get("extract_tables", True)
        self.page_ids = config_dict.get("page_ids", None)

    def __call__(self, *, pdf_paths: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Read PDFs and produce Documents + page-level Chunks.

        Args:
            pdf_paths: List of paths to PDF files.

        Returns:
            {"chunks": List[Chunk], "documents": List[Document]}
        """
        if not pdf_paths:
            return {"chunks": [], "documents": []}

        try:
            import pdfplumber
            from pdfminer.high_level import extract_pages
        except ImportError as e:
            raise ImportError(
                "PDF processing requires pdfminer.six and pdfplumber. "
                "Install them with: pip install 'retrico[pdf]'"
            ) from e

        all_chunks: List[Chunk] = []
        all_documents: List[Document] = []

        for pdf_path in pdf_paths:
            path = Path(pdf_path)
            filename = path.name

            doc = Document(
                source=filename,
                metadata={"source_pdf": filename, "pdf_path": str(path)},
            )

            page_numbers = self.page_ids
            pdf = pdfplumber.open(pdf_path)
            pages = pdf.pages

            try:
                extracted_pages = list(extract_pages(
                    pdf_path, page_numbers=page_numbers,
                ))
            except Exception as e:
                logger.error(f"Failed to extract pages from {pdf_path}: {e}")
                pdf.close()
                continue

            page_texts = []
            for page_idx, extracted_page in enumerate(extracted_pages):
                page_id = extracted_page.pageid
                # pdfminer pageids are 1-based; pdfplumber pages are 0-based
                plumber_idx = page_id - 1
                if plumber_idx < 0 or plumber_idx >= len(pages):
                    logger.warning(
                        f"Page {page_id} out of range for {filename} "
                        f"(total pages: {len(pages)}), skipping."
                    )
                    continue

                page_content = _process_page(
                    pages[plumber_idx],
                    extracted_page,
                    text=self.extract_text,
                    table=self.extract_tables,
                )

                page_texts.append(page_content)

                chunk = Chunk(
                    document_id=doc.id,
                    text=page_content,
                    index=page_idx,
                    start_char=0,
                    end_char=len(page_content),
                    metadata={
                        "page_number": page_id,
                        "source_pdf": filename,
                    },
                )
                all_chunks.append(chunk)

            doc.text = "\n\n".join(page_texts)
            all_documents.append(doc)
            pdf.close()

        logger.info(
            f"PDF reader: {len(all_documents)} documents, "
            f"{len(all_chunks)} page chunks"
        )
        return {"chunks": all_chunks, "documents": all_documents}


@construct_registry.register("pdf_reader")
def create_pdf_reader(config_dict: dict, pipeline=None):
    return PDFReaderProcessor(config_dict, pipeline)
