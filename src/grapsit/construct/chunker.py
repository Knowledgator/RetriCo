"""Text chunking processor."""

from typing import Any, Dict, List, Literal
import re
import uuid
import logging

from ..core.base import BaseProcessor
from ..core.registry import processor_registry
from ..models.document import Chunk, Document

logger = logging.getLogger(__name__)


class ChunkerProcessor(BaseProcessor):
    """Split documents into chunks.

    Config keys:
        method: "sentence" | "fixed" | "paragraph"  (default: "sentence")
        chunk_size: int — max chars per chunk for fixed mode (default: 512)
        overlap: int — char overlap for fixed mode (default: 50)
    """

    def __init__(self, config_dict: Dict[str, Any], pipeline: Any = None):
        super().__init__(config_dict, pipeline)
        self.method: str = config_dict.get("method", "sentence")
        self.chunk_size: int = config_dict.get("chunk_size", 512)
        self.overlap: int = config_dict.get("overlap", 50)

    def __call__(self, *, texts: List[str] = None, documents: List[Document] = None, **kwargs) -> Dict[str, Any]:
        """Chunk texts or documents.

        Returns:
            {"chunks": List[Chunk], "documents": List[Document]}
        """
        if documents is None:
            documents = []
            if texts:
                for text in texts:
                    documents.append(Document(text=text))

        all_chunks: List[Chunk] = []
        for doc in documents:
            chunks = self._chunk_text(doc.text, doc.id)
            all_chunks.extend(chunks)

        return {"chunks": all_chunks, "documents": documents}

    def _chunk_text(self, text: str, document_id: str) -> List[Chunk]:
        if self.method == "sentence":
            return self._sentence_chunk(text, document_id)
        elif self.method == "paragraph":
            return self._paragraph_chunk(text, document_id)
        else:
            return self._fixed_chunk(text, document_id)

    def _sentence_chunk(self, text: str, document_id: str) -> List[Chunk]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks = []
        for i, sent in enumerate(sentences):
            if not sent.strip():
                continue
            start = text.find(sent)
            chunks.append(Chunk(
                document_id=document_id,
                text=sent.strip(),
                index=i,
                start_char=start if start >= 0 else 0,
                end_char=(start + len(sent)) if start >= 0 else len(sent),
            ))
        return chunks

    def _paragraph_chunk(self, text: str, document_id: str) -> List[Chunk]:
        paragraphs = re.split(r"\n\s*\n", text.strip())
        chunks = []
        for i, para in enumerate(paragraphs):
            if not para.strip():
                continue
            start = text.find(para)
            chunks.append(Chunk(
                document_id=document_id,
                text=para.strip(),
                index=i,
                start_char=start if start >= 0 else 0,
                end_char=(start + len(para)) if start >= 0 else len(para),
            ))
        return chunks

    def _fixed_chunk(self, text: str, document_id: str) -> List[Chunk]:
        chunks = []
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(Chunk(
                document_id=document_id,
                text=text[start:end],
                index=idx,
                start_char=start,
                end_char=end,
            ))
            start = end - self.overlap if end < len(text) else end
            idx += 1
        return chunks


@processor_registry.register("chunker")
def create_chunker(config_dict: dict, pipeline=None):
    return ChunkerProcessor(config_dict, pipeline)
