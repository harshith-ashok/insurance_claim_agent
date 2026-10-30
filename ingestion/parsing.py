import logging
import os
import re
import uuid

import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE
from models import Chunk, ChunkMetadata

logger = logging.getLogger(__name__)

SECTION_REGEX = re.compile(r"(?:Section|Clause|Article|Policy Provision)\s*([A-Za-z0-9\.]+)", re.IGNORECASE)

# Fixed namespace for content-hash chunk IDs (uuid5) — same (source, page,
# text) always yields the same point ID, so re-ingesting an unchanged page
# overwrites its own point instead of appending a duplicate (see refactor.md
# Phase 2; full incremental-ingest upsert logic is Phase 3).
CHUNK_ID_NAMESPACE = uuid.UUID("3f6f9b1a-6e6b-4e6a-9c3a-6c9a1f8b2d40")

_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def content_hash_id(source: str, page: int, text: str) -> str:
    return str(uuid.uuid5(CHUNK_ID_NAMESPACE, f"{source}|{page}|{text}"))


def parse_document(pdf_path: str) -> list[Chunk]:
    """Parse a single PDF into Chunks, section tracked across pages within
    this document (reset per call — ingestion/pipeline.py calls this once
    per PDF)."""
    source = os.path.basename(pdf_path)
    logger.info("Parsing %s...", source)
    pages_data = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)

    chunks: list[Chunk] = []
    current_section = "N/A"

    for page_idx, page in enumerate(pages_data):
        page_text = page["text"]
        page_num = int(page.get("metadata", {}).get("page_number", page_idx + 1))

        for chunk_text in _SPLITTER.split_text(page_text):
            match = SECTION_REGEX.search(chunk_text)
            if match:
                current_section = match.group(1).strip().rstrip(".")

            chunks.append(
                Chunk(
                    text=chunk_text,
                    metadata=ChunkMetadata(
                        page=page_num,
                        chunk_id=content_hash_id(source, page_num, chunk_text),
                        section=current_section,
                        source=source,
                    ),
                )
            )

    return chunks
