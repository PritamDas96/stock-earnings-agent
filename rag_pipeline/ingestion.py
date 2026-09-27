"""Document ingestion — load PDF and plain-text sources into ``Document``s.

PDFs are parsed with ``pdfplumber`` (pure-Python, no native ML deps). Text and
Markdown files are read directly. Each source becomes one or more
:class:`Document` objects carrying provenance metadata used later for citations.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber

from core import get_logger

log = get_logger("ingestion")

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


@dataclass(slots=True)
class Document:
    """A unit of source text with associated metadata.

    Attributes:
        content: The extracted text.
        metadata: Provenance and filtering fields (``source``, ``ticker``,
            ``page``, ``doc_type`` …).
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_path(path: str | Path, metadata: dict[str, Any] | None = None) -> list[Document]:
    """Load a single file into one or more :class:`Document` objects.

    Args:
        path: Path to a ``.pdf``, ``.txt``, ``.md`` or ``.markdown`` file.
        metadata: Extra metadata merged into every produced document (e.g.
            ``{"ticker": "AAPL", "doc_type": "10-Q"}``).

    Returns:
        A list of documents (one per PDF page, or a single document for text).

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file type is unsupported.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_SUFFIXES)}"
        )

    base_meta = {"source": path.name, **(metadata or {})}

    if suffix == ".pdf":
        docs = _load_pdf(path, base_meta)
    else:
        docs = _load_text(path, base_meta)

    log.info("Loaded {} document section(s) from {}", len(docs), path.name)
    return docs


def load_documents(
    paths: Iterable[str | Path], metadata: dict[str, Any] | None = None
) -> list[Document]:
    """Load multiple files, concatenating their documents.

    Unreadable individual files are logged and skipped so a single bad file does
    not abort a bulk ingest.
    """
    documents: list[Document] = []
    for path in paths:
        try:
            documents.extend(load_path(path, metadata))
        except (FileNotFoundError, ValueError, OSError) as exc:
            log.warning("Skipping {}: {}", path, exc)
    return documents


def _load_pdf(path: Path, base_meta: dict[str, Any]) -> list[Document]:
    documents: list[Document] = []
    with pdfplumber.open(path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            documents.append(
                Document(
                    content=text,
                    metadata={
                        **base_meta,
                        "page": page_number,
                        "doc_type": base_meta.get("doc_type", "pdf"),
                    },
                )
            )
    if not documents:
        log.warning("No extractable text in PDF {}", path.name)
    return documents


def _load_text(path: Path, base_meta: dict[str, Any]) -> list[Document]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    metadata = {**base_meta, "doc_type": base_meta.get("doc_type", "text")}
    return [Document(content=text, metadata=metadata)]
