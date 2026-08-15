"""Guideline corpus ingestion and chunking.

Loads the authoritative first-aid guideline documents (IFRC 2020, Resuscitation
Council UK 2021) from PDF or text, splits them into overlapping chunks, and
returns structured ``Document`` records ready for embedding.

Chunking strategy (thesis Chapter 4):
    We use recursive character splitting with overlap. Overlap preserves context
    across chunk boundaries so that a retrieved chunk rarely cuts a critical
    instruction in half. Chunk size and overlap are configurable hyper-parameters
    you should tune and report (a small ablation over chunk size makes a strong
    addition to the evaluation chapter).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Document:
    """A single retrievable chunk of guideline text."""

    id: str
    text: str
    source: str          # filename or citation of the origin document
    chunk_index: int     # position within the source document
    metadata: dict

    def to_json(self) -> dict:
        return asdict(self)


def _hash_id(source: str, chunk_index: int, text: str) -> str:
    h = hashlib.sha1(f"{source}:{chunk_index}:{text[:64]}".encode()).hexdigest()
    return h[:16]


def load_text_from_pdf(path: Path) -> str:
    """Extract raw text from a PDF using pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def load_text(path: Path) -> str:
    """Load text from a .pdf or .txt/.md file."""
    if path.suffix.lower() == ".pdf":
        return load_text_from_pdf(path)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    metadata: dict | None = None,
) -> list[Document]:
    """Split text into overlapping chunks using recursive character splitting."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    pieces = splitter.split_text(text)
    docs: list[Document] = []
    for i, piece in enumerate(pieces):
        piece = piece.strip()
        if not piece:
            continue
        docs.append(
            Document(
                id=_hash_id(source, i, piece),
                text=piece,
                source=source,
                chunk_index=i,
                metadata=metadata or {},
            )
        )
    return docs


def build_corpus(
    guidelines_dir: Path,
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Ingest every guideline file in a directory into a list of chunks."""
    guidelines_dir = Path(guidelines_dir)
    all_docs: list[Document] = []
    supported = {".pdf", ".txt", ".md"}
    files = sorted(p for p in guidelines_dir.iterdir() if p.suffix.lower() in supported)
    if not files:
        raise FileNotFoundError(
            f"No guideline documents (.pdf/.txt/.md) found in {guidelines_dir}. "
            f"Add the IFRC 2020 and Resuscitation Council UK 2021 guideline files."
        )
    for f in files:
        text = load_text(f)
        docs = chunk_text(
            text,
            source=f.name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            metadata={"path": str(f)},
        )
        all_docs.extend(docs)
    return all_docs


def save_corpus(docs: list[Document], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(d.to_json()) + "\n")


def load_corpus(in_path: Path) -> list[Document]:
    in_path = Path(in_path)
    docs: list[Document] = []
    with in_path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            docs.append(Document(**rec))
    return docs
