"""Dense retriever built on sentence-transformers + FAISS.

Encodes guideline chunks into dense vectors and performs similarity search to
find the passages most relevant to a user query. This is the "R" in RAG.

Design notes (thesis Chapter 4):
    * Embedding model is configurable. all-MiniLM-L6-v2 is a strong, fast default
      (384-dim). Reporting a comparison against a larger model (e.g. all-mpnet-base-v2)
      is a good ablation.
    * We use inner-product search on L2-normalised vectors, which is equivalent to
      cosine similarity and is the standard choice for sentence embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ingest import Document, load_corpus, save_corpus


@dataclass
class RetrievedChunk:
    document: Document
    score: float


class DenseRetriever:
    """Sentence-embedding retriever backed by a FAISS index.

    The default embedding model is a sentence-transformer downloaded from the
    HuggingFace Hub. In fully offline environments where the model cannot be
    downloaded, pass ``embedding_model="hashing"`` to use a dependency-free
    hashing embedder. The hashing embedder is a fallback for demos/CI only and is
    weaker than a real sentence-transformer — always use a real model for thesis
    results.
    """

    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.embedding_model_name = embedding_model
        self._model = None
        self._index = None
        self._docs: list[Document] = []

    # -- model ------------------------------------------------------------- #
    def _ensure_model(self):
        if self.embedding_model_name == "hashing":
            return None  # no model object needed
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def _embed(self, texts: list[str]) -> np.ndarray:
        if self.embedding_model_name == "hashing":
            return self._embed_hashing(texts)
        model = self._ensure_model()
        vecs = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine similarity via inner product
            show_progress_bar=False,
        )
        return vecs.astype("float32")

    @staticmethod
    def _embed_hashing(texts: list[str], dim: int = 512) -> np.ndarray:
        """Dependency-free hashing bag-of-words embedder (offline fallback only).

        Uses the hashing trick over word unigrams and bigrams, then L2-normalises
        so inner product equals cosine similarity. Not competitive with neural
        embeddings; provided so the pipeline is runnable without downloads.
        """
        import hashlib
        import re

        def stable_hash(s: str) -> int:
            # md5 is deterministic across processes, unlike builtin hash().
            return int(hashlib.md5(s.encode()).hexdigest(), 16)

        vecs = np.zeros((len(texts), dim), dtype="float32")
        for row, text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9]+", text.lower())
            grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for g in grams:
                h = stable_hash(g) % dim
                vecs[row, h] += 1.0
            norm = np.linalg.norm(vecs[row])
            if norm > 0:
                vecs[row] /= norm
        return vecs

    # -- build / persist --------------------------------------------------- #
    def build(self, docs: list[Document]) -> None:
        """Encode documents and build the FAISS index."""
        import faiss

        if not docs:
            raise ValueError("Cannot build retriever from an empty document list.")
        self._docs = docs
        embeddings = self._embed([d.text for d in docs])
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product on normalised vectors
        index.add(embeddings)
        self._index = index

    def save(self, directory: Path) -> None:
        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "index.faiss"))
        save_corpus(self._docs, directory / "docs.jsonl")
        (directory / "meta.txt").write_text(self.embedding_model_name, encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "DenseRetriever":
        import faiss

        directory = Path(directory)
        model_name = (directory / "meta.txt").read_text(encoding="utf-8").strip()
        retriever = cls(embedding_model=model_name)
        retriever._index = faiss.read_index(str(directory / "index.faiss"))
        retriever._docs = load_corpus(directory / "docs.jsonl")
        return retriever

    # -- query ------------------------------------------------------------- #
    def retrieve(self, query: str, k: int = 4) -> list[RetrievedChunk]:
        """Return the top-k most similar chunks to the query."""
        if self._index is None:
            raise RuntimeError("Retriever index not built or loaded.")
        q = self._embed([query])
        scores, idxs = self._index.search(q, k)
        results: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            results.append(RetrievedChunk(document=self._docs[idx], score=float(score)))
        return results
