"""Configuration loading (YAML) with sensible defaults.

Central config makes experiments reproducible and keeps hyper-parameters out of
the code. Record the resolved config alongside every results run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    provider: str = "huggingface"        # ollama | gemini | huggingface
    model: str | None = None             # None => backend default
    temperature: float = 0.2
    max_tokens: int = 512


@dataclass
class RetrieverConfig:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 4


@dataclass
class PathsConfig:
    guidelines_dir: str = "data/guidelines"
    index_dir: str = "data/processed/faiss_index"
    scenarios: str = "data/datasets/scenarios.json"
    intents: str = "data/datasets/intents.json"
    results_dir: str = "results"


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load config from YAML, falling back to defaults for any missing keys."""
    if path is None:
        return AppConfig()
    import yaml

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return AppConfig(
        llm=LLMConfig(**(data.get("llm") or {})),
        retriever=RetrieverConfig(**(data.get("retriever") or {})),
        paths=PathsConfig(**(data.get("paths") or {})),
    )
