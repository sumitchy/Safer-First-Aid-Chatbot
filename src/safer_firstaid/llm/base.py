"""Abstract LLM backend interface.

All generator backends (Ollama, Gemini, HuggingFace) implement this interface so
the RAG pipeline and baselines can swap models without code changes. This is a
key design decision for the thesis: it allows controlled comparison of generators
while holding retrieval and evaluation constant.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class GenerationConfig:
    """Generation hyper-parameters shared across backends.

    Keeping these in one place makes experiments reproducible: record this object
    alongside results so any run can be repeated exactly.
    """

    temperature: float = 0.2  # low temperature: we want deterministic, safe answers
    max_tokens: int = 512
    top_p: float = 0.9
    seed: int | None = 42
    stop: list[str] = field(default_factory=list)


class LLMBackend(ABC):
    """Common interface for all text-generation backends."""

    #: Human-readable identifier recorded in results (e.g. "ollama:mistral").
    name: str

    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        """Return the model's completion for a single prompt.

        Implementations must be synchronous and return plain text. Any backend
        specific error should be raised as ``LLMBackendError`` so callers can
        handle all backends uniformly.
        """
        raise NotImplementedError

    def health_check(self) -> bool:
        """Return True if the backend is reachable and ready.

        Default implementation runs a tiny generation. Override for a cheaper check.
        """
        try:
            out = self.generate("Reply with the single word: ready.", GenerationConfig(max_tokens=8))
            return bool(out and out.strip())
        except Exception:
            return False


class LLMBackendError(RuntimeError):
    """Raised when a backend fails to generate (network, auth, model missing...)."""
