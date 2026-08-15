"""Baseline A: vanilla LLM (no retrieval).

Represents a general-purpose chatbot answering first-aid questions from its
parametric knowledge alone. This is the condition prior studies found unsafe
(7-17% guideline congruence). It shares the SAME safety layer as the RAG system,
so the ONLY difference measured is the presence/absence of guideline grounding.

Keeping the safety layer identical is a deliberate experimental control: it isolates
the effect of retrieval, which is exactly what RQ1 asks about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.base import GenerationConfig, LLMBackend
from ..pipeline.safety import SafetyDecision, SafetyLayer

VANILLA_SYSTEM = """You are a first-aid information assistant for members of the public.
Give clear, step-by-step first-aid guidance. If a situation may be life-threatening,
the most important instruction is to call emergency services immediately."""

VANILLA_TEMPLATE = """{system}

USER QUESTION:
{question}

ANSWER:"""


@dataclass
class BaselineResponse:
    query: str
    answer: str
    raw_answer: str
    safety: SafetyDecision | None = None
    backend_name: str = ""
    retrieved: list = field(default_factory=list)  # always empty; kept for a uniform schema

    def provenance(self) -> list[str]:
        return []  # no retrieval, no sources


class VanillaLLMBaseline:
    """LLM answering with no retrieved context."""

    def __init__(
        self,
        backend: LLMBackend,
        safety: SafetyLayer | None = None,
        gen_config: GenerationConfig | None = None,
    ) -> None:
        self.backend = backend
        self.safety = safety or SafetyLayer()
        self.gen_config = gen_config or GenerationConfig()

    def answer(self, query: str) -> BaselineResponse:
        prompt = VANILLA_TEMPLATE.format(system=VANILLA_SYSTEM, question=query)
        raw = self.backend.generate(prompt, self.gen_config)
        final, decision = self.safety.apply(query, raw)
        return BaselineResponse(
            query=query,
            answer=final,
            raw_answer=raw,
            safety=decision,
            backend_name=self.backend.name,
        )
