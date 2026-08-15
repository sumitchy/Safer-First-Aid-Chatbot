"""The RAG chatbot: retrieval-augmented, guideline-grounded generation.

This ties the pieces together:
    user query
      -> retriever finds relevant guideline chunks
      -> prompt builder grounds the LLM in those chunks
      -> LLM generates an answer
      -> safety layer enforces escalation and blocks dangerous advice
      -> final answer + provenance returned

This is the system under test (SUT) for the thesis. Baseline A (no retrieval) and
Baseline B (intent classifier) are defined in the ``baselines`` package and share
the same safety layer so comparisons are fair.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..llm.base import GenerationConfig, LLMBackend
from .retriever import DenseRetriever, RetrievedChunk
from .safety import SafetyDecision, SafetyLayer

SYSTEM_INSTRUCTION = """You are a careful first-aid information assistant for members of the public.
You must follow these rules without exception:
1. Base your answer ONLY on the provided guideline context. If the context does not
   cover the question, say so plainly and advise contacting a medical professional.
2. Give clear, numbered, step-by-step instructions a layperson can follow.
3. Never invent drug doses, techniques, or procedures not present in the context.
4. Keep language simple and calm. Avoid jargon.
5. If the situation could be life-threatening, the single most important instruction
   is to call the emergency services immediately.
"""

PROMPT_TEMPLATE = """{system}

GUIDELINE CONTEXT:
{context}

USER QUESTION:
{question}

Answer using only the guideline context above. If the context is insufficient, say
so and recommend professional medical help.

ANSWER:"""


@dataclass
class RAGResponse:
    """Full, auditable result of a RAG query — everything the thesis needs to log."""

    query: str
    answer: str                       # final, safety-wrapped answer shown to user
    raw_answer: str                   # LLM output before the safety layer
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    safety: SafetyDecision | None = None
    backend_name: str = ""

    def provenance(self) -> list[str]:
        """Return the source citations for the retrieved chunks."""
        seen, out = set(), []
        for r in self.retrieved:
            src = r.document.source
            if src not in seen:
                seen.add(src)
                out.append(src)
        return out


class RAGChatbot:
    """Retrieval-augmented first-aid chatbot."""

    def __init__(
        self,
        retriever: DenseRetriever,
        backend: LLMBackend,
        safety: SafetyLayer | None = None,
        top_k: int = 4,
        gen_config: GenerationConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.backend = backend
        self.safety = safety or SafetyLayer()
        self.top_k = top_k
        self.gen_config = gen_config or GenerationConfig()

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        blocks = []
        for i, ch in enumerate(chunks, 1):
            blocks.append(f"[{i}] (source: {ch.document.source})\n{ch.document.text}")
        return "\n\n".join(blocks) if blocks else "(no relevant guideline text found)"

    def answer(self, query: str) -> RAGResponse:
        # 1. Retrieve
        chunks = self.retriever.retrieve(query, k=self.top_k)

        # 2. Build grounded prompt
        prompt = PROMPT_TEMPLATE.format(
            system=SYSTEM_INSTRUCTION,
            context=self._build_context(chunks),
            question=query,
        )

        # 3. Generate
        raw = self.backend.generate(prompt, self.gen_config)

        # 4. Enforce safety
        final, decision = self.safety.apply(query, raw)

        return RAGResponse(
            query=query,
            answer=final,
            raw_answer=raw,
            retrieved=chunks,
            safety=decision,
            backend_name=self.backend.name,
        )
