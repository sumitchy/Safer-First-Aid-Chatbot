"""A deterministic mock backend for testing the pipeline without any real model.

Useful for CI, for verifying the RAG plumbing end-to-end, and for demonstrating the
harness offline. It "answers" by echoing the retrieved guideline context, which lets
us confirm that retrieval + safety + evaluation all work before plugging in a real
model (Ollama/Gemini/HuggingFace).

This is NOT for producing thesis results — it is a development scaffold only.
"""

from __future__ import annotations

import re

from .base import GenerationConfig, LLMBackend


class MockBackend(LLMBackend):
    def __init__(self, model: str = "mock") -> None:
        self.model = model
        self.name = f"mock:{model}"

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        # If a guideline context is present (RAG prompt), summarise it into a
        # step-like answer. Otherwise (vanilla prompt), give a generic reply.
        m = re.search(r"GUIDELINE CONTEXT:\n(.*?)\n\nUSER QUESTION:", prompt, re.S)
        if m:
            context = m.group(1)
            # Take the most content-bearing sentences from the retrieved context.
            sentences = re.split(r"(?<=[.!?])\s+", context)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
            picked = sentences[:4]
            return " ".join(picked) if picked else "Follow first-aid guidance and call for help if needed."
        # vanilla path
        return (
            "Try to stay calm and help the person. If it looks serious, consider "
            "getting medical help."
        )
