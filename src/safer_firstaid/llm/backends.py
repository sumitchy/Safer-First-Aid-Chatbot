"""Concrete LLM backends: Ollama, Gemini, HuggingFace Transformers.

Each backend is lazily imported so that installing only one provider's
dependencies is enough to run the project. This keeps the environment light and
lets you run fully offline (Ollama / HF) or via API (Gemini).
"""

from __future__ import annotations

import os

from .base import GenerationConfig, LLMBackend, LLMBackendError


# --------------------------------------------------------------------------- #
# Ollama  (local, free, offline)                                              #
# --------------------------------------------------------------------------- #
class OllamaBackend(LLMBackend):
    """Local generation via an Ollama server (https://ollama.com).

    Prerequisite: `ollama serve` running and the model pulled, e.g.
        ollama pull mistral
    """

    def __init__(self, model: str = "mistral", host: str | None = None) -> None:
        self.model = model
        self.host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.name = f"ollama:{model}"

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        config = config or GenerationConfig()
        try:
            import ollama

            client = ollama.Client(host=self.host)
            response = client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": config.temperature,
                    "num_predict": config.max_tokens,
                    "top_p": config.top_p,
                    "seed": config.seed if config.seed is not None else 0,
                    "stop": config.stop,
                },
            )
            return response.get("response", "").strip()
        except Exception as exc:  # noqa: BLE001
            raise LLMBackendError(f"Ollama generation failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Gemini  (Google Generative AI API)                                          #
# --------------------------------------------------------------------------- #
class GeminiBackend(LLMBackend):
    """Generation via Google's Gemini API.

    Prerequisite: set the API key in the environment:
        export GOOGLE_API_KEY="your-key"
    """

    def __init__(self, model: str = "gemini-1.5-flash", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.name = f"gemini:{model}"
        if not self.api_key:
            # Do not raise here: allows constructing the object for --dry-run.
            # The error surfaces on first generate() call instead.
            pass

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        config = config or GenerationConfig()
        if not self.api_key:
            raise LLMBackendError("GOOGLE_API_KEY not set for Gemini backend.")
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model)
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": config.temperature,
                    "max_output_tokens": config.max_tokens,
                    "top_p": config.top_p,
                },
            )
            return (response.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            raise LLMBackendError(f"Gemini generation failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# HuggingFace Transformers  (local, free, offline after download)             #
# --------------------------------------------------------------------------- #
class HuggingFaceBackend(LLMBackend):
    """Local generation via HuggingFace Transformers.

    Loads a text-generation pipeline. Small instruct models such as
    ``google/flan-t5-base`` or ``Qwen/Qwen2.5-0.5B-Instruct`` run on CPU; larger
    models benefit from a GPU. The model is loaded once and cached on the object.
    """

    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str | None = None) -> None:
        self.model = model
        self.device = device
        self.name = f"hf:{model}"
        self._pipe = None  # lazily initialised

    def _ensure_pipe(self):
        if self._pipe is not None:
            return
        try:
            import torch
            from transformers import pipeline

            device = self.device
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self._pipe = pipeline(
                "text-generation",
                model=self.model,
                device_map="auto" if device == "cuda" else None,
                torch_dtype="auto",
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMBackendError(f"Failed to load HuggingFace model '{self.model}': {exc}") from exc

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        config = config or GenerationConfig()
        self._ensure_pipe()
        try:
            # Use chat template if the tokenizer provides one (instruct models).
            messages = [{"role": "user", "content": prompt}]
            try:
                formatted = self._pipe.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                formatted = prompt

            outputs = self._pipe(
                formatted,
                max_new_tokens=config.max_tokens,
                temperature=max(config.temperature, 1e-2),
                top_p=config.top_p,
                do_sample=config.temperature > 0,
                return_full_text=False,
            )
            return outputs[0]["generated_text"].strip()
        except Exception as exc:  # noqa: BLE001
            raise LLMBackendError(f"HuggingFace generation failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# OpenAI-compatible  (LM Studio, vLLM, or real OpenAI)                         #
# --------------------------------------------------------------------------- #
class OpenAICompatibleBackend(LLMBackend):
    """Generation via any OpenAI-compatible ``/v1/chat/completions`` endpoint.

    One backend covers several providers by only changing base_url:
        * LM Studio : http://localhost:1234/v1  (no key; pass any dummy string)
        * vLLM      : http://localhost:8000/v1
        * OpenAI    : https://api.openai.com/v1 (real key required)

    Uses plain HTTP (stdlib ``urllib``) so no extra dependency is required. The
    model id must match what the server exposes; for LM Studio fetch it from
    ``GET /v1/models`` or read it from the Server tab.
    """

    def __init__(
        self,
        model: str = "local-model",
        base_url: str = "http://localhost:1234/v1",
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "not-needed")
        self.name = f"openai-compat:{model}"

    def list_models(self) -> list[str]:
        """Return model ids the server currently exposes (handy for LM Studio)."""
        import json
        import urllib.request

        req = urllib.request.Request(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
            data = json.loads(resp.read().decode())
        return [m["id"] for m in data.get("data", [])]

    def generate(self, prompt: str, config: GenerationConfig | None = None) -> str:
        import json
        import urllib.error
        import urllib.request

        config = config or GenerationConfig()
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p,
            "stream": False,
        }
        if config.stop:
            payload["stop"] = config.stop

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:  # noqa: S310
                body = json.loads(resp.read().decode())
            return body["choices"][0]["message"]["content"].strip()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="ignore")
            raise LLMBackendError(
                f"OpenAI-compatible server returned HTTP {exc.code}: {detail}. "
                f"If using LM Studio, check the model id matches GET {self.base_url}/models."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise LLMBackendError(
                f"OpenAI-compatible generation failed ({self.base_url}): {exc}. "
                f"Is the server running?"
            ) from exc


class LMStudioBackend(OpenAICompatibleBackend):
    """Convenience wrapper for LM Studio's default local server.

    Prerequisites:
        1. Open LM Studio and download a model (e.g. Mistral 7B Instruct).
        2. Go to the Developer/Server tab and click "Start Server".
        3. Load a model in the SERVER tab (the Chat-tab model does NOT carry over).
        4. The server listens on http://localhost:1234/v1 by default.
    """

    def __init__(self, model: str = "local-model", base_url: str | None = None,
                 api_key: str | None = None) -> None:
        super().__init__(
            model=model,
            base_url=base_url or os.environ.get("LMSTUDIO_HOST", "http://localhost:1234/v1"),
            api_key=api_key or "lm-studio",
        )
        self.name = f"lmstudio:{model}"


# --------------------------------------------------------------------------- #
# Factory                                                                     #
# --------------------------------------------------------------------------- #
def _mock_backend(model: str | None = None, **kwargs):
    # Imported lazily to keep the mock out of the main import path.
    from .mock import MockBackend

    return MockBackend(model=model or "mock")


_BACKENDS = {
    "ollama": OllamaBackend,
    "gemini": GeminiBackend,
    "huggingface": HuggingFaceBackend,
    "hf": HuggingFaceBackend,
    "lmstudio": LMStudioBackend,
    "lm-studio": LMStudioBackend,
    "openai": OpenAICompatibleBackend,
    "openai-compatible": OpenAICompatibleBackend,
    "vllm": OpenAICompatibleBackend,
    "mock": _mock_backend,  # offline testing only
}


def build_backend(provider: str, model: str | None = None, **kwargs) -> LLMBackend:
    """Instantiate a backend by name.

    Args:
        provider: one of "ollama", "gemini", "huggingface" (alias "hf").
        model: model identifier; if None, the backend's sensible default is used.
        **kwargs: forwarded to the backend constructor (host, api_key, device...).
    """
    provider = provider.lower().strip()
    if provider not in _BACKENDS:
        raise ValueError(
            f"Unknown provider '{provider}'. Choose from: {sorted(set(_BACKENDS))}"
        )
    cls = _BACKENDS[provider]
    if model:
        return cls(model=model, **kwargs)
    return cls(**kwargs)
