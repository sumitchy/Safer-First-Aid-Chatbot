"""Command-line interface.

Commands:
    build-index   Ingest guideline docs and build the FAISS retriever.
    ask           Ask the RAG chatbot a single question (interactive-friendly).
    evaluate      Run RAG + baselines over the scenario set and write results.

Run `python -m safer_firstaid.cli --help` for details.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print as rprint
from rich.panel import Panel

from .baselines import IntentClassifierBaseline, VanillaLLMBaseline
from .config import load_config
from .evaluation import SystemSpec, load_scenarios, run_evaluation
from .llm import GenerationConfig, build_backend
from .pipeline import DenseRetriever, RAGChatbot, SafetyLayer, build_corpus

app = typer.Typer(add_completion=False, help="Safer First-Aid Chatbots CLI")


def _make_backend(cfg):
    return build_backend(cfg.llm.provider, cfg.llm.model)


def _gen_config(cfg):
    return GenerationConfig(temperature=cfg.llm.temperature, max_tokens=cfg.llm.max_tokens)


@app.command("build-index")
def build_index(config: str = typer.Option(None, help="Path to YAML config")):
    """Ingest guideline documents and build the FAISS retriever index."""
    cfg = load_config(config)
    rprint(Panel.fit("Building guideline corpus + FAISS index", style="cyan"))
    docs = build_corpus(
        Path(cfg.paths.guidelines_dir),
        chunk_size=cfg.retriever.chunk_size,
        chunk_overlap=cfg.retriever.chunk_overlap,
    )
    rprint(f"Ingested [bold]{len(docs)}[/bold] chunks.")
    retriever = DenseRetriever(embedding_model=cfg.retriever.embedding_model)
    retriever.build(docs)
    retriever.save(Path(cfg.paths.index_dir))
    rprint(f"Saved index to [green]{cfg.paths.index_dir}[/green]")


@app.command("ask")
def ask(
    question: str = typer.Argument(..., help="First-aid question"),
    config: str = typer.Option(None, help="Path to YAML config"),
):
    """Ask the RAG chatbot a single question."""
    cfg = load_config(config)
    retriever = DenseRetriever.load(Path(cfg.paths.index_dir))
    backend = _make_backend(cfg)
    bot = RAGChatbot(
        retriever=retriever,
        backend=backend,
        safety=SafetyLayer(),
        top_k=cfg.retriever.top_k,
        gen_config=_gen_config(cfg),
    )
    resp = bot.answer(question)
    rprint(Panel(resp.answer, title="Answer", style="green"))
    rprint(f"[dim]Sources: {', '.join(resp.provenance()) or 'none'}[/dim]")
    rprint(f"[dim]Backend: {resp.backend_name}[/dim]")


@app.command("evaluate")
def evaluate(
    config: str = typer.Option(None, help="Path to YAML config"),
    no_nlp: bool = typer.Option(False, help="Skip NLP metrics (faster)"),
):
    """Run RAG + both baselines over the scenario set and write results."""
    cfg = load_config(config)
    scenarios = load_scenarios(Path(cfg.paths.scenarios))
    rprint(f"Loaded [bold]{len(scenarios)}[/bold] scenarios.")

    backend = _make_backend(cfg)
    safety = SafetyLayer()

    retriever = DenseRetriever.load(Path(cfg.paths.index_dir))
    rag = RAGChatbot(retriever, backend, safety, cfg.retriever.top_k, _gen_config(cfg))
    vanilla = VanillaLLMBaseline(backend, safety, _gen_config(cfg))

    intents_path = Path(cfg.paths.intents)
    intent_bot = IntentClassifierBaseline.from_json(intents_path, safety=safety)

    systems = [
        SystemSpec("RAG", rag),
        SystemSpec("VanillaLLM", vanilla),
        SystemSpec("IntentClassifier", intent_bot),
    ]

    summary = run_evaluation(
        systems, scenarios, Path(cfg.paths.results_dir), compute_nlp=not no_nlp
    )
    rprint(Panel.fit("Evaluation complete", style="cyan"))
    rprint(json.dumps(summary, indent=2))
    rprint(f"[green]Results written to {cfg.paths.results_dir}/[/green]")


if __name__ == "__main__":
    app()
