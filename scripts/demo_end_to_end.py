"""End-to-end demonstration of the full pipeline using the mock backend.

Run:  python scripts/demo_end_to_end.py

This proves the plumbing works: ingest -> index -> RAG + baselines -> evaluate.
Swap MockBackend for a real backend (Ollama/Gemini/HF) to get real results.
"""

from pathlib import Path
import sys

# Make src importable when running from the repo root without installing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from safer_firstaid.baselines import IntentClassifierBaseline, VanillaLLMBaseline
from safer_firstaid.evaluation import SystemSpec, load_scenarios, run_evaluation
from safer_firstaid.llm import GenerationConfig
from safer_firstaid.llm.mock import MockBackend
from safer_firstaid.pipeline import (
    DenseRetriever,
    RAGChatbot,
    SafetyLayer,
    build_corpus,
)

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    gen = GenerationConfig(temperature=0.2, max_tokens=256)
    safety = SafetyLayer()
    backend = MockBackend()

    print("1) Ingesting guideline corpus...")
    docs = build_corpus(ROOT / "data" / "guidelines", chunk_size=500, chunk_overlap=100)
    print(f"   -> {len(docs)} chunks")

    print("2) Building FAISS retriever...")
    # NOTE: 'hashing' is an offline fallback embedder so this demo runs without
    # downloading a model. For real thesis results use the default sentence
    # transformer: DenseRetriever() with no arguments.
    retriever = DenseRetriever(embedding_model="hashing")
    retriever.build(docs)
    index_dir = ROOT / "data" / "processed" / "faiss_index"
    retriever.save(index_dir)
    print(f"   -> index saved to {index_dir}")

    print("3) Assembling systems (RAG + 2 baselines)...")
    rag = RAGChatbot(retriever, backend, safety, top_k=3, gen_config=gen)
    vanilla = VanillaLLMBaseline(backend, safety, gen)
    intent_bot = IntentClassifierBaseline.from_json(
        ROOT / "data" / "datasets" / "intents.json", safety=safety
    )

    print("4) Loading scenarios...")
    scenarios = load_scenarios(ROOT / "data" / "datasets" / "scenarios.json")
    print(f"   -> {len(scenarios)} scenarios")

    print("5) Demonstrating a single RAG answer:")
    demo = rag.answer(scenarios[1].query)  # cardiac arrest
    print("   QUERY:", scenarios[1].query)
    print("   ANSWER:\n" + "\n".join("      " + l for l in demo.answer.splitlines()))
    print("   SOURCES:", demo.provenance())
    print("   ESCALATED:", demo.safety.escalate)

    print("\n6) Running full evaluation over all systems...")
    systems = [
        SystemSpec("RAG", rag),
        SystemSpec("VanillaLLM", vanilla),
        SystemSpec("IntentClassifier", intent_bot),
    ]
    # NLP metrics off for the mock demo (BERTScore download is heavy); the harness
    # supports them for real runs.
    summary = run_evaluation(systems, scenarios, ROOT / "results", compute_nlp=False)

    print("\n7) SUMMARY (guideline congruence):")
    print(f"   {'System':<18}{'FullCong':<10}{'MeanCong':<10}{'DangerousOut':<14}")
    for name, s in summary.items():
        print(
            f"   {name:<18}"
            f"{s['full_congruence_rate']:<10}"
            f"{s['mean_congruence']:<10}"
            f"{s['dangerous_output_count']:<14}"
        )
    print("\nDone. Results in results/summary.csv and results/responses.jsonl")


if __name__ == "__main__":
    main()
