# Safer First-Aid Chatbots

**Reducing Guideline-Discordant Advice Through Retrieval-Augmented Generation Grounded in Red Cross and Resuscitation Council Guidelines**

Postgraduate Computing Project — Arden University
Sumit Chaudhary (24161337) · Project P20435

---

> **Building or extending this with an AI coding agent?** See **[`BUILD_GUIDE.md`](BUILD_GUIDE.md)** — a complete, phased specification (with acceptance tests) that a coding agent can follow to construct the whole project from scratch.

## What this is

A research codebase that builds and evaluates a **Retrieval-Augmented Generation (RAG)** first-aid chatbot and compares it, under identical conditions, against two baselines:

- **Baseline A — Vanilla LLM**: the same language model with *no* retrieval (represents current general-purpose chatbots).
- **Baseline B — Intent classifier**: a TF-IDF + Logistic Regression FAQ bot (represents the rule-based state of the art).

The central research question (RQ1): *does grounding a first-aid chatbot in authoritative guidelines via RAG reduce guideline-discordant and dangerous advice compared to a vanilla LLM?*

The primary metric is **guideline congruence** (does the answer contain the guideline-required actions and avoid forbidden ones), supported by secondary NLP metrics (BLEU, ROUGE, BERTScore, readability).

> ⚠️ This is a research prototype. It must **never** be deployed publicly or used in a real emergency. Every life-threatening query triggers a hardcoded instruction to call emergency services.

---

## Architecture

```
              ┌─────────────────────────────────────────────────────┐
  user query  │                                                     │
  ───────────▶│  SafetyLayer.assess_query()  ── escalation trigger  │
              │            │                                        │
              │            ▼                                        │
              │   DenseRetriever (FAISS)  ◀── guideline corpus      │
              │            │  top-k chunks                          │
              │            ▼                                        │
              │   PROMPT (system + context + question)              │
              │            │                                        │
              │            ▼                                        │
              │   LLMBackend.generate()  ── Ollama│Gemini│HF        │
              │            │  raw answer                            │
              │            ▼                                        │
              │   SafetyLayer.apply()  ── escalation + danger scan  │
              │            │                                        │
  final ◀─────│            ▼   final answer + provenance + flags    │
  answer      └─────────────────────────────────────────────────────┘
```

Everything is behind clean interfaces so components swap without touching the rest:
`LLMBackend`, `DenseRetriever`, `SafetyLayer`, and the `.answer()` contract shared by the RAG bot and both baselines.

---

## Install

```bash
cd safer-firstaid-chatbot
python -m venv .venv && source .venv/bin/activate     # optional
pip install -e .          # installs the package + dependencies
# or: pip install -r requirements.txt
```

Python 3.10+ required.

---

## Choose your LLM backend

The pipeline is **backend-agnostic**. Pick one:

### LM Studio (local, free, OpenAI-compatible GUI)
```bash
# 1. Install LM Studio, download a model (e.g. Mistral 7B Instruct v0.3)
# 2. Developer/Server tab -> load the model there -> Start Server
# 3. Get the exact model id:
curl http://localhost:1234/v1/models
# 4. config: provider: lmstudio, model: <id from step 3>
#    (no API key needed; see configs/lmstudio.yaml)
```

### Ollama (local, free, offline)
```bash
# install from https://ollama.com, then:
ollama pull mistral
export OLLAMA_HOST=http://localhost:11434
# config: provider: ollama, model: mistral
```

### Gemini (API)
```bash
export GOOGLE_API_KEY="your-key"
# config: provider: gemini, model: gemini-1.5-flash
```

### HuggingFace Transformers (local, free)
```bash
# config: provider: huggingface, model: Qwen/Qwen2.5-0.5B-Instruct
# small models run on CPU; larger ones want a GPU
```

Set your choice in `configs/default.yaml`.

---

## Quick start

```bash
# 1. Add guideline documents (PDF/txt) to data/guidelines/
#    - IFRC International First Aid, Resuscitation & Education Guidelines 2020
#    - Resuscitation Council UK Guidelines 2021
#    (A small SAMPLE file is included for testing only — replace it.)

# 2. Build the retrieval index
python -m safer_firstaid.cli build-index --config configs/default.yaml

# 3. Ask a question
python -m safer_firstaid.cli ask "someone is choking, what do I do?" --config configs/default.yaml

# 4. Run the full evaluation (RAG vs. both baselines)
python -m safer_firstaid.cli evaluate --config configs/default.yaml
```

Results are written to `results/`:
- `responses.jsonl` — every response with scores and provenance
- `summary.json` / `summary.csv` — headline metrics per system

### Offline smoke test (no downloads, no API)
```bash
python scripts/demo_end_to_end.py
```
Uses a mock LLM and a hashing embedder to prove the plumbing works end-to-end.

---

## Using the real datasets

1. Download the datasets:
   - Kaggle: `elvisblitti/first-aid`
   - FirstAidQA (arXiv:2511.01289)
2. Convert them:
   ```bash
   python -m safer_firstaid.data.loaders path/to/dataset.csv \
       --question-col question --answer-col answer --source firstaidqa
   ```
   This emits an `intents.json` (for Baseline B) and a **scenarios skeleton**.
3. **By hand**, add a guideline-grounded `checklist` to each scenario in
   `scenarios.json`, citing the IFRC / Resuscitation Council guideline it comes from.
   *This is your intellectual contribution and keeps the evaluation independent.*

---

## Evaluation methodology

**Layer 1 — Guideline congruence (primary).** Each scenario has a checklist of
required actions and forbidden (dangerous) actions drawn from the guidelines. The
automatic scorer does keyword matching for fast iteration; for the reported thesis
results, ≥2 first-aid-qualified assessors score responses and inter-rater agreement
is computed with **Cohen's Kappa** (`evaluation.congruence.cohens_kappa`).

**Layer 2 — NLP metrics (secondary).** BLEU-4, ROUGE-1/2/L, BERTScore-F1, and Flesch
reading ease against reference answers. These measure fluency/similarity, **not
safety**, and must never be reported alone for a safety-critical system.

---

## Project layout

```
src/safer_firstaid/
  llm/            pluggable backends (base, ollama, gemini, hf, mock) + factory
  pipeline/       ingest, retriever (FAISS), safety layer, rag
  baselines/      vanilla_llm, intent_classifier
  evaluation/     congruence (primary), nlp_metrics (secondary), runner
  data/           dataset loaders/converters
  config.py       YAML config
  cli.py          build-index / ask / evaluate
data/
  guidelines/     put guideline PDFs/txt here
  datasets/       scenarios.json, intents.json
  processed/      built FAISS index
configs/          default.yaml
scripts/          demo_end_to_end.py
tests/            pytest unit tests
results/          evaluation outputs
```

---

## Reproducibility

- All hyper-parameters live in `configs/*.yaml`; record the config with each run.
- Generation uses low temperature (0.2) and a fixed seed by default.
- Guideline corpus is version-locked — note the guideline edition/date used.

---

## Ethics & safety

- No human participants; secondary data only.
- Prototype is never publicly deployed.
- Hardcoded emergency-services escalation on life-threatening queries.
- Dangerous-advice patterns are flagged and the response is replaced with a safe fallback.
- All work proceeds only after Arden ARMS ethics approval.

---

## Tests

```bash
pytest -q
```

---

## License / attribution

Academic project. Respect dataset licences (e.g. FirstAidQA is CC BY-NC-SA 4.0) and
guideline copyright — store guideline documents locally; do not redistribute.
