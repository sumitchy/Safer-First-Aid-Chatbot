# BUILD GUIDE — Safer First-Aid Chatbots

**A complete, self-contained specification for an AI coding agent (Claude Code) to build, run, verify, and extend this research project.**

Project: *Safer First-Aid Chatbots: Reducing Guideline-Discordant Advice Through Retrieval-Augmented Generation Grounded in Red Cross and Resuscitation Council Guidelines*
Author: Sumit Chaudhary (24161337) · Arden University · Project P20435 (Postgraduate)

---

## 0. How to use this document

This guide is written so that a coding agent can reconstruct the entire project from nothing, OR extend the existing codebase. Work through the phases **in order**. Each phase has: a goal, the files to create, the exact acceptance test to run, and the expected output. Do not move to the next phase until the acceptance test passes.

**Golden rules for the agent:**
1. Never deploy this chatbot publicly or connect it to real users — it is a research prototype for a safety-critical domain.
2. The safety layer (`pipeline/safety.py`) is non-negotiable. Every generation path must pass through it. Never add a code path that returns a raw LLM answer to the user without the safety wrapper.
3. Do not auto-generate the evaluation "gold" checklists with an LLM. Those must be human-written from the guidelines to keep the evaluation independent. Leave them as `[]` for a human to fill.
4. Keep the three systems (RAG, VanillaLLM, IntentClassifier) sharing the *same* safety layer and the *same* `.answer()` contract so comparisons are fair.
5. Prefer local, free backends (Ollama / LM Studio / HuggingFace) so the project runs with zero API cost.

---

## 1. What you are building (mental model)

Three question-answering systems that all expose the same method `.answer(query) -> response`:

| System | What it does | Represents |
|---|---|---|
| **RAG** | retrieves guideline passages, grounds an LLM in them, generates an answer | the proposed solution |
| **VanillaLLM** | same LLM, no retrieval | current general-purpose chatbots |
| **IntentClassifier** | TF-IDF + Logistic Regression → canned answer | rule-based FAQ bots |

All three are wrapped by a deterministic **SafetyLayer** that (a) forces an emergency-services escalation banner on life-threatening queries and (b) blocks known-dangerous advice.

An **evaluation harness** runs all three over a set of **scenarios**, scoring each answer on **guideline congruence** (primary) and **NLP metrics** (secondary), and writes CSV/JSON results.

Data flow:

```
query ──▶ SafetyLayer.assess_query ──▶ Retriever(FAISS) ──▶ prompt ──▶ LLMBackend
                                                                          │
result ◀── SafetyLayer.apply ◀──────────────────── raw answer ◀──────────┘
```

---

## 2. Environment and dependencies

- Python **3.10+** (3.12 recommended).
- OS: Linux/macOS/Windows. GPU optional (only speeds up HuggingFace local models).

Install (editable, so `src/` changes take effect immediately):

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -e .                    # uses pyproject.toml
```

If `pip install -e .` is not desired, `pip install -r requirements.txt` then run modules with `PYTHONPATH=src`.

Core dependencies and why:

| Package | Purpose |
|---|---|
| `sentence-transformers` | dense embeddings for retrieval |
| `faiss-cpu` | vector similarity search |
| `langchain-text-splitters` | recursive chunking of guideline docs |
| `pypdf` | read guideline PDFs |
| `scikit-learn` | TF-IDF + Logistic Regression intent baseline |
| `sacrebleu`, `rouge-score`, `bert-score` | evaluation metrics |
| `ollama`, `google-generativeai`, `transformers` | LLM backends (all optional) |
| `pyyaml`, `pydantic` | config |
| `rich`, `typer` | CLI |

LM Studio / vLLM / OpenAI need **no extra package** — the backend uses stdlib `urllib`.

---

## 3. Directory layout (target end state)

```
safer-firstaid-chatbot/
├── pyproject.toml
├── requirements.txt
├── README.md
├── BUILD_GUIDE.md                 ← this file
├── configs/
│   ├── default.yaml
│   ├── lmstudio.yaml
│   └── offline_test.yaml
├── data/
│   ├── guidelines/                ← put IFRC 2020 + Resus Council UK 2021 here
│   │   └── sample_first_aid_guidelines.txt   (replace for real results)
│   ├── datasets/
│   │   ├── scenarios.json         ← eval cases + guideline checklists
│   │   └── intents.json           ← intents for the FAQ baseline
│   └── processed/                 ← built FAISS index (generated)
├── notebooks/
│   └── quickstart.ipynb
├── scripts/
│   └── demo_end_to_end.py
├── src/safer_firstaid/
│   ├── __init__.py
│   ├── config.py
│   ├── cli.py
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py                ← LLMBackend ABC + GenerationConfig
│   │   ├── backends.py            ← Ollama, Gemini, HF, LM Studio, OpenAI-compat, factory
│   │   └── mock.py                ← offline test backend
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── ingest.py              ← load + chunk guideline docs
│   │   ├── retriever.py           ← sentence-transformers + FAISS (+ hashing fallback)
│   │   ├── safety.py              ← escalation + dangerous-advice guard
│   │   └── rag.py                 ← the RAG chatbot
│   ├── baselines/
│   │   ├── __init__.py
│   │   ├── vanilla_llm.py         ← Baseline A
│   │   └── intent_classifier.py   ← Baseline B
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── congruence.py          ← primary metric + Cohen's Kappa
│   │   ├── nlp_metrics.py         ← BLEU/ROUGE/BERTScore/readability
│   │   └── runner.py              ← runs all systems, writes results
│   └── data/
│       ├── __init__.py
│       └── loaders.py             ← convert Kaggle/FirstAidQA into project formats
├── tests/
│   └── test_core.py
└── results/                       ← evaluation outputs (generated)
```

---

## 4. Build order (phased, each with an acceptance test)

### Phase 1 — LLM backend interface
**Goal:** a pluggable generation layer so any model can be swapped in.

Create `src/safer_firstaid/llm/base.py`:
- `GenerationConfig` dataclass: `temperature=0.2, max_tokens=512, top_p=0.9, seed=42, stop=[]`.
- `LLMBackend` ABC with abstract `generate(prompt, config) -> str`, a `name` attribute, and a default `health_check()`.
- `LLMBackendError(RuntimeError)`.

Create `src/safer_firstaid/llm/backends.py` with these classes, all subclassing `LLMBackend`:
- `OllamaBackend(model="mistral", host=...)` → uses the `ollama` package; reads `OLLAMA_HOST`.
- `GeminiBackend(model="gemini-1.5-flash", api_key=...)` → uses `google.generativeai`; reads `GOOGLE_API_KEY`.
- `HuggingFaceBackend(model="Qwen/Qwen2.5-0.5B-Instruct", device=...)` → uses `transformers.pipeline("text-generation")`, applies chat template if available, lazy-loads the model.
- `OpenAICompatibleBackend(model, base_url="http://localhost:1234/v1", api_key)` → **stdlib `urllib` only**, POSTs to `/chat/completions`, parses `choices[0].message.content`; has `list_models()` hitting `GET /v1/models`.
- `LMStudioBackend(...)` → subclass of `OpenAICompatibleBackend` defaulting to `http://localhost:1234/v1` and dummy key `"lm-studio"`; reads `LMSTUDIO_HOST`.
- `MockBackend` lives in `mock.py`; echoes retrieved context (for offline tests).
- `build_backend(provider, model=None, **kwargs)` factory mapping: `ollama, gemini, huggingface/hf, lmstudio/lm-studio, openai/openai-compatible, vllm, mock`.

All backends must raise `LLMBackendError` on failure (never a raw provider exception) so callers handle them uniformly. All heavy imports are **lazy** (inside methods) so installing one provider is enough.

**Acceptance test:**
```bash
python -c "
import sys; sys.path.insert(0,'src')
from safer_firstaid.llm import build_backend
for p in ['ollama','gemini','hf','lmstudio','vllm','mock']:
    b = build_backend(p, 'x') if p!='vllm' else build_backend(p,'x',base_url='http://localhost:8000/v1')
    print(p, '->', b.name)
"
```
Expect one line per provider, no exceptions.

---

### Phase 2 — Document ingestion + retriever
**Goal:** turn guideline documents into a searchable FAISS index.

`pipeline/ingest.py`:
- `Document` dataclass: `id, text, source, chunk_index, metadata`.
- `load_text(path)` handles `.pdf` (via `pypdf`) and `.txt/.md`.
- `chunk_text(text, source, chunk_size=800, chunk_overlap=150, metadata)` uses `RecursiveCharacterTextSplitter` with separators `["\n\n","\n",". "," ",""]`; ids via short sha1.
- `build_corpus(guidelines_dir, ...)` ingests every `.pdf/.txt/.md`; raises if the dir is empty.
- `save_corpus` / `load_corpus` as JSONL.

`pipeline/retriever.py`:
- `RetrievedChunk` dataclass: `document, score`.
- `DenseRetriever(embedding_model="sentence-transformers/all-MiniLM-L6-v2")`.
  - `_embed` uses `SentenceTransformer.encode(..., normalize_embeddings=True)`.
  - **Offline fallback:** if `embedding_model == "hashing"`, use a deterministic md5-hashing bag-of-words embedder (no download) — for demos/CI only.
  - `build(docs)` → FAISS `IndexFlatIP` (cosine via normalised inner product).
  - `save(dir)` / `load(dir)` persist `index.faiss`, `docs.jsonl`, `meta.txt`.
  - `retrieve(query, k=4) -> list[RetrievedChunk]`.

**Acceptance test** (offline embedder, uses the sample guideline file):
```bash
python -c "
import sys; sys.path.insert(0,'src'); from pathlib import Path
from safer_firstaid.pipeline import build_corpus, DenseRetriever
docs = build_corpus(Path('data/guidelines'), chunk_size=500, chunk_overlap=100)
r = DenseRetriever('hashing'); r.build(docs)
hits = r.retrieve('someone is choking', k=2)
print('chunks:', len(docs), '| top source:', hits[0].document.source)
assert 'choking' in hits[0].document.text.lower() or hits
print('OK')
"
```

---

### Phase 3 — Safety layer
**Goal:** deterministic escalation + dangerous-advice guard. This is the project's core contribution — treat it as critical code.

`pipeline/safety.py`:
- Constants: `EMERGENCY_NUMBER`, `LIFE_THREATENING_TRIGGERS` (choking, not breathing, cardiac arrest, severe bleeding, anaphylaxis, stroke, seizure, unconscious, suicide, …), `DANGEROUS_ADVICE_PATTERNS` (regex: "do not call 999", "induce vomiting", "remove the knife/blade/object", giving food/water to the unconscious, tourniquet on neck, …).
- `SafetyDecision` dataclass: `escalate, triggered_terms, dangerous_flags`, plus `is_dangerous`.
- `SafetyLayer`:
  - `assess_query(query)` → sets `escalate` if any trigger present.
  - `escalation_banner()` → the mandatory "Call 999…" message.
  - `scan_response(response)` → flags dangerous patterns.
  - `apply(query, response) -> (final_text, SafetyDecision)`: prepends banner when escalating; if dangerous flags present, **replaces** the answer with a safe fallback; otherwise appends a "general guidance, not a substitute for professional care" disclaimer.

**Acceptance test:**
```bash
python -c "
import sys; sys.path.insert(0,'src')
from safer_firstaid.pipeline.safety import SafetyLayer
s = SafetyLayer()
f1,d1 = s.apply('he is not breathing','Do chest compressions.')
assert d1.escalate and '999' in f1
f2,d2 = s.apply('cut with glass','remove the knife and clean it')
assert d2.is_dangerous and 'can' in f2.lower()
print('OK')
"
```

---

### Phase 4 — RAG chatbot + baselines
**Goal:** the three systems, all sharing safety and a common `.answer()`.

`pipeline/rag.py`:
- `SYSTEM_INSTRUCTION` (answer ONLY from context; numbered steps; never invent doses; escalate if life-threatening).
- `PROMPT_TEMPLATE` (system + guideline context + question).
- `RAGResponse` dataclass: `query, answer, raw_answer, retrieved, safety, backend_name`, with `provenance()`.
- `RAGChatbot(retriever, backend, safety, top_k=4, gen_config)` with `answer(query)`: retrieve → build context → generate → `safety.apply`.

`baselines/vanilla_llm.py`:
- `VanillaLLMBaseline(backend, safety, gen_config)` — same as RAG but **no retrieval**; `VANILLA_TEMPLATE` has no context block. `BaselineResponse` mirrors `RAGResponse` schema (empty `retrieved`, empty `provenance`).

`baselines/intent_classifier.py`:
- `IntentClassifierBaseline(safety, confidence_threshold=0.35)`.
- `fit(intents_dict)` → `TfidfVectorizer(ngram_range=(1,2))` + `LogisticRegression(class_weight="balanced")`; stores canned answers.
- `from_json(path)` loads `{intent: {examples:[...], answer:"..."}}`.
- `answer(query)`: predict intent + confidence; if below threshold, abstain with a safe referral; else return the canned answer — all through `safety.apply`.

**Design invariant:** RAG and VanillaLLM must be constructed with the **same backend instance and same SafetyLayer** in experiments, so the only difference is retrieval.

**Acceptance test** (offline, mock backend + hashing embedder):
```bash
python scripts/demo_end_to_end.py
```
Expect: "ESCALATED: True" for the cardiac-arrest demo, and a SUMMARY table with three systems and `DangerousOut = 0`.

---

### Phase 5 — Evaluation harness
**Goal:** score every system on guideline congruence (primary) + NLP metrics (secondary).

`evaluation/congruence.py`:
- `ChecklistItem(text, keywords, forbidden=False)`.
- `Scenario(id, query, reference_answer, checklist, source)` with `required_items` / `forbidden_items`.
- `CongruenceResult(...)` with `congruence`, `full_congruence`, `dangerous`.
- `score_response_automatic(scenario, response_text)` — keyword matching (word-boundary aware).
- `aggregate(results)` → `mean_congruence`, `full_congruence_rate`, `dangerous_output_rate/count`.
- `cohens_kappa(rater_a, rater_b)` for inter-rater agreement of human assessors.
- `load_scenarios(path)`.

`evaluation/nlp_metrics.py`:
- `NLPMetrics` dataclass.
- `compute_nlp_metrics(prediction, reference)` — sacrebleu BLEU, rouge-score ROUGE-1/2/L, bert-score F1 (lazy; NaN if unavailable), Flesch reading ease.
- `average_metrics(list)`.

`evaluation/runner.py`:
- `SystemSpec(name, system)`.
- `run_evaluation(systems, scenarios, out_dir, compute_nlp=True)`:
  - For each system × scenario: time it, get `resp`, **score the `raw_answer`** (not the safety-wrapped text, so the always-present "call 999" banner doesn't inflate congruence — document this choice in the thesis).
  - Write `responses.jsonl`, `summary.json`, `summary.csv`.

**Acceptance test:**
```bash
python -m pytest tests/ -q     # 7 tests must pass
```

---

### Phase 6 — Config + CLI
**Goal:** run everything from the command line, driven by YAML.

`config.py`: `LLMConfig, RetrieverConfig, PathsConfig, AppConfig`; `load_config(path)` merges YAML over defaults.

`cli.py` (typer app):
- `build-index --config` → ingest + build + save FAISS.
- `ask "question" --config` → RAG answer with sources.
- `evaluate --config [--no-nlp]` → run all three systems, write results.

Register `safer-firstaid = "safer_firstaid.cli:app"` in `pyproject.toml [project.scripts]`.

**Acceptance test** (offline config provided):
```bash
safer-firstaid build-index --config configs/offline_test.yaml
safer-firstaid ask "someone is choking on food" --config configs/offline_test.yaml
safer-firstaid evaluate --config configs/offline_test.yaml --no-nlp
```
Expect the choking guideline retrieved, an escalation banner, and a 3-system summary.

---

## 5. LLM backend setup (all options)

Pick ONE and set it in your chosen `configs/*.yaml`.

### Option A — LM Studio (local, free, GUI, OpenAI-compatible)  ★ requested
1. Install LM Studio, download a model (e.g. *Mistral 7B Instruct v0.3*).
2. Open the **Developer / Server** tab, **load the model there** (the Chat-tab model does not carry over), click **Start Server**.
3. Confirm it runs at `http://localhost:1234/v1`.
4. Get the exact model id: `curl http://localhost:1234/v1/models` (it may include a quantization suffix).
5. In `configs/lmstudio.yaml` set `provider: lmstudio` and `model: <that id>`.
6. Run: `safer-firstaid ask "someone is choking" --config configs/lmstudio.yaml`.

No API key is needed; LM Studio ignores it. Common gotcha: a **500 error** means the `model` id doesn't match `/v1/models`.

### Option B — Ollama (local, free, CLI/daemon)
```bash
ollama pull mistral
# provider: ollama, model: mistral  (reads OLLAMA_HOST, default http://localhost:11434)
```

### Option C — HuggingFace Transformers (local, free, in-process)
```yaml
llm: { provider: huggingface, model: Qwen/Qwen2.5-0.5B-Instruct }
```
Small models run on CPU; larger want a GPU. First run downloads weights.

### Option D — Gemini (API, needs key)
```bash
export GOOGLE_API_KEY="your-key"
# provider: gemini, model: gemini-1.5-flash
```

### Option E — vLLM / any OpenAI-compatible server
```yaml
llm: { provider: vllm, model: <served-model> }   # set base_url via code/kwargs, default :8000/v1
```

### Backend comparison as a thesis result
Because the backend is pluggable, running the **same** evaluation across LM Studio, Ollama, and Gemini and comparing guideline congruence is a strong, low-effort addition to Chapter 5 (generator sensitivity analysis).

---

## 6. Getting to REAL (thesis-grade) results

The sample data makes the pipeline run; it does **not** produce reportable results. To get there:

1. **Guideline corpus.** Replace `data/guidelines/sample_first_aid_guidelines.txt` with the full authoritative documents:
   - IFRC *International First Aid, Resuscitation and Education Guidelines* (2020)
   - Resuscitation Council UK Guidelines (2021)
   Store locally; note the exact edition/date (version-lock). Rebuild the index.

2. **Datasets.** Download and convert:
   ```bash
   python -m safer_firstaid.data.loaders path/to/firstaidqa.csv \
       --question-col question --answer-col answer --source firstaidqa
   ```
   This emits `intents.json` (baseline B) and a **scenarios skeleton** with empty checklists.
   - Kaggle `elvisblitti/first-aid` — **verify size/format on download**; if < ~200 usable pairs, rely on FirstAidQA (5,500 pairs, CC BY-NC-SA 4.0) instead. (This is risk **R05** in the risk assessment.)

3. **Write checklists by hand.** For each scenario, add `checklist` items (required + forbidden) grounded in a specific guideline, with the `source` cited. **Do not** generate these with an LLM.

4. **Choose a real backend** (Section 5) and rebuild the index with a real embedder (drop `hashing`).

5. **Human evaluation.** Have ≥2 first-aid-qualified assessors score responses; compute **Cohen's Kappa** (`evaluation.congruence.cohens_kappa`). Report the automatic scores as a development proxy and the human scores as the headline result.

6. **Ablations** (each a results subsection): chunk size, `top_k`, embedding model, backend model, with/without safety layer.

**Success criterion (RQ1):** RAG full-congruence should substantially exceed the ~7–17% baseline reported for general chatbots and beat the VanillaLLM baseline by a defensible margin.

---

## 7. Verification checklist (run before declaring done)

```bash
pip install -e .
python -m pytest tests/ -q                      # 7 passed
python scripts/demo_end_to_end.py               # runs, ESCALATED: True, 3-system summary
safer-firstaid build-index --config configs/offline_test.yaml
safer-firstaid evaluate  --config configs/offline_test.yaml --no-nlp
# then with a real backend:
safer-firstaid ask "someone is choking" --config configs/lmstudio.yaml
```

All must succeed with no tracebacks. `results/summary.csv` should contain rows for `RAG`, `VanillaLLM`, `IntentClassifier`.

---

## 8. Extension roadmap (optional, for a stronger thesis)

- **Reranking:** add a cross-encoder reranker after FAISS retrieval; measure congruence delta.
- **Offline/edge variant:** quantised small model via LM Studio/Ollama for low-connectivity settings (ties to the FirstAidQA motivation).
- **CrisisMMD input filter (optional):** a classifier that routes/annotates incoming messages; keep it clearly optional (risk **R09** — scope creep).
- **Streaming CLI:** add token streaming for the `ask` command (LM Studio/Ollama support it).
- **Structured output:** ask the model for JSON steps and validate against a schema.

Each extension should be gated behind a config flag and evaluated with the same harness so results stay comparable.

---

## 9. Academic-integrity guardrails (important)

This codebase is a **foundation you extend**, not a finished submission. The examinable research contribution is yours:
- the hand-written guideline checklists,
- the choice/justification of architecture and hyper-parameters,
- the evaluation design and human assessment,
- the analysis and discussion.

The agent should build and wire the components and leave clearly-marked `TODO(human)` where independent scholarly judgement is required (checklists, guideline curation, interpretation). Do not fabricate results, citations, or evaluation gold data.

---

## 10. Quick reference — key APIs

```python
from safer_firstaid.llm import build_backend, GenerationConfig
from safer_firstaid.pipeline import (
    build_corpus, DenseRetriever, RAGChatbot, SafetyLayer,
)
from safer_firstaid.baselines import VanillaLLMBaseline, IntentClassifierBaseline
from safer_firstaid.evaluation import (
    load_scenarios, run_evaluation, SystemSpec, cohens_kappa,
)

backend   = build_backend("lmstudio", "local-model")      # or ollama/gemini/hf/vllm
retriever = DenseRetriever(); retriever.build(build_corpus("data/guidelines"))
rag       = RAGChatbot(retriever, backend, SafetyLayer(), top_k=4)

print(rag.answer("someone is choking, what do I do?").answer)
```

---

*End of build guide. Build in phase order, run each acceptance test, and keep the safety layer sacred.*
