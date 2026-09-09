# Ablation results (§8)

Fixed backend for all ablations: **LM Studio `qwen/qwen3-vl-8b`** — chosen because,
of the backends evaluated so far, it had the lowest RAG dangerous-output rate (0.125
on the earlier n=8 set) and runs locally with no API rate limits or cost, which
matters for a 6-run sweep. Gemini (free tier, 20 req/day) and Groq (8000 TPM) were
ruled out for this reason. Scenario set: the full n=26 set (`data/datasets/scenarios.json`),
scored on the RAG system only — VanillaLLM/IntentClassifier are held fixed as controls
and don't depend on retriever settings.

Raw summaries: `results/ablation_chunk500/`, `results/ablation_chunk800/`,
`results/ablation_chunk1200/`, `results/ablation_topk2/`, `results/ablation_topk6/`
(top_k=4 shares the chunk800 run), `results/ablation_mpnet/`. Configs:
`configs/ablation_*.yaml`. Regenerate tables with `python scripts/ablation_summary.py`;
regenerate the safety-layer statistic with `python scripts/safety_layer_contribution.py`.

## 8a. Chunk size (RAG, n=26)

| chunk_size | mean_congruence | full_congruence_rate | dangerous_output_rate | dangerous_output_count | mean_latency_s |
|---|---|---|---|---|---|
| 500 | 0.8109 | 0.500 | 0.0769 | 2 | 8.83 |
| **800 (default)** | 0.8109 | 0.500 | **0.0385** | **1** | 11.68 |
| 1200 | 0.8013 | 0.3846 | 0.0769 | 2 | 12.96 |

**Interpretation.** 500 and 800 tie on mean congruence and full-congruence rate, but
800 halves the dangerous-output rate (1/26 vs 2/26) — smaller chunks split guideline
steps across more chunk boundaries, so a scenario's answer can retrieve a chunk that
is topically relevant but missing a safety-critical caveat that lived in the
neighbouring chunk. 1200 is worst on full-congruence rate (0.385): larger chunks
dilute the retrieved context with more off-topic guideline text per chunk, crowding
out the specific instruction the query needs within the model's effective attention.
1200 is also slowest (fewer, larger chunks to embed does not offset the larger
per-chunk context fed to the LLM). 800 is the best balance of the three and the right
default.

## 8b(i). top_k (RAG, n=26, chunk_size=800 fixed)

| top_k | mean_congruence | full_congruence_rate | dangerous_output_rate | dangerous_output_count | mean_latency_s |
|---|---|---|---|---|---|
| 2 | 0.7436 | 0.3077 | 0.0769 | 2 | 9.94 |
| **4 (default)** | 0.8109 | 0.500 | **0.0385** | **1** | 11.68 |
| 6 | **0.8590** | **0.6154** | 0.0769 | 2 | 12.35 |

**Interpretation.** Congruence rises monotonically with top_k (2→4→6:
0.744→0.811→0.859 mean, 0.308→0.500→0.615 full-congruence), as expected — more
retrieved chunks means more guideline coverage per query, at a modest latency cost
(~2.4s from top_k=2 to top_k=6). But dangerous-output rate is *not* monotonic:
top_k=4 has the lowest rate (0.0385), while both top_k=2 and top_k=6 tie at 0.0769.
Retrieving more chunks does not guarantee they are the *right* chunks — at top_k=6
the model has more raw material to draw on, including chunks that are topically
adjacent but not the correct instruction, which appears to occasionally introduce a
dangerous phrasing that would not have been retrieved at top_k=4. This is a
congruence/safety trade-off rather than a single winner: top_k=6 is best if
congruence is the priority metric, top_k=4 is best if minimising dangerous-output
rate is the priority (which, per this project's central safety claim, it is) — hence
top_k=4 remains the default.

## 8b(ii). Embedding model (RAG, n=26, chunk_size=800, top_k=4 fixed)

| embedding_model | mean_congruence | full_congruence_rate | dangerous_output_rate | dangerous_output_count | mean_latency_s |
|---|---|---|---|---|---|
| **all-MiniLM-L6-v2 (default)** | **0.8109** | 0.500 | 0.0385 | 1 | 11.68 |
| all-mpnet-base-v2 | 0.7788 | 0.500 | **0.0** | **0** | 12.51 |

**Interpretation.** Both embedders tie on full-congruence rate. MiniLM scores higher
on mean congruence (0.811 vs 0.779), but mpnet eliminates the single dangerous output
MiniLM produced (0/26 vs 1/26). At n=26 a 1-vs-0 count difference is directional, not
statistically robust — it should be reported as suggestive rather than conclusive.
MiniLM (384-dim, faster to embed) remains the reasonable default given it matches or
beats mpnet (768-dim) on every metric except this single dangerous-output count; a
larger scenario set would be needed to confirm whether mpnet's safety edge holds up.

## 8c. Safety-layer contribution (no re-run — read from existing responses.jsonl)

Across all 6 evaluated backends (`lmstudio`, `gemini`, `real`, `qwen`,
`groq_gpt-oss-120b`, `groq_qwen3.6-27b`), **80/80 (100%)** of life-threatening
(escalate-triggering) queries received the mandatory escalation banner in the final
answer — this is expected and confirms the deterministic guarantee: the banner is
hardcoded, so once `safety.assess_query()` flags a query, `SafetyLayer.apply()`
prepends it unconditionally regardless of what the model generated.

The more informative comparison is what the **raw, pre-safety-layer** model output
would have said on its own:

- **VanillaLLM** (short prompt, one line nudging escalation): raw output already
  mentioned calling emergency services unprompted in **25/25 (100%)** of its
  escalate-triggering queries.
- **RAG** (longer prompt: same escalation nudge, plus retrieved guideline context):
  raw output mentioned it in only **20/30 (66.7%)** of its escalate-triggering
  queries.

Both prompts explicitly instruct the model to prioritise calling emergency
services, so neither is a naive zero-instruction baseline — the honest claim is not
"the safety layer raises escalation from 0% to 100%" but that **prompted compliance
is model- and prompt-dependent and only 66.7% for the RAG system** (the added
retrieval context appears to sometimes crowd out the escalation instruction), **while
the safety layer makes escalation a 100% architectural guarantee for every system,
independent of what the LLM decides to say.** That gap (66.7% → 100% for RAG
specifically) is the concrete, quantitative demonstration of the safety layer's
value: it does not rely on model behaviour holding up under distraction from
retrieved content.

*Caveat:* `results/lmstudio/` is a partial run (RAG records only, no
VanillaLLM/IntentClassifier — an earlier aborted run) and contributes only to the
RAG-side counts above.
