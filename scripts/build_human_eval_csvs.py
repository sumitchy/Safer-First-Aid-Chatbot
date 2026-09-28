#!/usr/bin/env python3
"""Build the blinded human-eval CSVs from human_eval_template.csv + real responses.

Reads the checklist template (scenario_id, query, checklist_item_id,
guideline_statement, item_type) and two full-26-scenario responses.jsonl
files (RAG/VanillaLLM/IntentClassifier x qwen3-vl-8b, deepseek-r1). Samples
45 (system, backend, scenario_id) response triples stratified across the
three systems and across scenario_ids, expands each into per-checklist-item
rows with response_text filled in and a blinded sequential response_id, then
writes two independently-shuffled assessor CSVs plus a private answer key.
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

TEMPLATE = Path("results/human_eval_template.csv")
OUT_DIR = Path("results/human_eval")
RESPONSE_FILES = {
    "qwen3-vl-8b": Path("results/qwen/responses.jsonl"),
    "deepseek-r1": Path("results/deepseek/responses.jsonl"),
}
SYSTEMS = ["RAG", "VanillaLLM", "IntentClassifier"]
PER_SYSTEM = 15
SAMPLE_SEED = 42
ASSESSOR_SEEDS = {"assessor_1.csv": 43, "assessor_2.csv": 44}


def load_template() -> tuple[dict[str, dict], dict[str, list[dict]]]:
    rows = list(csv.DictReader(TEMPLATE.open(encoding="utf-8")))
    queries: dict[str, str] = {}
    items_by_scenario: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        sid = r["scenario_id"]
        queries[sid] = r["query"]
        items_by_scenario[sid].append(
            {"guideline_statement": r["guideline_statement"], "item_type": r["item_type"]}
        )
    return queries, items_by_scenario


def load_responses() -> dict[str, dict[str, dict[str, str]]]:
    """backend -> system -> scenario_id -> raw_answer (skips empty raw_answer)."""
    out: dict[str, dict[str, dict[str, str]]] = {}
    for backend, path in RESPONSE_FILES.items():
        by_system: dict[str, dict[str, str]] = defaultdict(dict)
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = (rec.get("raw_answer") or "").strip()
                if not text:
                    continue
                by_system[rec["system"]][rec["scenario_id"]] = text
        out[backend] = by_system
    return out


def sample_pairs(responses: dict, queries: dict) -> list[dict]:
    rng = random.Random(SAMPLE_SEED)
    all_scenario_ids = sorted(queries.keys())
    selected: list[dict] = []

    for system in SYSTEMS:
        if system == "IntentClassifier":
            # backend-independent (verified identical across backends) -- dedupe to one
            # canonical backend, spread picks across distinct scenario_ids.
            available = [
                sid for sid in all_scenario_ids
                if sid in responses["qwen3-vl-8b"].get("IntentClassifier", {})
            ]
            rng.shuffle(available)
            picks = available[:PER_SYSTEM]
            for sid in picks:
                selected.append({
                    "system": system,
                    "backend": "n/a (backend-independent)",
                    "scenario_id": sid,
                    "raw_answer": responses["qwen3-vl-8b"][system][sid],
                })
            continue

        # RAG / VanillaLLM: alternate backend per scenario pick so each backend
        # gets roughly half, falling back to whichever backend actually has
        # that scenario_id (covers the 2 empty deepseek RAG records).
        scenario_order = all_scenario_ids[:]
        rng.shuffle(scenario_order)
        backend_cycle = ["qwen3-vl-8b", "deepseek-r1"]
        rng.shuffle(backend_cycle)

        picks = []
        bi = 0
        for sid in scenario_order:
            if len(picks) >= PER_SYSTEM:
                break
            preferred = backend_cycle[bi % 2]
            fallback = backend_cycle[(bi + 1) % 2]
            for backend in (preferred, fallback):
                if sid in responses[backend].get(system, {}):
                    picks.append((sid, backend))
                    bi += 1
                    break

        for sid, backend in picks:
            selected.append({
                "system": system,
                "backend": backend,
                "scenario_id": sid,
                "raw_answer": responses[backend][system][sid],
            })

    return selected


def main() -> None:
    queries, items_by_scenario = load_template()
    responses = load_responses()
    selected = sample_pairs(responses, queries)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Assign blinded sequential response_ids in a deterministic order.
    selected.sort(key=lambda s: (s["system"], s["scenario_id"], s["backend"]))
    for i, s in enumerate(selected, start=1):
        s["response_id"] = f"RESP-{i:04d}"

    # Build full checklist-item rows per response.
    all_rows: list[dict] = []
    for s in selected:
        items = items_by_scenario[s["scenario_id"]]
        for j, item in enumerate(items, start=1):
            all_rows.append({
                "response_id": s["response_id"],
                "scenario_id": s["scenario_id"],
                "query": queries[s["scenario_id"]],
                "response_text": s["raw_answer"],
                "checklist_item_id": f"{s['response_id']}_item_{j}",
                "guideline_statement": item["guideline_statement"],
                "item_type": item["item_type"],
                "assessor_judgment": "",
                "assessor_notes": "",
            })

    # answer_key.csv -- private, keeps system/backend mapping.
    with (OUT_DIR / "answer_key.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["response_id", "system", "backend", "scenario_id", "source_file"])
        w.writeheader()
        for s in selected:
            source = (
                "both (identical across backends)"
                if s["system"] == "IntentClassifier"
                else str(RESPONSE_FILES[s["backend"]])
            )
            w.writerow({
                "response_id": s["response_id"],
                "system": s["system"],
                "backend": s["backend"],
                "scenario_id": s["scenario_id"],
                "source_file": source,
            })

    # Blinded assessor CSVs -- shuffled at response level, no system/backend column.
    blinded_fields = [
        "response_id", "scenario_id", "query", "response_text",
        "checklist_item_id", "guideline_statement", "item_type",
        "assessor_judgment", "assessor_notes",
    ]
    rows_by_response: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        rows_by_response[r["response_id"]].append(r)

    response_ids = [s["response_id"] for s in selected]
    for fname, seed in ASSESSOR_SEEDS.items():
        order = response_ids[:]
        random.Random(seed).shuffle(order)
        with (OUT_DIR / fname).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=blinded_fields)
            w.writeheader()
            for rid in order:
                for row in rows_by_response[rid]:
                    w.writerow({k: row[k] for k in blinded_fields})

    # Summary counts.
    by_system = defaultdict(int)
    by_backend = defaultdict(int)
    for s in selected:
        by_system[s["system"]] += 1
        by_backend[s["backend"]] += 1
    print(f"responses sampled: {len(selected)}")
    print(f"total checklist rows: {len(all_rows)}")
    print(f"per-system: {dict(by_system)}")
    print(f"per-backend: {dict(by_backend)}")
    print(f"distinct scenario_ids covered: {len(set(s['scenario_id'] for s in selected))} / {len(queries)}")


if __name__ == "__main__":
    main()
