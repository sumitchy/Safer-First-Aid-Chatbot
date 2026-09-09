#!/usr/bin/env python3
"""Export blinded (response, checklist-item) pairs for independent human assessment.

Reads scenarios.json and one or more responses.jsonl files, and for each
response writes a row containing a random non-identifying response_id, the
scenario query, the raw response text, and a blank Yes/No answer column per
checklist item (required and forbidden items are mixed together and their
order shuffled, so neither is labelled as such). Writes one CSV per assessor,
each independently shuffled. A private mapping.json (NOT for assessors)
records response_id -> system/scenario/backend and the automatic scorer's
per-item judgement, for later comparison in import_human_eval.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from safer_firstaid.evaluation.congruence import load_scenarios, score_response_automatic  # noqa: E402

INSTRUCTIONS = """Instructions for human assessment

For each row, read the response text and, for every listed item, write "Yes"
or "No" in its matching answer column for whether the response includes
that item. Leave an answer cell blank only if its matching item_N_text cell
is also blank (that response's scenario has fewer checklist items than this
sheet has columns for). Don't try to guess what system produced the
response -- you don't have that information, which is intentional. Judge
only what is actually written in the response text, not what you believe
the correct first-aid answer should be.
"""


def load_responses(paths: list[Path]) -> list[dict]:
    records = []
    for p in paths:
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["_source_file"] = str(p)
                records.append(rec)
    return records


def build_rows(scenarios_by_id: dict, responses: list[dict], rng: random.Random) -> list[dict]:
    rows = []
    for rec in responses:
        scen = scenarios_by_id.get(rec.get("scenario_id"))
        if scen is None:
            print(f"warn: skipping unknown scenario_id {rec.get('scenario_id')!r}", file=sys.stderr)
            continue

        text = rec.get("raw_answer") or rec.get("final_answer") or ""
        auto = score_response_automatic(scen, text)
        matched_required = set(auto.details["matched_required_items"])
        matched_forbidden = set(auto.details["dangerous_items_present"])

        order = list(range(len(scen.checklist)))
        rng.shuffle(order)

        items = []
        for idx in order:
            item = scen.checklist[idx]
            hit = (item.text in matched_forbidden) if item.forbidden else (item.text in matched_required)
            items.append({"text": item.text, "forbidden": item.forbidden, "automatic": int(hit)})

        rows.append({
            "response_id": "R-" + secrets.token_hex(4).upper(),
            "system": rec.get("system", ""),
            "backend": rec.get("backend", ""),
            "scenario_id": scen.id,
            "scenario_query": scen.query,
            "response_text": text,
            "source_file": rec.get("_source_file", ""),
            "items": items,
        })
    return rows


def write_assessor_csv(path: Path, rows: list[dict], max_items: int, rng: random.Random) -> None:
    order = list(range(len(rows)))
    rng.shuffle(order)

    fieldnames = ["response_id", "scenario_query", "response_text"]
    for i in range(1, max_items + 1):
        fieldnames += [f"item_{i}_text", f"item_{i}_answer"]

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i in order:
            row = rows[i]
            out = {
                "response_id": row["response_id"],
                "scenario_query": row["scenario_query"],
                "response_text": row["response_text"],
            }
            for j in range(max_items):
                if j < len(row["items"]):
                    out[f"item_{j + 1}_text"] = row["items"][j]["text"]
                else:
                    out[f"item_{j + 1}_text"] = ""
                out[f"item_{j + 1}_answer"] = ""
            writer.writerow(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", type=Path, default=Path("data/datasets/scenarios.json"))
    ap.add_argument("--responses", type=Path, nargs="+", required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("results/human_eval"))
    ap.add_argument("--assessors", nargs="+", default=["assessor1", "assessor2"])
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    scenarios = load_scenarios(args.scenarios)
    scenarios_by_id = {s.id: s for s in scenarios}

    responses = load_responses(args.responses)
    rows = build_rows(scenarios_by_id, responses, random.Random(args.seed))
    if not rows:
        ap.error("no rows built -- check --responses scenario_ids match --scenarios")
    max_items = max(len(r["items"]) for r in rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(args.assessors):
        assessor_rng = random.Random(args.seed * 1000 + i + 1)
        write_assessor_csv(args.out_dir / f"assessor_{name}.csv", rows, max_items, assessor_rng)

    mapping = {
        row["response_id"]: {
            "system": row["system"],
            "backend": row["backend"],
            "scenario_id": row["scenario_id"],
            "source_file": row["source_file"],
            "items": row["items"],
        }
        for row in rows
    }
    (args.out_dir / "mapping.json").write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    (args.out_dir / "instructions.txt").write_text(INSTRUCTIONS, encoding="utf-8")

    print(f"Wrote {len(rows)} rows for {len(args.assessors)} assessor(s) to {args.out_dir}")
    print(f"Max checklist items per response: {max_items}")
    print("mapping.json is PRIVATE -- do not send it to assessors.")


if __name__ == "__main__":
    main()
