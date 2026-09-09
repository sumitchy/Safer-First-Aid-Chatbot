#!/usr/bin/env python3
"""Import completed human-assessment CSVs and score them.

Computes, per system and overall:
    - inter-rater Cohen's Kappa (assessor vs. assessor), per checklist item
    - automatic-vs-human agreement (agreement rate + Kappa), per assessor

This is the primary-metric report for the human evaluation -- see
scripts/export_for_human_eval.py for how mapping.json and the assessor
CSVs are produced.

Usage:
    python scripts/import_human_eval.py \
        --mapping results/human_eval/mapping.json \
        --assessor-csv alice=results/human_eval/assessor_alice_done.csv \
        --assessor-csv bob=results/human_eval/assessor_bob_done.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from safer_firstaid.evaluation.congruence import cohens_kappa  # noqa: E402


def parse_yes_no(value: str) -> int | None:
    v = (value or "").strip().lower()
    if v in ("yes", "y", "1", "true"):
        return 1
    if v in ("no", "n", "0", "false"):
        return 0
    return None


def load_assessor_csv(path: Path, mapping: dict) -> dict[tuple[str, int], int]:
    """Returns {(response_id, item_index): label}, item_index into mapping[rid]['items']."""
    out: dict[tuple[str, int], int] = {}
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rid = row.get("response_id", "")
            if rid not in mapping:
                print(f"warn: unknown response_id {rid!r} in {path}", file=sys.stderr)
                continue
            n_items = len(mapping[rid]["items"])
            for idx in range(n_items):
                text_col, ans_col = f"item_{idx + 1}_text", f"item_{idx + 1}_answer"
                expected_text = mapping[rid]["items"][idx]["text"]
                if row.get(text_col, "").strip() != expected_text.strip():
                    print(
                        f"warn: item text mismatch for {rid} idx {idx} in {path} "
                        f"(expected {expected_text!r}, got {row.get(text_col)!r})",
                        file=sys.stderr,
                    )
                label = parse_yes_no(row.get(ans_col, ""))
                if label is None:
                    print(f"warn: missing/unparseable answer for {rid} idx {idx} in {path}", file=sys.stderr)
                    continue
                out[(rid, idx)] = label
    return out


def kappa_and_agreement(a: list[int], b: list[int]) -> dict:
    agreement = sum(1 for x, y in zip(a, b) if x == y) / len(a)
    return {"agreement_rate": round(agreement, 4), "kappa": round(cohens_kappa(a, b), 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mapping", type=Path, required=True)
    ap.add_argument(
        "--assessor-csv", action="append", required=True, metavar="NAME=PATH",
        help="repeatable: assessor name and their completed CSV",
    )
    ap.add_argument("--out", type=Path, default=Path("results/human_eval/report.json"))
    args = ap.parse_args()

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))

    assessors: dict[str, dict[tuple[str, int], int]] = {}
    for spec in args.assessor_csv:
        name, _, path = spec.partition("=")
        if not path:
            ap.error(f"--assessor-csv must be NAME=PATH, got {spec!r}")
        assessors[name] = load_assessor_csv(Path(path), mapping)
    assessor_names = list(assessors.keys())
    if len(assessor_names) < 2:
        ap.error("need at least two --assessor-csv to compute inter-rater Kappa")

    all_keys = [(rid, idx) for rid, entry in mapping.items() for idx in range(len(entry["items"]))]
    complete_keys = [k for k in all_keys if all(k in assessors[n] for n in assessor_names)]
    missing = len(all_keys) - len(complete_keys)
    if missing:
        print(f"warn: {missing}/{len(all_keys)} item-judgements missing from >=1 assessor; excluded", file=sys.stderr)
    if not complete_keys:
        ap.error("no item-judgements complete across all assessors -- nothing to score")

    def system_of(k):
        return mapping[k[0]]["system"]

    def automatic_of(k):
        return mapping[k[0]]["items"][k[1]]["automatic"]

    def score(keys: list[tuple[str, int]]) -> dict:
        rep = {"n_items_judged": len(keys), "inter_rater_kappa": {}, "automatic_vs_human": {}}
        for a, b in combinations(assessor_names, 2):
            ra = [assessors[a][k] for k in keys]
            rb = [assessors[b][k] for k in keys]
            rep["inter_rater_kappa"][f"{a}_vs_{b}"] = round(cohens_kappa(ra, rb), 4)
        for a in assessor_names:
            ra = [assessors[a][k] for k in keys]
            rauto = [automatic_of(k) for k in keys]
            rep["automatic_vs_human"][a] = kappa_and_agreement(ra, rauto)
        return rep

    report = {
        "n_items_judged": len(complete_keys),
        "assessors": assessor_names,
        "overall": score(complete_keys),
        "per_system": {},
    }
    for sys_name in sorted({system_of(k) for k in complete_keys}):
        sys_keys = [k for k in complete_keys if system_of(k) == sys_name]
        report["per_system"][sys_name] = score(sys_keys)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nWrote report to {args.out}")


if __name__ == "__main__":
    main()
