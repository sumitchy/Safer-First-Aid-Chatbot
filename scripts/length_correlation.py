"""Per-record correlation of raw_answer word count against congruence and
bertscore_f1, across every results directory's responses_with_nlp.jsonl.

Section 14.4.1 of documentation_of_code.md found, at the per-system/per-backend
mean level (n=8 each), that whichever system wrote longer answers had lower
BERTScore. Section 14.7 item 3 flagged that this had not been checked at the
individual-response level, where the aggregate pattern could be an artefact
of averaging rather than a real per-record relationship. This script checks
that directly: one row per stored record (n~128 across all backends/systems),
word count of raw_answer vs. congruence and vs. bertscore_f1, Pearson and
Spearman correlations for each pair.

Usage:
    python scripts/length_correlation.py <results_dir> [<results_dir> ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scipy import stats


def load_records(results_dir: Path) -> list[dict]:
    path = results_dir / "responses_with_nlp.jsonl"
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dirs", nargs="+", type=Path)
    args = parser.parse_args()

    rows = []
    for results_dir in args.results_dirs:
        for rec in load_records(results_dir):
            nlp = rec.get("nlp") or {}
            congruence = rec.get("congruence") or {}
            bertscore = nlp.get("bertscore_f1")
            cong = congruence.get("congruence")
            raw_answer = rec.get("raw_answer") or ""
            if bertscore is None or cong is None or not raw_answer:
                continue
            rows.append(
                {
                    "results_dir": results_dir.name,
                    "system": rec.get("system"),
                    "scenario_id": rec.get("scenario_id"),
                    "word_count": len(raw_answer.split()),
                    "congruence": cong,
                    "bertscore_f1": bertscore,
                }
            )

    print(f"n = {len(rows)} records across {len(args.results_dirs)} results dirs\n")

    words = [r["word_count"] for r in rows]
    cong = [r["congruence"] for r in rows]
    bert = [r["bertscore_f1"] for r in rows]

    for label, other in (("congruence", cong), ("bertscore_f1", bert)):
        pearson_r, pearson_p = stats.pearsonr(words, other)
        spearman_r, spearman_p = stats.spearmanr(words, other)
        print(f"word_count vs {label}:")
        print(f"  Pearson  r = {pearson_r:+.4f}  p = {pearson_p:.4g}")
        print(f"  Spearman r = {spearman_r:+.4f}  p = {spearman_p:.4g}")
        print()

    out_path = Path("results") / "length_correlation_records.csv"
    import csv

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
