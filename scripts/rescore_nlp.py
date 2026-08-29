"""Retroactively compute NLP metrics (BLEU/ROUGE/BERTScore) for existing results.

BERTScore previously segfaulted this machine mid-run (Section 6.3/10.5 of
documentation_of_code.md), forcing every evaluation to use --no-nlp. The fix
(evaluation/bertscore_subprocess.py) isolates BERTScore in a subprocess so a
crash yields NaN instead of killing the run -- but it has never been exercised
against a real results directory.

Since responses.jsonl already stores each system's raw_answer, and
scenarios.json stores each scenario's reference_answer, we don't need to
re-run any LLM/API to get NLP metrics -- just score the same text that's
already on disk. Zero cost, zero re-run.

Usage:
    python scripts/rescore_nlp.py <results_dir> [<results_dir> ...] --scenarios <path>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from safer_firstaid.evaluation.congruence import load_scenarios
from safer_firstaid.evaluation.nlp_metrics import average_metrics, compute_nlp_metrics


def rescore_dir(results_dir: Path, scenarios_path: Path) -> None:
    responses_path = results_dir / "responses.jsonl"
    if not responses_path.exists():
        print(f"[skip] {results_dir}: no responses.jsonl")
        return

    scenarios = {s.id: s for s in load_scenarios(scenarios_path)}
    records = [json.loads(line) for line in responses_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    print(f"\n=== {results_dir} ===")
    by_system_metrics: dict[str, list] = {}

    for record in records:
        scen = scenarios.get(record["scenario_id"])
        if scen is None or not scen.reference_answer:
            print(f"  [skip] {record['scenario_id']!r}: no reference_answer")
            continue

        raw_answer = record.get("raw_answer") or record.get("final_answer", "")
        nlp = compute_nlp_metrics(raw_answer, scen.reference_answer)
        record["nlp"] = nlp.__dict__

        flag = " [BERTSCORE=NaN]" if nlp.bertscore_f1 != nlp.bertscore_f1 else ""
        print(f"  [{record['system']}/{record['scenario_id']}] bleu={nlp.bleu} rouge1={nlp.rouge1} bertscore_f1={nlp.bertscore_f1}{flag}")

        by_system_metrics.setdefault(record["system"], []).append(nlp)

    out_path = results_dir / "responses_with_nlp.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    print(f"  Wrote {out_path}")

    summary = {name: average_metrics(metrics) for name, metrics in by_system_metrics.items()}
    summary_path = results_dir / "summary_nlp.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Wrote {summary_path}")

    csv_path = results_dir / "summary_nlp.csv"
    _write_csv(summary, csv_path)
    print(f"  Wrote {csv_path}")


def _write_csv(summary: dict, path: Path) -> None:
    cols = ["system", "bleu", "rouge1", "rouge2", "rougeL", "bertscore_f1", "flesch_reading_ease"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for name, s in summary.items():
            row = {"system": name}
            for c in cols[1:]:
                row[c] = s.get(c, "")
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dirs", nargs="+", type=Path)
    parser.add_argument("--scenarios", required=True, type=Path)
    args = parser.parse_args()

    for results_dir in args.results_dirs:
        rescore_dir(results_dir, args.scenarios)


if __name__ == "__main__":
    main()
