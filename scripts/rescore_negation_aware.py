"""Retroactively re-score existing results with the negation-aware congruence scorer.

Session 3 / Section 10.8 found that the old `score_response_automatic` flagged
forbidden phrases by plain substring presence, with no check for negation. This
produced false "dangerous output" flags on responses correctly warning the user
AGAINST an unsafe action (see CHANGELOG.md).

Since `responses.jsonl` already stores each system's `raw_answer`, we don't need
to re-run any LLM to fix historical results -- we just re-score the same text
with the fixed logic. This is a zero-cost audit: no API budget spent.

Usage:
    python scripts/rescore_negation_aware.py <results_dir> [<results_dir> ...] --scenarios <path>
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from safer_firstaid.evaluation.congruence import (
    aggregate,
    load_scenarios,
    score_response_automatic,
)


def rescore_dir(results_dir: Path, scenarios_path: Path) -> None:
    responses_path = results_dir / "responses.jsonl"
    if not responses_path.exists():
        print(f"[skip] {results_dir}: no responses.jsonl")
        return

    scenarios = {s.id: s for s in load_scenarios(scenarios_path)}

    records = [json.loads(line) for line in responses_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    print(f"\n=== {results_dir} ===")
    changed = 0
    by_system_results: dict[str, list] = {}

    for record in records:
        scen = scenarios.get(record["scenario_id"])
        if scen is None:
            print(f"  [warn] unknown scenario_id {record['scenario_id']!r}, skipping")
            continue

        old_congruence = record["congruence"]
        raw_answer = record.get("raw_answer") or record.get("final_answer", "")
        new_result = score_response_automatic(scen, raw_answer)
        new_congruence = new_result.to_json()

        if bool(old_congruence.get("dangerous")) != bool(new_congruence["dangerous"]):
            changed += 1
            print(
                f"  CHANGED [{record['system']}/{record['scenario_id']}]: "
                f"dangerous {old_congruence.get('dangerous')} -> {new_congruence['dangerous']} "
                f"(warned_against={new_congruence['details'].get('safely_warned_against')})"
            )

        record["congruence_original"] = old_congruence
        record["congruence"] = new_congruence
        by_system_results.setdefault(record["system"], []).append(new_result)

    if changed == 0:
        print("  No dangerous-flag changes.")

    rescored_path = results_dir / "responses_rescored.jsonl"
    with rescored_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    print(f"  Wrote {rescored_path}")

    new_summary = {name: aggregate(results) for name, results in by_system_results.items()}
    summary_path = results_dir / "summary_rescored.json"
    summary_path.write_text(json.dumps(new_summary, indent=2), encoding="utf-8")
    print(f"  Wrote {summary_path}")

    csv_path = results_dir / "summary_rescored.csv"
    _write_csv(new_summary, csv_path)
    print(f"  Wrote {csv_path}")

    old_summary_path = results_dir / "summary.json"
    if old_summary_path.exists():
        old_summary = json.loads(old_summary_path.read_text(encoding="utf-8"))
        print(f"\n  {'System':<18}{'OldDangerRate':<16}{'NewDangerRate':<16}")
        for name in new_summary:
            old_rate = old_summary.get(name, {}).get("dangerous_output_rate", "n/a")
            new_rate = new_summary[name].get("dangerous_output_rate", "n/a")
            print(f"  {name:<18}{str(old_rate):<16}{str(new_rate):<16}")


def _write_csv(summary: dict, path: Path) -> None:
    cols = [
        "system", "n_scenarios", "mean_congruence", "full_congruence_rate",
        "dangerous_output_rate", "dangerous_output_count",
    ]
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
