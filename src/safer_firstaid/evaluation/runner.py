"""Evaluation runner: runs all systems over all scenarios and writes results.

This is the experiment harness for the thesis. It:
    1. takes a set of scenarios,
    2. runs each system (RAG, vanilla LLM, intent classifier) over every query,
    3. scores each response with guideline-congruence (primary) and NLP metrics
       (secondary),
    4. writes per-response JSONL and a summary table (CSV) for your results chapter.

The output schema is uniform across systems so the comparison in Chapter 5 is a
straightforward group-by.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .congruence import (
    Scenario,
    aggregate,
    score_response_automatic,
)
from .nlp_metrics import average_metrics, compute_nlp_metrics


class Answerable(Protocol):
    """Any system exposing .answer(query) -> object with .answer/.raw_answer."""

    def answer(self, query: str): ...


@dataclass
class SystemSpec:
    name: str            # label recorded in results, e.g. "RAG", "VanillaLLM"
    system: Answerable


def run_evaluation(
    systems: list[SystemSpec],
    scenarios: list[Scenario],
    out_dir: Path,
    compute_nlp: bool = True,
) -> dict:
    """Run the full evaluation and persist results. Returns the summary dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_response_path = out_dir / "responses.jsonl"
    summary_path = out_dir / "summary.json"
    csv_path = out_dir / "summary.csv"

    summary: dict[str, dict] = {}

    with per_response_path.open("w", encoding="utf-8") as fh:
        for spec in systems:
            congruence_results = []
            nlp_results = []
            latencies = []

            for scen in scenarios:
                t0 = time.perf_counter()
                resp = spec.system.answer(scen.query)
                dt = time.perf_counter() - t0
                latencies.append(dt)

                # We score the RAW answer for congruence so the safety banner text
                # (which always contains "call 999") does not inflate scores. This is
                # an important methodological choice — document it in the thesis.
                text_for_scoring = getattr(resp, "raw_answer", None) or resp.answer

                cong = score_response_automatic(scen, text_for_scoring)
                congruence_results.append(cong)

                nlp = None
                if compute_nlp and scen.reference_answer:
                    nlp = compute_nlp_metrics(text_for_scoring, scen.reference_answer)
                    nlp_results.append(nlp)

                record = {
                    "system": spec.name,
                    "backend": getattr(resp, "backend_name", ""),
                    "scenario_id": scen.id,
                    "query": scen.query,
                    "final_answer": resp.answer,
                    "raw_answer": getattr(resp, "raw_answer", ""),
                    "provenance": resp.provenance() if hasattr(resp, "provenance") else [],
                    "congruence": cong.to_json(),
                    "nlp": nlp.__dict__ if nlp else None,
                    "latency_s": round(dt, 4),
                    "safety": {
                        "escalate": getattr(getattr(resp, "safety", None), "escalate", None),
                        "dangerous_flags": getattr(getattr(resp, "safety", None), "dangerous_flags", []),
                    },
                }
                fh.write(json.dumps(record) + "\n")

            sys_summary = aggregate(congruence_results)
            if nlp_results:
                sys_summary["nlp"] = average_metrics(nlp_results)
            sys_summary["mean_latency_s"] = round(sum(latencies) / len(latencies), 4)
            summary[spec.name] = sys_summary

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(summary, csv_path)
    return summary


def _write_csv(summary: dict, path: Path) -> None:
    import csv

    cols = [
        "system", "n_scenarios", "mean_congruence", "full_congruence_rate",
        "dangerous_output_rate", "dangerous_output_count", "mean_latency_s",
        "bleu", "rouge1", "rouge2", "rougeL", "bertscore_f1", "flesch_reading_ease",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for name, s in summary.items():
            row = {"system": name}
            for c in cols[1:]:
                if c in s:
                    row[c] = s[c]
                elif "nlp" in s and c in s["nlp"]:
                    row[c] = s["nlp"][c]
                else:
                    row[c] = ""
            writer.writerow(row)
