"""Collect ablation summary.json files into per-dimension tables (thesis §8).

Reads results/ablation_*/summary.json (RAG row only -- RAG is the system under
ablation; VanillaLLM/IntentClassifier are held fixed as controls and don't
change with retriever settings) and prints one table per dimension: chunk
size, top_k, embedding model.

Usage:
    python scripts/ablation_summary.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

CHUNK_SIZE_RUNS = [
    ("500", "results/ablation_chunk500"),
    ("800 (default)", "results/ablation_chunk800"),
    ("1200", "results/ablation_chunk1200"),
]

TOP_K_RUNS = [
    ("2", "results/ablation_topk2"),
    ("4 (default)", "results/ablation_chunk800"),  # same run as chunk=800 baseline
    ("6", "results/ablation_topk6"),
]

EMBEDDING_RUNS = [
    ("all-MiniLM-L6-v2 (default)", "results/ablation_chunk800"),
    ("all-mpnet-base-v2", "results/ablation_mpnet"),
]

COLS = [
    "mean_congruence",
    "full_congruence_rate",
    "dangerous_output_rate",
    "dangerous_output_count",
    "mean_latency_s",
]


def load_rag_row(results_dir: str) -> dict | None:
    path = Path(results_dir) / "summary.json"
    if not path.exists():
        print(f"missing: {path}", file=sys.stderr)
        return None
    data = json.loads(path.read_text())
    return data.get("RAG")


def print_table(title: str, label_header: str, runs: list[tuple[str, str]]) -> list[dict]:
    print(f"\n== {title} ==")
    header = [label_header] + COLS
    print(",".join(header))
    rows = []
    for label, results_dir in runs:
        row = load_rag_row(results_dir)
        if row is None:
            continue
        out = {label_header: label, **{c: row.get(c, "") for c in COLS}}
        rows.append(out)
        print(",".join(str(out[c]) for c in header))
    return rows


def write_csv(path: Path, label_header: str, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[label_header] + COLS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    out_dir = Path("results")
    chunk_rows = print_table("Chunk size ablation (RAG, n=26, backend=qwen3-vl-8b)", "chunk_size", CHUNK_SIZE_RUNS)
    write_csv(out_dir / "ablation_chunk_size_table.csv", "chunk_size", chunk_rows)

    topk_rows = print_table("top_k ablation (RAG, n=26, backend=qwen3-vl-8b)", "top_k", TOP_K_RUNS)
    write_csv(out_dir / "ablation_top_k_table.csv", "top_k", topk_rows)

    emb_rows = print_table("Embedding model ablation (RAG, n=26, backend=qwen3-vl-8b)", "embedding_model", EMBEDDING_RUNS)
    write_csv(out_dir / "ablation_embedding_table.csv", "embedding_model", emb_rows)


if __name__ == "__main__":
    main()
