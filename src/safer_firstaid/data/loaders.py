"""Utilities to load and convert real first-aid datasets into project formats.

Supports:
    * Kaggle elvisblitti/first-aid  (CSV / JSON of Q&A or intents)
    * FirstAidQA                    (CSV / JSON of question/answer pairs)

Because the exact column names in third-party datasets vary, these loaders accept
column-name overrides. After downloading a dataset, inspect it and pass the right
column names. The functions emit:
    * intents.json     for the IntentClassifierBaseline
    * scenarios.json   skeleton for evaluation (you then add guideline checklists)

IMPORTANT (thesis integrity): the guideline checklists in scenarios.json must be
written by you, grounded in the IFRC / Resuscitation Council guidelines. Do not
auto-generate the "correct answer" checklists from an LLM — that would undermine the
independence of your evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_qa_csv(
    path: Path,
    question_col: str = "question",
    answer_col: str = "answer",
) -> list[dict]:
    """Load a generic question/answer CSV into a list of dicts."""
    import pandas as pd

    df = pd.read_csv(path)
    missing = {question_col, answer_col} - set(df.columns)
    if missing:
        raise KeyError(
            f"Columns {missing} not in CSV. Available columns: {list(df.columns)}. "
            f"Pass question_col/answer_col overrides."
        )
    pairs = []
    for _, row in df.iterrows():
        q = str(row[question_col]).strip()
        a = str(row[answer_col]).strip()
        if q and a and q.lower() != "nan":
            pairs.append({"question": q, "answer": a})
    return pairs


def qa_to_scenarios_skeleton(pairs: list[dict], out_path: Path, source: str) -> None:
    """Write a scenarios.json skeleton (WITHOUT checklists) for you to complete.

    You must add a 'checklist' to each scenario by hand, grounded in the guidelines.
    """
    scenarios = []
    for i, p in enumerate(pairs):
        scenarios.append(
            {
                "id": f"{source}_{i:04d}",
                "query": p["question"],
                "reference_answer": p["answer"],
                "source": source,
                "checklist": [],  # <-- YOU fill this in from the guidelines
            }
        )
    Path(out_path).write_text(json.dumps(scenarios, indent=2), encoding="utf-8")


def qa_to_intents(
    pairs: list[dict],
    out_path: Path,
    intent_col: str | None = None,
    csv_path: Path | None = None,
) -> None:
    """Convert Q&A (optionally with an intent/category column) into intents.json.

    If no intent column is available, each unique answer becomes its own intent with
    its associated questions as examples. This is a reasonable default for FAQ-style
    datasets; refine groupings for a stronger classifier baseline.
    """
    intents: dict[str, dict] = {}

    if intent_col and csv_path:
        import pandas as pd

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            intent = str(row[intent_col]).strip()
            q = str(row.get("question", "")).strip()
            a = str(row.get("answer", "")).strip()
            if not intent:
                continue
            intents.setdefault(intent, {"examples": [], "answer": a})
            if q:
                intents[intent]["examples"].append(q)
    else:
        # group by identical answer text
        by_answer: dict[str, list[str]] = {}
        answer_text: dict[str, str] = {}
        for p in pairs:
            key = p["answer"][:80]
            by_answer.setdefault(key, []).append(p["question"])
            answer_text[key] = p["answer"]
        for i, (key, questions) in enumerate(by_answer.items()):
            intents[f"intent_{i:03d}"] = {
                "examples": questions,
                "answer": answer_text[key],
            }

    Path(out_path).write_text(json.dumps(intents, indent=2), encoding="utf-8")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Convert first-aid Q&A datasets.")
    ap.add_argument("csv", type=Path, help="Path to the dataset CSV")
    ap.add_argument("--question-col", default="question")
    ap.add_argument("--answer-col", default="answer")
    ap.add_argument("--source", default="dataset")
    ap.add_argument("--out-scenarios", type=Path, default=Path("data/datasets/scenarios_from_data.json"))
    ap.add_argument("--out-intents", type=Path, default=Path("data/datasets/intents_from_data.json"))
    args = ap.parse_args()

    pairs = load_qa_csv(args.csv, args.question_col, args.answer_col)
    print(f"Loaded {len(pairs)} Q&A pairs.")
    qa_to_scenarios_skeleton(pairs, args.out_scenarios, args.source)
    qa_to_intents(pairs, args.out_intents)
    print(f"Wrote scenarios skeleton -> {args.out_scenarios}")
    print(f"Wrote intents            -> {args.out_intents}")
    print("Next: add guideline-grounded checklists to each scenario by hand.")
