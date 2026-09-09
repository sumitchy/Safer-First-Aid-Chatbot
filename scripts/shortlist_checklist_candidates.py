"""Shortlist scenario-skeleton candidates for hand checklist authoring.

Reads one or more scenario-skeleton pools (checklist: [] entries produced by
safer_firstaid.data.loaders.qa_to_scenarios_skeleton) plus the existing
hand-authored scenarios.json, classifies every candidate by
LIFE_THREATENING_TOPICS (scripts/_scenario_topics.py), and picks up to
--per-topic candidates per topic — prioritising topics with the smallest
existing coverage in scenarios.json first (zero-coverage life-threatening
topics come first).

This tool only picks WHICH queries are worth authoring checklists for. It
does not write checklists — reference_answer in the output is the pool's
raw answer (LLM-generated for FirstAidQA, uncertain provenance for Kaggle)
and must not be trusted or copied into a checklist; per BUILD_GUIDE.md
Section 9 / loaders.py, checklists must be written by hand against the real
guideline text in data/guidelines/real/.

Usage:
    python scripts/shortlist_checklist_candidates.py \\
        data/datasets/scenarios_from_data.json \\
        data/datasets/scenarios_from_firstaidqa.json \\
        --existing data/datasets/scenarios.json \\
        --per-topic 3 \\
        --out data/datasets/checklist_shortlist.json

Also writes a batches-of-5 Markdown companion next to --out (same stem,
.md extension).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _scenario_topics import LIFE_THREATENING_TOPICS, classify_topics


def load_pool(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def existing_coverage(existing: list[dict]) -> dict[str, int]:
    counts = {topic: 0 for topic in LIFE_THREATENING_TOPICS}
    for s in existing:
        for t in classify_topics(s.get("query", ""), s.get("reference_answer", "")):
            counts[t] += 1
    return counts


def build_shortlist(
    pools: list[tuple[str, list[dict]]],
    existing: list[dict],
    per_topic: int,
) -> tuple[list[dict], dict[str, int]]:
    coverage = existing_coverage(existing)
    existing_queries = {s.get("query", "").strip().lower() for s in existing}

    # candidate -> topics it matches; skip anything already authored, and
    # anything whose checklist is already non-empty (already done elsewhere).
    by_topic: dict[str, list[dict]] = {t: [] for t in LIFE_THREATENING_TOPICS}
    seen_queries: set[str] = set()
    for pool_name, pool in pools:
        for item in pool:
            if item.get("checklist"):
                continue
            q = item.get("query", "").strip()
            if not q or q.lower() in existing_queries or q.lower() in seen_queries:
                continue
            # Query only, not reference_answer: a candidate must be about
            # the topic itself, not merely mention it in passing within the
            # pool's (unverified) answer text.
            topics = classify_topics(q)
            if not topics:
                continue
            seen_queries.add(q.lower())
            candidate = {
                "pool": pool_name,
                "id": item.get("id"),
                "query": q,
                "reference_answer": item.get("reference_answer", ""),
                "source": item.get("source", pool_name),
            }
            for t in topics:
                by_topic[t].append(candidate)

    # Order topics by coverage gap: zero-coverage first, then ascending count.
    ordered_topics = sorted(LIFE_THREATENING_TOPICS, key=lambda t: coverage[t])

    shortlist = []
    used_ids: set[tuple[str, str]] = set()
    for topic in ordered_topics:
        picked = []
        for cand in by_topic[topic]:
            key = (cand["pool"], cand["id"])
            if key in used_ids:
                continue
            picked.append(cand)
            used_ids.add(key)
            if len(picked) >= per_topic:
                break
        shortlist.append(
            {
                "topic": topic,
                "existing_coverage": coverage[topic],
                "candidates": picked,
            }
        )

    return shortlist, coverage


def write_markdown(shortlist: list[dict], out_path: Path) -> None:
    lines = ["# Checklist authoring shortlist", ""]
    lines.append(
        "Batches of 5, ordered by coverage gap (life-threatening topics with "
        "the least existing coverage first). For each candidate: read the "
        "query, open the real guideline text in `data/guidelines/real/`, and "
        "write the checklist against the guideline — not against the pool's "
        "reference_answer, which is not guideline-verified."
    )
    lines.append("")

    flat: list[tuple[str, int, dict]] = []
    for entry in shortlist:
        for cand in entry["candidates"]:
            flat.append((entry["topic"], entry["existing_coverage"], cand))

    batch_size = 5
    for batch_start in range(0, len(flat), batch_size):
        batch = flat[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        lines.append(f"## Batch {batch_num}")
        lines.append("")
        for topic, existing_count, cand in batch:
            lines.append(
                f"### [{topic}, existing={existing_count}] {cand['pool']}/{cand['id']}"
            )
            lines.append("")
            lines.append(f"**Query:** {cand['query']}")
            lines.append("")
            lines.append(
                f"**Pool reference_answer (unverified, do not copy as-is):** "
                f"{cand['reference_answer']}"
            )
            lines.append("")
            lines.append("**Checklist (fill in against the real guideline):**")
            lines.append("- [ ] required items:")
            lines.append("- [ ] forbidden items:")
            lines.append("- [ ] source citation (specific section, not just doc name):")
            lines.append("")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pools", type=Path, nargs="+", help="Scenario-skeleton pool JSON files")
    parser.add_argument("--existing", type=Path, required=True, help="Existing scenarios.json")
    parser.add_argument("--per-topic", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    pools = [(p.stem, load_pool(p)) for p in args.pools]
    existing = json.loads(args.existing.read_text(encoding="utf-8"))

    shortlist, coverage = build_shortlist(pools, existing, args.per_topic)

    args.out.write_text(json.dumps(shortlist, indent=2), encoding="utf-8")
    md_path = args.out.with_suffix(".md")
    write_markdown(shortlist, md_path)

    total_candidates = sum(len(e["candidates"]) for e in shortlist)
    print(f"Existing coverage: {coverage}")
    print(f"Shortlisted {total_candidates} candidates across {len(shortlist)} topics")
    print(f"Wrote {args.out}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
