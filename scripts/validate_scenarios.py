"""Validate data/datasets/scenarios.json structure and print topic coverage.

Hard errors (nonzero exit): missing/empty required fields, duplicate ids,
empty checklist, checklist items missing text/keywords, keywords not a
non-empty list of strings, forbidden not a bool when present.

Soft warnings (printed, don't fail the run): source that looks like a bare
document name rather than a specific citation (e.g. "IFRC 2020" with no
section), since loaders.py and BUILD_GUIDE.md Section 9 require a specific
citation per scenario.

Coverage table: how many scenarios hit each LIFE_THREATENING_TOPICS category
(scripts/_scenario_topics.py), so you know which topic to prioritise next.

Usage:
    python scripts/validate_scenarios.py data/datasets/scenarios.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from _scenario_topics import LIFE_THREATENING_TOPICS, classify_topics

REQUIRED_FIELDS = ("id", "query", "reference_answer", "source", "checklist")

# A source that is just a bare doc name with no section/topic qualifier,
# e.g. "IFRC 2020" or "RCUK 2025 First Aid Guidelines" with nothing after it.
BARE_SOURCE_RE = re.compile(
    r"^(IFRC\s*\d{4}|RCUK\s*\d{4}(\s+\w+)*\s+Guidelines?)\s*$", re.IGNORECASE
)


def validate(scenarios: list[dict]) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()

    for i, s in enumerate(scenarios):
        where = f"scenario[{i}] (id={s.get('id', '?')!r})"

        for field in REQUIRED_FIELDS:
            if field not in s:
                errors.append(f"{where}: missing required field '{field}'")

        sid = s.get("id")
        if sid:
            if sid in seen_ids:
                errors.append(f"{where}: duplicate id '{sid}'")
            seen_ids.add(sid)

        for field in ("query", "reference_answer", "source"):
            val = s.get(field)
            if field in s and (not isinstance(val, str) or not val.strip()):
                errors.append(f"{where}: '{field}' must be a non-empty string")

        checklist = s.get("checklist")
        if "checklist" in s:
            if not isinstance(checklist, list) or len(checklist) == 0:
                errors.append(f"{where}: 'checklist' must be a non-empty list")
            else:
                for j, item in enumerate(checklist):
                    iw = f"{where} checklist[{j}]"
                    if not isinstance(item, dict):
                        errors.append(f"{iw}: must be an object")
                        continue
                    text = item.get("text")
                    if not isinstance(text, str) or not text.strip():
                        errors.append(f"{iw}: missing/empty 'text'")
                    keywords = item.get("keywords")
                    if not isinstance(keywords, list) or len(keywords) == 0:
                        errors.append(f"{iw}: 'keywords' must be a non-empty list")
                    elif not all(isinstance(k, str) and k.strip() for k in keywords):
                        errors.append(f"{iw}: all 'keywords' must be non-empty strings")
                    if "forbidden" in item and not isinstance(item["forbidden"], bool):
                        errors.append(f"{iw}: 'forbidden' must be a bool if present")

        source = s.get("source")
        if isinstance(source, str) and BARE_SOURCE_RE.match(source.strip()):
            warnings.append(
                f"{where}: source '{source}' looks like a bare document name, "
                f"not a specific citation (e.g. 'IFRC 2020, section on seizures')"
            )

    return errors, warnings


def coverage_table(scenarios: list[dict]) -> list[tuple[str, int]]:
    counts = {topic: 0 for topic in LIFE_THREATENING_TOPICS}
    for s in scenarios:
        topics = classify_topics(s.get("query", ""), s.get("reference_answer", ""))
        for t in topics:
            counts[t] += 1
    return sorted(counts.items(), key=lambda kv: kv[1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenarios", type=Path)
    args = parser.parse_args()

    scenarios = json.loads(args.scenarios.read_text(encoding="utf-8"))
    errors, warnings = validate(scenarios)

    print(f"Loaded {len(scenarios)} scenarios from {args.scenarios}\n")

    if warnings:
        print(f"Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"  ! {w}")
        print()

    if errors:
        print(f"Hard errors ({len(errors)}):")
        for e in errors:
            print(f"  x {e}")
        print()
    else:
        print("No hard errors.\n")

    print("Coverage against LIFE_THREATENING_TOPICS:")
    zero = []
    for topic, count in coverage_table(scenarios):
        flag = " <-- ZERO" if count == 0 else ""
        if count == 0:
            zero.append(topic)
        print(f"  {topic:<20} {count}{flag}")

    if zero:
        print(f"\n{len(zero)} topic(s) with zero coverage: {', '.join(zero)}")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
