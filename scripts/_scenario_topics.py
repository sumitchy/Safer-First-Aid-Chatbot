"""Shared life-threatening topic classification for scenario coverage tooling.

Canonical topic list follows documentation_of_code.md Section 3's coverage
table, which maps `pipeline/safety.py`'s LIFE_THREATENING_TRIGGERS phrases
against the guideline corpus (data/guidelines/real/). Trigger phrases below
are taken verbatim from LIFE_THREATENING_TRIGGERS and grouped into the same
9 topics that section already uses, so "coverage" here means the same thing
it means in that section. Breathing-distress phrases ("can't breathe",
"cannot breathe") are grouped under not_breathing since they describe the
same underlying emergency for a lay reporter.

Used by both scripts/validate_scenarios.py (coverage table) and
scripts/shortlist_checklist_candidates.py (gap-prioritised candidate
selection) so the two tools can't drift out of sync on what counts as
covering a topic.
"""

from __future__ import annotations

import re

# "stroke" as a bare substring also matches "heat stroke" / "heatstroke",
# a distinct heat-illness condition not covered by the FAST-assessment
# stroke guideline. pipeline/safety.py's own LIFE_THREATENING_TRIGGERS
# matching has this same false-positive (plain `"stroke" in query.lower()`),
# which is safe-direction there (over-escalating isn't dangerous) but would
# mislabel shortlist/coverage candidates here, so this tool excludes it.
_HEAT_STROKE_RE = re.compile(r"heat\s*stroke", re.IGNORECASE)

TOPIC_TRIGGERS: dict[str, tuple[str, ...]] = {
    "choking": ("choking",),
    "not_breathing": (
        "not breathing", "stopped breathing", "no pulse",
        "can't breathe", "cannot breathe",
    ),
    "cardiac_arrest": ("cardiac arrest", "heart attack", "chest pain"),
    "severe_bleeding": ("severe bleeding", "won't stop bleeding"),
    "anaphylaxis": ("anaphylaxis", "anaphylactic"),
    "stroke": ("stroke",),
    "seizure": ("seizure",),
    "unconscious": ("unconscious", "unresponsive"),
    "suicide": ("suicide", "suicidal"),
}

LIFE_THREATENING_TOPICS: tuple[str, ...] = tuple(TOPIC_TRIGGERS.keys())


def classify_topics(*texts: str) -> list[str]:
    """Return every topic whose trigger phrase appears in any of `texts`.

    Matching is case-insensitive substring matching, same as
    LIFE_THREATENING_TRIGGERS' own matching behaviour in safety.py.
    """
    blob = " ".join(t or "" for t in texts).lower()
    stroke_safe_blob = _HEAT_STROKE_RE.sub("", blob)
    hits = []
    for topic, phrases in TOPIC_TRIGGERS.items():
        haystack = stroke_safe_blob if topic == "stroke" else blob
        if any(phrase in haystack for phrase in phrases):
            hits.append(topic)
    return hits
