"""Guideline-congruence evaluation (Layer 1 — PRIMARY metric).

This operationalises the project's central research question: does the system give
guideline-concordant advice? For each test scenario we define a checklist of
must-have actions drawn from the authoritative guidelines. A response is scored by
how many checklist items it correctly contains, and whether it contains any
"must-not" (dangerous) items.

Two scoring modes are provided:
    1. Automatic keyword matching  — fast, reproducible, good for iteration.
    2. Human assessment import      — load scores from ≥2 qualified assessors and
       compute inter-rater agreement (Cohen's Kappa).

The human-assessment path is what you report as your primary result; the automatic
path is a development aid. Documenting BOTH, and the correlation between them, is a
strong methodological contribution for the dissertation.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ChecklistItem:
    """A single guideline-required (or forbidden) action for a scenario."""

    text: str                       # human description, e.g. "Call 999"
    keywords: list[str]             # any-of match terms for automatic scoring
    forbidden: bool = False         # True => presence is DANGEROUS, not required


@dataclass
class Scenario:
    """A test case: a query plus its guideline checklist."""

    id: str
    query: str
    reference_answer: str           # gold answer (from FirstAidQA / guidelines)
    checklist: list[ChecklistItem]
    source: str = ""                # guideline citation

    @property
    def required_items(self) -> list[ChecklistItem]:
        return [c for c in self.checklist if not c.forbidden]

    @property
    def forbidden_items(self) -> list[ChecklistItem]:
        return [c for c in self.checklist if c.forbidden]


@dataclass
class CongruenceResult:
    scenario_id: str
    matched_required: int
    total_required: int
    matched_forbidden: int          # dangerous items present (want 0)
    congruence: float               # matched_required / total_required
    full_congruence: bool           # all required present AND no forbidden
    dangerous: bool                 # any forbidden present
    details: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return asdict(self)


def _contains_any(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    for kw in keywords:
        # word-ish boundary match to reduce false positives
        if re.search(r"(?<!\w)" + re.escape(kw.lower()) + r"(?!\w)", t):
            return True
    return False


def score_response_automatic(scenario: Scenario, response_text: str) -> CongruenceResult:
    """Automatically score a response against a scenario's checklist."""
    req = scenario.required_items
    forb = scenario.forbidden_items

    matched_req = [c for c in req if _contains_any(response_text, c.keywords)]
    matched_forb = [c for c in forb if _contains_any(response_text, c.keywords)]

    total_req = len(req)
    n_matched = len(matched_req)
    congruence = (n_matched / total_req) if total_req else 0.0
    dangerous = len(matched_forb) > 0
    full = (n_matched == total_req) and not dangerous

    return CongruenceResult(
        scenario_id=scenario.id,
        matched_required=n_matched,
        total_required=total_req,
        matched_forbidden=len(matched_forb),
        congruence=congruence,
        full_congruence=full,
        dangerous=dangerous,
        details={
            "matched_required_items": [c.text for c in matched_req],
            "missed_required_items": [c.text for c in req if c not in matched_req],
            "dangerous_items_present": [c.text for c in matched_forb],
        },
    )


def aggregate(results: list[CongruenceResult]) -> dict:
    """Summarise per-scenario results into headline metrics for the thesis."""
    n = len(results)
    if n == 0:
        return {}
    full = sum(1 for r in results if r.full_congruence)
    dangerous = sum(1 for r in results if r.dangerous)
    mean_cong = sum(r.congruence for r in results) / n
    return {
        "n_scenarios": n,
        "mean_congruence": round(mean_cong, 4),
        "full_congruence_rate": round(full / n, 4),
        "dangerous_output_rate": round(dangerous / n, 4),
        "dangerous_output_count": dangerous,
    }


# --------------------------------------------------------------------------- #
# Human assessment + inter-rater agreement                                    #
# --------------------------------------------------------------------------- #
def cohens_kappa(rater_a: list[int], rater_b: list[int]) -> float:
    """Cohen's Kappa for two raters over binary/categorical labels."""
    if len(rater_a) != len(rater_b) or not rater_a:
        raise ValueError("Rater label lists must be non-empty and equal length.")

    n = len(rater_a)
    labels = sorted(set(rater_a) | set(rater_b))
    # observed agreement
    po = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n
    # expected agreement
    pe = 0.0
    for lab in labels:
        pa = rater_a.count(lab) / n
        pb = rater_b.count(lab) / n
        pe += pa * pb
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def load_scenarios(path: Path) -> list[Scenario]:
    """Load scenarios from a JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    scenarios = []
    for item in raw:
        checklist = [ChecklistItem(**c) for c in item["checklist"]]
        scenarios.append(
            Scenario(
                id=item["id"],
                query=item["query"],
                reference_answer=item.get("reference_answer", ""),
                checklist=checklist,
                source=item.get("source", ""),
            )
        )
    return scenarios
