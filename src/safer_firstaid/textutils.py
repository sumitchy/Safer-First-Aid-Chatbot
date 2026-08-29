"""Position-aware negation detection shared by the safety layer and congruence scorer.

Both `pipeline/safety.py` and `evaluation/congruence.py` used to flag forbidden
phrases by plain substring/regex presence, with no check for whether the phrase
was asserted or negated ("do not remove the object" was flagged the same as
"remove the object"). This module fixes that with a clause-scoped, position-based
heuristic: a negation cue only cancels a match if it appears BEFORE the matched
phrase within the same clause.

This is a heuristic, not a full NLP negation parser. It is clause-scoped and
position-based, and can be fooled by unusual phrasing such as double negatives
or negation split across a long subordinate clause. See CHANGELOG.md.
"""

from __future__ import annotations

import re

NEGATION_CUES: tuple[str, ...] = (
    "do not", "don't", "does not", "doesn't", "did not", "didn't",
    "never", "avoid", "should not", "shouldn't", "must not", "mustn't",
    "cannot", "can't", "no need to", "without", "refrain from",
    "you should not", "make sure not to", "be careful not to", "stop",
)

_CLAUSE_SPLIT_RE = re.compile(r"[.!?:;\n]+")


def split_clauses(text: str) -> list[str]:
    """Split text into clauses on sentence-ending punctuation, colon, semicolon, or newline."""
    return [c.strip() for c in _CLAUSE_SPLIT_RE.split(text) if c.strip()]


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(keyword.lower()) + r"(?!\w)", re.IGNORECASE)


def has_negation_before(clause: str, position: int) -> bool:
    """True if a negation cue appears in clause[:position]."""
    prefix = clause[:position].lower()
    for cue in NEGATION_CUES:
        if re.search(r"(?<!\w)" + re.escape(cue) + r"(?!\w)", prefix):
            return True
    return False


def clause_contains_keyword_unnegated(clause: str, keyword: str) -> bool:
    """Keyword match exists AND no negation cue precedes it."""
    pattern = _keyword_pattern(keyword)
    for match in pattern.finditer(clause):
        if not has_negation_before(clause, match.start()):
            return True
    return False


def clause_contains_keyword_negated(clause: str, keyword: str) -> bool:
    """Keyword match exists AND a negation cue precedes it."""
    pattern = _keyword_pattern(keyword)
    for match in pattern.finditer(clause):
        if has_negation_before(clause, match.start()):
            return True
    return False


def contains_any_unnegated(text: str, keywords: list[str]) -> bool:
    """True if ANY keyword matches unnegated in ANY clause."""
    clauses = split_clauses(text)
    return any(
        clause_contains_keyword_unnegated(clause, kw)
        for clause in clauses
        for kw in keywords
    )


def contains_any_negated(text: str, keywords: list[str]) -> bool:
    """True if ANY keyword matches negated in ANY clause."""
    clauses = split_clauses(text)
    return any(
        clause_contains_keyword_negated(clause, kw)
        for clause in clauses
        for kw in keywords
    )
