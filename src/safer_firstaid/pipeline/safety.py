"""Safety layer: mandatory escalation and dangerous-content guarding.

This module encodes the project's central safety commitment: any response to a
potentially life-threatening scenario MUST prepend an emergency-services
escalation instruction, and this cannot be overridden by the language model.

Design rationale (thesis Chapter 4):
    LLMs are probabilistic and can omit critical instructions. Rather than trust
    the model to always escalate, we enforce escalation deterministically in code.
    This converts a "hope the model says it" behaviour into a guarantee, which is
    exactly the kind of engineered safeguard the literature (Birkun & Gautam 2023)
    finds missing in general-purpose chatbots.

The lists below are intentionally conservative and auditable. They are a starting
point: extending and justifying them with reference to the IFRC / Resuscitation
Council guidelines is part of your research contribution.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Emergency numbers shown to the user. UK-first per the guideline corpus.
EMERGENCY_NUMBER = "999 (or 112 in the EU / 911 in the US)"

# Scenarios that always require immediate professional emergency response.
# Keyed phrases are matched case-insensitively against the user query.
LIFE_THREATENING_TRIGGERS: tuple[str, ...] = (
    "not breathing", "stopped breathing", "no pulse", "cardiac arrest",
    "heart attack", "chest pain", "choking", "can't breathe", "cannot breathe",
    "unconscious", "unresponsive", "severe bleeding", "won't stop bleeding",
    "stroke", "seizure", "anaphylaxis", "anaphylactic", "overdose",
    "drowning", "electric shock", "severe burn", "broken neck", "spinal injury",
    "suicide", "suicidal",
)

# Phrases that would be dangerous coming from a first-aid assistant to a layperson.
# If a generated answer contains these, the response is flagged for review and,
# depending on policy, blocked or annotated. This supports the "dangerous-output
# count" primary evaluation metric.
DANGEROUS_ADVICE_PATTERNS: tuple[str, ...] = (
    r"\bdo not call\b.*\b(999|112|911|emergency)\b",
    r"\bno need (to|for)\b.*\b(999|112|911|ambulance|emergency)\b",
    r"\binduce vomiting\b",            # contraindicated for many poisonings
    r"\bgive (them )?(water|food|drink)\b.*\bunconscious\b",
    r"\bremove the (knife|blade|object)\b",  # impaled objects should not be removed
    r"\btourniquet\b.*\bneck\b",
)


@dataclass
class SafetyDecision:
    """Result of running the safety layer over a query/response pair."""

    escalate: bool
    triggered_terms: list[str] = field(default_factory=list)
    dangerous_flags: list[str] = field(default_factory=list)

    @property
    def is_dangerous(self) -> bool:
        return bool(self.dangerous_flags)


class SafetyLayer:
    """Deterministic pre/post processing that wraps the generator."""

    def __init__(
        self,
        triggers: tuple[str, ...] = LIFE_THREATENING_TRIGGERS,
        dangerous_patterns: tuple[str, ...] = DANGEROUS_ADVICE_PATTERNS,
        emergency_number: str = EMERGENCY_NUMBER,
    ) -> None:
        self.triggers = tuple(t.lower() for t in triggers)
        self.dangerous_patterns = [re.compile(p, re.IGNORECASE) for p in dangerous_patterns]
        self.emergency_number = emergency_number

    # -- Pre-generation ---------------------------------------------------- #
    def assess_query(self, query: str) -> SafetyDecision:
        """Decide whether a query describes a life-threatening scenario."""
        q = query.lower()
        hits = [t for t in self.triggers if t in q]
        return SafetyDecision(escalate=bool(hits), triggered_terms=hits)

    def escalation_banner(self) -> str:
        """The mandatory, non-overridable escalation message."""
        return (
            f"\u26a0\ufe0f  This may be a medical emergency. Call {self.emergency_number} "
            f"immediately before doing anything else. If the person is unresponsive and "
            f"not breathing normally, start CPR if you are able and stay on the line with "
            f"the operator, who will guide you."
        )

    # -- Post-generation --------------------------------------------------- #
    def scan_response(self, response: str) -> SafetyDecision:
        """Flag dangerous advice patterns in a generated response."""
        flags = [p.pattern for p in self.dangerous_patterns if p.search(response)]
        return SafetyDecision(escalate=False, dangerous_flags=flags)

    def apply(self, query: str, response: str) -> tuple[str, SafetyDecision]:
        """Wrap a generated response with escalation and dangerous-content checks.

        Returns the final user-facing text and a combined SafetyDecision for logging
        and evaluation.
        """
        pre = self.assess_query(query)
        post = self.scan_response(response)

        combined = SafetyDecision(
            escalate=pre.escalate,
            triggered_terms=pre.triggered_terms,
            dangerous_flags=post.dangerous_flags,
        )

        parts: list[str] = []
        if pre.escalate:
            parts.append(self.escalation_banner())
            parts.append("")  # blank line

        if post.dangerous_flags:
            # Conservative policy: replace the flagged answer with a safe fallback.
            parts.append(
                "I can't safely answer that in a way I'm confident follows first-aid "
                "guidelines. Please call the emergency number above and follow the "
                "operator's instructions."
            )
            return "\n".join(parts).strip(), combined

        parts.append(response.strip())
        parts.append("")
        parts.append(
            "\u2139\ufe0f  This is general first-aid guidance based on published "
            "guidelines, not a substitute for professional medical care."
        )
        return "\n".join(parts).strip(), combined
