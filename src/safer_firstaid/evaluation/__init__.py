"""Evaluation: guideline congruence (primary) + NLP metrics (secondary)."""

from .congruence import (
    ChecklistItem,
    CongruenceResult,
    Scenario,
    aggregate,
    cohens_kappa,
    load_scenarios,
    score_response_automatic,
)
from .nlp_metrics import NLPMetrics, average_metrics, compute_nlp_metrics
from .runner import SystemSpec, run_evaluation

__all__ = [
    "ChecklistItem",
    "Scenario",
    "CongruenceResult",
    "score_response_automatic",
    "aggregate",
    "cohens_kappa",
    "load_scenarios",
    "NLPMetrics",
    "compute_nlp_metrics",
    "average_metrics",
    "SystemSpec",
    "run_evaluation",
]
