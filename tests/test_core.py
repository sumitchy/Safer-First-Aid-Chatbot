"""Core unit tests. Run: pytest -q

These cover the parts that must be correct for the evaluation to be trustworthy:
the safety layer, guideline-congruence scoring, and Cohen's Kappa.
"""

import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from safer_firstaid.evaluation.bertscore_subprocess import (
    DEFAULT_BERTSCORE_MODEL,
    compute_bertscore_f1,
)
from safer_firstaid.evaluation.congruence import (
    ChecklistItem,
    Scenario,
    cohens_kappa,
    score_response_automatic,
)
from safer_firstaid.pipeline.safety import SafetyLayer


def _scenario():
    return Scenario(
        id="t1",
        query="someone is not breathing",
        reference_answer="Call 999 and start CPR.",
        checklist=[
            ChecklistItem("Call 999", ["999", "emergency"]),
            ChecklistItem("Start CPR", ["cpr", "compressions"]),
            ChecklistItem("Do not give water", ["give water"], forbidden=True),
        ],
    )


def test_safety_escalates_on_life_threatening():
    safety = SafetyLayer()
    final, decision = safety.apply("he is not breathing", "Do chest compressions.")
    assert decision.escalate is True
    assert "999" in final


def test_safety_blocks_dangerous_advice():
    safety = SafetyLayer()
    final, decision = safety.apply(
        "cut with glass in it", "You should remove the knife and clean it."
    )
    assert decision.is_dangerous is True
    assert "can't safely answer" in final.lower()


def test_congruence_full_match():
    scen = _scenario()
    resp = "Call 999 immediately and start CPR with chest compressions."
    result = score_response_automatic(scen, resp)
    assert result.matched_required == 2
    assert result.full_congruence is True
    assert result.dangerous is False


def test_congruence_detects_danger():
    scen = _scenario()
    resp = "Call 999, start CPR, and give water to the person."
    result = score_response_automatic(scen, resp)
    assert result.dangerous is True
    assert result.full_congruence is False


def test_congruence_partial():
    scen = _scenario()
    resp = "Call 999."
    result = score_response_automatic(scen, resp)
    assert result.matched_required == 1
    assert 0.0 < result.congruence < 1.0


def test_cohens_kappa_perfect_agreement():
    a = [1, 0, 1, 1, 0]
    assert abs(cohens_kappa(a, a) - 1.0) < 1e-9


def test_cohens_kappa_range():
    a = [1, 1, 0, 0, 1, 0]
    b = [1, 0, 0, 1, 1, 0]
    k = cohens_kappa(a, b)
    assert -1.0 <= k <= 1.0


def test_bertscore_subprocess_falls_back_to_nan_on_crash(monkeypatch):
    calls = {}

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="SIGKILL")

    monkeypatch.setattr("subprocess.run", fake_run)
    result = compute_bertscore_f1("Call 999 immediately.", "Call 999.")

    assert math.isnan(result)
    assert DEFAULT_BERTSCORE_MODEL in " ".join(calls["cmd"])
