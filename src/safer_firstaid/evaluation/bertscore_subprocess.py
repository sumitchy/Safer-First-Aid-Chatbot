"""Safe BERTScore wrapper that isolates the model in a subprocess.

The default BERTScore checkpoint is intentionally smaller than the historical
``roberta-large`` default to avoid OOM-level process death in evaluation runs.
If the model process crashes or fails to download, the wrapper logs a warning and
returns NaN instead of taking down the entire evaluation job.
"""

from __future__ import annotations

import json
import logging
import math
import subprocess
import sys

DEFAULT_BERTSCORE_MODEL = "distilbert-base-uncased"
LOGGER = logging.getLogger(__name__)

_BERTSCORE_WORKER = r'''
import json
import sys

payload = json.loads(sys.argv[1])

try:
    from bert_score import score

    _, _, f1 = score(
        [payload["prediction"]],
        [payload["reference"]],
        model_type=payload["model_name"],
        lang="en",
        verbose=False,
    )
    print(float(f1.mean()))
except Exception as exc:  # pragma: no cover - exercised via subprocess failure
    print(f"BERTScore worker error: {exc}", file=sys.stderr)
    raise
'''


def compute_bertscore_f1(prediction: str, reference: str, model_name: str = DEFAULT_BERTSCORE_MODEL) -> float:
    """Compute BERTScore-F1 in a separate Python process.

    Returns NaN if the subprocess exits non-zero, is killed, or cannot load the
    model. The caller can continue without crashing the whole evaluation run.
    """
    payload = json.dumps(
        {
            "prediction": prediction,
            "reference": reference,
            "model_name": model_name,
        },
        ensure_ascii=False,
    )

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _BERTSCORE_WORKER, payload],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except Exception as exc:  # e.g. timeout or OS-level spawn failure
        LOGGER.warning("BERTScore subprocess could not start: %s", exc)
        return math.nan

    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "unknown subprocess failure").strip()
        LOGGER.warning("BERTScore subprocess failed (rc=%s): %s", proc.returncode, detail[:500])
        return math.nan

    out = (proc.stdout or "").strip()
    if not out:
        LOGGER.warning("BERTScore subprocess exited cleanly but produced no score output")
        return math.nan

    try:
        return float(out.splitlines()[-1])
    except ValueError:
        LOGGER.warning("BERTScore subprocess returned non-numeric output: %s", out[:500])
        return math.nan
