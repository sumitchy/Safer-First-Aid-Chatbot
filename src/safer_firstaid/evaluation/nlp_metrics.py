"""Automated NLP metrics (Layer 2 — SECONDARY).

Overlap and semantic-similarity metrics comparing generated answers to reference
answers from FirstAidQA / the guidelines. These complement the primary
guideline-congruence metric: they measure fluency and similarity, NOT safety, so
they must never be reported alone for a safety-critical system. That caveat itself
is worth stating explicitly in the dissertation.

Metrics:
    * BLEU-4         (sacrebleu)   — n-gram precision
    * ROUGE-1/2/L    (rouge-score) — recall-oriented overlap
    * BERTScore-F1   (bert-score)  — contextual embedding similarity
    * Flesch reading ease          — layperson readability
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NLPMetrics:
    bleu: float
    rouge1: float
    rouge2: float
    rougeL: float
    bertscore_f1: float
    flesch_reading_ease: float


def _flesch_reading_ease(text: str) -> float:
    """Approximate Flesch Reading Ease (higher = easier)."""
    import re

    sentences = max(1, len(re.findall(r"[.!?]+", text)))
    words = re.findall(r"[A-Za-z]+", text)
    n_words = max(1, len(words))

    def syllables(word: str) -> int:
        word = word.lower()
        groups = re.findall(r"[aeiouy]+", word)
        count = len(groups)
        if word.endswith("e") and count > 1:
            count -= 1
        return max(1, count)

    n_syll = sum(syllables(w) for w in words)
    return round(
        206.835 - 1.015 * (n_words / sentences) - 84.6 * (n_syll / n_words), 2
    )


def compute_nlp_metrics(prediction: str, reference: str) -> NLPMetrics:
    """Compute all automated metrics for a single prediction/reference pair."""
    import sacrebleu
    from rouge_score import rouge_scorer

    from .bertscore_subprocess import compute_bertscore_f1

    # BLEU (sentence-level via sacrebleu, scaled 0-1)
    bleu = sacrebleu.sentence_bleu(prediction, [reference]).score / 100.0

    # ROUGE
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge = scorer.score(reference, prediction)

    # BERTScore runs in an isolated subprocess so a model crash becomes NaN
    # rather than killing the whole evaluation run.
    bert_f1 = compute_bertscore_f1(prediction, reference)

    return NLPMetrics(
        bleu=round(bleu, 4),
        rouge1=round(rouge["rouge1"].fmeasure, 4),
        rouge2=round(rouge["rouge2"].fmeasure, 4),
        rougeL=round(rouge["rougeL"].fmeasure, 4),
        bertscore_f1=round(bert_f1, 4) if bert_f1 == bert_f1 else float("nan"),
        flesch_reading_ease=_flesch_reading_ease(prediction),
    )


def average_metrics(metrics: list[NLPMetrics]) -> dict:
    """Average a list of NLPMetrics into headline numbers."""
    if not metrics:
        return {}

    def _avg(attr: str) -> float:
        vals = [getattr(m, attr) for m in metrics]
        vals = [v for v in vals if v == v]  # drop NaN
        return round(sum(vals) / len(vals), 4) if vals else float("nan")

    return {
        "bleu": _avg("bleu"),
        "rouge1": _avg("rouge1"),
        "rouge2": _avg("rouge2"),
        "rougeL": _avg("rougeL"),
        "bertscore_f1": _avg("bertscore_f1"),
        "flesch_reading_ease": _avg("flesch_reading_ease"),
    }
