"""Model threshold selection and evaluation metrics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)


def calibration_diagnostics(
    labels: Iterable[int],
    probabilities: Iterable[float],
    *,
    bins: int = 10,
) -> dict[str, Any]:
    """Calculate proper scoring rules, ECE, and reliability-bin data."""

    y_true = np.asarray(list(labels), dtype=int)
    scores = np.clip(np.asarray(list(probabilities), dtype=float), 0.0, 1.0)
    if y_true.shape != scores.shape:
        raise ValueError("Labels and probabilities must have identical shapes")
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(scores, edges[1:-1]), bins - 1)
    reliability = []
    expected_calibration_error = 0.0
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        mean_probability = float(scores[mask].mean()) if count else None
        observed_rate = float(y_true[mask].mean()) if count else None
        if count:
            expected_calibration_error += (
                count / len(y_true) * abs(float(mean_probability) - float(observed_rate))
            )
        reliability.append(
            {
                "bin": index + 1,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "rows": count,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
            }
        )
    return {
        "brier_score": float(brier_score_loss(y_true, scores)),
        "log_loss": float(log_loss(y_true, scores, labels=[0, 1])),
        "expected_calibration_error": float(expected_calibration_error),
        "calibration_bins": bins,
        "reliability": reliability,
    }


def tune_threshold(
    labels: Iterable[int],
    probabilities: Iterable[float],
) -> tuple[float, dict[str, float]]:
    """Choose a threshold by macro F1, phishing recall, then proximity to 0.5."""

    y_true = np.asarray(list(labels), dtype=int)
    scores = np.asarray(list(probabilities), dtype=float)
    if y_true.shape != scores.shape:
        raise ValueError("Labels and probabilities must have identical shapes")
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Threshold tuning requires both classes")

    best_key: tuple[float, float, float] | None = None
    best_threshold = 0.5
    best_metrics: dict[str, float] = {}
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (scores >= threshold).astype(int)
        macro_f1 = float(f1_score(y_true, predictions, average="macro"))
        phishing_recall = float(recall_score(y_true, predictions, pos_label=1))
        key = (macro_f1, phishing_recall, -abs(float(threshold) - 0.5))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = {
                "macro_f1": macro_f1,
                "phishing_recall": phishing_recall,
            }
    return best_threshold, best_metrics


def evaluate_probabilities(
    labels: Iterable[int],
    probabilities: Iterable[float],
    threshold: float,
    sources: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Calculate security-focused binary classification metrics."""

    y_true = np.asarray(list(labels), dtype=int)
    scores = np.asarray(list(probabilities), dtype=float)
    predictions = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=[1],
        average=None,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    result: dict[str, Any] = {
        "threshold": float(threshold),
        "rows": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "phishing_precision": float(precision[0]),
        "phishing_recall": float(recall[0]),
        "phishing_f1": float(f1[0]),
        "pr_auc": (
            float(average_precision_score(y_true, scores))
            if len(np.unique(y_true)) > 1
            else None
        ),
        "roc_auc": (
            float(roc_auc_score(y_true, scores))
            if len(np.unique(y_true)) > 1
            else None
        ),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "calibration": calibration_diagnostics(y_true, scores),
    }

    if sources is not None:
        source_values = np.asarray(list(sources), dtype=object)
        if source_values.shape != y_true.shape:
            raise ValueError("Sources must have the same shape as labels")
        per_source = {}
        for source in sorted(set(source_values)):
            mask = source_values == source
            per_source[str(source)] = evaluate_probabilities(
                y_true[mask],
                scores[mask],
                threshold,
            )
        result["per_source"] = per_source
    return result


def metrics_table(candidate_metrics: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Create a compact comparison table for report output."""

    rows = []
    for name, metrics in candidate_metrics.items():
        rows.append(
            {
                "candidate": name,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "phishing_precision": metrics["phishing_precision"],
                "phishing_recall": metrics["phishing_recall"],
                "phishing_f1": metrics["phishing_f1"],
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "threshold": metrics["threshold"],
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["macro_f1", "phishing_recall"],
        ascending=False,
        kind="stable",
    )
