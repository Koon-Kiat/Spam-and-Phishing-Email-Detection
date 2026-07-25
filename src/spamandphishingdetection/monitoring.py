"""Content-free batch monitoring for score, slice, and optional performance drift."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import ProjectConfig
from .evaluation import evaluate_probabilities
from .inference import resolve_artifact_dir

_FORBIDDEN_CONTENT_KEYS = {
    "text",
    "body",
    "subject",
    "message",
    "email",
    "raw",
    "content",
}


def _psi(observed: np.ndarray, expected: np.ndarray) -> float:
    epsilon = 1e-6
    observed = np.clip(observed, epsilon, None)
    expected = np.clip(expected, epsilon, None)
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def _private_dimension_value(key: str, value: object) -> str:
    text = str(value)
    if key in {"domain", "sender_domain", "url_domain"}:
        return "domain_" + hashlib.sha256(text.casefold().encode()).hexdigest()[:16]
    return text


def load_monitoring_batch(path: Path) -> list[dict[str, Any]]:
    """Load JSONL while rejecting message-content fields."""

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Line {line_number} must be a JSON object")
            forbidden = _FORBIDDEN_CONTENT_KEYS & {key.casefold() for key in value}
            if forbidden:
                raise ValueError(
                    f"Line {line_number} contains forbidden message-content fields: "
                    f"{sorted(forbidden)}"
                )
            missing = {"timestamp", "probability", "predicted_label"} - set(value)
            if missing:
                raise ValueError(f"Line {line_number} is missing fields: {sorted(missing)}")
            datetime.fromisoformat(str(value["timestamp"]).replace("Z", "+00:00"))
            probability = float(value["probability"])
            if not 0.0 <= probability <= 1.0:
                raise ValueError(f"Line {line_number} probability must be in [0, 1]")
            label = value["predicted_label"]
            if label in {"safe", 0, "0"}:
                predicted = 0
            elif label in {"phishing", 1, "1"}:
                predicted = 1
            else:
                raise ValueError(f"Line {line_number} has an invalid predicted_label")
            slices = value.get("slices", {})
            if not isinstance(slices, dict):
                raise ValueError(f"Line {line_number} slices must be an object")
            records.append(
                {
                    "timestamp": str(value["timestamp"]),
                    "probability": probability,
                    "predicted_label": predicted,
                    "true_label": value.get("true_label"),
                    "slices": {
                        str(key): _private_dimension_value(str(key), item)
                        for key, item in slices.items()
                    },
                }
            )
    if not records:
        raise ValueError("Monitoring batch contains no records")
    return records


def monitor_batch(
    config: ProjectConfig,
    batch_path: Path,
    *,
    artifact: str | Path | None = None,
) -> dict[str, Any]:
    """Compare a content-free batch with the schema-2 monitoring baseline."""

    artifact_dir = resolve_artifact_dir(config, artifact)
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    baseline = manifest.get("monitoring_baseline")
    if not baseline:
        raise ValueError("Artifact does not contain a schema-2 monitoring baseline")
    records = load_monitoring_batch(batch_path)
    probabilities = np.asarray([record["probability"] for record in records])
    predictions = np.asarray([record["predicted_label"] for record in records])
    edges = np.asarray(baseline["score_bin_edges"], dtype=float)
    observed_counts, _ = np.histogram(probabilities, bins=edges)
    observed_fractions = observed_counts / len(records)
    expected_fractions = np.asarray(baseline["score_bin_fractions"], dtype=float)
    thresholds = baseline["alert_thresholds"]
    score_psi = _psi(observed_fractions, expected_fractions)
    prevalence = float(predictions.mean())
    prevalence_change = abs(prevalence - float(baseline["predicted_phishing_fraction"]))

    alerts = []
    if score_psi >= float(thresholds["score_population_stability_index"]):
        alerts.append("score_distribution_drift")
    if prevalence_change >= float(thresholds["predicted_prevalence_absolute_change"]):
        alerts.append("predicted_prevalence_drift")

    slice_result: dict[str, Any] = {}
    dimensions = sorted(
        {
            dimension
            for record in records
            for dimension in record["slices"]
        }
    )
    for dimension in dimensions:
        counts = Counter(
            record["slices"].get(dimension, "<missing>") for record in records
        )
        fractions = {
            value: count / len(records)
            for value, count in sorted(counts.items())
        }
        expected = baseline.get("slice_fractions", {}).get(dimension, {})
        maximum_change = max(
            (
                abs(fractions.get(value, 0.0) - float(expected.get(value, 0.0)))
                for value in set(fractions) | set(expected)
            ),
            default=0.0,
        )
        slice_result[dimension] = {
            "fractions": fractions,
            "maximum_absolute_change": maximum_change,
        }
        if maximum_change >= float(thresholds["slice_fraction_absolute_change"]):
            alerts.append(f"slice_drift:{dimension}")

    labeled = [record for record in records if record["true_label"] is not None]
    performance = None
    if labeled:
        labels = [int(record["true_label"]) for record in labeled]
        scores = [float(record["probability"]) for record in labeled]
        performance = evaluate_probabilities(
            labels,
            scores,
            float(manifest["threshold"]),
        )
        baseline_f1 = float(manifest["test_metrics"]["macro_f1"])
        if baseline_f1 - float(performance["macro_f1"]) >= float(
            thresholds["performance_macro_f1_drop"]
        ):
            alerts.append("performance_degradation")

    return {
        "schema_version": 1,
        "artifact": manifest["model_version"],
        "rows": len(records),
        "content_retained": False,
        "score_population_stability_index": score_psi,
        "predicted_phishing_fraction": prevalence,
        "predicted_prevalence_absolute_change": prevalence_change,
        "slice_drift": slice_result,
        "performance": performance,
        "alerts": sorted(set(alerts)),
        "status": "alert" if alerts else "ok",
        "finite": all(
            math.isfinite(value)
            for value in (score_psi, prevalence, prevalence_change)
        ),
    }
