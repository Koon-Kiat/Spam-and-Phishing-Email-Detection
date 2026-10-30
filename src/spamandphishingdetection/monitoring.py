"""Content-free batch monitoring for score, slice, and optional performance drift."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .config import ProjectConfig
from .evaluation import evaluate_probabilities
from .grouping import registrable_domain, stable_private_hash
from .integrity import load_trusted_manifest

MAX_RECORD_CHARS = 8_192
MAX_TIMESTAMP_CHARS = 64
MAX_CAMPAIGN_CHARS = 128

# Rejected at any depth, even where a stricter rule would also reject them, so the
# error can name the problem without echoing the submitted value.
_FORBIDDEN_CONTENT_KEYS = frozenset(
    {
        "address",
        "attachment",
        "attachments",
        "body",
        "content",
        "email",
        "filename",
        "from",
        "headers",
        "html",
        "message",
        "raw",
        "recipient",
        "recipients",
        "sender",
        "sender_address",
        "subject",
        "text",
        "to",
        "url",
        "urls",
    }
)
_RECORD_FIELDS = frozenset(
    {"timestamp", "probability", "predicted_label", "true_label", "slices"}
)
_ENUMERATED_SLICES = {
    "length_slice": frozenset({"short", "medium", "long", "very_long"}),
    "attachment_slice": frozenset({"has_attachment", "no_attachment", "unknown"}),
    "language_slice": frozenset({"spanish", "other_or_unknown"}),
}
_BOOLEAN_SLICES = frozenset({"has_html", "has_url", "has_obfuscation"})
_HASHED_GROUP_SLICES = {
    "domain_group": re.compile(r"domain_(?:[0-9a-f]{16}|none)"),
    "campaign_group": re.compile(r"campaign_[0-9a-f]{16}"),
}
_RAW_DOMAIN_SLICES = frozenset({"domain", "sender_domain", "url_domain"})
_CATEGORY_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
_HOSTNAME = re.compile(
    r"(?=.{1,253}\Z)[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
)
ALLOWED_SLICE_DIMENSIONS = frozenset(
    {
        *_ENUMERATED_SLICES,
        *_BOOLEAN_SLICES,
        *_HASHED_GROUP_SLICES,
        *_RAW_DOMAIN_SLICES,
        "campaign",
        "source",
    }
)
_BINARY_LABELS = {"safe": 0, "0": 0, "phishing": 1, "1": 1}


class MonitoringInputError(ValueError):
    """Raised for an invalid batch. Messages never echo submitted values."""


def _psi(observed: np.ndarray, expected: np.ndarray) -> float:
    epsilon = 1e-6
    observed = np.clip(observed, epsilon, None)
    expected = np.clip(expected, epsilon, None)
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def _forbidden_keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str) and key.casefold() in _FORBIDDEN_CONTENT_KEYS:
                found.add(key.casefold())
            found |= _forbidden_keys(item)
    elif isinstance(value, list):
        for item in value:
            found |= _forbidden_keys(item)
    return found


def _binary_label(value: object) -> int | None:
    """Return 0 or 1 for a strictly binary label; booleans are not labels."""

    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value in {0, 1}:
        return value
    if isinstance(value, str):
        return _BINARY_LABELS.get(value)
    return None


def _slice_value(dimension: str, value: object) -> str | None:
    """Validate one slice value and return its content-free aggregation form."""

    if dimension in _BOOLEAN_SLICES:
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, str) and value.casefold() in {"true", "false"}:
            return value.capitalize()
        return None
    if not isinstance(value, str):
        return None
    if dimension in _ENUMERATED_SLICES:
        return value if value in _ENUMERATED_SLICES[dimension] else None
    if dimension in _HASHED_GROUP_SLICES:
        return value if _HASHED_GROUP_SLICES[dimension].fullmatch(value) else None
    if dimension in _RAW_DOMAIN_SLICES:
        if not _HOSTNAME.fullmatch(value):
            return None
        return stable_private_hash(registrable_domain(value), namespace="domain")
    if dimension == "campaign":
        if not 0 < len(value) <= MAX_CAMPAIGN_CHARS or not value.isprintable():
            return None
        return stable_private_hash(value.strip().casefold(), namespace="campaign")
    if dimension == "source":
        return value if _CATEGORY_LABEL.fullmatch(value) else None
    return None


def _parse_record(value: object, line_number: int, threshold: float | None) -> dict[str, Any]:
    def invalid(reason: str) -> MonitoringInputError:
        return MonitoringInputError(f"Line {line_number} {reason}")

    if not isinstance(value, dict):
        raise invalid("must be a JSON object")
    forbidden = _forbidden_keys(value)
    if forbidden:
        raise invalid(f"contains forbidden message-content fields: {sorted(forbidden)}")
    if set(value) - _RECORD_FIELDS:
        raise invalid("contains unsupported fields")
    missing = {"timestamp", "probability", "predicted_label"} - set(value)
    if missing:
        raise invalid(f"is missing fields: {sorted(missing)}")

    timestamp = value["timestamp"]
    if not isinstance(timestamp, str) or len(timestamp) > MAX_TIMESTAMP_CHARS:
        raise invalid("has an invalid timestamp")
    try:
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        raise invalid("has an invalid timestamp") from None

    probability = value["probability"]
    if (
        isinstance(probability, bool)
        or not isinstance(probability, int | float)
        or not math.isfinite(probability)
        or not 0.0 <= probability <= 1.0
    ):
        raise invalid("probability must be a number in [0, 1]")
    probability = float(probability)

    predicted = _binary_label(value["predicted_label"])
    if predicted is None:
        raise invalid("has an invalid predicted_label")
    if threshold is not None and predicted != int(probability >= threshold):
        raise invalid("predicted_label is inconsistent with the probability and model threshold")

    true_label = value.get("true_label")
    if true_label is not None:
        true_label = _binary_label(true_label)
        if true_label is None:
            raise invalid("true_label must be 0, 1, safe, or phishing when supplied")

    slices = value.get("slices", {})
    if not isinstance(slices, dict):
        raise invalid("slices must be an object")
    if set(slices) - ALLOWED_SLICE_DIMENSIONS:
        raise invalid("contains unsupported slice dimensions")
    aggregated = {}
    for dimension, item in slices.items():
        normalized = _slice_value(dimension, item)
        if normalized is None:
            raise invalid(f"has an invalid value for slice {dimension}")
        aggregated[dimension] = normalized
    return {
        "timestamp": timestamp,
        "probability": probability,
        "predicted_label": predicted,
        "true_label": true_label,
        "slices": aggregated,
    }


def load_monitoring_batch(
    path: Path,
    *,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Load content-free JSONL, accepting only the documented schema and limits.

    Slice values must be scalar categories from an explicit allowlist. Raw domains and
    campaign names are hashed before aggregation, so no accepted field can carry
    message content into a report. When ``threshold`` is supplied, each
    ``predicted_label`` must equal ``probability >= threshold``.
    """

    records = []
    with path.open("r", encoding="utf-8") as handle:
        line_number = 0
        while line := handle.readline(MAX_RECORD_CHARS + 1):
            line_number += 1
            if len(line) > MAX_RECORD_CHARS:
                raise MonitoringInputError(
                    f"Line {line_number} exceeds {MAX_RECORD_CHARS} characters"
                )
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                raise MonitoringInputError(f"Line {line_number} is not valid JSON") from None
            records.append(_parse_record(value, line_number, threshold))
    if not records:
        raise MonitoringInputError("Monitoring batch contains no records")
    return records


def monitor_batch(
    config: ProjectConfig,
    batch_path: Path,
    *,
    artifact: str | Path | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Compare a content-free batch with the schema-2 monitoring baseline."""

    manifest = load_trusted_manifest(
        config,
        artifact,
        expected_manifest_sha256=expected_manifest_sha256,
    ).manifest
    baseline = manifest.get("monitoring_baseline")
    if not baseline:
        raise ValueError("Artifact does not contain a schema-2 monitoring baseline")
    records = load_monitoring_batch(batch_path, threshold=float(manifest["threshold"]))
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
