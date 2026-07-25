"""Post-lock generalization, calibration, drift, and robustness reporting."""

from __future__ import annotations

import hashlib
import html
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ProjectConfig
from .datasets import create_dataset_splits, file_sha256, load_and_audit_datasets
from .evaluation import evaluate_probabilities
from .grouping import add_message_metadata
from .inference import Predictor, resolve_artifact_dir
from .preprocessing import prepare_text
from .training import (
    _calibration_folds,
    _predict_positive_probability,
    build_classical_candidate,
)

_URL = re.compile(r"https?://[^\s<>'\"]+", re.IGNORECASE)


def load_spaphish_v5(path: Path, config: ProjectConfig) -> pd.DataFrame:
    """Adapt the exact SpaPhish v5 CSV to the canonical evaluation schema."""

    frame = pd.read_csv(path, encoding="utf-8-sig")
    normalized = {str(column).strip().casefold(): column for column in frame.columns}
    required = {"subject", "body", "date", "label"}
    missing = required - set(normalized)
    if missing:
        raise ValueError(f"SpaPhish v5 is missing columns: {sorted(missing)}")
    if len(frame) != 1_395:
        raise ValueError(f"SpaPhish v5 must contain 1,395 rows; found {len(frame)}")

    labels = pd.to_numeric(frame[normalized["label"]], errors="raise").astype(int)
    if set(labels.unique()) != {0, 1}:
        raise ValueError("SpaPhish labels must contain exactly 0 and 1")
    subjects = frame[normalized["subject"]].fillna("").astype(str)
    bodies = frame[normalized["body"]].fillna("").astype(str)
    raw_text = [
        f"{subject}\n\n{body}" if subject.strip() else body
        for subject, body in zip(subjects, bodies, strict=True)
    ]
    hashes = (
        frame[normalized["hash"]].fillna("").astype(str)
        if "hash" in normalized
        else pd.Series(raw_text).map(
            lambda value: hashlib.sha256(value.encode()).hexdigest()
        )
    )
    canonical = pd.DataFrame(
        {
            "raw_text": raw_text,
            "label": labels,
            "source": "spaphish_v5",
            "row_id": hashes.map(
                lambda value: "spaphish:"
                + hashlib.sha256(value.encode()).hexdigest()[:20]
            ),
            "sender": "",
            "timestamp": pd.to_datetime(
                frame[normalized["date"]],
                errors="coerce",
                utc=True,
            ),
            "attachment_slice": "unknown",
        }
    )
    if "attachments_count" in normalized:
        attachment_count = pd.to_numeric(
            frame[normalized["attachments_count"]],
            errors="coerce",
        ).fillna(0)
        canonical["attachment_slice"] = np.where(
            attachment_count > 0,
            "has_attachment",
            "no_attachment",
        )
    canonical = add_message_metadata(canonical)
    prepared = canonical["raw_text"].map(
        lambda value: prepare_text(value, config.max_text_chars)[0]
    )
    canonical["text"] = prepared
    counts = canonical["label"].value_counts().to_dict()
    if counts != {1: 731, 0: 664}:
        raise ValueError(f"Unexpected SpaPhish prevalence: {counts}")
    return canonical.drop(columns=["raw_text", "sender"])


def _probabilities(predictor: Predictor, frame: pd.DataFrame) -> np.ndarray:
    if predictor.backend == "classical":
        return _predict_positive_probability(predictor.classical_model, frame["text"])
    return np.asarray(
        [predictor.predict_text(text).probability for text in frame["text"]],
        dtype=float,
    )


def _slice_report(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    columns: tuple[str, ...],
    *,
    minimum_rows: int = 20,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for column in columns:
        column_result: dict[str, Any] = {}
        for value, group in frame.groupby(column, dropna=False, sort=True):
            if len(group) < minimum_rows:
                continue
            indices = group.index.to_numpy()
            group_probabilities = probabilities[indices]
            column_result[str(value)] = evaluate_probabilities(
                group["label"],
                group_probabilities,
                threshold,
            )
        output[column] = column_result
    return output


def _reliability_svg(named_metrics: dict[str, dict[str, Any]]) -> str:
    width, height = 760, 570
    left, top, size = 100, 90, 400
    colors = ("#a2472f", "#c77b18", "#51623d")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Probability reliability diagram</title>',
        '<desc id="desc">Predicted phishing probability compared with observed rate.</desc>',
        f'<rect width="{width}" height="{height}" fill="#f5efe3"/>',
        '<text x="48" y="43" font-family="Georgia,serif" font-size="25" fill="#25231f">'
        "Probability reliability</text>",
    ]
    for tick in range(6):
        value = tick / 5
        x = left + value * size
        y = top + size - value * size
        parts.extend(
            [
                f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top + size}" '
                'stroke="#d2c7b6"/>',
                f'<line x1="{left}" y1="{y}" x2="{left + size}" y2="{y}" '
                'stroke="#d2c7b6"/>',
                f'<text x="{x}" y="{top + size + 25}" text-anchor="middle" '
                'font-family="Segoe UI,sans-serif" font-size="12" fill="#6d675d">'
                f"{value:.1f}</text>",
                f'<text x="{left - 15}" y="{y + 4}" text-anchor="end" '
                'font-family="Segoe UI,sans-serif" font-size="12" fill="#6d675d">'
                f"{value:.1f}</text>",
            ]
        )
    parts.append(
        f'<line x1="{left}" y1="{top + size}" x2="{left + size}" y2="{top}" '
        'stroke="#81786b" stroke-dasharray="6 5"/>'
    )
    for series_index, (name, metrics) in enumerate(named_metrics.items()):
        points = []
        for item in metrics["calibration"]["reliability"]:
            if item["rows"]:
                x = left + float(item["mean_probability"]) * size
                y = top + size - float(item["observed_rate"]) * size
                points.append((x, y))
        color = colors[series_index % len(colors)]
        if points:
            coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
            parts.append(
                f'<polyline points="{coordinates}" fill="none" stroke="{color}" '
                'stroke-width="3"/>'
            )
            for x, y in points:
                parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        legend_y = 125 + series_index * 32
        parts.extend(
            [
                f'<line x1="545" y1="{legend_y}" x2="575" y2="{legend_y}" '
                f'stroke="{color}" stroke-width="3"/>',
                f'<text x="585" y="{legend_y + 4}" font-family="Segoe UI,sans-serif" '
                f'font-size="13" fill="#25231f">{html.escape(name)}</text>',
            ]
        )
    parts.extend(
        [
            '<text x="300" y="550" text-anchor="middle" font-family="Segoe UI,sans-serif" '
            'font-size="13" fill="#25231f">Mean predicted probability</text>',
            '<text x="27" y="290" text-anchor="middle" font-family="Segoe UI,sans-serif" '
            'font-size="13" fill="#25231f" transform="rotate(-90 27 290)">'
            "Observed phishing rate</text>",
            "</svg>",
        ]
    )
    return "\n".join(parts)


def _whitespace_perturbation(text: str) -> str:
    words = text.split()
    return " ".join(
        f"{word}\n" if index and index % 12 == 0 else word
        for index, word in enumerate(words)
    )


def _homoglyph_perturbation(text: str) -> str:
    translations = str.maketrans({"a": "а", "e": "е", "o": "о"})
    words = text.split()
    return " ".join(
        word.translate(translations) if index % 11 == 0 else word
        for index, word in enumerate(words)
    )


def _url_perturbation(text: str) -> str:
    return _URL.sub(lambda match: f"{match.group(0)}?tracking=<IDENTIFIER>", text)


def _perturbation_report(
    model: Any,
    frame: pd.DataFrame,
    threshold: float,
) -> dict[str, Any]:
    transforms = {
        "whitespace": _whitespace_perturbation,
        "homoglyph": _homoglyph_perturbation,
        "url_query": _url_perturbation,
        "call_to_action": lambda text: f"{text}\nAct now to confirm this request.",
    }
    baseline = _predict_positive_probability(model, frame["text"])
    result: dict[str, Any] = {
        "baseline": evaluate_probabilities(frame["label"], baseline, threshold)
    }
    for name, transform in transforms.items():
        probabilities = _predict_positive_probability(
            model,
            frame["text"].map(transform),
        )
        metrics = evaluate_probabilities(frame["label"], probabilities, threshold)
        metrics["mean_absolute_probability_shift"] = float(
            np.mean(np.abs(probabilities - baseline))
        )
        metrics["decision_flip_fraction"] = float(
            np.mean((probabilities >= threshold) != (baseline >= threshold))
        )
        result[name] = metrics
    return result


def generate_generalization_report(
    config: ProjectConfig,
    *,
    artifact: str | Path | None = None,
    external_path: str | Path | None = None,
    include_external: bool = True,
) -> Path:
    """Generate post-lock grouped, calibration, drift, and robustness evidence."""

    artifact_dir = resolve_artifact_dir(config, artifact)
    predictor = Predictor.load(config, artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = str(manifest["model_version"])
    report_dir = config.reports_dir / run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    lock_path = report_dir / "external_evaluation.lock.json"
    if include_external and lock_path.exists():
        raise RuntimeError(
            "External evaluation is already locked for this artifact; refusing to run it again"
        )

    audited = load_and_audit_datasets(config)
    splits = create_dataset_splits(audited.dataframe, config)
    test_probabilities = _probabilities(predictor, splits.test)
    test_metrics = evaluate_probabilities(
        splits.test["label"],
        test_probabilities,
        predictor.threshold,
        sources=splits.test["source"],
    )

    validation_probabilities: np.ndarray
    validation_model = None
    if predictor.backend == "classical":
        selected_name = str(manifest["selected_model"])
        validation_model = build_classical_candidate(
            selected_name,
            config,
            calibration_cv=(
                _calibration_folds(splits.train, config)
                if selected_name == "word_char_linear_svm"
                else 3
            ),
        )
        validation_model.fit(
            splits.train["text"].tolist(),
            splits.train["label"].to_numpy(),
        )
        validation_probabilities = _predict_positive_probability(
            validation_model,
            splits.validation["text"],
        )
    else:
        validation_probabilities = _probabilities(predictor, splits.validation)
    validation_metrics = evaluate_probabilities(
        splits.validation["label"],
        validation_probabilities,
        predictor.threshold,
        sources=splits.validation["source"],
    )

    slice_columns = (
        "length_slice",
        "has_html",
        "has_url",
        "has_obfuscation",
        "attachment_slice",
        "language_slice",
        "source",
        "domain_group",
        "campaign_group",
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "selection_isolation": {
            "untouched_test_used_for_selection": False,
            "external_used_for_selection": False,
            "grouped_cv_source": "grouped_cv_summary.csv",
        },
        "near_duplicate_grouping": audited.audit["near_duplicate_grouping"],
        "split_method": splits.summary["method"],
        "validation": validation_metrics,
        "untouched_test": test_metrics,
        "slices": {
            "validation": _slice_report(
                splits.validation.reset_index(drop=True),
                validation_probabilities,
                predictor.threshold,
                slice_columns,
            ),
            "untouched_test": _slice_report(
                splits.test.reset_index(drop=True),
                test_probabilities,
                predictor.threshold,
                slice_columns,
            ),
        },
    }
    if validation_model is not None:
        payload["robustness_perturbations"] = _perturbation_report(
            validation_model,
            splits.validation,
            predictor.threshold,
        )

    reliability_metrics = {
        "Validation": validation_metrics,
        "Untouched test": test_metrics,
    }
    if include_external:
        path = (
            Path(external_path)
            if external_path
            else config.project_root / "data/external/spaphish_v5.csv"
        )
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"External corpus not found: {path}. Run fetch-data --accept-licenses --external."
            )
        external = load_spaphish_v5(path, config).reset_index(drop=True)
        external_probabilities = _probabilities(predictor, external)
        external_metrics = evaluate_probabilities(
            external["label"],
            external_probabilities,
            predictor.threshold,
        )
        external["year"] = external["timestamp"].dt.year.astype("Int64")
        external["period"] = np.select(
            [
                external["year"].between(2014, 2023).fillna(False).to_numpy(dtype=bool),
                external["year"].eq(2024).fillna(False).to_numpy(dtype=bool),
                external["year"].eq(2025).fillna(False).to_numpy(dtype=bool),
            ],
            ["2014-2023", "2024", "2025"],
            default="date_missing_or_outside_range",
        )
        payload["external"] = {
            "dataset": "SpaPhish v5",
            "doi": "10.17632/hz2d6gz7pc.5",
            "sha256": file_sha256(path),
            "observed_prevalence": float(external["label"].mean()),
            "prevalence_limitation": (
                "Observed SpaPhish prevalence is preserved and is not representative "
                "of every production inbox."
            ),
            "overall": external_metrics,
            "time_periods": _slice_report(
                external,
                external_probabilities,
                predictor.threshold,
                ("period",),
                minimum_rows=1,
            )["period"],
            "spanish_language_drift": _slice_report(
                external,
                external_probabilities,
                predictor.threshold,
                ("language_slice",),
                minimum_rows=1,
            )["language_slice"],
            "domain_and_campaign_groups": _slice_report(
                external,
                external_probabilities,
                predictor.threshold,
                ("domain_group", "campaign_group"),
                minimum_rows=20,
            ),
            "slices": _slice_report(
                external,
                external_probabilities,
                predictor.threshold,
                slice_columns,
            ),
        }
        reliability_metrics["External SpaPhish v5"] = external_metrics

    output_path = report_dir / "generalization_report.json"
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    reliability_path = report_dir / "reliability_diagram.svg"
    reliability_path.write_text(
        _reliability_svg(reliability_metrics),
        encoding="utf-8",
    )

    if include_external:
        lock = {
            "artifact": run_id,
            "artifact_manifest_sha256_before_report": file_sha256(manifest_path),
            "external_sha256": payload["external"]["sha256"],
            "evaluated_at": payload["generated_at"],
            "report_sha256": file_sha256(output_path),
        }
        lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    manifest.setdefault("report_hashes", {})
    manifest["report_hashes"]["generalization_report.json"] = file_sha256(output_path)
    manifest["report_hashes"]["reliability_diagram.svg"] = file_sha256(reliability_path)
    if include_external:
        manifest["external_evaluation"] = {
            "dataset": "SpaPhish v5",
            "dataset_sha256": payload["external"]["sha256"],
            "evaluated_at": payload["generated_at"],
            "report_sha256": file_sha256(output_path),
            "used_for_selection": False,
        }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return output_path
