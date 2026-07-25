"""Generate reproducible model diagnostic tables and SVG charts."""

from __future__ import annotations

import html
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ProjectConfig
from .datasets import create_dataset_splits, load_and_audit_datasets
from .evaluation import evaluate_probabilities
from .inference import resolve_artifact_dir
from .training import _calibration_folds, build_classical_candidate

DEFAULT_LEARNING_FRACTIONS = (0.05, 0.10, 0.25, 0.50, 0.75, 1.00)
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiagnosticReport:
    """Files generated for one trained model run."""

    run_id: str
    learning_curve_csv: Path
    charts: tuple[Path, ...]


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _positive_probability(model: Any, texts: pd.Series) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(texts.tolist()), dtype=float)
    return probabilities[:, 1]


def _stratified_training_sample(
    frame: pd.DataFrame,
    fraction: float,
    seed: int,
) -> pd.DataFrame:
    if not 0 < fraction <= 1:
        raise ValueError("Learning-curve fractions must be greater than 0 and at most 1")
    if math.isclose(fraction, 1.0):
        return frame.copy()
    strata = frame["source"].astype(str) + ":" + frame["label"].astype(str)
    sample, _ = train_test_split(
        frame,
        train_size=fraction,
        random_state=seed,
        stratify=strata,
    )
    return sample.reset_index(drop=True)


def compute_learning_curve(
    config: ProjectConfig,
    model_name: str,
    threshold: float,
    fractions: tuple[float, ...] = DEFAULT_LEARNING_FRACTIONS,
) -> pd.DataFrame:
    """Refit a classical model at increasing sample sizes and compare train/validation."""

    audited = load_and_audit_datasets(config)
    splits = create_dataset_splits(audited.dataframe, config)
    rows: list[dict[str, float | int]] = []
    for fraction in fractions:
        sample = _stratified_training_sample(
            splits.train,
            fraction,
            config.random_seed,
        )
        LOGGER.info(
            "Fitting %s learning-curve model with %s rows (%.0f%%)",
            model_name,
            f"{len(sample):,}",
            fraction * 100,
        )
        started = time.perf_counter()
        model = build_classical_candidate(
            model_name,
            config,
            calibration_cv=(
                _calibration_folds(sample, config)
                if model_name == "word_char_linear_svm"
                else 3
            ),
        )
        model.fit(sample["text"].tolist(), sample["label"].to_numpy())
        fit_seconds = time.perf_counter() - started

        train_metrics = evaluate_probabilities(
            sample["label"],
            _positive_probability(model, sample["text"]),
            threshold,
        )
        validation_metrics = evaluate_probabilities(
            splits.validation["label"],
            _positive_probability(model, splits.validation["text"]),
            threshold,
        )
        rows.append(
            {
                "training_fraction": float(fraction),
                "training_rows": int(len(sample)),
                "validation_rows": int(len(splits.validation)),
                "train_accuracy": train_metrics["accuracy"],
                "validation_accuracy": validation_metrics["accuracy"],
                "accuracy_gap": train_metrics["accuracy"]
                - validation_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "validation_macro_f1": validation_metrics["macro_f1"],
                "macro_f1_gap": train_metrics["macro_f1"]
                - validation_metrics["macro_f1"],
                "train_phishing_recall": train_metrics["phishing_recall"],
                "validation_phishing_recall": validation_metrics["phishing_recall"],
                "fit_seconds": fit_seconds,
            }
        )
    return pd.DataFrame(rows)


def _svg_document(
    title: str,
    description: str,
    body: str,
    *,
    width: int,
    height: int,
) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">\n'
        f"<title id=\"title\">{html.escape(title)}</title>\n"
        f"<desc id=\"desc\">{html.escape(description)}</desc>\n"
        "<style>"
        "text{font-family:Segoe UI,Arial,sans-serif;fill:#25231f}"
        ".title{font-size:24px;font-weight:700}.subtitle{font-size:13px;fill:#6d675d}"
        ".axis{font-size:12px;fill:#6d675d}.label{font-size:12px;font-weight:600}"
        ".grid{stroke:#d2c7b6;stroke-width:1}.frame{fill:#fffaf0;stroke:#bdb19f}"
        "</style>\n"
        f'<rect width="{width}" height="{height}" fill="#f5efe3"/>\n{body}\n</svg>\n'
    )


def _line_panel(
    frame: pd.DataFrame,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    metric: str,
    panel_title: str,
) -> str:
    train_values = frame[f"train_{metric}"].astype(float).tolist()
    validation_values = frame[f"validation_{metric}"].astype(float).tolist()
    all_values = train_values + validation_values
    value_span = max(all_values) - min(all_values)
    tick_step = 0.005 if value_span <= 0.03 else 0.01
    y_min = max(
        0.0,
        math.floor((min(all_values) - tick_step / 2) / tick_step) * tick_step,
    )
    y_max = min(1.0, math.ceil(max(all_values) / tick_step) * tick_step)
    if y_max <= y_min:
        y_min = max(0.0, y_min - tick_step)
        y_max = min(1.0, y_max + tick_step)
    tick_count = int(round((y_max - y_min) / tick_step)) + 1

    left = x + 58
    top = y + 42
    chart_width = width - 78
    chart_height = height - 116
    training_rows = frame["training_rows"].astype(int).tolist()
    x_tick_step = 10_000
    x_max = max(
        x_tick_step,
        math.ceil(max(training_rows) / x_tick_step) * x_tick_step,
    )

    def point(row_count: int, value: float) -> tuple[float, float]:
        point_x = left + row_count / x_max * chart_width
        point_y = top + (y_max - value) / (y_max - y_min) * chart_height
        return point_x, point_y

    parts = [
        f'<rect class="frame" x="{x}" y="{y}" width="{width}" height="{height}" rx="8"/>',
        f'<text class="label" x="{x + 18}" y="{y + 27}">{html.escape(panel_title)}</text>',
    ]
    for tick in range(tick_count):
        value = y_min + tick_step * tick
        tick_y = top + chart_height - (value - y_min) / (y_max - y_min) * chart_height
        parts.append(
            f'<line class="grid" x1="{left}" y1="{tick_y:.1f}" '
            f'x2="{left + chart_width}" y2="{tick_y:.1f}"/>'
        )
        parts.append(
            f'<text class="axis" x="{left - 8}" y="{tick_y + 4:.1f}" '
            f'text-anchor="end">{value * 100:.1f}%</text>'
        )
    for row_count in range(0, x_max + x_tick_step, x_tick_step):
        point_x = left + row_count / x_max * chart_width
        parts.extend(
            [
                f'<line class="grid" x1="{point_x:.1f}" y1="{top}" '
                f'x2="{point_x:.1f}" y2="{top + chart_height}"/>',
                f'<text class="axis" x="{point_x:.1f}" y="{top + chart_height + 22}" '
                f'text-anchor="middle">{row_count:,}</text>',
            ]
        )

    series = (
        ("Training", train_values, "#51623d"),
        ("Validation", validation_values, "#a2472f"),
    )
    for label, values, color in series:
        coordinates = [
            point(row_count, value)
            for row_count, value in zip(training_rows, values, strict=True)
        ]
        polyline = " ".join(f"{px:.1f},{py:.1f}" for px, py in coordinates)
        parts.append(
            f'<polyline points="{polyline}" fill="none" stroke="{color}" '
            'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for (point_x, point_y), row_count, value in zip(
            coordinates,
            training_rows,
            values,
            strict=True,
        ):
            parts.append(
                f'<circle cx="{point_x:.1f}" cy="{point_y:.1f}" r="4" '
                f'fill="{color}"><title>{label} at {row_count:,} rows: '
                f"{value * 100:.3f}%</title></circle>"
            )
    parts.extend(
        [
            f'<line x1="{x + width - 220}" y1="{y + 24}" '
            f'x2="{x + width - 194}" y2="{y + 24}" '
            'stroke="#51623d" stroke-width="3"/>',
            f'<text class="axis" x="{x + width - 188}" y="{y + 28}">Training</text>',
            f'<line x1="{x + width - 118}" y1="{y + 24}" '
            f'x2="{x + width - 92}" y2="{y + 24}" '
            'stroke="#a2472f" stroke-width="3"/>',
            f'<text class="axis" x="{x + width - 86}" y="{y + 28}">Validation</text>',
            f'<text class="axis" x="{left + chart_width / 2:.1f}" y="{y + height - 17}" '
            'text-anchor="middle">Training rows</text>',
        ]
    )
    return "\n".join(parts)


def render_learning_curve(frame: pd.DataFrame, run_id: str) -> str:
    body = [
        '<text class="title" x="48" y="44">Selected-model learning curves</text>',
        (
            '<text class="subtitle" x="48" y="69">'
            f"Training run: {html.escape(_display_run_time(run_id))} · "
            "fixed promoted threshold"
            "</text>"
        ),
        _line_panel(
            frame,
            x=42,
            y=96,
            width=590,
            height=450,
            metric="accuracy",
            panel_title="Accuracy",
        ),
        _line_panel(
            frame,
            x=652,
            y=96,
            width=590,
            height=450,
            metric="macro_f1",
            panel_title="Macro F1",
        ),
        (
            '<text class="subtitle" x="642" y="590" text-anchor="middle">'
            "A shrinking training-validation gap indicates improving generalization as "
            "training data increases."
            "</text>"
        ),
    ]
    return _svg_document(
        "Selected-model learning curves",
        "Training and validation accuracy and macro F1 at increasing training set sizes.",
        "\n".join(body),
        width=1284,
        height=620,
    )


def _display_name(name: str) -> str:
    names = {
        "majority": "Majority baseline",
        "word_only_logistic": "Word TF-IDF + logistic",
        "word_char_logistic": "Word/char TF-IDF + logistic",
        "word_char_nb": "Word/char TF-IDF + NB",
        "word_char_linear_svm": "Word/char TF-IDF + SVM",
        "distilbert": "DistilBERT",
    }
    return names.get(name, name.replace("_", " ").title())


def _display_source(name: str) -> str:
    names = {
        "ceas_08": "CEAS 2008",
        "phishing_email": "Phishing Email",
    }
    return names.get(name, name.replace("_", " ").title())


def _display_run_time(run_id: str) -> str:
    try:
        timestamp = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except ValueError:
        return run_id
    return timestamp.strftime("%d %b %Y, %H:%M:%S UTC")


def render_candidate_comparison(frame: pd.DataFrame, run_id: str) -> str:
    frame = frame.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    width, height = 1150, 590
    left, top, chart_width = 210, 120, 750
    group_height = 68
    metrics = (
        ("accuracy", "Accuracy", "#c77b18"),
        ("macro_f1", "Macro F1", "#a2472f"),
        ("phishing_recall", "Phishing recall", "#51623d"),
    )
    parts = [
        '<text class="title" x="48" y="44">Validation model comparison</text>',
        (
            f'<text class="subtitle" x="48" y="69">Training run: '
            f"{html.escape(_display_run_time(run_id))} · validation data only</text>"
        ),
    ]
    for tick in range(6):
        value = tick / 5
        tick_x = left + value * chart_width
        parts.append(
            f'<line class="grid" x1="{tick_x:.1f}" y1="{top - 15}" '
            f'x2="{tick_x:.1f}" y2="{top + len(frame) * group_height - 12}"/>'
        )
        parts.append(
            f'<text class="axis" x="{tick_x:.1f}" y="{top - 24}" '
            f'text-anchor="middle">{value * 100:.0f}%</text>'
        )
    for row_index, row in frame.iterrows():
        group_y = top + row_index * group_height
        parts.append(
            f'<text class="label" x="{left - 14}" y="{group_y + 27}" '
            f'text-anchor="end">{html.escape(_display_name(str(row["candidate"])))}</text>'
        )
        for metric_index, (column, label, color) in enumerate(metrics):
            value = float(row[column])
            bar_y = group_y + metric_index * 17
            parts.append(
                f'<rect x="{left}" y="{bar_y}" width="{value * chart_width:.1f}" '
                f'height="12" rx="3" fill="{color}">'
                f"<title>{html.escape(label)}: {value * 100:.4f}%</title></rect>"
            )
            parts.append(
                f'<text class="axis" x="{left + chart_width + 10}" y="{bar_y + 10}">'
                f"{value * 100:.2f}%</text>"
            )
    legend_x = 585
    for index, (_, label, color) in enumerate(metrics):
        item_x = legend_x + index * 150
        parts.extend(
            [
                f'<rect x="{item_x}" y="38" width="14" height="14" rx="2" fill="{color}"/>',
                f'<text class="axis" x="{item_x + 20}" y="50">{html.escape(label)}</text>',
            ]
        )
    return _svg_document(
        "Validation model comparison",
        "Accuracy, macro F1, and phishing recall for every benchmark candidate.",
        "\n".join(parts),
        width=width,
        height=height,
    )


def render_confusion_matrix(metrics: dict[str, Any], run_id: str) -> str:
    matrix = metrics["confusion_matrix"]
    cells = (
        ("True negative (TN)", int(matrix[0][0]), "#e0e7d7"),
        ("False positive (FP)", int(matrix[0][1]), "#f3ded4"),
        ("False negative (FN)", int(matrix[1][0]), "#f3ded4"),
        ("True positive (TP)", int(matrix[1][1]), "#e0e7d7"),
    )
    parts = [
        '<text class="title" x="48" y="44">Untouched test confusion matrix</text>',
        (
            f'<text class="subtitle" x="48" y="69">Training run: '
            f"{html.escape(_display_run_time(run_id))} · rows are actual labels</text>"
        ),
        '<text class="label" x="280" y="115" text-anchor="middle">Predicted safe</text>',
        '<text class="label" x="540" y="115" text-anchor="middle">Predicted phishing</text>',
        '<text class="label" x="140" y="235" text-anchor="end">Actually safe</text>',
        '<text class="label" x="140" y="400" text-anchor="end">Actually phishing</text>',
    ]
    positions = ((155, 135), (415, 135), (155, 300), (415, 300))
    for (label, value, color), (x, y) in zip(cells, positions, strict=True):
        parts.extend(
            [
                f'<rect x="{x}" y="{y}" width="250" height="145" rx="8" '
                f'fill="{color}" stroke="#81786b"/>',
                f'<text x="{x + 125}" y="{y + 67}" text-anchor="middle" '
                f'style="font-size:34px;font-weight:700">{value:,}</text>',
                f'<text class="axis" x="{x + 125}" y="{y + 99}" '
                f'text-anchor="middle">{html.escape(label)}</text>',
            ]
        )
    return _svg_document(
        "Untouched test confusion matrix",
        "Counts of true negatives, false positives, false negatives, and true positives.",
        "\n".join(parts),
        width=820,
        height=485,
    )


def render_per_source_metrics(metrics: dict[str, Any], run_id: str) -> str:
    per_source = metrics["per_source"]
    metric_specs = (
        ("accuracy", "Accuracy", "#c77b18"),
        ("macro_f1", "Macro F1", "#a2472f"),
        ("phishing_precision", "Phishing precision", "#87500d"),
        ("phishing_recall", "Phishing recall", "#51623d"),
    )
    sources = sorted(per_source)
    parts = [
        '<text class="title" x="48" y="44">Untouched test metrics by dataset</text>',
        (
            f'<text class="subtitle" x="48" y="69">Training run: '
            f"{html.escape(_display_run_time(run_id))} · horizontal scale starts at 75%</text>"
        ),
    ]
    left, chart_width = 160, 760
    for tick in range(6):
        value = 0.75 + tick * 0.05
        tick_x = left + (value - 0.75) / 0.25 * chart_width
        parts.extend(
            [
                f'<line class="grid" x1="{tick_x:.1f}" y1="100" '
                f'x2="{tick_x:.1f}" y2="420"/>',
                f'<text class="axis" x="{tick_x:.1f}" y="90" '
                f'text-anchor="middle">{value * 100:.0f}%</text>',
            ]
        )
    for source_index, source in enumerate(sources):
        start_y = 120 + source_index * 155
        parts.append(
            f'<text class="label" x="{left - 16}" y="{start_y + 36}" '
            f'text-anchor="end">{html.escape(_display_source(source))}</text>'
        )
        for metric_index, (key, label, color) in enumerate(metric_specs):
            value = float(per_source[source][key])
            bar_y = start_y + metric_index * 27
            bar_width = max(0.0, value - 0.75) / 0.25 * chart_width
            parts.extend(
                [
                    f'<rect x="{left}" y="{bar_y}" width="{bar_width:.1f}" '
                    f'height="18" rx="3" fill="{color}">'
                    f"<title>{html.escape(label)}: {value * 100:.4f}%</title></rect>",
                    f'<text class="axis" x="{left + bar_width + 7:.1f}" y="{bar_y + 14}">'
                    f"{value * 100:.2f}%</text>",
                ]
            )
    legend_y = 465
    for index, (_, label, color) in enumerate(metric_specs):
        item_x = 145 + index * 235
        parts.extend(
            [
                f'<rect x="{item_x}" y="{legend_y}" width="14" height="14" '
                f'rx="2" fill="{color}"/>',
                f'<text class="axis" x="{item_x + 20}" y="{legend_y + 12}">'
                f"{html.escape(label)}</text>",
            ]
        )
    return _svg_document(
        "Untouched test metrics by source dataset",
        "Accuracy, macro F1, phishing precision, and phishing recall for each source.",
        "\n".join(parts),
        width=1120,
        height=520,
    )


def generate_diagnostic_report(
    config: ProjectConfig,
    *,
    artifact: str | Path | None = None,
    output_dir: str | Path = "docs/images",
    fractions: tuple[float, ...] = DEFAULT_LEARNING_FRACTIONS,
) -> DiagnosticReport:
    """Generate learning-curve data and all model evaluation charts."""

    artifact_dir = resolve_artifact_dir(config, artifact)
    manifest = _read_json(artifact_dir / "manifest.json")
    run_id = str(manifest["model_version"])
    if manifest["backend"] != "classical":
        raise ValueError("Learning-curve generation currently requires a classical winner")

    report_dir = config.reports_dir / run_id
    metrics_path = report_dir / "metrics.json"
    candidates_path = report_dir / "candidate_metrics.csv"
    if not metrics_path.is_file() or not candidates_path.is_file():
        raise FileNotFoundError(f"Training report files are missing for run {run_id}")

    target = Path(output_dir)
    target = target.resolve() if target.is_absolute() else (config.project_root / target).resolve()
    target.mkdir(parents=True, exist_ok=True)

    learning_curve = compute_learning_curve(
        config,
        str(manifest["selected_model"]),
        float(manifest["threshold"]),
        fractions,
    )
    learning_curve_csv = report_dir / "learning_curve.csv"
    learning_curve.to_csv(learning_curve_csv, index=False)

    report = _read_json(metrics_path)
    candidates = pd.read_csv(candidates_path)
    charts_and_content = (
        (
            target / "learning_curve.svg",
            render_learning_curve(learning_curve, run_id),
        ),
        (
            target / "validation_model_comparison.svg",
            render_candidate_comparison(candidates, run_id),
        ),
        (
            target / "test_confusion_matrix.svg",
            render_confusion_matrix(report["test_metrics"], run_id),
        ),
        (
            target / "test_metrics_by_source.svg",
            render_per_source_metrics(report["test_metrics"], run_id),
        ),
    )
    for path, content in charts_and_content:
        path.write_text(content, encoding="utf-8")

    summary = {
        "run_id": run_id,
        "selected_model": manifest["selected_model"],
        "threshold": manifest["threshold"],
        "learning_curve_method": (
            "Refit the promoted classical model on deterministic, source-and-label-stratified "
            "subsets of the original training partition. Evaluate training and validation at "
            "the fixed promoted threshold. The untouched test partition is not used."
        ),
        "fractions": list(fractions),
        "charts": [path.name for path, _ in charts_and_content],
    }
    (report_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return DiagnosticReport(
        run_id=run_id,
        learning_curve_csv=learning_curve_csv,
        charts=tuple(path for path, _ in charts_and_content),
    )
