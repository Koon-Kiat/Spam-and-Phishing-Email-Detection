"""Generate reproducible model diagnostic tables and Matplotlib SVG charts."""

from __future__ import annotations

import html
import io
import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import colors as mpl_colors
from matplotlib import ticker as mpl_ticker
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split

from .config import ProjectConfig
from .datasets import create_dataset_splits, load_and_audit_datasets
from .evaluation import evaluate_probabilities
from .inference import resolve_artifact_dir
from .training import _calibration_folds, build_classical_candidate

DEFAULT_LEARNING_FRACTIONS = (0.05, 0.10, 0.25, 0.50, 0.75, 1.00)
LOGGER = logging.getLogger(__name__)

_PLOT_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.edgecolor": "#B6B8B7",
    "axes.labelcolor": "#313432",
    "axes.linewidth": 0.8,
    "axes.titlecolor": "#252725",
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.facecolor": "#FFFFFF",
    "figure.facecolor": "#FFFFFF",
    "grid.alpha": 0.8,
    "grid.color": "#DCDDDB",
    "grid.linewidth": 0.65,
    "legend.frameon": False,
    "text.color": "#252725",
    "xtick.color": "#565A57",
    "ytick.color": "#565A57",
    "svg.fonttype": "none",
}
_COLORS = {
    "primary": "#3D6258",
    "accent": "#B65335",
    "neutral": "#6B706D",
    "guide": "#B9BCBA",
}


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


def _display_name(name: str) -> str:
    names = {
        "majority": "Majority baseline",
        "word_only_logistic": "Word logistic",
        "word_char_logistic": "Word + character logistic",
        "word_char_nb": "Word + character NB",
        "word_char_linear_svm": "Word + character SVM",
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
    return timestamp.strftime("%d %b %Y, %H:%M UTC")


def _figure_to_svg(figure: Figure, title: str, description: str) -> str:
    """Serialize a Matplotlib figure as an accessible, deterministic SVG."""

    buffer = io.StringIO()
    figure.savefig(
        buffer,
        format="svg",
        facecolor="white",
        metadata={
            "Title": title,
            "Description": description,
            "Creator": f"Matplotlib {mpl.__version__}",
            "Date": None,
        },
    )
    svg = buffer.getvalue()
    svg = svg[svg.index("<svg") :]
    opening_end = svg.index(">")
    opening = svg[:opening_end]
    opening += ' role="img" aria-labelledby="chart-title chart-description"'
    escaped_title = html.escape(title)
    svg = svg.replace(
        f"<title>{escaped_title}</title>",
        f'<title id="chart-title">{escaped_title}</title>',
        1,
    )
    title_end = svg.index("</title>") + len("</title>")
    svg = (
        f"{opening}>{svg[opening_end + 1 : title_end]}"
        f'\n <desc id="chart-description">{html.escape(description)}</desc>'
        f"{svg[title_end:]}"
    )
    return "\n".join(line.rstrip() for line in svg.splitlines()) + "\n"


def _prepare_axis(axis: Axes, *, grid_axis: str) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, axis=grid_axis)
    axis.set_axisbelow(True)


def _add_heading(figure: Figure, title: str, subtitle: str) -> None:
    figure.suptitle(
        title,
        x=0.06,
        y=0.955,
        ha="left",
        fontsize=16,
        fontweight="semibold",
        color="#252725",
    )
    figure.text(
        0.06,
        0.885,
        subtitle,
        ha="left",
        va="top",
        color="#666A67",
        fontsize=9,
    )


def render_learning_curve(frame: pd.DataFrame, run_id: str) -> str:
    title = "Learning curves"
    description = (
        "Training and validation accuracy and macro F1 at increasing training set sizes."
    )
    with mpl.rc_context(_PLOT_STYLE):
        figure = Figure(figsize=(11.6, 5.2))
        FigureCanvasAgg(figure)
        axes = figure.subplots(1, 2)
        figure.subplots_adjust(
            left=0.08,
            right=0.975,
            bottom=0.14,
            top=0.76,
            wspace=0.2,
        )
        _add_heading(
            figure,
            title,
            f"Training run: {_display_run_time(run_id)} · fixed decision threshold",
        )
        training_rows = frame["training_rows"].astype(int).to_numpy()
        series = (
            ("Training", "train", _COLORS["primary"]),
            ("Validation", "validation", _COLORS["accent"]),
        )
        for axis, (metric, panel_title) in zip(
            axes,
            (("accuracy", "Accuracy"), ("macro_f1", "Macro F1")),
            strict=True,
        ):
            values: list[float] = []
            for label, prefix, color in series:
                metric_values = frame[f"{prefix}_{metric}"].astype(float).to_numpy()
                values.extend(metric_values.tolist())
                axis.plot(
                    training_rows,
                    metric_values,
                    marker="o",
                    markersize=4.5,
                    linewidth=2,
                    color=color,
                    label=label,
                )
            lower = max(0.0, math.floor((min(values) - 0.0025) / 0.005) * 0.005)
            axis.set_ylim(lower, 1.001)
            axis.set_title(panel_title, loc="left")
            axis.set_xlabel("Training rows")
            axis.set_ylabel("Score")
            axis.yaxis.set_major_formatter(mpl_ticker.PercentFormatter(1.0, decimals=1))
            axis.xaxis.set_major_formatter(mpl_ticker.StrMethodFormatter("{x:,.0f}"))
            axis.yaxis.set_major_locator(mpl_ticker.MaxNLocator(6))
            _prepare_axis(axis, grid_axis="y")
            axis.legend(loc="lower right")
        return _figure_to_svg(figure, title, description)


def render_candidate_comparison(frame: pd.DataFrame, run_id: str) -> str:
    title = "Model comparison — validation set"
    description = (
        "Accuracy, macro F1, and phishing recall for every benchmark candidate."
    )
    ordered = frame.sort_values("macro_f1", ascending=False).reset_index(drop=True)
    metrics = (
        ("accuracy", "Accuracy", _COLORS["primary"]),
        ("macro_f1", "Macro F1", _COLORS["accent"]),
        ("phishing_recall", "Phishing recall", _COLORS["neutral"]),
    )
    with mpl.rc_context(_PLOT_STYLE):
        figure = Figure(figsize=(10.8, 5.6))
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        figure.subplots_adjust(left=0.205, right=0.945, bottom=0.13, top=0.74)
        _add_heading(
            figure,
            title,
            f"Training run: {_display_run_time(run_id)} · validation partition only",
        )
        positions = np.arange(len(ordered), dtype=float)
        bar_height = 0.2
        offsets = (-bar_height, 0.0, bar_height)
        for offset, (column, label, color) in zip(offsets, metrics, strict=True):
            values = ordered[column].astype(float).to_numpy()
            bars = axis.barh(
                positions + offset,
                values,
                height=bar_height * 0.78,
                color=color,
                label=label,
            )
            for bar, value in zip(bars, values, strict=True):
                axis.text(
                    max(value + 0.008, 0.012),
                    bar.get_y() + bar.get_height() / 2,
                    f"{value:.1%}",
                    va="center",
                    ha="left",
                    fontsize=8,
                    color="#404040",
                )
        axis.set_yticks(
            positions,
            [_display_name(str(name)) for name in ordered["candidate"]],
        )
        axis.invert_yaxis()
        axis.set_xlim(0.0, 1.06)
        axis.set_xlabel("Validation score")
        axis.xaxis.set_major_formatter(mpl_ticker.PercentFormatter(1.0))
        axis.xaxis.set_major_locator(mpl_ticker.MultipleLocator(0.2))
        _prepare_axis(axis, grid_axis="x")
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.945, 0.84),
            ncol=3,
            columnspacing=1.25,
            handlelength=1.8,
        )
        return _figure_to_svg(figure, title, description)


def render_confusion_matrix(metrics: dict[str, Any], run_id: str) -> str:
    title = "Confusion matrix — untouched test"
    description = (
        "Counts and row percentages for true negatives, false positives, "
        "false negatives, and true positives."
    )
    matrix = np.asarray(metrics["confusion_matrix"], dtype=int)
    labels = (
        ("True negative (TN)", "False positive (FP)"),
        ("False negative (FN)", "True positive (TP)"),
    )
    row_totals = matrix.sum(axis=1, keepdims=True)
    row_percentages = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    with mpl.rc_context(_PLOT_STYLE):
        figure = Figure(figsize=(8.8, 5.25))
        FigureCanvasAgg(figure)
        grid = figure.add_gridspec(
            1,
            2,
            width_ratios=(1, 0.035),
            left=0.165,
            right=0.84,
            bottom=0.14,
            top=0.74,
            wspace=0.08,
        )
        axis = figure.add_subplot(grid[0, 0])
        color_axis = figure.add_subplot(grid[0, 1])
        _add_heading(
            figure,
            title,
            f"Training run: {_display_run_time(run_id)} · rows are actual labels",
        )
        color_map = mpl_colors.LinearSegmentedColormap.from_list(
            "neutral_green",
            ("#F4F3EF", "#C9D2CE", _COLORS["primary"]),
        )
        image = axis.imshow(matrix, cmap=color_map)
        for row in range(2):
            for column in range(2):
                color = "white" if row_percentages[row, column] > 0.55 else "#202020"
                axis.text(
                    column,
                    row,
                    (
                        f"{matrix[row, column]:,}\n"
                        f"{row_percentages[row, column]:.2%} of actual class\n"
                        f"{labels[row][column]}"
                    ),
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=9,
                    linespacing=1.4,
                )
        axis.set_xticks((0, 1), ("Predicted safe", "Predicted phishing"))
        axis.set_yticks((0, 1), ("Actually safe", "Actually phishing"))
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        color_bar = figure.colorbar(image, cax=color_axis)
        color_bar.set_label("Messages")
        color_bar.ax.yaxis.set_major_formatter(
            mpl_ticker.StrMethodFormatter("{x:,.0f}")
        )
        return _figure_to_svg(figure, title, description)


def render_per_source_metrics(metrics: dict[str, Any], run_id: str) -> str:
    per_source = metrics["per_source"]
    metric_specs = (
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1"),
        ("phishing_precision", "Phishing precision"),
        ("phishing_recall", "Phishing recall"),
    )
    sources = sorted(per_source)
    title = "Metrics by dataset — untouched test"
    description = (
        "Accuracy, macro F1, phishing precision, and phishing recall for each source."
    )
    values = [
        float(per_source[source][metric])
        for source in sources
        for metric, _ in metric_specs
    ]
    lower = max(0.0, math.floor((min(values) - 0.005) / 0.01) * 0.01)
    with mpl.rc_context(_PLOT_STYLE):
        figure = Figure(figsize=(10.8, 4.8))
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        figure.subplots_adjust(left=0.17, right=0.945, bottom=0.16, top=0.72)
        _add_heading(
            figure,
            title,
            f"Training run: {_display_run_time(run_id)} · familiar-source test partition",
        )
        y_positions = np.arange(len(metric_specs), dtype=float)
        offsets = np.linspace(-0.12, 0.12, len(sources))
        source_colors = (_COLORS["primary"], _COLORS["accent"])
        for y_position, (metric, _) in zip(
            y_positions,
            metric_specs,
            strict=True,
        ):
            paired_values = [float(per_source[source][metric]) for source in sources]
            axis.hlines(
                y_position,
                min(paired_values),
                max(paired_values),
                color=_COLORS["guide"],
                linewidth=1,
                zorder=1,
            )
        for offset, source, color in zip(
            offsets,
            sources,
            source_colors,
            strict=True,
        ):
            source_values = np.asarray(
                [float(per_source[source][metric]) for metric, _ in metric_specs]
            )
            axis.scatter(
                source_values,
                y_positions + offset,
                s=54,
                color=color,
                label=_display_source(source),
                zorder=3,
            )
            for value, y_position in zip(
                source_values,
                y_positions + offset,
                strict=True,
            ):
                axis.annotate(
                    f"{value:.2%}",
                    (value, y_position),
                    xytext=(-7, 0),
                    textcoords="offset points",
                    va="center",
                    ha="right",
                    fontsize=8,
                    color="#303030",
                )
        axis.set_yticks(y_positions, [label for _, label in metric_specs])
        axis.invert_yaxis()
        axis.set_xlim(lower, 1.003)
        axis.set_xlabel("Score (expanded scale)")
        axis.xaxis.set_major_formatter(mpl_ticker.PercentFormatter(1.0, decimals=1))
        axis.xaxis.set_major_locator(mpl_ticker.MaxNLocator(7))
        _prepare_axis(axis, grid_axis="x")
        handles, labels = axis.get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="upper right",
            bbox_to_anchor=(0.945, 0.825),
            ncol=2,
            columnspacing=1.25,
            handletextpad=0.55,
        )
        return _figure_to_svg(figure, title, description)


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
    target = (
        target.resolve()
        if target.is_absolute()
        else (config.project_root / target).resolve()
    )
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
        "chart_renderer": f"Matplotlib {mpl.__version__}",
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
