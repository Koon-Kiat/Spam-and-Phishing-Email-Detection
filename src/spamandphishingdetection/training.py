"""End-to-end model benchmarking, selection, refitting, and artifact creation."""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .config import ProjectConfig
from .datasets import (
    DatasetSplits,
    create_dataset_splits,
    file_sha256,
    load_and_audit_datasets,
)
from .evaluation import evaluate_probabilities, metrics_table, tune_threshold
from .privacy import scan_model_artifact

LOGGER = logging.getLogger(__name__)


@dataclass
class CandidateResult:
    """Validation result and fitted state for a candidate model."""

    name: str
    backend: str
    probabilities: np.ndarray
    threshold: float
    metrics: dict[str, Any]
    model: Any = None
    tokenizer: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingResult:
    """Locations and metrics produced by a successful training run."""

    artifact_dir: Path
    report_dir: Path
    selected_model: str
    test_metrics: dict[str, Any]


CLASSICAL_SIMPLICITY = {
    "word_only_logistic": 0,
    "word_char_nb": 1,
    "word_char_logistic": 2,
    "word_char_linear_svm": 3,
}


def configure_runtime_cache(config: ProjectConfig) -> None:
    """Keep package/model caches on the project drive."""

    cache_root = config.project_root / ".cache"
    environment = {
        "PIP_CACHE_DIR": cache_root / "pip",
        "HF_HOME": cache_root / "huggingface",
        "TORCH_HOME": cache_root / "torch",
    }
    for name, path in environment.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(name, str(path))


def _word_vectorizer(config: ProjectConfig, max_features: int | None = None) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=config.min_document_frequency,
        max_df=0.995,
        max_features=max_features or config.word_max_features,
        strip_accents="unicode",
        sublinear_tf=True,
        dtype=np.float32,
    )


def _word_char_features(config: ProjectConfig) -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", _word_vectorizer(config)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=max(2, config.min_document_frequency),
                    max_df=0.995,
                    max_features=config.char_max_features,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )


def build_classical_candidate(
    name: str,
    config: ProjectConfig,
    *,
    calibration_cv: int | list[tuple[np.ndarray, np.ndarray]] = 3,
    calibration_jobs: int = -1,
) -> Pipeline:
    """Build a reproducible sparse-text candidate by name."""

    if name == "word_only_logistic":
        return Pipeline(
            [
                ("features", _word_vectorizer(config)),
                (
                    "classifier",
                    LogisticRegression(
                        C=2.0,
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=config.random_seed,
                        solver="liblinear",
                    ),
                ),
            ]
        )
    if name == "word_char_logistic":
        classifier: BaseEstimator = LogisticRegression(
            C=2.0,
            class_weight="balanced",
            max_iter=1_000,
            random_state=config.random_seed,
            solver="liblinear",
        )
    elif name == "word_char_nb":
        classifier = ComplementNB(alpha=0.3)
    elif name == "word_char_linear_svm":
        classifier = CalibratedClassifierCV(
            estimator=LinearSVC(
                C=1.0,
                class_weight="balanced",
                random_state=config.random_seed,
            ),
            method="sigmoid",
            cv=calibration_cv,
            n_jobs=calibration_jobs,
        )
    else:
        raise ValueError(f"Unknown classical candidate: {name}")
    return Pipeline([("features", _word_char_features(config)), ("classifier", classifier)])


def _strata(frame: pd.DataFrame) -> pd.Series:
    return frame["source"].astype(str) + ":" + frame["label"].astype(str)


def _grouped_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    return [
        (np.asarray(train_index), np.asarray(test_index))
        for train_index, test_index in splitter.split(
            frame["text"],
            _strata(frame),
            frame["similarity_group"],
        )
    ]


def _calibration_folds(
    frame: pd.DataFrame,
    config: ProjectConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups_per_stratum = (
        frame.assign(_stratum=_strata(frame))
        .groupby("_stratum")["similarity_group"]
        .nunique()
    )
    n_splits = min(3, int(groups_per_stratum.min()))
    if n_splits < 2:
        raise ValueError("Group-aware calibration requires two groups per stratum")
    return _grouped_folds(frame, n_splits=n_splits, seed=config.random_seed + 17)


def _evaluate_grouped_fold(
    training: pd.DataFrame,
    config: ProjectConfig,
    repetition: int,
    fold: int,
    development_index: np.ndarray,
    evaluation_index: np.ndarray,
) -> list[dict[str, Any]]:
    names = (
        "word_only_logistic",
        "word_char_logistic",
        "word_char_nb",
        "word_char_linear_svm",
    )
    development = training.iloc[development_index].reset_index(drop=True)
    evaluation = training.iloc[evaluation_index].reset_index(drop=True)
    threshold_folds = _grouped_folds(
        development,
        n_splits=config.cv_folds,
        seed=config.random_seed + 100 + repetition * config.cv_folds + fold,
    )
    fit_index, threshold_index = threshold_folds[0]
    fit_frame = development.iloc[fit_index].reset_index(drop=True)
    threshold_frame = development.iloc[threshold_index].reset_index(drop=True)
    if set(fit_frame["similarity_group"]) & set(threshold_frame["similarity_group"]):
        raise ValueError("Inner threshold split contains group leakage")

    rows: list[dict[str, Any]] = []
    for name in names:
        LOGGER.info(
            "Grouped CV repetition %d/%d fold %d/%d: %s",
            repetition + 1,
            config.cv_repetitions,
            fold + 1,
            config.cv_folds,
            name,
        )
        model = build_classical_candidate(
            name,
            config,
            calibration_cv=(
                _calibration_folds(fit_frame, config)
                if name == "word_char_linear_svm"
                else 3
            ),
            calibration_jobs=1,
        )
        model.fit(fit_frame["text"].tolist(), fit_frame["label"].to_numpy())
        threshold_probabilities = _predict_positive_probability(
            model,
            threshold_frame["text"],
        )
        threshold, _ = tune_threshold(
            threshold_frame["label"],
            threshold_probabilities,
        )
        probabilities = _predict_positive_probability(model, evaluation["text"])
        metrics = evaluate_probabilities(
            evaluation["label"],
            probabilities,
            threshold,
        )
        rows.append(
            {
                "candidate": name,
                "repetition": repetition + 1,
                "fold": fold + 1,
                "fit_rows": len(fit_frame),
                "threshold_rows": len(threshold_frame),
                "evaluation_rows": len(evaluation),
                "threshold": threshold,
                "macro_f1": metrics["macro_f1"],
                "phishing_recall": metrics["phishing_recall"],
                "brier_score": metrics["calibration"]["brier_score"],
                "log_loss": metrics["calibration"]["log_loss"],
                "expected_calibration_error": metrics["calibration"][
                    "expected_calibration_error"
                ],
            }
        )
    return rows


def grouped_candidate_cross_validation(
    training: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run repeated grouped CV with an inner group-disjoint threshold split."""

    tasks = []
    for repetition in range(config.cv_repetitions):
        outer_folds = _grouped_folds(
            training,
            n_splits=config.cv_folds,
            seed=config.random_seed + repetition,
        )
        for fold, (development_index, evaluation_index) in enumerate(outer_folds):
            tasks.append(
                joblib.delayed(_evaluate_grouped_fold)(
                    training,
                    config,
                    repetition,
                    fold,
                    development_index,
                    evaluation_index,
                )
            )
    parallel_jobs = (
        1
        if len(training) < 1_000
        else min(8, max(1, (os.cpu_count() or 2) // 2), len(tasks))
    )
    completed = joblib.Parallel(
        n_jobs=parallel_jobs,
        prefer="processes",
        verbose=5,
    )(tasks)
    fold_rows = [row for fold_result in completed for row in fold_result]
    folds = pd.DataFrame(fold_rows)
    folds.sort_values(["repetition", "fold", "candidate"], inplace=True)
    summary = (
        folds.groupby("candidate", as_index=False)
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            worst_fold_macro_f1=("macro_f1", "min"),
            mean_phishing_recall=("phishing_recall", "mean"),
            mean_brier_score=("brier_score", "mean"),
            mean_log_loss=("log_loss", "mean"),
            mean_expected_calibration_error=("expected_calibration_error", "mean"),
            folds=("macro_f1", "size"),
        )
    )
    summary["simplicity_rank"] = summary["candidate"].map(CLASSICAL_SIMPLICITY)
    return folds, summary.sort_values(
        [
            "mean_macro_f1",
            "worst_fold_macro_f1",
            "mean_phishing_recall",
            "simplicity_rank",
        ],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _predict_positive_probability(model: Any, texts: pd.Series | list[str]) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(list(texts)), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Expected binary predict_proba output")
    return probabilities[:, 1]


def _fit_classical_candidates(
    splits: DatasetSplits,
    config: ProjectConfig,
) -> dict[str, CandidateResult]:
    results: dict[str, CandidateResult] = {}
    train_text = splits.train["text"].tolist()
    train_labels = splits.train["label"].to_numpy()
    validation_text = splits.validation["text"].tolist()
    validation_labels = splits.validation["label"].to_numpy()

    majority_probability = float(np.mean(train_labels))
    majority_probabilities = np.full(len(validation_labels), majority_probability)
    majority_metrics = evaluate_probabilities(
        validation_labels,
        majority_probabilities,
        threshold=0.5,
        sources=splits.validation["source"],
    )
    results["majority"] = CandidateResult(
        name="majority",
        backend="baseline",
        probabilities=majority_probabilities,
        threshold=0.5,
        metrics=majority_metrics,
        metadata={"training_prevalence": majority_probability},
    )

    names = (
        "word_only_logistic",
        "word_char_logistic",
        "word_char_nb",
        "word_char_linear_svm",
    )
    for name in names:
        LOGGER.info("Fitting classical candidate: %s", name)
        started = time.perf_counter()
        model = build_classical_candidate(
            name,
            config,
            calibration_cv=(
                _calibration_folds(splits.train, config)
                if name == "word_char_linear_svm"
                else 3
            ),
        )
        model.fit(train_text, train_labels)
        probabilities = _predict_positive_probability(model, validation_text)
        threshold, _ = tune_threshold(validation_labels, probabilities)
        metrics = evaluate_probabilities(
            validation_labels,
            probabilities,
            threshold,
            sources=splits.validation["source"],
        )
        results[name] = CandidateResult(
            name=name,
            backend="classical",
            model=model,
            probabilities=probabilities,
            threshold=threshold,
            metrics=metrics,
            metadata={"fit_seconds": time.perf_counter() - started},
        )
    return results


def _select_with_tolerance(
    candidates: list[CandidateResult],
    tolerance: float,
) -> CandidateResult:
    best_macro_f1 = max(candidate.metrics["macro_f1"] for candidate in candidates)
    eligible = [
        candidate
        for candidate in candidates
        if candidate.metrics["macro_f1"] >= best_macro_f1 - tolerance
    ]
    return max(
        eligible,
        key=lambda candidate: (
            candidate.metrics["phishing_recall"],
            -CLASSICAL_SIMPLICITY.get(candidate.name, 100),
        ),
    )


def _head_tail_token_ids(tokenizer: Any, text: str, max_length: int) -> list[int]:
    token_ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]
    available = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if len(token_ids) > available:
        head_length = int(available * 0.75)
        token_ids = token_ids[:head_length] + token_ids[-(available - head_length) :]
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("Transformer tokenizer must define CLS and SEP token IDs")
    return [int(tokenizer.cls_token_id), *token_ids, int(tokenizer.sep_token_id)]


class _HeadTailDataset:
    """Minimal torch-compatible dataset with deterministic head-tail tokenization."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int] | None,
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.encodings = [
            {
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
            }
            for ids in (_head_tail_token_ids(tokenizer, text, max_length) for text in texts)
        ]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = dict(self.encodings[index])
        if self.labels is not None:
            item["labels"] = int(self.labels[index])
        return item


def _transformer_probabilities(predictions: Any) -> np.ndarray:
    logits = predictions.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return (exponentials / exponentials.sum(axis=1, keepdims=True))[:, 1]


def _train_transformer(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame | None,
    config: ProjectConfig,
    output_dir: Path,
    epochs: int | None = None,
) -> tuple[Any, Any, Any, int]:
    """Fine-tune DistilBERT, importing the heavy stack only when requested."""

    configure_runtime_cache(config)
    import torch
    from sklearn.metrics import f1_score, recall_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.transformer_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.transformer_model,
        num_labels=2,
        id2label={0: "safe", 1: "phishing"},
        label2id={"safe": 0, "phishing": 1},
    )
    train_dataset = _HeadTailDataset(
        train_frame["text"].tolist(),
        train_frame["label"].astype(int).tolist(),
        tokenizer,
        config.transformer_max_length,
    )
    validation_dataset = (
        _HeadTailDataset(
            validation_frame["text"].tolist(),
            validation_frame["label"].astype(int).tolist(),
            tokenizer,
            config.transformer_max_length,
        )
        if validation_frame is not None
        else None
    )

    def compute_metrics(prediction: Any) -> dict[str, float]:
        logits, labels = prediction
        predicted = np.argmax(logits, axis=-1)
        return {
            "macro_f1": float(f1_score(labels, predicted, average="macro")),
            "phishing_recall": float(recall_score(labels, predicted, pos_label=1)),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    has_cuda = torch.cuda.is_available()
    supports_bf16 = has_cuda and torch.cuda.is_bf16_supported()
    has_validation = validation_dataset is not None
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=config.transformer_batch_size,
        per_device_eval_batch_size=config.transformer_batch_size,
        num_train_epochs=float(epochs or config.transformer_epochs),
        learning_rate=config.transformer_learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch" if has_validation else "no",
        save_strategy="epoch" if has_validation else "no",
        load_best_model_at_end=has_validation,
        metric_for_best_model="macro_f1" if has_validation else None,
        greater_is_better=True if has_validation else None,
        save_total_limit=1,
        logging_strategy="epoch",
        report_to="none",
        seed=config.random_seed,
        data_seed=config.random_seed,
        bf16=supports_bf16,
        fp16=has_cuda and not supports_bf16,
        tf32=has_cuda,
        dataloader_num_workers=0,
    )
    callbacks = [EarlyStoppingCallback(early_stopping_patience=1)] if has_validation else []
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics if has_validation else None,
        callbacks=callbacks,
    )
    trainer.train()

    best_epoch = int(epochs or config.transformer_epochs)
    if has_validation:
        evaluations = [
            row for row in trainer.state.log_history if "eval_macro_f1" in row and "epoch" in row
        ]
        if evaluations:
            best_epoch = max(evaluations, key=lambda row: row["eval_macro_f1"])["epoch"]
            best_epoch = max(1, int(round(float(best_epoch))))
    return trainer, model, tokenizer, best_epoch


def _fit_transformer_candidate(
    splits: DatasetSplits,
    config: ProjectConfig,
    candidate_dir: Path,
) -> CandidateResult:
    LOGGER.info("Fine-tuning transformer candidate: %s", config.transformer_model)
    started = time.perf_counter()
    trainer, model, tokenizer, best_epoch = _train_transformer(
        splits.train,
        splits.validation,
        config,
        candidate_dir,
    )
    validation_dataset = _HeadTailDataset(
        splits.validation["text"].tolist(),
        splits.validation["label"].astype(int).tolist(),
        tokenizer,
        config.transformer_max_length,
    )
    probabilities = _transformer_probabilities(trainer.predict(validation_dataset))
    threshold, _ = tune_threshold(splits.validation["label"], probabilities)
    metrics = evaluate_probabilities(
        splits.validation["label"],
        probabilities,
        threshold,
        sources=splits.validation["source"],
    )
    return CandidateResult(
        name="distilbert",
        backend="transformer",
        model=model,
        tokenizer=tokenizer,
        probabilities=probabilities,
        threshold=threshold,
        metrics=metrics,
        metadata={
            "fit_seconds": time.perf_counter() - started,
            "best_epoch": best_epoch,
            "model_name": config.transformer_model,
        },
    )


def _maybe_build_ensemble(
    classical: CandidateResult,
    transformer: CandidateResult,
    validation: pd.DataFrame,
    config: ProjectConfig,
) -> CandidateResult | None:
    best_single = _select_with_tolerance(
        [classical, transformer],
        config.selection_tolerance,
    )
    candidates: list[CandidateResult] = []
    for classical_weight in (0.25, 0.5, 0.75):
        probabilities = (
            classical_weight * classical.probabilities
            + (1.0 - classical_weight) * transformer.probabilities
        )
        threshold, _ = tune_threshold(validation["label"], probabilities)
        metrics = evaluate_probabilities(
            validation["label"],
            probabilities,
            threshold,
            sources=validation["source"],
        )
        candidates.append(
            CandidateResult(
                name=f"ensemble_{classical_weight:.2f}",
                backend="ensemble",
                probabilities=probabilities,
                threshold=threshold,
                metrics=metrics,
                metadata={
                    "classical_weight": classical_weight,
                    "transformer_weight": 1.0 - classical_weight,
                    "classical_name": classical.name,
                },
            )
        )
    best_ensemble = max(
        candidates,
        key=lambda candidate: (
            candidate.metrics["macro_f1"],
            candidate.metrics["phishing_recall"],
        ),
    )
    improvement = best_ensemble.metrics["macro_f1"] - best_single.metrics["macro_f1"]
    if (
        improvement >= config.ensemble_min_gain
        and best_ensemble.metrics["phishing_recall"] >= best_single.metrics["phishing_recall"]
    ):
        return best_ensemble
    return None


def _package_versions() -> dict[str, str]:
    packages = [
        "accelerate",
        "beautifulsoup4",
        "datasketch",
        "Flask",
        "joblib",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "torch",
        "tldextract",
        "transformers",
    ]
    installed = {}
    for package in packages:
        try:
            installed[package] = version(package)
        except PackageNotFoundError:
            continue
    return installed


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def _monitoring_baseline(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    counts, edges = np.histogram(probabilities, bins=np.linspace(0.0, 1.0, 11))
    slices = {}
    for column in (
        "length_slice",
        "has_html",
        "has_url",
        "has_obfuscation",
        "attachment_slice",
        "language_slice",
        "source",
    ):
        values = frame[column].astype(str).value_counts(normalize=True, dropna=False)
        slices[column] = {str(name): float(value) for name, value in values.items()}
    return {
        "rows": len(frame),
        "score_bin_edges": [float(value) for value in edges],
        "score_bin_fractions": [float(value / len(frame)) for value in counts],
        "predicted_phishing_fraction": float(np.mean(probabilities >= threshold)),
        "mean_probability": float(np.mean(probabilities)),
        "slice_fractions": slices,
        "alert_thresholds": {
            "score_population_stability_index": 0.20,
            "predicted_prevalence_absolute_change": 0.10,
            "slice_fraction_absolute_change": 0.15,
            "performance_macro_f1_drop": 0.05,
        },
    }


def _save_split_assignments(splits: DatasetSplits, report_dir: Path) -> Path:
    rows = []
    for split_name, frame in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        selection = frame[
            [
                "row_id",
                "source",
                "label",
                "text_hash",
                "similarity_group",
                "split_fold",
            ]
        ].copy()
        selection.insert(0, "split", split_name)
        rows.append(selection)
    path = report_dir / "split_assignments.csv"
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)
    return path


def _cross_source_diagnostic(
    dataframe: pd.DataFrame,
    candidate_name: str,
    threshold: float,
    config: ProjectConfig,
) -> dict[str, Any]:
    diagnostics = {}
    sources = sorted(dataframe["source"].unique())
    for train_source in sources:
        test_source = next(source for source in sources if source != train_source)
        train = dataframe.loc[dataframe["source"] == train_source]
        test = dataframe.loc[dataframe["source"] == test_source]
        model = build_classical_candidate(
            candidate_name,
            config,
            calibration_cv=(
                _calibration_folds(train.reset_index(drop=True), config)
                if candidate_name == "word_char_linear_svm"
                else 3
            ),
        )
        model.fit(train["text"].tolist(), train["label"].to_numpy())
        probabilities = _predict_positive_probability(model, test["text"])
        diagnostics[f"{train_source}_to_{test_source}"] = evaluate_probabilities(
            test["label"],
            probabilities,
            threshold,
        )
    return diagnostics


def _predict_transformer_frame(
    model: Any,
    tokenizer: Any,
    frame: pd.DataFrame,
    config: ProjectConfig,
) -> np.ndarray:
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

    dataset = _HeadTailDataset(
        frame["text"].tolist(),
        frame["label"].astype(int).tolist(),
        tokenizer,
        config.transformer_max_length,
    )
    arguments = TrainingArguments(
        output_dir=str(config.artifacts_dir / ".prediction"),
        per_device_eval_batch_size=config.transformer_batch_size,
        report_to="none",
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    return _transformer_probabilities(trainer.predict(dataset))


def run_training(
    config: ProjectConfig,
    *,
    include_transformer: bool | None = None,
) -> TrainingResult:
    """Run the complete audit, benchmark, selection, refit, evaluation, and save workflow."""

    configure_runtime_cache(config)
    config.ensure_output_directories()
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    artifact_dir = config.artifacts_dir / run_id
    report_dir = config.reports_dir / run_id
    artifact_dir.mkdir(parents=True, exist_ok=False)
    report_dir.mkdir(parents=True, exist_ok=False)

    audited = load_and_audit_datasets(config)
    _save_json(report_dir / "data_audit.json", audited.audit)
    splits = create_dataset_splits(audited.dataframe, config)
    _save_json(report_dir / "split_summary.json", splits.summary)
    _save_split_assignments(splits, report_dir)

    grouped_cv_folds, grouped_cv_summary = grouped_candidate_cross_validation(
        splits.train,
        config,
    )
    grouped_cv_folds.to_csv(report_dir / "grouped_cv_folds.csv", index=False)
    grouped_cv_summary.to_csv(report_dir / "grouped_cv_summary.csv", index=False)

    classical_results = _fit_classical_candidates(splits, config)
    classical_candidates = [
        result for result in classical_results.values() if result.backend == "classical"
    ]
    cv_selected_name = str(grouped_cv_summary.iloc[0]["candidate"])
    best_classical = next(
        candidate for candidate in classical_candidates if candidate.name == cv_selected_name
    )

    transformer_result: CandidateResult | None = None
    transformer_error: str | None = None
    should_train_transformer = (
        config.transformer_enabled if include_transformer is None else include_transformer
    )
    candidate_transformer_dir = artifact_dir / ".transformer_candidate"
    if should_train_transformer:
        try:
            transformer_result = _fit_transformer_candidate(
                splits,
                config,
                candidate_transformer_dir,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            transformer_error = f"{type(error).__name__}: {error}"
            LOGGER.exception("Transformer experiment failed; continuing with classical models")

    selectable = [best_classical]
    ensemble_result = None
    if transformer_result is not None:
        selectable.append(transformer_result)
        ensemble_result = _maybe_build_ensemble(
            best_classical,
            transformer_result,
            splits.validation,
            config,
        )
        if ensemble_result is not None:
            selectable.append(ensemble_result)
    selected = (
        ensemble_result
        if ensemble_result is not None
        else _select_with_tolerance(selectable, config.selection_tolerance)
    )

    train_and_validation = pd.concat(
        [splits.train, splits.validation],
        ignore_index=True,
    )
    final_classical = None
    final_transformer_model = None
    final_transformer_tokenizer = None
    test_classical_probabilities = None
    test_transformer_probabilities = None
    prediction_started = time.perf_counter()

    if selected.backend in {"classical", "ensemble"}:
        classical_name = (
            selected.name
            if selected.backend == "classical"
            else selected.metadata["classical_name"]
        )
        final_classical = build_classical_candidate(
            classical_name,
            config,
            calibration_cv=(
                _calibration_folds(train_and_validation, config)
                if classical_name == "word_char_linear_svm"
                else 3
            ),
        )
        final_classical.fit(
            train_and_validation["text"].tolist(),
            train_and_validation["label"].to_numpy(),
        )
        test_classical_probabilities = _predict_positive_probability(
            final_classical,
            splits.test["text"],
        )
        joblib.dump(final_classical, artifact_dir / "model.joblib")

    if selected.backend in {"transformer", "ensemble"}:
        if transformer_result is None:
            raise RuntimeError("Transformer state is missing for selected model")
        final_dir = artifact_dir / "transformer"
        final_trainer, final_transformer_model, final_transformer_tokenizer, _ = _train_transformer(
            train_and_validation,
            None,
            config,
            final_dir / "checkpoints",
            epochs=int(transformer_result.metadata["best_epoch"]),
        )
        del final_trainer
        final_transformer_model.save_pretrained(final_dir / "model")
        final_transformer_tokenizer.save_pretrained(final_dir / "model")
        test_transformer_probabilities = _predict_transformer_frame(
            final_transformer_model,
            final_transformer_tokenizer,
            splits.test,
            config,
        )

    if selected.backend == "classical":
        test_probabilities = test_classical_probabilities
    elif selected.backend == "transformer":
        test_probabilities = test_transformer_probabilities
    else:
        weight = float(selected.metadata["classical_weight"])
        test_probabilities = (
            weight * test_classical_probabilities + (1.0 - weight) * test_transformer_probabilities
        )
    if test_probabilities is None:
        raise RuntimeError("Selected model did not produce test probabilities")

    prediction_seconds = time.perf_counter() - prediction_started
    test_metrics = evaluate_probabilities(
        splits.test["label"],
        test_probabilities,
        selected.threshold,
        sources=splits.test["source"],
    )
    test_metrics["batch_prediction_seconds"] = prediction_seconds
    test_metrics["milliseconds_per_email"] = 1_000 * prediction_seconds / len(splits.test)

    candidate_metrics = {name: result.metrics for name, result in classical_results.items()}
    if transformer_result is not None:
        candidate_metrics[transformer_result.name] = transformer_result.metrics
    if ensemble_result is not None:
        candidate_metrics[ensemble_result.name] = ensemble_result.metrics
    comparison = metrics_table(candidate_metrics)
    comparison.to_csv(report_dir / "candidate_metrics.csv", index=False)

    cross_source = _cross_source_diagnostic(
        audited.dataframe,
        best_classical.name,
        best_classical.threshold,
        config,
    )
    report = {
        "run_id": run_id,
        "selected_model": selected.name,
        "selected_backend": selected.backend,
        "selection_threshold": selected.threshold,
        "candidate_metrics": candidate_metrics,
        "test_metrics": test_metrics,
        "cross_source_diagnostic": cross_source,
        "grouped_cv_selection": grouped_cv_summary.to_dict(orient="records"),
        "transformer_error": transformer_error,
    }
    _save_json(report_dir / "metrics.json", report)

    report_names = (
        "data_audit.json",
        "split_summary.json",
        "split_assignments.csv",
        "grouped_cv_folds.csv",
        "grouped_cv_summary.csv",
        "candidate_metrics.csv",
        "metrics.json",
    )
    report_hashes = {
        name: file_sha256(report_dir / name)
        for name in report_names
        if (report_dir / name).is_file()
    }
    privacy_scan = scan_model_artifact(artifact_dir)
    provenance_path = config.project_root / "config" / "datasets.v1.json"
    dependency_lock = config.project_root / "requirements.lock"

    manifest = {
        "schema_version": 2,
        "model_version": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "backend": selected.backend,
        "selected_model": selected.name,
        "threshold": selected.threshold,
        "positive_class": 1,
        "labels": {"0": "safe", "1": "phishing"},
        "max_text_chars": config.max_text_chars,
        "max_request_bytes": config.max_request_bytes,
        "transformer_model": config.transformer_model,
        "transformer_max_length": config.transformer_max_length,
        "transformer_batch_size": config.transformer_batch_size,
        "ensemble": selected.metadata if selected.backend == "ensemble" else None,
        "test_metrics": test_metrics,
        "calibration": {
            "method": (
                "group-aware sigmoid"
                if selected.name == "word_char_linear_svm"
                else "native probability"
            ),
            "validation": selected.metrics["calibration"],
            "untouched_test": test_metrics["calibration"],
        },
        "grouping": {
            **audited.audit["near_duplicate_grouping"],
            "split_method": splits.summary["method"],
        },
        "monitoring_baseline": _monitoring_baseline(
            splits.test,
            np.asarray(test_probabilities),
            selected.threshold,
        ),
        "privacy_scan": privacy_scan,
        "dataset_fingerprints": audited.audit["dataset_fingerprints"],
        "dataset_manifest": {
            "path": "config/datasets.v1.json",
            "sha256": file_sha256(provenance_path) if provenance_path.is_file() else None,
        },
        "report_hashes": report_hashes,
        "dependency_lock": {
            "path": "requirements.lock",
            "sha256": file_sha256(dependency_lock) if dependency_lock.is_file() else None,
        },
        "config": config.manifest_dict(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
    }
    if (artifact_dir / "model.joblib").is_file():
        manifest["model_sha256"] = file_sha256(artifact_dir / "model.joblib")
    _save_json(artifact_dir / "manifest.json", manifest)

    latest = {
        "artifact": artifact_dir.relative_to(config.artifacts_dir).as_posix(),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    temporary_latest = config.artifacts_dir / "latest.json.tmp"
    _save_json(temporary_latest, latest)
    temporary_latest.replace(config.artifacts_dir / "latest.json")

    if candidate_transformer_dir.exists():
        resolved_candidate = candidate_transformer_dir.resolve()
        if artifact_dir.resolve() not in resolved_candidate.parents:
            raise RuntimeError("Refusing to remove transformer candidate outside artifact run")
        shutil.rmtree(resolved_candidate)

    return TrainingResult(
        artifact_dir=artifact_dir,
        report_dir=report_dir,
        selected_model=selected.name,
        test_metrics=test_metrics,
    )


def finalize_completed_classical_run(
    config: ProjectConfig,
    run_id: str,
) -> TrainingResult:
    """Validate and finalize a completed classical run after a packaging-stage failure.

    This recovery path never fits or selects a model. It is intentionally limited
    to a run that already contains the complete report set and one serialized
    classical model. The untouched-test predictions are recomputed and must match
    the recorded metrics before a manifest or latest pointer is written.
    """

    configure_runtime_cache(config)
    artifact_dir = config.artifacts_dir / run_id
    report_dir = config.reports_dir / run_id
    model_path = artifact_dir / "model.joblib"
    metrics_path = report_dir / "metrics.json"
    if not artifact_dir.is_dir() or not report_dir.is_dir():
        raise FileNotFoundError(f"Run directories are incomplete: {run_id}")
    if not model_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError(f"Completed model and metrics are required: {run_id}")
    if (artifact_dir / "manifest.json").exists():
        raise FileExistsError(f"Run is already finalized: {run_id}")

    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    if report.get("run_id") != run_id:
        raise ValueError("Run ID in metrics does not match the requested artifact")
    if report.get("selected_backend") != "classical":
        raise ValueError("Recovery finalization is limited to classical artifacts")
    selected_name = str(report["selected_model"])
    threshold = float(report["selection_threshold"])
    if selected_name not in CLASSICAL_SIMPLICITY:
        raise ValueError(f"Unknown selected classical model: {selected_name}")

    required_reports = (
        "data_audit.json",
        "split_summary.json",
        "split_assignments.csv",
        "grouped_cv_folds.csv",
        "grouped_cv_summary.csv",
        "candidate_metrics.csv",
        "metrics.json",
    )
    missing_reports = [
        name for name in required_reports if not (report_dir / name).is_file()
    ]
    if missing_reports:
        raise FileNotFoundError(
            "Completed run is missing reports: " + ", ".join(missing_reports)
        )

    audited = load_and_audit_datasets(config)
    splits = create_dataset_splits(audited.dataframe, config)
    saved_split_summary = json.loads(
        (report_dir / "split_summary.json").read_text(encoding="utf-8")
    )
    if saved_split_summary != splits.summary:
        raise ValueError("Re-derived split summary does not match the completed run")

    model = joblib.load(model_path)
    test_probabilities = _predict_positive_probability(model, splits.test["text"])
    recomputed_test = evaluate_probabilities(
        splits.test["label"],
        test_probabilities,
        threshold,
        sources=splits.test["source"],
    )
    recorded_test = report["test_metrics"]
    for metric_name in (
        "accuracy",
        "macro_f1",
        "phishing_precision",
        "phishing_recall",
        "phishing_f1",
        "false_positive_rate",
        "roc_auc",
        "pr_auc",
    ):
        if not np.isclose(
            float(recomputed_test[metric_name]),
            float(recorded_test[metric_name]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"Recomputed untouched-test {metric_name} does not match")

    candidate_transformer_dir = artifact_dir / ".transformer_candidate"
    if candidate_transformer_dir.exists():
        resolved_candidate = candidate_transformer_dir.resolve()
        if artifact_dir.resolve() not in resolved_candidate.parents:
            raise RuntimeError("Refusing to remove transformer candidate outside artifact run")
        shutil.rmtree(resolved_candidate)

    report_hashes = {
        name: file_sha256(report_dir / name)
        for name in required_reports
    }
    privacy_scan = scan_model_artifact(artifact_dir)
    provenance_path = config.project_root / "config" / "datasets.v1.json"
    dependency_lock = config.project_root / "requirements.lock"
    selected_metrics = report["candidate_metrics"][selected_name]
    manifest = {
        "schema_version": 2,
        "model_version": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "backend": "classical",
        "selected_model": selected_name,
        "threshold": threshold,
        "positive_class": 1,
        "labels": {"0": "safe", "1": "phishing"},
        "max_text_chars": config.max_text_chars,
        "max_request_bytes": config.max_request_bytes,
        "transformer_model": config.transformer_model,
        "transformer_max_length": config.transformer_max_length,
        "transformer_batch_size": config.transformer_batch_size,
        "ensemble": None,
        "test_metrics": recorded_test,
        "calibration": {
            "method": (
                "group-aware sigmoid"
                if selected_name == "word_char_linear_svm"
                else "native probability"
            ),
            "validation": selected_metrics["calibration"],
            "untouched_test": recorded_test["calibration"],
        },
        "grouping": {
            **audited.audit["near_duplicate_grouping"],
            "split_method": splits.summary["method"],
        },
        "monitoring_baseline": _monitoring_baseline(
            splits.test,
            np.asarray(test_probabilities),
            threshold,
        ),
        "privacy_scan": privacy_scan,
        "dataset_fingerprints": audited.audit["dataset_fingerprints"],
        "dataset_manifest": {
            "path": "config/datasets.v1.json",
            "sha256": file_sha256(provenance_path) if provenance_path.is_file() else None,
        },
        "report_hashes": report_hashes,
        "dependency_lock": {
            "path": "requirements.lock",
            "sha256": file_sha256(dependency_lock) if dependency_lock.is_file() else None,
        },
        "config": config.manifest_dict(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        "recovery": {
            "reason": "packaging-stage privacy scan included an unselected hidden checkpoint",
            "validation": "untouched-test metrics re-derived from the serialized selected model",
        },
        "model_sha256": file_sha256(model_path),
    }
    _save_json(artifact_dir / "manifest.json", manifest)

    latest = {
        "artifact": artifact_dir.relative_to(config.artifacts_dir).as_posix(),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    temporary_latest = config.artifacts_dir / "latest.json.tmp"
    _save_json(temporary_latest, latest)
    temporary_latest.replace(config.artifacts_dir / "latest.json")
    return TrainingResult(
        artifact_dir=artifact_dir,
        report_dir=report_dir,
        selected_model=selected_name,
        test_metrics=recorded_test,
    )
