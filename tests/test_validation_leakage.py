from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from spamandphishingdetection import generalization, training
from spamandphishingdetection.config import ProjectConfig
from spamandphishingdetection.datasets import (
    create_dataset_splits,
    file_sha256,
    load_and_audit_datasets,
)
from spamandphishingdetection.evaluation import evaluate_probabilities, tune_threshold
from spamandphishingdetection.generalization import generate_generalization_report
from spamandphishingdetection.inference import Prediction
from spamandphishingdetection.integrity import load_trusted_manifest
from spamandphishingdetection.training import (
    VALIDATION_PREDICTIONS_FILE,
    CandidateResult,
    run_training,
)

_HEADER = json.dumps({"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
_SAFETENSORS = struct.pack("<Q", len(_HEADER)) + _HEADER + b"\x00\x00\x80\x3f"


class _SavedComponent:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files

    def save_pretrained(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, payload in self.files.items():
            (directory / name).write_bytes(payload)


def _candidate_probabilities(frame: pd.DataFrame) -> np.ndarray:
    # Deterministic, imperfect scores so threshold tuning and metrics are non-trivial.
    jitter = (np.arange(len(frame)) % 7) / 20
    return np.where(frame["label"].to_numpy() == 1, 0.55 + jitter / 2, 0.15 + jitter)


def _mock_transformer(
    monkeypatch: pytest.MonkeyPatch,
    events: list[tuple[str, Any]],
) -> dict[str, CandidateResult]:
    created: dict[str, CandidateResult] = {}

    def fit_candidate(splits, config, candidate_dir: Path) -> CandidateResult:
        candidate_dir.mkdir(parents=True)
        (candidate_dir / "checkpoint.json").write_text("{}", encoding="utf-8")
        probabilities = _candidate_probabilities(splits.validation)
        threshold, _ = tune_threshold(splits.validation["label"], probabilities)
        created["transformer"] = CandidateResult(
            name="distilbert",
            backend="transformer",
            probabilities=probabilities,
            threshold=threshold,
            metrics=evaluate_probabilities(
                splits.validation["label"],
                probabilities,
                threshold,
                sources=splits.validation["source"],
            ),
            metadata={"best_epoch": 2, "model_name": config.transformer_model},
        )
        events.append(("select", set(splits.train["row_id"])))
        return created["transformer"]

    def train_final(train_frame, validation_frame, config, output_dir: Path, epochs=None):
        output_dir.mkdir(parents=True)
        (output_dir / "trainer_state.json").write_text("{}", encoding="utf-8")
        events.append(
            ("refit", {"rows": set(train_frame["row_id"]), "validation": validation_frame,
                       "epochs": epochs})
        )
        model = _SavedComponent(
            {"config.json": b'{"model_type": "distilbert"}', "model.safetensors": _SAFETENSORS}
        )
        tokenizer = _SavedComponent({"tokenizer.json": b'{"model": {"type": "WordPiece"}}'})
        return None, model, tokenizer, epochs

    def predict_frame(model, tokenizer, frame, config) -> np.ndarray:
        return _candidate_probabilities(frame)

    monkeypatch.setattr(training, "_fit_transformer_candidate", fit_candidate)
    monkeypatch.setattr(training, "_train_transformer", train_final)
    monkeypatch.setattr(training, "_predict_transformer_frame", predict_frame)
    return created


def _force_backend(monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    def forced_ensemble(classical, transformer, validation, config):
        if backend != "ensemble":
            return None
        probabilities = 0.5 * classical.probabilities + 0.5 * transformer.probabilities
        threshold, _ = tune_threshold(validation["label"], probabilities)
        return CandidateResult(
            name="ensemble_0.50",
            backend="ensemble",
            probabilities=probabilities,
            threshold=threshold,
            metrics=evaluate_probabilities(
                validation["label"], probabilities, threshold, sources=validation["source"]
            ),
            metadata={
                "classical_weight": 0.5,
                "transformer_weight": 0.5,
                "classical_name": classical.name,
            },
        )

    def prefer_transformer(candidates, tolerance):
        return next(candidate for candidate in candidates if candidate.backend == "transformer")

    monkeypatch.setattr(training, "_maybe_build_ensemble", forced_ensemble)
    monkeypatch.setattr(training, "_select_with_tolerance", prefer_transformer)


class _RefitPredictor:
    """Stands in for the train-plus-validation refit, which is certain about every row."""

    def __init__(self, config: ProjectConfig, artifact=None, **options: Any) -> None:
        trusted = load_trusted_manifest(config, artifact, **options)
        self.artifact_dir = trusted.artifact_dir
        self.manifest = trusted.manifest
        self.manifest_sha256 = trusted.sha256
        self.backend = str(self.manifest["backend"])
        self.threshold = float(self.manifest["threshold"])
        self.model_version = str(self.manifest["model_version"])

    def predict_text(self, text: str) -> Prediction:
        return Prediction("phishing", True, 1.0, self.threshold, self.model_version)


@pytest.mark.parametrize("backend", ["transformer", "ensemble"])
def test_validation_metrics_come_only_from_training_only_models(
    trained_project: ProjectConfig,
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
) -> None:
    config = trained_project
    events: list[tuple[str, Any]] = []
    created = _mock_transformer(monkeypatch, events)
    _force_backend(monkeypatch, backend)

    result = run_training(config, include_transformer=True)

    splits = create_dataset_splits(load_and_audit_datasets(config).dataframe, config)
    manifest = json.loads((result.artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((result.report_dir / "metrics.json").read_text(encoding="utf-8"))
    evidence_path = result.report_dir / VALIDATION_PREDICTIONS_FILE
    evidence = pd.read_csv(evidence_path, dtype={"row_id": str})
    candidate = created["transformer"]

    # Selection evidence is frozen, hashed, and produced by training-only models.
    assert manifest["backend"] == backend
    assert manifest["report_hashes"][VALIDATION_PREDICTIONS_FILE] == file_sha256(evidence_path)
    assert evidence["row_id"].tolist() == splits.validation["row_id"].astype(str).tolist()
    if backend == "transformer":
        assert np.array_equal(evidence["probability"].to_numpy(), candidate.probabilities)
        assert manifest["threshold"] == candidate.threshold
    else:
        assert np.array_equal(
            evidence["transformer_probability"].to_numpy(), candidate.probabilities
        )
        assert np.allclose(
            evidence["probability"],
            0.5 * evidence["classical_probability"] + 0.5 * evidence["transformer_probability"],
            rtol=0,
            atol=1e-15,
        )
    provenance = manifest["metric_provenance"]
    assert provenance == report["metric_provenance"]
    assert provenance["validation"]["fitted_on"] == ["train"]
    assert provenance["untouched_test"]["fitted_on"] == ["train", "validation"]

    # The refit happens once, after selection, on train plus validation only.
    assert [name for name, _detail in events] == ["select", "refit"]
    selection_rows = events[0][1]
    refit = events[1][1]
    test_rows = set(splits.test["row_id"])
    assert selection_rows == set(splits.train["row_id"])
    assert refit["rows"] == set(splits.train["row_id"]) | set(splits.validation["row_id"])
    assert not refit["rows"] & test_rows
    assert refit["validation"] is None
    assert refit["epochs"] == 2
    assert not (result.artifact_dir / ".transformer_candidate").exists()
    assert not (result.artifact_dir / "transformer" / "checkpoints").exists()

    # Generalization reports the frozen evidence, never the refit scored on validation.
    monkeypatch.setattr(generalization.Predictor, "load", _RefitPredictor)
    internal = generate_generalization_report(config, include_external=False)
    payload = json.loads(internal.read_text(encoding="utf-8"))
    expected = evaluate_probabilities(
        splits.validation["label"],
        evidence["probability"].to_numpy(),
        manifest["threshold"],
        sources=splits.validation["source"],
    )
    assert payload["validation"]["macro_f1"] == expected["macro_f1"]
    assert payload["validation"]["false_positive_rate"] == expected["false_positive_rate"]
    assert payload["validation"]["false_positive_rate"] < 1.0
    assert payload["untouched_test"]["false_positive_rate"] == 1.0
    assert "recorded pre-refit predictions" in payload["metric_provenance"]["validation"][
        "source"
    ]
    if backend == "transformer":
        assert payload["validation"]["macro_f1"] == candidate.metrics["macro_f1"]


@pytest.mark.parametrize("problem", ["missing", "tampered"])
def test_generalization_refuses_unverifiable_validation_evidence(
    trained_project: ProjectConfig,
    monkeypatch: pytest.MonkeyPatch,
    problem: str,
) -> None:
    config = trained_project
    _mock_transformer(monkeypatch, [])
    _force_backend(monkeypatch, "transformer")
    result = run_training(config, include_transformer=True)
    evidence_path = result.report_dir / VALIDATION_PREDICTIONS_FILE
    if problem == "missing":
        evidence_path.unlink()
    else:
        evidence = pd.read_csv(evidence_path, dtype={"row_id": str})
        evidence["probability"] = 0.99
        evidence.to_csv(evidence_path, index=False)
    monkeypatch.setattr(generalization.Predictor, "load", _RefitPredictor)

    with pytest.raises(RuntimeError, match="validation predictions"):
        generate_generalization_report(config, include_external=False)
