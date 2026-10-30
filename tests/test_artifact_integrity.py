from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Callable
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from spamandphishingdetection import integrity
from spamandphishingdetection.api import create_app
from spamandphishingdetection.config import ProjectConfig
from spamandphishingdetection.datasets import file_sha256
from spamandphishingdetection.inference import ArtifactError, Predictor
from spamandphishingdetection.integrity import bind_latest


def _write_manifest(artifact: Path, manifest: dict) -> None:
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _classical_artifact(config: ProjectConfig, run_id: str = "run-a") -> Path:
    model = Pipeline(
        [
            ("features", TfidfVectorizer()),
            ("classifier", LogisticRegression()),
        ]
    ).fit(
        [
            "urgent verify your password",
            "team lunch agenda",
            "confirm account now",
            "project notes",
        ],
        [1, 0, 1, 0],
    )
    artifact = config.artifacts_dir / run_id
    artifact.mkdir(parents=True)
    joblib.dump(model, artifact / "model.joblib")
    _write_manifest(
        artifact,
        {
            "schema_version": 2,
            "model_version": run_id,
            "backend": "classical",
            "threshold": 0.5,
            "max_text_chars": 1_000,
            "model_sha256": file_sha256(artifact / "model.joblib"),
        },
    )
    return artifact


@pytest.fixture
def published(config_factory: Callable[..., ProjectConfig]) -> tuple[ProjectConfig, Path]:
    config = config_factory()
    artifact = _classical_artifact(config)
    bind_latest(config, artifact, action="promote")
    return config, artifact


def _refuse_deserialization(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("joblib.load must not run for untrusted bytes")

    monkeypatch.setattr(integrity.joblib, "load", refuse)


def test_valid_published_artifact_loads(published: tuple[ProjectConfig, Path]) -> None:
    config, artifact = published

    predictor = Predictor.load(config)

    assert predictor.model_version == artifact.name
    assert predictor.manifest_sha256 == file_sha256(artifact / "manifest.json")
    assert predictor.predict_text("urgent verify your password").label in {"safe", "phishing"}


def test_modified_payload_is_rejected_before_deserialization(
    published: tuple[ProjectConfig, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, artifact = published
    with (artifact / "model.joblib").open("ab") as handle:
        handle.write(b"tampered")
    _refuse_deserialization(monkeypatch)

    with pytest.raises(ArtifactError, match="checksum does not match"):
        Predictor.load(config)


def test_modified_manifest_is_rejected(
    published: tuple[ProjectConfig, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, artifact = published
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    manifest["threshold"] = 0.01
    _write_manifest(artifact, manifest)
    _refuse_deserialization(monkeypatch)

    with pytest.raises(ArtifactError, match="trusted latest.json record"):
        Predictor.load(config)


def test_mismatched_latest_hash_is_rejected(
    published: tuple[ProjectConfig, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, _artifact = published
    latest_path = config.artifacts_dir / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    latest["manifest_sha256"] = "0" * 64
    latest_path.write_text(json.dumps(latest), encoding="utf-8")
    _refuse_deserialization(monkeypatch)

    with pytest.raises(ArtifactError, match="does not match"):
        Predictor.load(config)


@pytest.mark.parametrize(
    "corruption",
    ["latest_without_hash", "malformed_latest", "manifest_without_model_hash", "invalid_json"],
)
def test_missing_or_malformed_integrity_metadata_fails_closed(
    published: tuple[ProjectConfig, Path],
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    config, artifact = published
    latest_path = config.artifacts_dir / "latest.json"
    if corruption == "latest_without_hash":
        latest_path.write_text(json.dumps({"artifact": artifact.name}), encoding="utf-8")
    elif corruption == "malformed_latest":
        latest_path.write_text("{not json", encoding="utf-8")
    elif corruption == "manifest_without_model_hash":
        manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        del manifest["model_sha256"]
        _write_manifest(artifact, manifest)
        bind_latest(config, artifact, action="promote")
    else:
        (artifact / "manifest.json").write_text("{not json", encoding="utf-8")
        bind_latest(config, artifact, action="promote")
    _refuse_deserialization(monkeypatch)

    with pytest.raises(ArtifactError):
        Predictor.load(config)


def test_legacy_pointer_is_trusted_through_promotion_history(
    published: tuple[ProjectConfig, Path],
) -> None:
    config, artifact = published
    latest_path = config.artifacts_dir / "latest.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    (config.artifacts_dir / "promotion_history.json").write_text(
        json.dumps([latest]),
        encoding="utf-8",
    )
    del latest["manifest_sha256"]
    latest_path.write_text(json.dumps(latest), encoding="utf-8")

    assert Predictor.load(config).model_version == artifact.name


def test_explicit_artifacts_require_an_external_trust_record(
    published: tuple[ProjectConfig, Path],
    tmp_path: Path,
) -> None:
    config, _published_artifact = published
    unbound = _classical_artifact(config, "run-b")
    outside = tmp_path / "downloaded-release"
    shutil.copytree(unbound, outside)
    manifest_hash = file_sha256(unbound / "manifest.json")

    with pytest.raises(ArtifactError, match="not bound to a trusted record"):
        Predictor.load(config, "run-b")
    with pytest.raises(ArtifactError, match="not bound to a trusted record"):
        Predictor.load(config, outside)
    with pytest.raises(ArtifactError, match="does not match"):
        Predictor.load(config, "run-b", expected_manifest_sha256="f" * 64)
    with pytest.raises(ArtifactError, match="64 hexadecimal"):
        Predictor.load(config, "run-b", expected_manifest_sha256="not-a-hash")

    assert Predictor.load(
        config, "run-b", expected_manifest_sha256=manifest_hash
    ).model_version == "run-b"
    assert Predictor.load(
        config, outside, expected_manifest_sha256=manifest_hash.upper()
    ).model_version == "run-b"


def test_deserialization_failure_is_reported_as_unavailable(
    published: tuple[ProjectConfig, Path],
) -> None:
    config, artifact = published
    (artifact / "model.joblib").write_bytes(b"harmless bytes that are not a pickle")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    manifest["model_sha256"] = file_sha256(artifact / "model.joblib")
    _write_manifest(artifact, manifest)
    bind_latest(config, artifact, action="promote")

    with pytest.raises(ArtifactError, match="could not be deserialized"):
        Predictor.load(config)


@pytest.mark.parametrize("corruption", ["payload", "manifest", "latest", "deserialization"])
def test_api_reports_stable_unavailable_responses_for_corrupted_artifacts(
    published: tuple[ProjectConfig, Path],
    caplog: pytest.LogCaptureFixture,
    corruption: str,
) -> None:
    config, artifact = published
    if corruption == "payload":
        (artifact / "model.joblib").write_bytes(b"tampered")
    elif corruption == "manifest":
        (artifact / "manifest.json").write_text("{}", encoding="utf-8")
    elif corruption == "latest":
        (config.artifacts_dir / "latest.json").write_text("[]", encoding="utf-8")
    else:
        (artifact / "model.joblib").write_bytes(b"not a pickle")
        manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
        manifest["model_sha256"] = file_sha256(artifact / "model.joblib")
        _write_manifest(artifact, manifest)
        bind_latest(config, artifact, action="promote")

    with caplog.at_level(logging.WARNING, logger="spamandphishingdetection.api"):
        client = create_app(config).test_client()
    health = client.get("/health")
    prediction = client.post("/api/v1/predict", json={"text": "verify your password"})

    assert health.status_code == 503
    assert health.get_json() == {
        "status": "unavailable",
        "model_loaded": False,
        "error": "Prediction model is unavailable",
    }
    assert prediction.status_code == 503
    assert prediction.get_json() == {"error": "Prediction model is unavailable"}
    for body in (health.get_data(as_text=True), prediction.get_data(as_text=True)):
        assert str(config.project_root) not in body
        assert "checksum" not in body.lower()
        assert "Traceback" not in body
    assert "ArtifactError" in caplog.text
