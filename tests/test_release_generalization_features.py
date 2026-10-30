from __future__ import annotations

import json
import struct
import zipfile
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from spamandphishingdetection.config import ProjectConfig
from spamandphishingdetection.datasets import file_sha256
from spamandphishingdetection.evaluation import calibration_diagnostics
from spamandphishingdetection.generalization import load_spaphish_v5
from spamandphishingdetection.grouping import build_near_duplicate_groups
from spamandphishingdetection.integrity import build_transformer_file_manifest
from spamandphishingdetection.monitoring import monitor_batch
from spamandphishingdetection.provenance import ProvenanceError, fetch_data
from spamandphishingdetection.release import (
    ReleaseValidationError,
    package_release,
    promote_artifact,
)

_SAFETENSORS_HEADER = json.dumps(
    {"classifier.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
).encode()
_FIXTURE_SAFETENSORS = (
    struct.pack("<Q", len(_SAFETENSORS_HEADER)) + _SAFETENSORS_HEADER + b"\x00\x00\x80\x3f"
)


def test_near_duplicate_candidates_are_exactly_verified() -> None:
    texts = [
        "Urgent verify your account at https://one.example/login now",
        "Urgent verify your account at https://two.example/login now",
        "Quarterly planning agenda and approved project notes",
    ]
    hashes = ["a" * 64, "b" * 64, "c" * 64]
    groups, audit = build_near_duplicate_groups(texts, hashes)

    assert groups[0] == groups[1]
    assert groups[0] != groups[2]
    assert audit["minhash_permutations"] == 128
    assert audit["verified_pairs"] >= 1


def test_calibration_metrics_include_reliability_bins() -> None:
    diagnostics = calibration_diagnostics(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        bins=10,
    )

    assert diagnostics["brier_score"] < 0.1
    assert diagnostics["log_loss"] > 0
    assert diagnostics["expected_calibration_error"] >= 0
    assert len(diagnostics["reliability"]) == 10


def test_fetch_data_verifies_normalization_and_refuses_overwrite(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    source = b"column\r\nvalue\r\n"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("source.csv", source)
    normalized = source.replace(b"\r\n", b"\n")
    import hashlib

    manifest = {
        "schema_version": 1,
        "policy": {"license_acceptance_required": True},
        "datasets": [
            {
                "id": "fixture",
                "download_url": archive.as_uri(),
                "archive_sha256": file_sha256(archive),
                "archive_bytes": archive.stat().st_size,
                "members": [
                    {
                        "archive_path": "source.csv",
                        "destination": "data/fixture.csv",
                        "source_sha256": hashlib.sha256(source).hexdigest(),
                        "source_bytes": len(source),
                        "sha256": hashlib.sha256(normalized).hexdigest(),
                        "bytes": len(normalized),
                        "normalization": "crlf_to_lf",
                    }
                ],
            }
        ],
        "external": [],
    }
    manifest_path = tmp_path / "datasets.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = fetch_data(
        tmp_path,
        accept_licenses=True,
        manifest_path=manifest_path,
    )
    destination = tmp_path / "data/fixture.csv"
    assert result[0]["status"] == "installed"
    assert destination.read_bytes() == normalized

    destination.write_bytes(b"unexpected")
    with pytest.raises(ProvenanceError, match="Refusing to overwrite"):
        fetch_data(
            tmp_path,
            accept_licenses=True,
            manifest_path=manifest_path,
        )


def test_spaphish_adapter_preserves_rows_and_prevalence(
    config_factory: Callable[..., ProjectConfig],
    tmp_path: Path,
) -> None:
    config = config_factory()
    rows = []
    for index in range(1_395):
        label = 0 if index < 664 else 1
        rows.append(
            {
                "hash": f"hash-{index}",
                "subject": f"Mensaje {index}",
                "body": "Verifique su cuenta" if label else "Agenda de proyecto",
                "date": "2025-01-01",
                "attachments_count": index % 2,
                "Label": label,
            }
        )
    path = tmp_path / "spaphish.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    adapted = load_spaphish_v5(path, config)

    assert len(adapted) == 1_395
    assert adapted["label"].value_counts().to_dict() == {1: 731, 0: 664}
    assert set(adapted["attachment_slice"]) == {"has_attachment", "no_attachment"}
    assert adapted["row_id"].str.startswith("spaphish:").all()


def _write_transformer_payload(artifact: Path) -> None:
    model_dir = artifact / "transformer" / "model"
    model_dir.mkdir(parents=True)
    for name, content in (
        ("config.json", b'{"model_type": "distilbert"}'),
        ("model.safetensors", _FIXTURE_SAFETENSORS),
        ("tokenizer.json", b'{"model": {"type": "WordPiece"}}'),
        ("tokenizer_config.json", b'{"do_lower_case": true}'),
    ):
        (model_dir / name).write_bytes(content)


def _write_release_artifact(
    config: ProjectConfig,
    run_id: str,
    *,
    backend: str = "classical",
) -> Path:
    artifact = config.artifacts_dir / run_id
    report = config.reports_dir / run_id
    artifact.mkdir(parents=True)
    report.mkdir(parents=True)
    if backend in {"classical", "ensemble"}:
        (artifact / "model.joblib").write_bytes(b"fixture model payload")
    if backend in {"transformer", "ensemble"}:
        _write_transformer_payload(artifact)
    dataset_manifest = config.project_root / "config" / "datasets.v1.json"
    dataset_manifest.write_text('{"schema_version": 1}\n', encoding="utf-8")
    dependency_lock = config.project_root / "requirements.lock"
    dependency_lock.write_text("fixture==1.0\n", encoding="utf-8")
    for name, content in (
        ("metrics.json", "{}"),
        ("generalization_report.json", "{}"),
        ("reliability_diagram.svg", "<svg></svg>"),
        ("grouped_cv_summary.csv", "candidate,mean_macro_f1\nfixture,0.9\n"),
    ):
        (report / name).write_text(content, encoding="utf-8")
    report_hashes = {
        name: file_sha256(report / name)
        for name in (
            "metrics.json",
            "generalization_report.json",
            "reliability_diagram.svg",
            "grouped_cv_summary.csv",
        )
    }
    external_hash = "a" * 64
    manifest = {
        "schema_version": 2,
        "model_version": run_id,
        "backend": backend,
        "threshold": 0.5,
        "external_evaluation": {
            "dataset": "SpaPhish v5",
            "dataset_sha256": external_hash,
            "used_for_selection": False,
        },
        "report_hashes": report_hashes,
        "dataset_manifest": {"sha256": file_sha256(dataset_manifest)},
        "dependency_lock": {"sha256": file_sha256(dependency_lock)},
        "test_metrics": {"macro_f1": 0.9},
        "monitoring_baseline": {
            "score_bin_edges": [index / 10 for index in range(11)],
            "score_bin_fractions": [0.1] * 10,
            "predicted_phishing_fraction": 0.5,
            "slice_fractions": {},
            "alert_thresholds": {
                "score_population_stability_index": 0.2,
                "predicted_prevalence_absolute_change": 0.1,
                "slice_fraction_absolute_change": 0.15,
                "performance_macro_f1_drop": 0.05,
            },
        },
    }
    if backend in {"classical", "ensemble"}:
        manifest["model_sha256"] = file_sha256(artifact / "model.joblib")
    if backend in {"transformer", "ensemble"}:
        manifest["transformer_files"] = build_transformer_file_manifest(artifact)
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (report / "external_evaluation.lock.json").write_text(
        json.dumps(
            {
                "artifact": run_id,
                "report_sha256": report_hashes["generalization_report.json"],
                "external_sha256": external_hash,
            }
        ),
        encoding="utf-8",
    )
    return artifact


def test_monitoring_packaging_promotion_and_rollback(
    config_factory: Callable[..., ProjectConfig],
) -> None:
    config = config_factory()
    first = _write_release_artifact(config, "run-one")
    second = _write_release_artifact(config, "run-two")
    for name in (
        "MODEL_CARD.md",
        "DATASET_NOTICES.md",
        "MODEL_LICENSE.md",
        "SECURITY.md",
    ):
        (config.project_root / name).write_text(f"# {name}\n", encoding="utf-8")

    batch = config.project_root / "batch.jsonl"
    batch.write_text(
        "\n".join(
            json.dumps(
                {
                    "timestamp": f"2026-01-{index + 1:02d}T00:00:00Z",
                    "probability": 0.95,
                    "predicted_label": "phishing",
                    "slices": {"source": "gateway"},
                }
            )
            for index in range(5)
        ),
        encoding="utf-8",
    )
    promote_artifact(config, artifact=first)
    monitoring = monitor_batch(config, batch, artifact=first)
    assert monitoring["content_retained"] is False
    assert monitoring["status"] == "alert"

    promote_artifact(config, artifact=second)
    rolled_back = promote_artifact(config, rollback=True)
    assert rolled_back["artifact"] == "run-one"

    packaged = package_release(
        config,
        artifact=first,
        output_dir="release-test",
    )
    archive = Path(packaged["archive"])
    assert archive.is_file()
    assert file_sha256(archive) == packaged["sha256"]
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        assert any(name.endswith("/model.joblib") for name in names)
        assert any(name.endswith("/SHA256SUMS") for name in names)
        assert not any(name.endswith(".eml") for name in names)


def test_promotion_binds_manifest_and_rollback_rejects_later_changes(
    config_factory: Callable[..., ProjectConfig],
) -> None:
    config = config_factory()
    first = _write_release_artifact(config, "run-one")
    _write_release_artifact(config, "run-two")

    promoted = promote_artifact(config, artifact=first)
    assert promoted["manifest_sha256"] == file_sha256(first / "manifest.json")
    promote_artifact(config, artifact="run-two")

    manifest = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    manifest["threshold"] = 0.01
    (first / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ReleaseValidationError, match="changed after it was promoted"):
        promote_artifact(config, rollback=True)
    latest = json.loads((config.artifacts_dir / "latest.json").read_text(encoding="utf-8"))
    assert latest["artifact"] == "run-two"


def _write_release_documents(config: ProjectConfig) -> None:
    for name in (
        "MODEL_CARD.md",
        "DATASET_NOTICES.md",
        "MODEL_LICENSE.md",
        "SECURITY.md",
    ):
        (config.project_root / name).write_text(f"# {name}\n", encoding="utf-8")


@pytest.mark.parametrize("backend", ["transformer", "ensemble"])
def test_transformer_release_packages_exactly_the_verified_file_set(
    config_factory: Callable[..., ProjectConfig],
    backend: str,
) -> None:
    config = config_factory()
    artifact = _write_release_artifact(config, f"run-{backend}", backend=backend)
    _write_release_documents(config)

    promote_artifact(config, artifact=artifact)
    packaged = package_release(config, artifact=artifact, output_dir="release-test")

    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(packaged["archive"]) as bundle:
        names = {name.split("/", 1)[1] for name in bundle.namelist()}
    assert set(manifest["transformer_files"]) <= names
    assert ("model.joblib" in names) is (backend == "ensemble")
    assert not any(name.startswith("transformer/checkpoints") for name in names)


@pytest.mark.parametrize(
    "tamper",
    ["weights", "tokenizer", "config", "missing", "unrecorded", "top_level"],
)
def test_promotion_and_packaging_reject_tampered_transformer_payloads(
    config_factory: Callable[..., ProjectConfig],
    tamper: str,
) -> None:
    config = config_factory()
    artifact = _write_release_artifact(config, "run-transformer", backend="transformer")
    _write_release_documents(config)
    model_dir = artifact / "transformer" / "model"
    if tamper == "weights":
        (model_dir / "model.safetensors").write_bytes(b"replaced weights")
    elif tamper == "tokenizer":
        (model_dir / "tokenizer.json").write_bytes(b'{"model": {"type": "BPE"}}')
    elif tamper == "config":
        (model_dir / "config.json").write_bytes(b'{"auto_map": {"AutoModel": "x.Model"}}')
    elif tamper == "missing":
        (model_dir / "tokenizer_config.json").unlink()
    elif tamper == "unrecorded":
        checkpoints = artifact / "transformer" / "checkpoints"
        checkpoints.mkdir()
        (checkpoints / "optimizer.json").write_bytes(b"{}")
    else:
        (artifact / "notes.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(ReleaseValidationError):
        promote_artifact(config, artifact=artifact)
    with pytest.raises(ReleaseValidationError):
        package_release(config, artifact=artifact, output_dir="release-test")
    assert not (config.artifacts_dir / "latest.json").exists()


def test_promotion_rejects_raw_content_without_echoing_it(
    config_factory: Callable[..., ProjectConfig],
) -> None:
    config = config_factory()
    artifact = _write_release_artifact(config, "run-leaky")
    sentence = b"Please review the attached invoice and confirm your payment today"
    (artifact / "model.joblib").write_bytes(b"\x80\x05" + sentence)
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    manifest["model_sha256"] = file_sha256(artifact / "model.joblib")
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ReleaseValidationError, match="natural_language=1") as raised:
        promote_artifact(config, artifact=artifact)

    assert "invoice" not in str(raised.value)
    assert not (config.artifacts_dir / "latest.json").exists()
