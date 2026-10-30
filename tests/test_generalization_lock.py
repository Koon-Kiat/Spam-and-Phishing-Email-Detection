from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from spamandphishingdetection import generalization, integrity
from spamandphishingdetection.config import ProjectConfig
from spamandphishingdetection.generalization import (
    EvaluationLockedError,
    generate_generalization_report,
)
from spamandphishingdetection.integrity import ArtifactError, resolve_artifact_dir
from spamandphishingdetection.release import (
    ReleaseValidationError,
    package_release,
    promote_artifact,
)


def _evidence_paths(config: ProjectConfig) -> dict[str, Path]:
    artifact_dir = resolve_artifact_dir(config)
    report_dir = config.reports_dir / artifact_dir.name
    return {
        "report": report_dir / "generalization_report.json",
        "diagram": report_dir / "reliability_diagram.svg",
        "lock": report_dir / "external_evaluation.lock.json",
        "manifest": artifact_dir / "manifest.json",
        "latest": config.artifacts_dir / "latest.json",
    }


def _snapshot(paths: dict[str, Path]) -> dict[str, bytes | None]:
    return {name: path.read_bytes() if path.exists() else None for name, path in paths.items()}


def _leftover_transaction_files(config: ProjectConfig) -> list[Path]:
    return [
        path
        for root in (config.artifacts_dir, config.reports_dir)
        for path in root.rglob(".*")
        if path.suffix in {".tmp", ".bak"}
    ]


def test_skip_external_after_lock_leaves_locked_evidence_untouched(
    trained_project: ProjectConfig,
) -> None:
    config = trained_project
    paths = _evidence_paths(config)
    generate_generalization_report(config)
    locked = _snapshot(paths)
    assert all(value is not None for value in locked.values())

    internal = generate_generalization_report(config, include_external=False)

    assert internal.name == "internal_generalization_report.json"
    assert (internal.parent / "internal_reliability_diagram.svg").is_file()
    assert json.loads(internal.read_text(encoding="utf-8"))["evidence_scope"] == (
        "internal_diagnostic"
    )
    assert _snapshot(paths) == locked

    promoted = promote_artifact(config, artifact=resolve_artifact_dir(config).name)
    assert promoted["action"] == "promote"
    assert Path(package_release(config, artifact=promoted["artifact"])["archive"]).is_file()


def test_second_external_evaluation_is_refused(trained_project: ProjectConfig) -> None:
    config = trained_project
    paths = _evidence_paths(config)
    generate_generalization_report(config)
    locked = _snapshot(paths)
    lock = json.loads(paths["lock"].read_text(encoding="utf-8"))
    assert lock["artifact_manifest_sha256"] == integrity.sha256_bytes(locked["manifest"])
    assert lock["reliability_diagram_sha256"] == integrity.sha256_bytes(locked["diagram"])

    with pytest.raises(EvaluationLockedError, match="train a new artifact"):
        generate_generalization_report(config)

    assert _snapshot(paths) == locked


@pytest.mark.parametrize(
    "failing_target",
    [
        "generalization_report.json",
        "reliability_diagram.svg",
        "manifest.json",
        "latest.json",
        "external_evaluation.lock.json",
    ],
)
def test_failure_at_any_commit_stage_restores_the_prior_state(
    trained_project: ProjectConfig,
    monkeypatch: pytest.MonkeyPatch,
    failing_target: str,
) -> None:
    config = trained_project
    paths = _evidence_paths(config)
    before = _snapshot(paths)
    real_replace = os.replace

    def failing_replace(source: str | os.PathLike, target: str | os.PathLike) -> None:
        if Path(source).name.endswith(".tmp") and Path(target).name == failing_target:
            raise OSError("simulated write failure")
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated write failure"):
        generate_generalization_report(config)
    monkeypatch.setattr(os, "replace", real_replace)

    assert _snapshot(paths) == before
    assert _leftover_transaction_files(config) == []
    generate_generalization_report(config)
    assert paths["lock"].is_file()


def test_checksum_mismatch_while_staging_writes_nothing(
    trained_project: ProjectConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = trained_project
    paths = _evidence_paths(config)
    before = _snapshot(paths)
    real_hash = integrity.hash_regular_file

    def corrupted_hash(path: Path) -> tuple[str, int]:
        digest, size = real_hash(path)
        return ("0" * 64, size) if "reliability_diagram" in path.name else (digest, size)

    monkeypatch.setattr(integrity, "hash_regular_file", corrupted_hash)
    with pytest.raises(ArtifactError, match="failed checksum verification"):
        generate_generalization_report(config)

    assert _snapshot(paths) == before
    assert _leftover_transaction_files(config) == []


def test_concurrent_manifest_change_aborts_without_writing(
    trained_project: ProjectConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = trained_project
    paths = _evidence_paths(config)
    real_loader = generalization.load_spaphish_v5

    def loader_with_concurrent_edit(*args: object, **kwargs: object):
        paths["manifest"].write_bytes(paths["manifest"].read_bytes() + b"\n")
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(generalization, "load_spaphish_v5", loader_with_concurrent_edit)
    with pytest.raises(RuntimeError, match="changed during evaluation"):
        generate_generalization_report(config)

    assert not paths["report"].exists()
    assert not paths["lock"].exists()


def test_editing_locked_evidence_blocks_promotion(trained_project: ProjectConfig) -> None:
    config = trained_project
    paths = _evidence_paths(config)
    generate_generalization_report(config)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["threshold"] = 0.01
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ReleaseValidationError, match="changed after the external evaluation"):
        promote_artifact(config, artifact=resolve_artifact_dir(config).name)
