"""Validated artifact promotion, rollback, and public release packaging."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .datasets import file_sha256
from .privacy import scan_model_artifact


class ReleaseValidationError(RuntimeError):
    """Raised when an artifact cannot safely be promoted or packaged."""


def _artifact_path(config: ProjectConfig, artifact: str | Path) -> Path:
    value = Path(artifact)
    if value.is_absolute():
        path = value.resolve()
    elif len(value.parts) == 1:
        path = (config.artifacts_dir / value).resolve()
    else:
        path = (config.project_root / value).resolve()
    if config.artifacts_dir.resolve() not in path.parents:
        raise ReleaseValidationError("Artifact must be inside the configured artifacts directory")
    return path


def validate_artifact(config: ProjectConfig, artifact: str | Path) -> dict[str, Any]:
    """Validate schema, payload checksums, reports, and privacy before promotion."""

    path = _artifact_path(config, artifact)
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise ReleaseValidationError(f"Model manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ReleaseValidationError("Public promotion requires manifest schema version 2")
    if manifest.get("model_version") != path.name:
        raise ReleaseValidationError("Model version does not match artifact directory")
    if manifest.get("backend") in {"classical", "ensemble"}:
        model_path = path / "model.joblib"
        if not model_path.is_file():
            raise ReleaseValidationError("Classical model payload is missing")
        if file_sha256(model_path) != manifest.get("model_sha256"):
            raise ReleaseValidationError("Classical model checksum does not match manifest")
    scan = scan_model_artifact(path)
    if scan["status"] != "passed":
        raise ReleaseValidationError("Artifact privacy scan did not pass")

    report_dir = config.reports_dir / path.name
    required_reports = (
        "metrics.json",
        "generalization_report.json",
        "reliability_diagram.svg",
    )
    report_hashes = manifest.get("report_hashes")
    if not isinstance(report_hashes, dict):
        raise ReleaseValidationError("Manifest report hashes are missing")
    for filename, expected_hash in report_hashes.items():
        relative = Path(filename)
        if relative.is_absolute() or len(relative.parts) != 1 or filename != relative.name:
            raise ReleaseValidationError(f"Unsafe report hash path: {filename}")
        report_path = report_dir / relative
        if not report_path.is_file():
            raise ReleaseValidationError(f"Hashed report is missing: {filename}")
        if file_sha256(report_path) != expected_hash:
            raise ReleaseValidationError(f"Report checksum does not match: {filename}")
    for required in required_reports:
        if not (report_dir / required).is_file():
            raise ReleaseValidationError(f"Required release report is missing: {required}")
        if required not in report_hashes:
            raise ReleaseValidationError(f"Required report hash is missing: {required}")
    if not manifest.get("external_evaluation"):
        raise ReleaseValidationError("Blind external evaluation is not recorded")
    lock_path = report_dir / "external_evaluation.lock.json"
    if not lock_path.is_file():
        raise ReleaseValidationError("Blind external evaluation lock is missing")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    external = manifest["external_evaluation"]
    if (
        lock.get("artifact") != path.name
        or lock.get("report_sha256") != report_hashes["generalization_report.json"]
        or lock.get("external_sha256") != external.get("dataset_sha256")
        or external.get("used_for_selection") is not False
    ):
        raise ReleaseValidationError("External evaluation lock does not match the manifest")
    for manifest_key, repository_path in (
        ("dataset_manifest", config.project_root / "config" / "datasets.v1.json"),
        ("dependency_lock", config.project_root / "requirements.lock"),
    ):
        record = manifest.get(manifest_key, {})
        if not repository_path.is_file() or file_sha256(repository_path) != record.get("sha256"):
            raise ReleaseValidationError(f"{manifest_key} checksum does not match")
    return {"artifact_dir": path, "manifest": manifest, "privacy_scan": scan}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def promote_artifact(
    config: ProjectConfig,
    *,
    artifact: str | Path | None = None,
    rollback: bool = False,
) -> dict[str, Any]:
    """Atomically promote a validated artifact or roll back to the prior promotion."""

    history_path = config.artifacts_dir / "promotion_history.json"
    history = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if history_path.is_file()
        else []
    )
    if rollback:
        if len(history) < 2:
            raise ReleaseValidationError("No prior promoted artifact is available for rollback")
        target = history[-2]["artifact"]
        action = "rollback"
    else:
        if artifact is None:
            raise ReleaseValidationError("--artifact is required unless --rollback is used")
        target = str(artifact)
        action = "promote"
    validated = validate_artifact(config, target)
    artifact_dir = validated["artifact_dir"]
    latest = {
        "artifact": artifact_dir.relative_to(config.artifacts_dir).as_posix(),
        "updated_at": datetime.now(UTC).isoformat(),
        "action": action,
    }
    _atomic_json(config.artifacts_dir / "latest.json", latest)
    history.append({**latest, "manifest_sha256": file_sha256(artifact_dir / "manifest.json")})
    _atomic_json(history_path, history)
    return latest


def package_release(
    config: ProjectConfig,
    *,
    artifact: str | Path,
    version: str = "v1.0.0",
    output_dir: str | Path = "release",
) -> dict[str, str]:
    """Build and verify a model-only release archive and checksums."""

    validated = validate_artifact(config, artifact)
    artifact_dir: Path = validated["artifact_dir"]
    report_dir = config.reports_dir / artifact_dir.name
    destination = Path(output_dir)
    if not destination.is_absolute():
        destination = config.project_root / destination
    destination = destination.resolve()
    if destination != config.project_root and config.project_root not in destination.parents:
        raise ReleaseValidationError("Release output must stay inside the project")
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / f"spam-phishing-model-{version}.zip"

    with tempfile.TemporaryDirectory(dir=destination) as temporary_name:
        stage = Path(temporary_name) / f"spam-phishing-model-{version}"
        stage.mkdir()
        for filename in ("manifest.json", "model.joblib"):
            source = artifact_dir / filename
            if source.is_file():
                shutil.copy2(source, stage / filename)
        if (artifact_dir / "transformer").is_dir():
            shutil.copytree(artifact_dir / "transformer", stage / "transformer")
        for filename in (
            "metrics.json",
            "generalization_report.json",
            "reliability_diagram.svg",
            "grouped_cv_summary.csv",
        ):
            shutil.copy2(report_dir / filename, stage / filename)
        documentation = {
            "MODEL_CARD.md": config.project_root / "MODEL_CARD.md",
            "DATASET_NOTICES.md": config.project_root / "DATASET_NOTICES.md",
            "MODEL_LICENSE.md": config.project_root / "MODEL_LICENSE.md",
            "SECURITY.md": config.project_root / "SECURITY.md",
        }
        for target_name, source in documentation.items():
            if not source.is_file():
                raise ReleaseValidationError(f"Required release document is missing: {target_name}")
            shutil.copy2(source, stage / target_name)
        checksums = []
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                checksums.append(
                    f"{file_sha256(path)}  {path.relative_to(stage).as_posix()}"
                )
        (stage / "SHA256SUMS").write_text(
            "\n".join(checksums) + "\n",
            encoding="utf-8",
        )
        temporary_archive = archive_path.with_suffix(".zip.tmp")
        temporary_archive.unlink(missing_ok=True)
        with zipfile.ZipFile(
            temporary_archive,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage.parent).as_posix())
        os.replace(temporary_archive, archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        bad_file = archive.testzip()
        if bad_file:
            raise ReleaseValidationError(f"Release archive checksum failed for {bad_file}")
        names = set(archive.namelist())
        if any(name.lower().endswith((".csv", ".eml")) for name in names):
            allowed = {"grouped_cv_summary.csv"}
            unexpected = [
                name for name in names if Path(name).name not in allowed and name.endswith(".csv")
            ]
            if unexpected or any(name.lower().endswith(".eml") for name in names):
                raise ReleaseValidationError("Raw CSV or EML found in release archive")
        checksum_members = [
            name for name in names if Path(name).name == "SHA256SUMS"
        ]
        if len(checksum_members) != 1:
            raise ReleaseValidationError("Release archive must contain one internal SHA256SUMS")
        checksum_member = checksum_members[0]
        archive_root = Path(checksum_member).parent
        checksum_lines = archive.read(checksum_member).decode("utf-8").splitlines()
        checked_members: set[str] = set()
        for line in checksum_lines:
            expected_hash, separator, relative_name = line.partition("  ")
            if (
                not separator
                or len(expected_hash) != 64
                or any(character not in "0123456789abcdef" for character in expected_hash)
            ):
                raise ReleaseValidationError("Malformed internal SHA256SUMS entry")
            relative = Path(relative_name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ReleaseValidationError("Unsafe path in internal SHA256SUMS")
            member = (archive_root / relative).as_posix()
            if member not in names:
                raise ReleaseValidationError(f"Checksummed member is missing: {relative_name}")
            actual_hash = hashlib.sha256(archive.read(member)).hexdigest()
            if actual_hash != expected_hash:
                raise ReleaseValidationError(
                    f"Internal release checksum failed for {relative_name}"
                )
            checked_members.add(member)
        payload_members = {
            name
            for name in names
            if not name.endswith("/") and name != checksum_member
        }
        if checked_members != payload_members:
            raise ReleaseValidationError("Internal SHA256SUMS does not cover every payload file")
    archive_hash = file_sha256(archive_path)
    checksum_path = destination / "SHA256SUMS"
    checksum_path.write_text(f"{archive_hash}  {archive_path.name}\n", encoding="utf-8")
    if file_sha256(archive_path) != checksum_path.read_text(encoding="utf-8").split()[0]:
        raise ReleaseValidationError("Final release archive checksum verification failed")
    return {
        "archive": str(archive_path),
        "sha256": archive_hash,
        "checksums": str(checksum_path),
    }
