"""Trusted artifact resolution and payload verification before deserialization.

Every model payload is a serialized program: joblib executes pickle opcodes while
loading. Nothing inside an artifact directory can vouch for itself, so trust is
anchored outside it, in records written by the training and promotion tooling
(``latest.json`` and ``promotion_history.json``), or in a SHA-256 supplied by the
operator. Manifest bytes are hashed before parsing, and payload bytes are hashed
before any deserializer sees them.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import math
import os
import re
import stat
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

import joblib

from .config import ProjectConfig

MAX_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_RECORD_BYTES = 16 * 1024 * 1024
MAX_CLASSICAL_MODEL_BYTES = 2 * 1024 * 1024 * 1024
SUPPORTED_BACKENDS = frozenset({"classical", "transformer", "ensemble"})
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class ArtifactError(RuntimeError):
    """Raised when a deployable artifact cannot be found, trusted, or loaded."""


class UnsafeFileError(OSError):
    """Raised when a path is a symlink or not a regular file."""


@dataclass(frozen=True)
class TrustedManifest:
    """A parsed manifest whose exact bytes matched a trusted record."""

    artifact_dir: Path
    manifest: dict[str, Any]
    sha256: str
    trust_source: str


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalize_sha256(value: object) -> str | None:
    """Return a lowercase SHA-256 hex digest, or None when the value is not one."""

    if not isinstance(value, str):
        return None
    candidate = value.strip().lower()
    return candidate if _SHA256_HEX.fullmatch(candidate) else None


def open_regular_file(path: Path) -> BinaryIO:
    """Open a regular file for reading without following symlinks or blocking on FIFOs."""

    if path.is_symlink():
        raise UnsafeFileError(f"Refusing to follow a symlink: {path}")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise UnsafeFileError(f"Not a regular file: {path}")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def read_regular_file(path: Path, max_bytes: int, description: str) -> bytes:
    """Read a bounded regular file, raising ArtifactError for anything unexpected."""

    try:
        with open_regular_file(path) as handle:
            payload = handle.read(max_bytes + 1)
    except FileNotFoundError as error:
        raise ArtifactError(f"{description} not found: {path}") from error
    except OSError as error:
        raise ArtifactError(f"{description} could not be read safely: {path}") from error
    if len(payload) > max_bytes:
        raise ArtifactError(f"{description} exceeds the supported size: {path}")
    return payload


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Durably replace one file so readers see either the old or the new bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(payload))


def resolve_artifact_dir(
    config: ProjectConfig,
    artifact: str | Path | None = None,
) -> Path:
    """Resolve an explicit artifact or the atomically published latest artifact."""

    if artifact is not None:
        supplied = Path(artifact)
        path = (
            supplied.resolve()
            if supplied.is_absolute()
            else (config.project_root / supplied).resolve()
        )
        if path.name == "manifest.json":
            path = path.parent
        if (
            not (path / "manifest.json").is_file()
            and not supplied.is_absolute()
            and len(supplied.parts) == 1
            and supplied.name not in {"", ".", ".."}
        ):
            path = (config.artifacts_dir / supplied.name).resolve()
    else:
        latest = _read_json_record(config.artifacts_dir / "latest.json")
        if latest is None:
            raise ArtifactError(f"No published model found at {config.artifacts_dir}")
        if not isinstance(latest, dict):
            raise ArtifactError("latest.json must contain a JSON object")
        relative = Path(str(latest.get("artifact", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ArtifactError("latest.json contains an unsafe artifact path")
        path = (config.artifacts_dir / relative).resolve()
        if config.artifacts_dir.resolve() not in path.parents:
            raise ArtifactError("Published artifact resolves outside the artifact directory")

    if not (path / "manifest.json").is_file():
        raise ArtifactError(f"Model manifest not found in {path}")
    return path


def _read_json_record(path: Path) -> Any:
    if not path.exists():
        return None
    payload = read_regular_file(path, MAX_RECORD_BYTES, path.name)
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactError(f"{path.name} is not valid JSON") from error


def registry_name(config: ProjectConfig, artifact_dir: Path) -> str | None:
    """Return the artifacts-root-relative name used by trust records, if any."""

    root = config.artifacts_dir.resolve()
    resolved = artifact_dir.resolve()
    if root not in resolved.parents:
        return None
    return resolved.relative_to(root).as_posix()


def promotion_history(config: ProjectConfig) -> list[dict[str, Any]]:
    history = _read_json_record(config.artifacts_dir / "promotion_history.json")
    if history is None:
        return []
    if not isinstance(history, list) or not all(isinstance(entry, dict) for entry in history):
        raise ArtifactError("promotion_history.json must contain a list of objects")
    return history


def trusted_manifest_record(
    config: ProjectConfig,
    artifact_dir: Path,
) -> tuple[str, str] | None:
    """Find the manifest SHA-256 recorded for an artifact outside its own directory.

    ``latest.json`` is authoritative for the artifact it points at. Older pointers
    that predate the ``manifest_sha256`` field fall back to the most recent
    promotion of the same artifact in ``promotion_history.json``.
    """

    name = registry_name(config, artifact_dir)
    if name is None:
        return None
    latest = _read_json_record(config.artifacts_dir / "latest.json")
    if isinstance(latest, dict) and latest.get("artifact") == name:
        digest = normalize_sha256(latest.get("manifest_sha256"))
        if digest:
            return digest, "latest.json"
    for entry in reversed(promotion_history(config)):
        if entry.get("artifact") == name:
            digest = normalize_sha256(entry.get("manifest_sha256"))
            if digest:
                return digest, "promotion_history.json"
    return None


def validate_manifest_fields(manifest: Any) -> None:
    """Reject manifests that the predictor cannot interpret safely."""

    if not isinstance(manifest, dict):
        raise ArtifactError("Model manifest must be a JSON object")
    if manifest.get("schema_version") not in {1, 2}:
        raise ArtifactError("Unsupported model manifest schema")
    if manifest.get("backend") not in SUPPORTED_BACKENDS:
        raise ArtifactError(f"Unsupported model backend: {manifest.get('backend')!r}")
    if not isinstance(manifest.get("model_version"), str) or not manifest["model_version"]:
        raise ArtifactError("Model manifest is missing its model version")
    threshold = manifest.get("threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not math.isfinite(threshold)
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise ArtifactError("Model manifest threshold must be a probability")


def load_trusted_manifest(
    config: ProjectConfig,
    artifact: str | Path | None = None,
    *,
    expected_manifest_sha256: str | None = None,
) -> TrustedManifest:
    """Resolve an artifact and parse its manifest only after its bytes are trusted.

    Explicitly supplied directories never vouch for themselves: they must be bound
    by a trust record in the artifacts root, or the operator must supply the
    expected manifest SHA-256 obtained out of band (for example from release
    checksums).
    """

    artifact_dir = resolve_artifact_dir(config, artifact)
    if expected_manifest_sha256 is not None:
        expected = normalize_sha256(expected_manifest_sha256)
        if expected is None:
            raise ArtifactError("Expected manifest SHA-256 must be 64 hexadecimal characters")
        trust_source = "operator"
    else:
        record = trusted_manifest_record(config, artifact_dir)
        if record is None:
            raise ArtifactError(
                "Artifact is not bound to a trusted record; promote it or supply its "
                "expected manifest SHA-256"
            )
        expected, trust_source = record

    payload = read_regular_file(
        artifact_dir / "manifest.json",
        MAX_MANIFEST_BYTES,
        "Model manifest",
    )
    if not hmac.compare_digest(sha256_bytes(payload), expected):
        raise ArtifactError(f"Model manifest does not match its trusted {trust_source} record")
    try:
        manifest = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactError("Model manifest is not valid JSON") from error
    validate_manifest_fields(manifest)
    return TrustedManifest(
        artifact_dir=artifact_dir,
        manifest=manifest,
        sha256=expected,
        trust_source=trust_source,
    )


def load_classical_model(artifact_dir: Path, manifest: dict[str, Any]) -> Any:
    """Verify the joblib payload bytes, then deserialize exactly those bytes."""

    expected = normalize_sha256(manifest.get("model_sha256"))
    if expected is None:
        raise ArtifactError("Model manifest does not record a classical model checksum")
    payload = read_regular_file(
        artifact_dir / "model.joblib",
        MAX_CLASSICAL_MODEL_BYTES,
        "Classical model payload",
    )
    if not hmac.compare_digest(sha256_bytes(payload), expected):
        raise ArtifactError("Classical model checksum does not match its manifest")
    try:
        # Unpickling is code execution. It is reached only for these exact bytes,
        # after they matched a checksum bound to an externally trusted manifest.
        return joblib.load(io.BytesIO(payload))
    except Exception as error:
        raise ArtifactError("Classical model payload could not be deserialized") from error


def bind_latest(
    config: ProjectConfig,
    artifact_dir: Path,
    *,
    action: str,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Atomically point latest.json at an artifact and bind its manifest bytes."""

    name = registry_name(config, artifact_dir)
    if name is None:
        raise ArtifactError("Only artifacts inside the artifacts directory can be published")
    digest = normalize_sha256(manifest_sha256) if manifest_sha256 else None
    if digest is None:
        digest = sha256_bytes(
            read_regular_file(artifact_dir / "manifest.json", MAX_MANIFEST_BYTES, "Model manifest")
        )
    latest = {
        "action": action,
        "artifact": name,
        "manifest_sha256": digest,
        "updated_at": datetime.now(UTC).isoformat(),
    }
    atomic_write_json(config.artifacts_dir / "latest.json", latest)
    return latest
