"""Versioned, hash-verified dataset acquisition."""

from __future__ import annotations

import json
import os
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from .datasets import file_sha256

DEFAULT_PROVENANCE_PATH = Path("config/datasets.v1.json")


class ProvenanceError(RuntimeError):
    """Raised when a source does not match the versioned provenance manifest."""


def load_provenance(path: str | Path = DEFAULT_PROVENANCE_PATH) -> dict[str, Any]:
    """Load and minimally validate the provenance manifest."""

    manifest_path = Path(path).resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != 1:
        raise ProvenanceError("Unsupported dataset provenance schema")
    if not manifest.get("policy", {}).get("license_acceptance_required"):
        raise ProvenanceError("Provenance manifest must require explicit license acceptance")
    return manifest


def _download(url: str, destination: Path, *, referer: str | None = None) -> None:
    headers = {"User-Agent": "Mozilla/5.0 (compatible; dataset-provenance-fetch/1.0)"}
    if referer:
        headers["Referer"] = referer
    request = urllib.request.Request(
        url,
        headers=headers,
    )
    with (
        urllib.request.urlopen(request, timeout=120) as response,
        destination.open("wb") as output,
    ):
        while chunk := response.read(1024 * 1024):
            output.write(chunk)


def _verify(path: Path, expected_hash: str, expected_bytes: int) -> None:
    size = path.stat().st_size
    if size != expected_bytes:
        raise ProvenanceError(
            f"Size mismatch for {path.name}: expected {expected_bytes}, found {size}"
        )
    actual_hash = file_sha256(path)
    if actual_hash != expected_hash:
        raise ProvenanceError(
            f"SHA-256 mismatch for {path.name}: expected {expected_hash}, found {actual_hash}"
        )


def _safe_destination(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ProvenanceError(f"Unsafe dataset destination: {relative}")
    destination = (root / relative_path).resolve()
    if destination != root and root not in destination.parents:
        raise ProvenanceError(f"Dataset destination escapes project root: {relative}")
    return destination


def _install_bytes(
    payload: bytes,
    destination: Path,
    *,
    expected_hash: str,
    expected_bytes: int,
) -> str:
    if destination.exists():
        try:
            _verify(destination, expected_hash, expected_bytes)
        except ProvenanceError as error:
            raise ProvenanceError(
                f"Refusing to overwrite mismatched dataset {destination}: {error}"
            ) from error
        return "verified_existing"

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        _verify(temporary, expected_hash, expected_bytes)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "installed"


def fetch_data(
    project_root: Path,
    *,
    accept_licenses: bool,
    include_external: bool = False,
    manifest_path: str | Path = DEFAULT_PROVENANCE_PATH,
) -> list[dict[str, str]]:
    """Fetch exact dataset versions and install only hash-matching files."""

    if not accept_licenses:
        raise ProvenanceError(
            "Dataset downloads require --accept-licenses after reviewing docs/DATASETS.md"
        )
    root = project_root.resolve()
    manifest = load_provenance(
        manifest_path
        if Path(manifest_path).is_absolute()
        else root / Path(manifest_path)
    )
    cache_dir = root / ".cache" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    outcomes: list[dict[str, str]] = []

    for dataset in manifest["datasets"]:
        archive_path = cache_dir / f"{dataset['id']}.zip"
        if archive_path.exists():
            try:
                _verify(
                    archive_path,
                    str(dataset["archive_sha256"]),
                    int(dataset["archive_bytes"]),
                )
            except ProvenanceError as error:
                raise ProvenanceError(
                    f"Refusing mismatched cached archive {archive_path}: {error}"
                ) from error
        else:
            temporary = archive_path.with_suffix(".zip.download")
            temporary.unlink(missing_ok=True)
            _download(str(dataset["download_url"]), temporary)
            _verify(
                temporary,
                str(dataset["archive_sha256"]),
                int(dataset["archive_bytes"]),
            )
            os.replace(temporary, archive_path)

        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            for member in dataset["members"]:
                archive_member = str(member["archive_path"])
                if archive_member not in names:
                    raise ProvenanceError(
                        f"Archive {dataset['id']} is missing {archive_member}"
                    )
                source = archive.read(archive_member)
                if "source_sha256" in member:
                    if len(source) != int(member["source_bytes"]):
                        raise ProvenanceError(f"Source size mismatch for {archive_member}")
                    import hashlib

                    if hashlib.sha256(source).hexdigest() != member["source_sha256"]:
                        raise ProvenanceError(f"Source SHA-256 mismatch for {archive_member}")
                if member["normalization"] == "crlf_to_lf":
                    source = source.replace(b"\r\n", b"\n")
                elif member["normalization"] != "none":
                    raise ProvenanceError(
                        f"Unsupported normalization: {member['normalization']}"
                    )
                destination = _safe_destination(root, str(member["destination"]))
                status = _install_bytes(
                    source,
                    destination,
                    expected_hash=str(member["sha256"]),
                    expected_bytes=int(member["bytes"]),
                )
                outcomes.append({"dataset": str(dataset["id"]), "status": status})

    if include_external:
        for dataset in manifest["external"]:
            destination = _safe_destination(root, str(dataset["destination"]))
            if destination.exists():
                _verify(
                    destination,
                    str(dataset["sha256"]),
                    int(dataset["bytes"]),
                )
                status = "verified_existing"
            else:
                with tempfile.NamedTemporaryFile(
                    dir=cache_dir,
                    prefix=f".{dataset['id']}.",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                try:
                    _download(
                        str(dataset["download_url"]),
                        temporary,
                        referer=str(dataset["source_url"]),
                    )
                    _verify(
                        temporary,
                        str(dataset["sha256"]),
                        int(dataset["bytes"]),
                    )
                    status = _install_bytes(
                        temporary.read_bytes(),
                        destination,
                        expected_hash=str(dataset["sha256"]),
                        expected_bytes=int(dataset["bytes"]),
                    )
                finally:
                    temporary.unlink(missing_ok=True)
            outcomes.append({"dataset": str(dataset["id"]), "status": status})
    return outcomes
