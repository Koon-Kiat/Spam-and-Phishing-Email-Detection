"""Release-time privacy checks for serialized model artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_PRINTABLE_RUN = re.compile(rb"[\x20-\x7e]{8,}")
_FULL_EMAIL = re.compile(
    rb"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]{2,}@[A-Za-z0-9.-]+\.[A-Za-z]{2,63}"
)
_LONG_NUMBER = re.compile(rb"(?<!\d)\d{9,}(?!\d)")


class PrivacyScanError(RuntimeError):
    """Raised when a release artifact appears to contain sensitive identifiers."""


def scan_model_artifact(artifact_dir: Path) -> dict[str, Any]:
    """Scan model payloads for full identifiers and suspicious raw text runs."""

    candidate_files = [
        path
        for path in artifact_dir.rglob("*")
        if (
            path.is_file()
            and path.name != "manifest.json"
            and not any(
                part.startswith(".")
                for part in path.relative_to(artifact_dir).parts
            )
        )
    ]
    email_findings: list[str] = []
    number_findings: list[str] = []
    longest_printable_run = 0
    scanned_bytes = 0
    for path in candidate_files:
        payload = path.read_bytes()
        scanned_bytes += len(payload)
        runs = _PRINTABLE_RUN.findall(payload)
        longest_printable_run = max(
            longest_printable_run,
            max((len(run) for run in runs), default=0),
        )
        for match in _FULL_EMAIL.finditer(payload):
            email_findings.append(match.group(0).decode("ascii", errors="replace"))
        for match in _LONG_NUMBER.finditer(payload):
            number_findings.append(match.group(0).decode("ascii", errors="replace"))
    result: dict[str, Any] = {
        "files_scanned": len(candidate_files),
        "bytes_scanned": scanned_bytes,
        "full_email_findings": len(set(email_findings)),
        "long_numeric_identifier_findings": len(set(number_findings)),
        "longest_printable_run": longest_printable_run,
        "status": "passed",
    }
    if email_findings or number_findings:
        result["status"] = "failed"
        raise PrivacyScanError(
            "Serialized model contains full email or long numeric identifier candidates"
        )
    return result
