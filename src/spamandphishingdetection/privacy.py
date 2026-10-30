"""Release-time privacy checks for serialized model artifacts.

The scan looks for raw message content that training could have memorized into a
payload: identifiers (email addresses, URLs, phone, card, IBAN, and long numbers),
credential assignments, street addresses, and runs of natural-language prose. It
streams every file in bounded chunks, with overlap so that matches spanning a chunk
boundary are still found, and never records the matched bytes themselves.

Format-aware handling keeps legitimate model metadata from tripping the rules:

- JSON files are parsed and only their keys and string values are scanned, so large
  numeric settings such as ``model_max_length`` are not mistaken for identifiers;
- safetensors headers are parsed and only tensor names and metadata strings are
  scanned, and the tensor data is scanned as binary;
- every other file is scanned as binary, where a printable ASCII run longer than
  ``MAX_BINARY_PRINTABLE_RUN`` also fails, because pickled vocabularies consist of
  short separated strings while embedded documents do not.

The scan cannot prove an artifact is free of personal data. Isolated tokens such as a
first name inside a TF-IDF vocabulary look the same as ordinary vocabulary terms.
"""

from __future__ import annotations

import json
import os
import re
import stat
import struct
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, BinaryIO

from .integrity import UnsafeFileError, open_regular_file

CHUNK_BYTES = 1024 * 1024
MAX_BINARY_PRINTABLE_RUN = 256
NATURAL_LANGUAGE_MIN_WORDS = 6
MAX_FILE_BYTES = 8 * 1024 * 1024 * 1024
MAX_STRUCTURED_BYTES = 64 * 1024 * 1024
MAX_RECORDED_FINDINGS = 50
_OVERLAP_BYTES = 512
_CONTEXT_BYTES = 64
_JSON_SUFFIXES = frozenset({".json"})
_PRINTABLE_BYTES = bytes(range(0x20, 0x7F))
_LONG_PRINTABLE_RUN = re.compile(rb"[\x20-\x7e]{8,}")
# Every pattern below matches only text bytes (tab through carriage return, and
# printable ASCII) and needs at least seven of them. Patterns therefore run only
# inside text runs of that length. Run boundaries are non-text bytes, which satisfy
# every lookaround exactly as they would in a whole-buffer search.
_TEXT_RUN = re.compile(rb"[\t-\r\x20-\x7e]{7,}")

# Every pattern is length-bounded below _OVERLAP_BYTES, and every lookbehind is
# shorter than _CONTEXT_BYTES, so chunked matching equals whole-file matching.
_PATTERNS: dict[str, re.Pattern[bytes]] = {
    "email_address": re.compile(
        rb"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]{2,64}@[A-Za-z0-9.-]{1,190}\.[A-Za-z]{2,24}"
    ),
    "url": re.compile(
        rb"(?i)(?<![a-z0-9])(?:(?:https?|ftp)://|www\.)[a-z0-9-]{1,63}(?:\.[a-z0-9-]{1,63}){1,4}"
    ),
    "phone_number": re.compile(
        rb"(?<![0-9A-Za-z+])(?:\+\d{1,3}[ .-]?)?(?:\(\d{3}\)[ .-]?|\d{3}[ .-])"
        rb"\d{3}[ .-]\d{4}(?![0-9A-Za-z])"
    ),
    "payment_card": re.compile(rb"(?<!\d)\d{4}(?:[ -]\d{4}){3}(?!\d)"),
    "long_numeric_identifier": re.compile(rb"(?<!\d)\d{9}"),
    "iban": re.compile(rb"(?<![A-Z0-9])[A-Z]{2}\d{2}[A-Z0-9]{11,30}(?![A-Z0-9])"),
    "credential_assignment": re.compile(
        rb"(?i)(?<![a-z])(?:(?:password|passwd|passcode|api[_-]?key|secret[_-]?key|"
        rb"access[_-]?token|auth[_-]?token|client[_-]?secret)[\"']?\s{0,3}[:=]\s{0,3}"
        rb"[\"']?(?:(?![\"',;])[\x21-\x7e]){4}|bearer\s{1,3}[a-z0-9._~+/-]{20})"
    ),
    "street_address": re.compile(
        rb"(?i)(?<![a-z0-9])\d{1,5}\s(?:[a-z]{2,20}\s){1,3}(?:street|st|avenue|ave|road|"
        rb"rd|boulevard|blvd|lane|ln|drive|dr|court|ct|place|pl|terrace|way)\.?(?![a-z])"
    ),
    "natural_language": re.compile(
        rb"(?<![A-Za-z'-])(?:[A-Za-z][A-Za-z'-]{1,23}[ \t]{1,2}){"
        + str(NATURAL_LANGUAGE_MIN_WORDS - 1).encode()
        + rb"}[A-Za-z][A-Za-z'-]{1,23}(?![A-Za-z'-])"
    ),
}
_STRUCTURE_CATEGORIES = (
    "binary_printable_run",
    "link",
    "malformed_structured_file",
    "oversized_file",
    "special_file",
    "unreadable_file",
)
CATEGORIES = tuple(sorted((*_PATTERNS, *_STRUCTURE_CATEGORIES)))


class PrivacyScanError(RuntimeError):
    """Raised when a release artifact appears to contain sensitive content."""

    def __init__(self, message: str, result: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.result = result


class _ScanState:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.findings: list[dict[str, Any]] = []
        self.longest_binary_run = 0

    def record(self, category: str, path: str, offset: int | None) -> None:
        self.counts[category] += 1
        if len(self.findings) < MAX_RECORDED_FINDINGS:
            self.findings.append({"category": category, "offset": offset, "path": path})


class _PrintableRunTracker:
    """Track printable ASCII runs across chunk boundaries in binary regions.

    Edge runs are measured with C-level strip operations and interior runs with a
    regex for runs of eight or more bytes, so random-looking tensor data costs
    almost no Python-level work.
    """

    def __init__(self, path: str, state: _ScanState) -> None:
        self.path = path
        self.state = state
        self.length = 0
        self.start = 0
        self.flagged = False

    def _extend(self, start: int, count: int) -> None:
        if self.length == 0:
            self.start = start
            self.flagged = False
        self.length += count
        self.state.longest_binary_run = max(self.state.longest_binary_run, self.length)
        if self.length > MAX_BINARY_PRINTABLE_RUN and not self.flagged:
            self.state.record("binary_printable_run", self.path, self.start)
            self.flagged = True

    def _close(self) -> None:
        self.length = 0
        self.flagged = False

    def feed(self, chunk: bytes, chunk_start: int) -> None:
        lead = len(chunk) - len(chunk.lstrip(_PRINTABLE_BYTES))
        if lead == len(chunk):
            self._extend(chunk_start, lead)
            return
        if lead:
            self._extend(chunk_start, lead)
        self._close()
        trail = len(chunk) - len(chunk.rstrip(_PRINTABLE_BYTES))
        for match in _LONG_PRINTABLE_RUN.finditer(chunk, lead, len(chunk) - trail):
            self._extend(chunk_start + match.start(), match.end() - match.start())
            self._close()
        if trail:
            self._extend(chunk_start + len(chunk) - trail, trail)


def _read_chunks(handle: BinaryIO, chunk_bytes: int) -> Iterator[bytes]:
    while chunk := handle.read(chunk_bytes):
        yield chunk


def _scan_stream(
    chunks: Iterable[bytes],
    *,
    base_offset: int,
    binary: bool,
    path: str,
    state: _ScanState,
) -> int:
    """Scan a byte stream chunk by chunk and return the number of bytes consumed."""

    pending = b""
    pending_start = base_offset
    accept_from = base_offset
    # Mirror finditer's non-overlapping semantics across buffer boundaries.
    match_end = dict.fromkeys(_PATTERNS, base_offset)
    runs = _PrintableRunTracker(path, state) if binary else None
    consumed = 0
    iterator = iter(chunks)
    chunk = next(iterator, None)
    while True:
        following = next(iterator, None) if chunk is not None else None
        final = following is None
        if chunk is not None:
            if runs is not None:
                runs.feed(chunk, base_offset + consumed)
            consumed += len(chunk)
            buffer = pending + chunk
        else:
            buffer = pending
        limit = len(buffer) if final else len(buffer) - _OVERLAP_BYTES
        for text_run in _TEXT_RUN.finditer(buffer):
            for category, pattern in _PATTERNS.items():
                for match in pattern.finditer(buffer, text_run.start(), text_run.end()):
                    absolute = pending_start + match.start()
                    if (
                        absolute < accept_from
                        or absolute < match_end[category]
                        or match.start() >= limit
                    ):
                        continue
                    match_end[category] = pending_start + match.end()
                    state.record(category, path, absolute)
        if final:
            return consumed
        if limit > 0:
            accept_from = pending_start + limit
            keep_from = max(0, limit - _CONTEXT_BYTES)
            pending_start += keep_from
            pending = buffer[keep_from:]
        else:
            pending = buffer
        chunk = following


def _scan_strings(values: Iterable[str], path: str, state: _ScanState) -> None:
    for value in values:
        encoded = value.encode("utf-8", errors="replace")
        for category, pattern in _PATTERNS.items():
            for _match in pattern.finditer(encoded):
                state.record(category, path, None)


def _json_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _json_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _json_strings(item)


def _valid_safetensors_header(header: Any) -> bool:
    if not isinstance(header, dict):
        return False
    for name, entry in header.items():
        if name == "__metadata__":
            if not isinstance(entry, dict) or not all(
                isinstance(key, str) and isinstance(item, str) for key, item in entry.items()
            ):
                return False
        elif not (
            isinstance(entry, dict)
            and isinstance(entry.get("dtype"), str)
            and isinstance(entry.get("shape"), list)
            and isinstance(entry.get("data_offsets"), list)
        ):
            return False
    return True


def _scan_file(
    path: Path,
    relative: str,
    size: int,
    chunk_bytes: int,
    state: _ScanState,
) -> int:
    with open_regular_file(path) as handle:
        suffix = path.suffix.lower()
        if suffix in _JSON_SUFFIXES:
            if size > MAX_STRUCTURED_BYTES:
                state.record("malformed_structured_file", relative, None)
                return 0
            try:
                document = json.loads(handle.read(MAX_STRUCTURED_BYTES + 1))
            except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
                state.record("malformed_structured_file", relative, None)
                return size
            _scan_strings(_json_strings(document), relative, state)
            return size
        if suffix == ".safetensors":
            prefix = handle.read(8)
            header_length = struct.unpack("<Q", prefix)[0] if len(prefix) == 8 else -1
            if not 0 < header_length <= min(MAX_STRUCTURED_BYTES, size - 8):
                state.record("malformed_structured_file", relative, None)
                return len(prefix)
            try:
                header = json.loads(handle.read(header_length))
            except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
                header = None
            if not _valid_safetensors_header(header):
                state.record("malformed_structured_file", relative, None)
                return 8 + header_length
            _scan_strings(_json_strings(header), relative, state)
            tensors = _scan_stream(
                _read_chunks(handle, chunk_bytes),
                base_offset=8 + header_length,
                binary=True,
                path=relative,
                state=state,
            )
            return 8 + header_length + tensors
        return _scan_stream(
            _read_chunks(handle, chunk_bytes),
            base_offset=0,
            binary=suffix != ".txt",
            path=relative,
            state=state,
        )


def scan_model_artifact(
    artifact_dir: Path,
    *,
    chunk_bytes: int = CHUNK_BYTES,
) -> dict[str, Any]:
    """Scan every payload file in bounded memory and fail on suspected raw content.

    Hidden files are scanned rather than skipped. Links and special files are never
    opened or followed. Findings carry only a category, artifact-relative path, and
    byte offset; matched values are never stored or reported.
    """

    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    state = _ScanState()
    files_scanned = 0
    hidden_files = 0
    bytes_scanned = 0
    if artifact_dir.is_symlink() or artifact_dir.is_junction():
        state.record("link", ".", None)
    walk = () if state.counts else os.walk(artifact_dir, followlinks=False)
    for directory, dirnames, filenames in walk:
        current = Path(directory)
        dirnames.sort()
        for name in list(dirnames):
            candidate = current / name
            if candidate.is_symlink() or candidate.is_junction():
                state.record("link", candidate.relative_to(artifact_dir).as_posix(), None)
                dirnames.remove(name)
        for name in sorted(filenames):
            path = current / name
            relative = path.relative_to(artifact_dir).as_posix()
            if relative == "manifest.json":
                continue
            status = path.lstat()
            if stat.S_ISLNK(status.st_mode):
                state.record("link", relative, None)
                continue
            if not stat.S_ISREG(status.st_mode):
                state.record("special_file", relative, None)
                continue
            if status.st_size > MAX_FILE_BYTES:
                state.record("oversized_file", relative, None)
                continue
            files_scanned += 1
            if any(part.startswith(".") for part in Path(relative).parts):
                hidden_files += 1
            try:
                bytes_scanned += _scan_file(path, relative, status.st_size, chunk_bytes, state)
            except UnsafeFileError:
                state.record("special_file", relative, None)
            except OSError:
                state.record("unreadable_file", relative, None)

    failed = sum(state.counts.values()) > 0
    result: dict[str, Any] = {
        "schema_version": 2,
        "status": "failed" if failed else "passed",
        "files_scanned": files_scanned,
        "hidden_files_scanned": hidden_files,
        "bytes_scanned": bytes_scanned,
        "longest_printable_run": state.longest_binary_run,
        "policy": {
            "chunk_bytes": chunk_bytes,
            "max_binary_printable_run": MAX_BINARY_PRINTABLE_RUN,
            "natural_language_min_words": NATURAL_LANGUAGE_MIN_WORDS,
            "max_recorded_findings": MAX_RECORDED_FINDINGS,
        },
        "finding_counts": {category: state.counts.get(category, 0) for category in CATEGORIES},
        "findings": state.findings,
        "findings_truncated": sum(state.counts.values()) > len(state.findings),
    }
    if failed:
        summary = ", ".join(
            f"{category}={count}" for category, count in sorted(state.counts.items())
        )
        raise PrivacyScanError(
            f"Artifact privacy scan failed ({summary}); matched values are not reported",
            result,
        )
    return result
