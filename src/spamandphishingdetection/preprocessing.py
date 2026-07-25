"""Shared text preparation and deterministic hashing."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRUNCATION_MARKER = "\n[...TRUNCATED...]\n"
_WHITESPACE = re.compile(r"\s+")
_EMAIL_ADDRESS = re.compile(
    r"(?<![\w.+-])([A-Z0-9._%+-]+)@([A-Z0-9.-]+\.[A-Z]{2,63})(?![\w.-])",
    re.IGNORECASE,
)
_URL = re.compile(r"https?://[^\s<>'\"\])}]+", re.IGNORECASE)
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d .()/-]{7,}\d)(?!\w)")
_LONG_IDENTIFIER = re.compile(
    r"(?<!\w)(?=[A-Z0-9-]{8,}\b)(?=[A-Z0-9-]*\d)[A-Z0-9-]+",
    re.IGNORECASE,
)


def _canonicalize_url(match: re.Match[str]) -> str:
    value = match.group(0)
    trailing = ""
    while value and value[-1] in ".,;:!?":
        trailing = value[-1] + trailing
        value = value[:-1]
    try:
        parsed = urlsplit(value)
        query = urlencode([(key, "<VALUE>") for key, _ in parse_qsl(parsed.query)])
        canonical = urlunsplit(
            (
                parsed.scheme.lower(),
                (parsed.hostname or "").lower(),
                parsed.path,
                query,
                "",
            )
        )
        return canonical + trailing
    except ValueError:
        return "<URL>" + trailing


def canonicalize_sensitive_identifiers(text: str) -> str:
    """Mask volatile personal identifiers while preserving useful context."""

    canonical = _EMAIL_ADDRESS.sub(r"<EMAIL_LOCAL>@\2", text)
    canonical = _URL.sub(_canonicalize_url, canonical)
    canonical = _PHONE.sub("<PHONE>", canonical)
    return _LONG_IDENTIFIER.sub("<IDENTIFIER>", canonical)


def normalize_text(value: object) -> str:
    """Normalize text without discarding punctuation, URLs, or casing."""

    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\x00", " ")
    return _WHITESPACE.sub(" ", text).strip()


def limit_text(text: str, max_chars: int) -> tuple[str, bool]:
    """Limit very large documents while retaining both the beginning and end."""

    if len(text) <= max_chars:
        return text, False
    marker_length = len(TRUNCATION_MARKER)
    available = max_chars - marker_length
    if available <= 1:
        return text[:max_chars], True
    head_length = int(available * 0.75)
    tail_length = available - head_length
    return f"{text[:head_length]}{TRUNCATION_MARKER}{text[-tail_length:]}", True


def normalized_text_hash(text: str) -> str:
    """Hash normalized, case-folded text for leakage-safe duplicate handling."""

    normalized = normalize_text(text).casefold()
    return hashlib.sha256(normalized.encode("utf-8", errors="replace")).hexdigest()


def prepare_text(value: object, max_chars: int) -> tuple[str, bool]:
    """Normalize and size-limit text for model input."""

    normalized = normalize_text(value)
    canonical = canonicalize_sensitive_identifiers(normalized)
    return limit_text(canonical, max_chars)
