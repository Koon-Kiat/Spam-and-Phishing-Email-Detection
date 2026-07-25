"""Privacy-safe similarity, campaign, and domain grouping helpers."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Iterable
from urllib.parse import urlsplit

import pandas as pd
import tldextract
from datasketch import MinHash, MinHashLSH

from .preprocessing import canonicalize_sensitive_identifiers, normalize_text

_WORD = re.compile(r"[\w<>]+", re.UNICODE)
_URL = re.compile(r"https?://[^\s<>'\"]+", re.IGNORECASE)
_EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Z]{2,63}", re.IGNORECASE)
_NUMBER = re.compile(r"\b\d+(?:[.,:/-]\d+)*\b")
_HTML = re.compile(r"<(?:html|body|a|img|table|div|span)\b", re.IGNORECASE)
_OBFUSCATION = re.compile(
    r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]|hxxps?|\[(?:dot|at)\]",
    re.IGNORECASE,
)
_SPANISH_MARKERS = re.compile(
    r"\b(?:usted|cuenta|correo|contrase(?:ñ|n)a|seguridad|factura|banco|"
    r"verifique|urgente|mensaje|gracias|hola|estimad[oa])\b",
    re.IGNORECASE,
)
_DOMAIN_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())


def stable_private_hash(value: str, *, namespace: str) -> str:
    """Return a report-safe stable identifier without exposing its raw value."""

    digest = hashlib.sha256(f"{namespace}\0{value}".encode()).hexdigest()
    return f"{namespace}_{digest[:16]}"


def registrable_domain(hostname: str) -> str:
    """Reduce a hostname to its registrable domain using a bundled suffix snapshot."""

    extracted = _DOMAIN_EXTRACTOR(hostname.strip().lower().rstrip("."))
    return extracted.top_domain_under_public_suffix or hostname.strip().lower().rstrip(".")


def extract_registrable_domains(text: str) -> list[str]:
    """Extract de-duplicated URL and email domains without returning local parts."""

    domains: set[str] = set()
    for match in _URL.finditer(text):
        try:
            hostname = urlsplit(match.group(0)).hostname
        except ValueError:
            hostname = None
        if hostname:
            domains.add(registrable_domain(hostname))
    for match in _EMAIL.finditer(text):
        domains.add(registrable_domain(match.group(0).rsplit("@", 1)[1]))
    return sorted(domain for domain in domains if domain)


def masked_similarity_text(text: str) -> str:
    """Create a volatile-token-masked view for near-duplicate detection."""

    masked = canonicalize_sensitive_identifiers(normalize_text(text)).casefold()
    masked = _URL.sub(" <URL> ", masked)
    masked = _EMAIL.sub(" <EMAIL> ", masked)
    masked = _NUMBER.sub(" <NUMBER> ", masked)
    return " ".join(masked.split())


def word_trigrams(text: str) -> set[str]:
    """Create word trigram shingles, with safe fallbacks for short messages."""

    tokens = _WORD.findall(masked_similarity_text(text))
    if len(tokens) < 3:
        return {" ".join(tokens)} if tokens else {"<EMPTY>"}
    return {" ".join(tokens[index : index + 3]) for index in range(len(tokens) - 2)}


def jaccard(left: set[str], right: set[str]) -> float:
    """Calculate exact set Jaccard similarity."""

    union = left | right
    return len(left & right) / len(union) if union else 1.0


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def build_near_duplicate_groups(
    texts: Iterable[str],
    text_hashes: Iterable[str],
    *,
    threshold: float = 0.85,
    num_perm: int = 128,
    seed: int = 42,
) -> tuple[list[str], dict[str, int | float]]:
    """Group exact-Jaccard-verified candidates found by 128-permutation MinHash LSH."""

    values = list(texts)
    hashes = list(text_hashes)
    if len(values) != len(hashes):
        raise ValueError("Texts and hashes must have identical lengths")
    shingles = [word_trigrams(text) for text in values]
    signatures: list[MinHash] = []
    for shingle_set in shingles:
        signature = MinHash(num_perm=num_perm, seed=seed)
        for shingle in sorted(shingle_set):
            signature.update(shingle.encode("utf-8"))
        signatures.append(signature)

    lsh = MinHashLSH(threshold=max(0.5, threshold - 0.10), num_perm=num_perm)
    candidates: set[tuple[int, int]] = set()
    for index, signature in enumerate(signatures):
        for candidate in lsh.query(signature):
            pair = (int(candidate), index)
            candidates.add(pair if pair[0] < pair[1] else (pair[1], pair[0]))
        lsh.insert(str(index), signature)

    sets = _DisjointSet(len(values))
    verified = 0
    for left, right in sorted(candidates):
        if jaccard(shingles[left], shingles[right]) >= threshold:
            sets.union(left, right)
            verified += 1

    members: dict[int, list[int]] = defaultdict(list)
    for index in range(len(values)):
        members[sets.find(index)].append(index)
    group_names: dict[int, str] = {}
    for root, indices in members.items():
        stable_hash = min(hashes[index] for index in indices)
        group_names[root] = f"similarity_{stable_hash[:20]}"
    groups = [group_names[sets.find(index)] for index in range(len(values))]
    summary: dict[str, int | float] = {
        "rows": len(values),
        "groups": len(members),
        "candidate_pairs": len(candidates),
        "verified_pairs": verified,
        "largest_group": max((len(indices) for indices in members.values()), default=0),
        "jaccard_threshold": threshold,
        "minhash_permutations": num_perm,
    }
    return groups, summary


def derive_message_metadata(text: str) -> dict[str, object]:
    """Derive privacy-safe report slices from message text."""

    domains = extract_registrable_domains(text)
    domain_ids = [stable_private_hash(domain, namespace="domain") for domain in domains]
    length = len(text)
    if length < 500:
        length_slice = "short"
    elif length < 2_000:
        length_slice = "medium"
    elif length < 10_000:
        length_slice = "long"
    else:
        length_slice = "very_long"
    return {
        "length_slice": length_slice,
        "has_html": bool(_HTML.search(text)),
        "has_url": bool(_URL.search(text)),
        "has_obfuscation": bool(_OBFUSCATION.search(text)),
        "language_slice": "spanish" if _SPANISH_MARKERS.search(text) else "other_or_unknown",
        "domain_group": domain_ids[0] if domain_ids else "domain_none",
        "campaign_group": stable_private_hash(
            "|".join(domain_ids) or "<none>",
            namespace="campaign",
        ),
    }


def add_message_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach deterministic non-content metadata columns to a canonical frame."""

    result = frame.copy()
    sender = result["sender"].fillna("").astype(str) if "sender" in result else [""] * len(result)
    metadata = pd.DataFrame(
        [
            derive_message_metadata(f"{sender_value}\n{text}")
            for sender_value, text in zip(sender, result["raw_text"], strict=True)
        ],
        index=result.index,
    )
    for column in metadata:
        result[column] = metadata[column]
    return result
