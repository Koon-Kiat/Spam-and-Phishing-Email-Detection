"""Load, audit, clean, fingerprint, and split the two local datasets."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import ProjectConfig
from .grouping import add_message_metadata, build_near_duplicate_groups
from .preprocessing import normalize_text, normalized_text_hash, prepare_text

PHISHING_LABELS = {"Safe Email": 0, "Phishing Email": 1}
CEAS_LABELS = {0: 0, 1: 1, "0": 0, "1": 1}
REQUIRED_COLUMNS = (
    "text",
    "label",
    "source",
    "row_id",
    "text_hash",
    "similarity_group",
)


@dataclass(frozen=True)
class AuditedDataset:
    """Clean canonical data and its audit metadata."""

    dataframe: pd.DataFrame
    audit: dict[str, Any]


@dataclass(frozen=True)
class DatasetSplits:
    """Deterministic train, validation, and test partitions."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    summary: dict[str, Any]


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a streaming SHA-256 fingerprint for a dataset."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_files(config: ProjectConfig) -> None:
    for path in (config.phishing_dataset, config.ceas_dataset):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset not found: {path}")


def _load_phishing_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["Email Text", "Email Type"])
    if frame["Email Type"].isna().any():
        raise ValueError("phishing_email contains missing labels")
    unknown = sorted(
        str(value)
        for value in frame["Email Type"].dropna().unique()
        if value not in PHISHING_LABELS
    )
    if unknown:
        raise ValueError(f"Unknown phishing_email labels: {unknown}")
    return pd.DataFrame(
        {
            "raw_text": frame["Email Text"],
            "label": frame["Email Type"].map(PHISHING_LABELS),
            "source": "phishing_email",
            "row_id": [f"phishing_email:{index}" for index in frame.index],
            "sender": "",
            "timestamp": pd.NaT,
            "attachment_slice": "unknown",
        }
    )


def _load_ceas_dataset(path: Path) -> pd.DataFrame:
    available = set(pd.read_csv(path, nrows=0).columns)
    requested = ["subject", "body", "label", "sender", "date"]
    frame = pd.read_csv(path, usecols=[column for column in requested if column in available])
    if frame["label"].isna().any():
        raise ValueError("CEAS contains missing labels")
    unknown = sorted(
        str(value) for value in frame["label"].dropna().unique() if value not in CEAS_LABELS
    )
    if unknown:
        raise ValueError(f"Unknown CEAS labels: {unknown}")

    subjects = frame["subject"].fillna("").map(normalize_text)
    bodies = frame["body"].fillna("").map(normalize_text)
    combined = [
        f"{subject}\n\n{body}" if subject else body
        for subject, body in zip(subjects, bodies, strict=True)
    ]
    return pd.DataFrame(
        {
            "raw_text": combined,
            "label": frame["label"].map(CEAS_LABELS),
            "source": "ceas_08",
            "row_id": [f"ceas_08:{index}" for index in frame.index],
            "sender": frame.get("sender", pd.Series("", index=frame.index)).fillna(""),
            "timestamp": pd.to_datetime(
                frame.get("date", pd.Series(pd.NaT, index=frame.index)),
                errors="coerce",
                utc=True,
            ),
            "attachment_slice": "unknown",
        }
    )


def _distribution(frame: pd.DataFrame) -> dict[str, Any]:
    by_source = {}
    for source, group in frame.groupby("source", sort=True):
        by_source[str(source)] = {
            "rows": int(len(group)),
            "labels": {
                str(int(label)): int(count)
                for label, count in group["label"].value_counts().sort_index().items()
            },
        }
    return {
        "rows": int(len(frame)),
        "labels": {
            str(int(label)): int(count)
            for label, count in frame["label"].value_counts().sort_index().items()
        },
        "by_source": by_source,
    }


def load_and_audit_datasets(config: ProjectConfig) -> AuditedDataset:
    """Load both datasets, canonicalize text, and remove leakage-prone duplicates."""

    _validate_files(config)
    phishing = _load_phishing_dataset(config.phishing_dataset)
    ceas = _load_ceas_dataset(config.ceas_dataset)
    combined = pd.concat([phishing, ceas], ignore_index=True)
    raw_distribution = _distribution(combined.dropna(subset=["label"]))

    combined["normalized_full_text"] = combined["raw_text"].map(normalize_text)
    empty_mask = combined["normalized_full_text"].eq("")
    empty_rows = combined.loc[empty_mask, ["row_id", "source"]].to_dict(orient="records")
    combined = combined.loc[~empty_mask].copy()

    combined["text_hash"] = combined["normalized_full_text"].map(normalized_text_hash)
    conflicting_hashes = set(
        combined.groupby("text_hash")["label"].nunique().loc[lambda values: values > 1].index
    )
    conflict_rows = combined.loc[
        combined["text_hash"].isin(conflicting_hashes),
        ["row_id", "source", "label", "text_hash"],
    ].to_dict(orient="records")
    combined = combined.loc[~combined["text_hash"].isin(conflicting_hashes)].copy()

    combined.sort_values(["source", "row_id"], kind="stable", inplace=True)
    duplicate_mask = combined.duplicated(subset=["text_hash"], keep="first")
    duplicate_rows = combined.loc[
        duplicate_mask, ["row_id", "source", "label", "text_hash"]
    ].to_dict(orient="records")
    combined = combined.loc[~duplicate_mask].copy()
    combined = add_message_metadata(combined)

    dataset_fingerprints = {
        "phishing_email": file_sha256(config.phishing_dataset),
        "ceas_08": file_sha256(config.ceas_dataset),
    }
    grouping_key_payload = {
        "cache_schema": 1,
        "dataset_fingerprints": dataset_fingerprints,
        "threshold": config.near_duplicate_jaccard,
        "permutations": config.minhash_permutations,
        "seed": config.random_seed,
        "algorithm": "masked_word_trigram_minhash_lsh_exact_jaccard",
    }
    grouping_key = hashlib.sha256(
        json.dumps(grouping_key_payload, sort_keys=True).encode()
    ).hexdigest()
    grouping_cache = config.project_root / ".cache" / "grouping" / f"{grouping_key}.json.gz"
    text_hashes = combined["text_hash"].astype(str).tolist()
    similarity_groups: list[str]
    grouping_audit: dict[str, int | float]
    if grouping_cache.is_file():
        with gzip.open(grouping_cache, "rt", encoding="utf-8") as handle:
            cached = json.load(handle)
        if cached.get("text_hashes") != text_hashes:
            raise ValueError("Derived grouping cache does not match canonical row hashes")
        similarity_groups = [str(value) for value in cached["groups"]]
        grouping_audit = dict(cached["audit"])
    else:
        similarity_groups, grouping_audit = build_near_duplicate_groups(
            combined["normalized_full_text"],
            combined["text_hash"],
            threshold=config.near_duplicate_jaccard,
            num_perm=config.minhash_permutations,
            seed=config.random_seed,
        )
        grouping_cache.parent.mkdir(parents=True, exist_ok=True)
        temporary_cache = grouping_cache.with_suffix(".json.gz.tmp")
        with gzip.open(temporary_cache, "wt", encoding="utf-8") as handle:
            json.dump(
                {
                    "cache_schema": 1,
                    "text_hashes": text_hashes,
                    "groups": similarity_groups,
                    "audit": grouping_audit,
                },
                handle,
                separators=(",", ":"),
            )
        os.replace(temporary_cache, grouping_cache)
    grouping_audit["cache_key"] = grouping_key
    combined["similarity_group"] = similarity_groups

    prepared = combined["normalized_full_text"].map(
        lambda text: prepare_text(text, config.max_text_chars)
    )
    combined["text"] = prepared.map(lambda result: result[0])
    combined["was_truncated"] = prepared.map(lambda result: result[1])
    combined["label"] = combined["label"].astype("int8")
    combined = combined[
        [
            "text",
            "label",
            "source",
            "row_id",
            "text_hash",
            "similarity_group",
            "was_truncated",
            "timestamp",
            "length_slice",
            "has_html",
            "has_url",
            "has_obfuscation",
            "attachment_slice",
            "language_slice",
            "domain_group",
            "campaign_group",
        ]
    ].reset_index(drop=True)

    if combined["label"].isna().any():
        raise ValueError("Missing labels remain after canonicalization")
    if set(combined["label"].unique()) != {0, 1}:
        raise ValueError("Both safe and phishing labels are required")

    audit = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_fingerprints": dataset_fingerprints,
        "raw": raw_distribution,
        "clean": _distribution(combined),
        "removed": {
            "empty_count": len(empty_rows),
            "empty_rows": empty_rows,
            "duplicate_count": len(duplicate_rows),
            "duplicate_rows": duplicate_rows,
            "conflicting_hash_count": len(conflicting_hashes),
            "conflicting_row_count": len(conflict_rows),
            "conflicting_rows": conflict_rows,
        },
        "truncated_count": int(combined["was_truncated"].sum()),
        "near_duplicate_grouping": grouping_audit,
    }
    return AuditedDataset(combined, audit)


def create_dataset_splits(dataframe: pd.DataFrame, config: ProjectConfig) -> DatasetSplits:
    """Assign deterministic source/label-stratified similarity groups to folds."""

    missing = set(REQUIRED_COLUMNS) - set(dataframe.columns)
    if missing:
        raise ValueError(f"Canonical dataframe is missing columns: {sorted(missing)}")

    working = dataframe.reset_index(drop=True).copy()
    stratification = working["source"].astype(str) + ":" + working["label"].astype(str)
    groups_per_stratum = (
        working.assign(_stratum=stratification)
        .groupby("_stratum")["similarity_group"]
        .nunique()
    )
    n_splits = min(
        config.group_folds,
        int(working["similarity_group"].nunique()),
        int(groups_per_stratum.min()),
    )
    if n_splits < 3:
        raise ValueError("At least three similarity groups per source/label stratum are required")

    fold_assignment = pd.Series(-1, index=working.index, dtype="int16")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=config.random_seed,
    )
    for fold, (_, fold_indices) in enumerate(
        splitter.split(working["text"], stratification, working["similarity_group"])
    ):
        fold_assignment.iloc[fold_indices] = fold
    if (fold_assignment < 0).any():
        raise ValueError("Not every row received a grouped fold assignment")
    working["split_fold"] = fold_assignment

    if n_splits == config.group_folds:
        train_count = config.train_folds
        validation_count = config.validation_folds
    else:
        validation_count = max(1, round(n_splits * config.validation_fraction))
        test_count = max(1, round(n_splits * config.test_fraction))
        train_count = n_splits - validation_count - test_count
        if train_count < 1:
            train_count = 1
            validation_count = 1
    train_folds = set(range(train_count))
    validation_folds = set(range(train_count, train_count + validation_count))
    test_folds = set(range(train_count + validation_count, n_splits))

    splits = {
        "train": working.loc[working["split_fold"].isin(train_folds)].reset_index(drop=True),
        "validation": working.loc[
            working["split_fold"].isin(validation_folds)
        ].reset_index(drop=True),
        "test": working.loc[working["split_fold"].isin(test_folds)].reset_index(drop=True),
    }
    _verify_splits(splits)
    summary = {name: _distribution(frame) for name, frame in splits.items()}
    summary["method"] = {
        "algorithm": "StratifiedGroupKFold",
        "folds": n_splits,
        "train_folds": sorted(train_folds),
        "validation_folds": sorted(validation_folds),
        "test_folds": sorted(test_folds),
        "group_column": "similarity_group",
    }
    return DatasetSplits(
        train=splits["train"],
        validation=splits["validation"],
        test=splits["test"],
        summary=summary,
    )


def _verify_splits(splits: dict[str, pd.DataFrame]) -> None:
    names = list(splits)
    for index, left_name in enumerate(names):
        left = splits[left_name]
        strata = set(zip(left["source"], left["label"], strict=True))
        expected = {
            ("phishing_email", 0),
            ("phishing_email", 1),
            ("ceas_08", 0),
            ("ceas_08", 1),
        }
        if strata != expected:
            raise ValueError(f"Split {left_name} does not contain every source/label stratum")
        left_hashes = set(left["text_hash"])
        left_groups = set(left["similarity_group"])
        for right_name in names[index + 1 :]:
            overlap = left_hashes.intersection(splits[right_name]["text_hash"])
            if overlap:
                raise ValueError(
                    f"Duplicate leakage between {left_name} and {right_name}: {len(overlap)} hashes"
                )
            group_overlap = left_groups.intersection(
                splits[right_name]["similarity_group"]
            )
            if group_overlap:
                raise ValueError(
                    "Near-duplicate leakage between "
                    f"{left_name} and {right_name}: {len(group_overlap)} groups"
                )


def write_audit_report(
    audited: AuditedDataset,
    config: ProjectConfig,
    filename: str = "data_audit.json",
) -> Path:
    """Write the full data audit report."""

    config.ensure_output_directories()
    output_path = config.reports_dir / filename
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(audited.audit, handle, indent=2, sort_keys=True)
    return output_path
