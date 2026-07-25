from __future__ import annotations

from collections.abc import Callable

from spamandphishingdetection.config import ProjectConfig
from spamandphishingdetection.datasets import (
    create_dataset_splits,
    load_and_audit_datasets,
)


def test_audit_maps_labels_and_removes_problem_rows(
    config_factory: Callable[..., ProjectConfig],
) -> None:
    config = config_factory(include_edge_rows=True)
    audited = load_and_audit_datasets(config)

    assert set(audited.dataframe["label"]) == {0, 1}
    assert audited.audit["removed"]["empty_count"] == 1
    assert audited.audit["removed"]["duplicate_count"] == 1
    assert audited.audit["removed"]["conflicting_hash_count"] == 1
    assert audited.audit["removed"]["conflicting_row_count"] == 2
    assert "ambiguous shared message" not in set(audited.dataframe["text"])


def test_splits_are_stratified_and_leakage_free(
    config_factory: Callable[..., ProjectConfig],
) -> None:
    config = config_factory()
    audited = load_and_audit_datasets(config)
    splits = create_dataset_splits(audited.dataframe, config)

    expected = {
        ("phishing_email", 0),
        ("phishing_email", 1),
        ("ceas_08", 0),
        ("ceas_08", 1),
    }
    for frame in (splits.train, splits.validation, splits.test):
        assert set(zip(frame["source"], frame["label"], strict=True)) == expected

    assert set(splits.train["text_hash"]).isdisjoint(splits.validation["text_hash"])
    assert set(splits.train["text_hash"]).isdisjoint(splits.test["text_hash"])
    assert set(splits.validation["text_hash"]).isdisjoint(splits.test["text_hash"])
    assert set(splits.train["similarity_group"]).isdisjoint(
        splits.validation["similarity_group"]
    )
    assert set(splits.train["similarity_group"]).isdisjoint(
        splits.test["similarity_group"]
    )
    assert set(splits.validation["similarity_group"]).isdisjoint(
        splits.test["similarity_group"]
    )
