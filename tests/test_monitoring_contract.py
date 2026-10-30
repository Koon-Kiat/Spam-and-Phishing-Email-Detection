from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from spamandphishingdetection.config import ProjectConfig
from spamandphishingdetection.grouping import derive_message_metadata
from spamandphishingdetection.integrity import bind_latest
from spamandphishingdetection.monitoring import (
    MAX_RECORD_CHARS,
    MonitoringInputError,
    monitor_batch,
)

SENSITIVE = "Dear Jane, your invoice from jane.doe@example.com is overdue"


@pytest.fixture
def monitored(config_factory: Callable[..., ProjectConfig]) -> ProjectConfig:
    config = config_factory()
    artifact = config.artifacts_dir / "run-monitored"
    artifact.mkdir(parents=True)
    manifest = {
        "schema_version": 2,
        "model_version": "run-monitored",
        "backend": "classical",
        "threshold": 0.5,
        "test_metrics": {"macro_f1": 0.9},
        "monitoring_baseline": {
            "score_bin_edges": [index / 10 for index in range(11)],
            "score_bin_fractions": [0.1] * 10,
            "predicted_phishing_fraction": 0.5,
            "slice_fractions": {"source": {"phishing_email": 0.5, "ceas_08": 0.5}},
            "alert_thresholds": {
                "score_population_stability_index": 0.2,
                "predicted_prevalence_absolute_change": 0.1,
                "slice_fraction_absolute_change": 0.15,
                "performance_macro_f1_drop": 0.05,
            },
        },
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    bind_latest(config, artifact, action="promote")
    return config


def _record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "timestamp": "2026-07-25T12:00:00Z",
        "probability": 0.73,
        "predicted_label": "phishing",
    }
    record.update(overrides)
    return record


def _monitor(config: ProjectConfig, *lines: object) -> dict[str, Any]:
    batch = config.project_root / "batch.jsonl"
    batch.write_text(
        "\n".join(line if isinstance(line, str) else json.dumps(line) for line in lines),
        encoding="utf-8",
    )
    return monitor_batch(config, batch)


def _rejected(config: ProjectConfig, *lines: object) -> str:
    with pytest.raises(MonitoringInputError) as raised:
        _monitor(config, *lines)
    message = str(raised.value)
    for fragment in ("Jane", "invoice", "example.com", "overdue"):
        assert fragment not in message
    return message


def test_valid_content_free_batch_is_aggregated(monitored: ProjectConfig) -> None:
    report = _monitor(
        monitored,
        _record(
            true_label=1,
            slices={
                "source": "review_queue",
                "length_slice": "short",
                "has_url": True,
                "has_html": "false",
                "attachment_slice": "no_attachment",
                "language_slice": "spanish",
                "domain_group": "domain_0123456789abcdef",
                "campaign_group": "campaign_0123456789abcdef",
            },
        ),
        _record(probability=0.2, predicted_label=0, true_label="safe"),
    )

    assert report["content_retained"] is False
    assert report["rows"] == 2
    assert report["performance"]["macro_f1"] == 1.0
    assert report["slice_drift"]["has_url"]["fractions"] == {"<missing>": 0.5, "True": 0.5}
    assert report["slice_drift"]["has_html"]["fractions"]["False"] == 0.5


@pytest.mark.parametrize("key", ["subject", "Body", "message", "sender", "url", "text"])
def test_forbidden_keys_nested_in_slices_are_rejected(
    monitored: ProjectConfig,
    key: str,
) -> None:
    message = _rejected(monitored, _record(slices={key: SENSITIVE}))
    assert "forbidden message-content fields" in message


@pytest.mark.parametrize(
    "value",
    [
        {"nested": "value"},
        ["review_queue"],
        None,
        3.5,
        "review queue with spaces",
        "a" * 65,
        SENSITIVE,
    ],
)
def test_non_scalar_free_form_and_oversized_slice_values_are_rejected(
    monitored: ProjectConfig,
    value: object,
) -> None:
    message = _rejected(monitored, _record(slices={"source": value}))
    assert "invalid value for slice source" in message


@pytest.mark.parametrize(
    "slices",
    [
        {"length_slice": "tiny"},
        {"has_url": "maybe"},
        {"domain_group": "domain_example.com"},
        {"sender_domain": "jane.doe@example.com"},
        {"campaign": "x" * 129},
        {"campaign": "line\nbreak"},
        {"comment": "free text"},
    ],
)
def test_slice_allowlist_and_limits_are_enforced(
    monitored: ProjectConfig,
    slices: dict[str, Any],
) -> None:
    message = _rejected(monitored, _record(slices=slices))
    assert "slice" in message


def test_raw_domains_and_campaigns_are_hashed_consistently(monitored: ProjectConfig) -> None:
    report = _monitor(
        monitored,
        _record(slices={"sender_domain": "Mail.Example.co.uk", "campaign": "Q3 Invoice Wave"}),
    )

    expected_domain = derive_message_metadata("contact jane@mail.example.co.uk")["domain_group"]
    assert report["slice_drift"]["sender_domain"]["fractions"] == {expected_domain: 1.0}
    rendered = json.dumps(report).lower()
    assert "example" not in rendered
    assert "invoice" not in rendered
    assert list(report["slice_drift"]["campaign"]["fractions"])[0].startswith("campaign_")


@pytest.mark.parametrize("true_label", [2, -1, True, "yes", 0.5, {"label": 1}])
def test_true_label_must_be_strictly_binary(monitored: ProjectConfig, true_label: object) -> None:
    message = _rejected(monitored, _record(true_label=true_label))
    assert "true_label" in message


def test_predicted_label_must_match_probability_and_threshold(
    monitored: ProjectConfig,
) -> None:
    message = _rejected(monitored, _record(probability=0.2, predicted_label="phishing"))
    assert "inconsistent" in message
    assert _monitor(monitored, _record(probability=0.5, predicted_label=1))["rows"] == 1


@pytest.mark.parametrize(
    "line",
    [
        _record(note=SENSITIVE),
        _record(timestamp=SENSITIVE),
        _record(probability=True),
        _record(probability="0.7"),
        _record(predicted_label=True),
        "{not json " + SENSITIVE,
        json.dumps(_record(slices={"source": "q"})) + " " * MAX_RECORD_CHARS,
        [SENSITIVE],
    ],
)
def test_invalid_records_fail_with_non_sensitive_messages(
    monitored: ProjectConfig,
    line: object,
) -> None:
    assert _rejected(monitored, line).startswith("Line 1 ")
