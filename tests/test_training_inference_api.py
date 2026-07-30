from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from server.flask_app import create_web_app
from spamandphishingdetection.api import create_app
from spamandphishingdetection.config import ProjectConfig
from spamandphishingdetection.inference import Predictor, extract_text_from_eml
from spamandphishingdetection.reporting import generate_diagnostic_report
from spamandphishingdetection.training import run_training


@pytest.fixture(scope="module")
def trained_state(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[ProjectConfig, Path]:
    # This module-scoped fixture builds its data directly because function-scoped
    # config_factory cannot be injected into a wider-scoped fixture.
    root = tmp_path_factory.mktemp("trained-project")
    import pandas as pd

    data_dir = root / "data"
    data_dir.mkdir()
    phishing = []
    ceas = []
    for index in range(24):
        unique = f"case{chr(97 + index % 26)}"
        phishing.extend(
            [
                {
                    "Email Text": (
                        f"Normal team meeting project {unique} reference {unique}"
                    ),
                    "Email Type": "Safe Email",
                },
                {
                    "Email Text": (
                            "URGENT account suspended click "
                            f"http://bad-{unique}.test verify {unique} reference {unique}"
                    ),
                    "Email Type": "Phishing Email",
                },
            ]
        )
        ceas.extend(
            [
                {
                    "subject": f"Calendar {index}",
                    "body": (
                        f"Approved appointment status {unique} reference {unique}"
                    ),
                    "label": 0,
                },
                {
                    "subject": f"Winner {index}",
                    "body": (
                        f"Claim prize and password at https://evil-{unique}.test "
                        f"for {unique} reference {unique}"
                    ),
                    "label": 1,
                },
            ]
        )
    pd.DataFrame(phishing).to_csv(data_dir / "phishing_email.csv", index=False)
    pd.DataFrame(ceas).to_csv(data_dir / "ceas_08.csv", index=False)
    config_dir = root / "config"
    config_dir.mkdir()
    payload = {
        "project_root": str(root),
        "data": {
            "phishing_email": "data/phishing_email.csv",
            "ceas_08": "data/ceas_08.csv",
        },
        "paths": {"artifacts": "artifacts", "reports": "reports"},
        "random_seed": 42,
        "splits": {
            "train": 0.60,
            "validation": 0.20,
            "test": 0.20,
            "group_folds": 5,
            "train_folds": 3,
            "validation_folds": 1,
            "test_folds": 1,
        },
        "generalization": {
            "minhash_permutations": 128,
            "near_duplicate_jaccard": 0.85,
            "cv_repetitions": 1,
            "cv_folds": 3,
            "calibration_bins": 10,
        },
        "max_text_chars": 1000,
        "max_request_bytes": 5000,
        "classical": {
            "word_max_features": 2000,
            "char_max_features": 3000,
            "min_document_frequency": 1,
        },
        "transformer": {
            "enabled": False,
            "model_name": "distilbert/distilbert-base-uncased",
            "max_length": 64,
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.00002,
        },
        "selection": {"tolerance": 0.005, "ensemble_min_gain": 0.005},
        "server": {"host": "127.0.0.1", "port": 5000},
    }
    config_path = config_dir / "config.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    from spamandphishingdetection.config import load_config

    config = load_config(config_path)
    result = run_training(config, include_transformer=False)
    return config, result.artifact_dir


def test_artifact_reloads_with_stable_prediction(
    trained_state: tuple[ProjectConfig, Path],
) -> None:
    config, artifact = trained_state
    first = Predictor.load(config, artifact)
    second = Predictor.load(config, artifact.name)

    text = "URGENT verify your password at http://fraud.test immediately"
    first_result = first.predict_text(text)
    second_result = second.predict_text(text)

    assert first_result == second_result
    assert 0.0 <= first_result.probability <= 1.0


def test_eml_extraction_prefers_text_and_supports_html() -> None:
    message = (
        b"Subject: Security alert\r\n"
        b"Content-Type: text/html; charset=utf-8\r\n\r\n"
        b"<html><body><b>Verify</b> your account</body></html>"
    )
    extracted = extract_text_from_eml(message)

    assert "Security alert" in extracted
    assert "Verify your account" in extracted
    assert "<b>" not in extracted


def test_api_json_eml_deprecated_routes_and_health(
    trained_state: tuple[ProjectConfig, Path],
) -> None:
    config, artifact = trained_state
    client = create_app(config, artifact=artifact).test_client()

    health = client.get("/health")
    assert health.status_code == 200
    assert health.get_json()["model_loaded"] is True

    json_response = client.post(
        "/api/v1/predict",
        json={"subject": "Alert", "text": "verify your password now"},
    )
    assert json_response.status_code == 200
    assert set(json_response.get_json()) == {
        "label",
        "is_phishing",
        "probability",
        "threshold",
        "model_version",
    }

    eml = b"Subject: Hello\r\n\r\nNormal project meeting tomorrow"
    eml_response = client.post(
        "/api/v1/predict",
        data=eml,
        content_type="message/rfc822",
    )
    assert eml_response.status_code == 200

    assert client.post("/evaluateEmail", data=eml).status_code == 404

    assert client.post("/api/v1/predict", json={"text": ""}).status_code == 400
    invalid_json = client.post(
        "/api/v1/predict",
        data=b"[]",
        content_type="application/json",
    )
    assert invalid_json.status_code == 400
    assert invalid_json.get_json() == {"error": "Invalid email input"}
    assert (
        client.post(
            "/api/v1/predict",
            data=b"x" * 5001,
            content_type="message/rfc822",
        ).status_code
        == 413
    )


def test_health_reports_missing_artifact(
    config_factory: Callable[..., ProjectConfig],
) -> None:
    config = config_factory()
    client = create_app(config).test_client()
    response = client.get("/health")

    assert response.status_code == 503
    assert response.get_json() == {
        "status": "unavailable",
        "model_loaded": False,
        "error": "Prediction model is unavailable",
    }

    prediction = client.post("/api/v1/predict", json={"text": "hello"})
    assert prediction.status_code == 503
    assert prediction.get_json() == {"error": "Prediction model is unavailable"}


def test_browser_test_page_and_diagnostic_charts(
    trained_state: tuple[ProjectConfig, Path],
    tmp_path: Path,
) -> None:
    config, artifact = trained_state
    client = create_web_app(config, artifact=artifact).test_client()

    page = client.get("/")
    assert page.status_code == 200
    assert b"Pause. Inspect. Then decide." in page.data
    assert b"Drop an" in page.data
    assert page.data.index(b"Paste the email") < page.data.index(b"Upload the original email")
    assert b"Model service" in page.data
    assert b"health-dot" not in page.data
    assert f'data-max-file-bytes="{config.max_request_bytes}"'.encode() in page.data
    assert client.get("/static/index.js").status_code == 200
    assert client.get("/static/index.css").status_code == 200
    assert client.get("/taskpane.html").status_code == 404
    assert client.get("/commands.html").status_code == 404
    javascript = client.get("/static/index.js").get_data(as_text=True)
    stylesheet = client.get("/static/index.css").get_data(as_text=True)
    assert "remove-file" in javascript
    assert "Saved, not active" in javascript
    assert 'setAttribute("aria-busy", "true")' in javascript
    assert "linear-gradient" not in stylesheet
    assert "#2563eb" not in stylesheet
    assert "#7c3aed" not in stylesheet

    diagnostics = generate_diagnostic_report(
        config,
        artifact=artifact,
        output_dir=tmp_path / "charts",
        fractions=(0.5, 1.0),
    )
    assert diagnostics.learning_curve_csv.is_file()
    assert len(diagnostics.charts) == 4
    for chart in diagnostics.charts:
        content = chart.read_text(encoding="utf-8")
        assert content.startswith("<svg")
        assert 'role="img"' in content
        assert "Matplotlib 3.11.1" in content
        assert "Training run:" in content
        assert diagnostics.run_id not in content
        assert "#2563eb" not in content
        assert "#7c3aed" not in content
    confusion = (tmp_path / "charts" / "test_confusion_matrix.svg").read_text(
        encoding="utf-8"
    )
    assert "True negative (TN)" in confusion
    assert "False positive (FP)" in confusion
    assert "False negative (FN)" in confusion
    assert "True positive (TP)" in confusion
