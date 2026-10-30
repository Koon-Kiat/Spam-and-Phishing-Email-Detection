from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from spamandphishingdetection.config import ProjectConfig, load_config


@pytest.fixture
def config_factory(tmp_path: Path) -> Callable[..., ProjectConfig]:
    def factory(
        *,
        include_edge_rows: bool = False,
        transformer_enabled: bool = False,
        max_request_bytes: int = 10_000,
    ) -> ProjectConfig:
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)

        phishing_rows = []
        ceas_rows = []
        for index in range(30):
            unique = f"topic{chr(97 + index % 26)}{chr(97 + index // 26)}"
            phishing_rows.extend(
                [
                    {
                        "Email Text": (
                            "Project meeting agenda and lunch schedule safe message "
                            f"{unique} reference {unique}"
                        ),
                        "Email Type": "Safe Email",
                    },
                    {
                        "Email Text": (
                            "URGENT verify your bank account password at "
                            f"http://fraud-{unique}.example now for {unique} reference {unique}!"
                        ),
                        "Email Type": "Phishing Email",
                    },
                ]
            )
            ceas_rows.extend(
                [
                    {
                        "subject": f"Team update {index}",
                        "body": (
                            "Here is the approved project status and calendar "
                            f"{unique} reference {unique}"
                        ),
                        "label": 0,
                    },
                    {
                        "subject": f"Security alert {index}",
                        "body": (
                            "Confirm your identity and claim a prize at "
                            f"https://phish-{unique}.example for {unique} reference {unique}"
                        ),
                        "label": 1,
                    },
                ]
            )

        if include_edge_rows:
            phishing_rows.extend(
                [
                    {
                        "Email Text": phishing_rows[0]["Email Text"],
                        "Email Type": "Safe Email",
                    },
                    {"Email Text": "   ", "Email Type": "Safe Email"},
                    {"Email Text": "ambiguous shared message", "Email Type": "Safe Email"},
                    {
                        "Email Text": "ambiguous shared message",
                        "Email Type": "Phishing Email",
                    },
                ]
            )

        pd.DataFrame(phishing_rows).to_csv(data_dir / "phishing_email.csv", index=False)
        pd.DataFrame(ceas_rows).to_csv(data_dir / "ceas_08.csv", index=False)

        raw_config = {
            "project_root": str(tmp_path),
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
            "max_text_chars": 1_000,
            "max_request_bytes": max_request_bytes,
            "classical": {
                "word_max_features": 2_000,
                "char_max_features": 3_000,
                "min_document_frequency": 1,
            },
            "transformer": {
                "enabled": transformer_enabled,
                "model_name": "distilbert/distilbert-base-uncased",
                "max_length": 64,
                "epochs": 1,
                "batch_size": 4,
                "learning_rate": 0.00002,
            },
            "selection": {"tolerance": 0.005, "ensemble_min_gain": 0.005},
            "server": {"host": "127.0.0.1", "port": 5000},
        }
        config_dir = tmp_path / "config"
        config_dir.mkdir(exist_ok=True)
        config_path = config_dir / "config.json"
        config_path.write_text(json.dumps(raw_config), encoding="utf-8")
        return load_config(config_path)

    return factory


def _write_trained_project_inputs(root: Path) -> Path:
    data_dir = root / "data"
    (data_dir / "external").mkdir(parents=True)
    phishing = []
    ceas = []
    for index in range(24):
        unique = f"case{chr(97 + index % 26)}"
        phishing.extend(
            [
                {
                    "Email Text": f"Normal team meeting project {unique} reference {unique}",
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
                    "body": f"Approved appointment status {unique} reference {unique}",
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
    pd.DataFrame(
        [
            {
                "hash": f"hash-{index}",
                "subject": f"Mensaje {index}",
                "body": "Verifique su cuenta" if index >= 664 else "Agenda de proyecto",
                "date": "2025-01-01",
                "attachments_count": index % 2,
                "Label": int(index >= 664),
            }
            for index in range(1_395)
        ]
    ).to_csv(data_dir / "external" / "spaphish_v5.csv", index=False)

    config_dir = root / "config"
    config_dir.mkdir()
    (config_dir / "datasets.v1.json").write_text('{"schema_version": 1}\n', encoding="utf-8")
    (root / "requirements.lock").write_text("fixture==1.0\n", encoding="utf-8")
    for name in ("MODEL_CARD.md", "DATASET_NOTICES.md", "MODEL_LICENSE.md", "SECURITY.md"):
        (root / name).write_text(f"# {name}\n", encoding="utf-8")
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
    return config_path


@pytest.fixture(scope="session")
def trained_project_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A project with one trained classical artifact, published as latest."""

    from spamandphishingdetection.training import run_training

    root = tmp_path_factory.mktemp("trained-project-template")
    run_training(load_config(_write_trained_project_inputs(root)), include_transformer=False)
    return root


@pytest.fixture
def trained_project(trained_project_template: Path, tmp_path: Path) -> ProjectConfig:
    """An isolated copy of the trained template that a test may modify freely."""

    root = tmp_path / "project"
    shutil.copytree(trained_project_template, root)
    config_path = root / "config" / "config.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload["project_root"] = str(root)
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    return load_config(config_path)
