from __future__ import annotations

import json
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
