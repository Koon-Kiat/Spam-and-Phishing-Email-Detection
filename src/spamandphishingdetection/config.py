"""Configuration loading and validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path("config/config.json")


@dataclass(frozen=True)
class ProjectConfig:
    """Validated project configuration with paths resolved from the repository root."""

    project_root: Path
    phishing_dataset: Path
    ceas_dataset: Path
    artifacts_dir: Path
    reports_dir: Path
    random_seed: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    group_folds: int
    train_folds: int
    validation_folds: int
    test_folds: int
    minhash_permutations: int
    near_duplicate_jaccard: float
    cv_repetitions: int
    cv_folds: int
    calibration_bins: int
    max_text_chars: int
    max_request_bytes: int
    word_max_features: int
    char_max_features: int
    min_document_frequency: int
    transformer_enabled: bool
    transformer_model: str
    transformer_max_length: int
    transformer_epochs: int
    transformer_batch_size: int
    transformer_learning_rate: float
    selection_tolerance: float
    ensemble_min_gain: float
    host: str
    port: int

    def ensure_output_directories(self) -> None:
        """Create directories used for generated artifacts and reports."""

        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def manifest_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation suitable for a model manifest."""

        values = asdict(self)
        for key, value in list(values.items()):
            if isinstance(value, Path):
                try:
                    values[key] = value.relative_to(self.project_root).as_posix()
                except ValueError:
                    values[key] = str(value)
        return values


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _required(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required configuration key: {key}")
    return mapping[key]


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    """Load a repository-relative JSON configuration file."""

    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    root_value = raw.get("project_root")
    root = (
        _resolve_path(config_path.parent, root_value)
        if root_value
        else config_path.parent.parent.resolve()
    )

    data = _required(raw, "data")
    paths = _required(raw, "paths")
    splits = _required(raw, "splits")
    classical = raw.get("classical", {})
    transformer = raw.get("transformer", {})
    selection = raw.get("selection", {})
    generalization = raw.get("generalization", {})
    server = raw.get("server", {})

    train_fraction = float(_required(splits, "train"))
    validation_fraction = float(_required(splits, "validation"))
    test_fraction = float(_required(splits, "test"))
    if abs(train_fraction + validation_fraction + test_fraction - 1.0) > 1e-9:
        raise ValueError("Train, validation, and test fractions must sum to 1.0")
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("All split fractions must be greater than zero")

    config = ProjectConfig(
        project_root=root,
        phishing_dataset=_resolve_path(root, _required(data, "phishing_email")),
        ceas_dataset=_resolve_path(root, _required(data, "ceas_08")),
        artifacts_dir=_resolve_path(root, _required(paths, "artifacts")),
        reports_dir=_resolve_path(root, _required(paths, "reports")),
        random_seed=int(raw.get("random_seed", 42)),
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        group_folds=int(splits.get("group_folds", 20)),
        train_folds=int(splits.get("train_folds", 14)),
        validation_folds=int(splits.get("validation_folds", 3)),
        test_folds=int(splits.get("test_folds", 3)),
        minhash_permutations=int(generalization.get("minhash_permutations", 128)),
        near_duplicate_jaccard=float(
            generalization.get("near_duplicate_jaccard", 0.85)
        ),
        cv_repetitions=int(generalization.get("cv_repetitions", 3)),
        cv_folds=int(generalization.get("cv_folds", 5)),
        calibration_bins=int(generalization.get("calibration_bins", 10)),
        max_text_chars=int(raw.get("max_text_chars", 100_000)),
        max_request_bytes=int(raw.get("max_request_bytes", 5_242_880)),
        word_max_features=int(classical.get("word_max_features", 120_000)),
        char_max_features=int(classical.get("char_max_features", 160_000)),
        min_document_frequency=int(classical.get("min_document_frequency", 2)),
        transformer_enabled=bool(transformer.get("enabled", True)),
        transformer_model=str(transformer.get("model_name", "distilbert/distilbert-base-uncased")),
        transformer_max_length=int(transformer.get("max_length", 256)),
        transformer_epochs=int(transformer.get("epochs", 3)),
        transformer_batch_size=int(transformer.get("batch_size", 16)),
        transformer_learning_rate=float(transformer.get("learning_rate", 2e-5)),
        selection_tolerance=float(selection.get("tolerance", 0.005)),
        ensemble_min_gain=float(selection.get("ensemble_min_gain", 0.005)),
        host=str(server.get("host", "127.0.0.1")),
        port=int(server.get("port", 5000)),
    )

    if config.max_text_chars < 1_000:
        raise ValueError("max_text_chars must be at least 1000")
    if config.max_request_bytes < config.max_text_chars:
        raise ValueError("max_request_bytes must be at least max_text_chars")
    if config.transformer_max_length < 32:
        raise ValueError("transformer.max_length must be at least 32")
    if (
        config.train_folds + config.validation_folds + config.test_folds
        != config.group_folds
    ):
        raise ValueError("Configured train/validation/test fold counts must sum to group_folds")
    if config.minhash_permutations != 128:
        raise ValueError("minhash_permutations must be 128 for release reproducibility")
    if not 0.0 < config.near_duplicate_jaccard <= 1.0:
        raise ValueError("near_duplicate_jaccard must be in (0, 1]")
    if config.cv_repetitions < 1 or config.cv_folds < 2:
        raise ValueError("Grouped cross-validation requires repetitions >= 1 and folds >= 2")
    if config.calibration_bins < 2:
        raise ValueError("calibration_bins must be at least 2")
    return config
