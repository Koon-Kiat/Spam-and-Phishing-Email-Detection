from __future__ import annotations

import json
import os
import sys
import types
from collections.abc import Callable
from pathlib import Path

import pytest

from spamandphishingdetection.config import ProjectConfig
from spamandphishingdetection.inference import ArtifactError, Predictor
from spamandphishingdetection.integrity import (
    bind_latest,
    build_transformer_file_manifest,
    hash_regular_file,
    verify_transformer_payload,
)

FIXTURE_FILES = {
    "config.json": b'{"model_type": "distilbert"}',
    "model.safetensors": b"fixture weights",
    "tokenizer.json": b'{"model": {"type": "WordPiece"}}',
    "tokenizer_config.json": b'{"do_lower_case": true}',
}


def _transformer_artifact(root: Path, run_id: str = "run-t") -> tuple[Path, dict]:
    artifact = root / run_id
    model_dir = artifact / "transformer" / "model"
    model_dir.mkdir(parents=True)
    for name, content in FIXTURE_FILES.items():
        (model_dir / name).write_bytes(content)
    manifest = {
        "schema_version": 2,
        "model_version": run_id,
        "backend": "transformer",
        "threshold": 0.5,
        "transformer_files": build_transformer_file_manifest(artifact),
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return artifact, manifest


def test_manifest_is_deterministic_sorted_and_relative(tmp_path: Path) -> None:
    artifact, manifest = _transformer_artifact(tmp_path)

    files = manifest["transformer_files"]
    assert list(files) == sorted(files)
    assert all(path.startswith("transformer/model/") for path in files)
    assert files["transformer/model/model.safetensors"]["bytes"] == len(b"fixture weights")
    assert json.dumps(build_transformer_file_manifest(artifact)) == json.dumps(files)
    assert verify_transformer_payload(artifact, manifest) == artifact / "transformer" / "model"


@pytest.mark.parametrize("name", sorted(FIXTURE_FILES))
def test_modified_files_are_rejected(tmp_path: Path, name: str) -> None:
    artifact, manifest = _transformer_artifact(tmp_path)
    with (artifact / "transformer" / "model" / name).open("ab") as handle:
        handle.write(b" ")

    with pytest.raises(ArtifactError, match="checksum does not match"):
        verify_transformer_payload(artifact, manifest)


def test_missing_and_unrecorded_files_are_rejected(tmp_path: Path) -> None:
    artifact, manifest = _transformer_artifact(tmp_path)
    (artifact / "transformer" / "model" / "tokenizer.json").unlink()
    with pytest.raises(ArtifactError, match="1 missing, 0 unrecorded"):
        verify_transformer_payload(artifact, manifest)

    artifact, manifest = _transformer_artifact(tmp_path, "run-extra")
    (artifact / "transformer" / "model" / ".hidden.json").write_bytes(b"{}")
    with pytest.raises(ArtifactError, match="0 missing, 1 unrecorded"):
        verify_transformer_payload(artifact, manifest)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "/etc/passwd",
        "../outside.json",
        "transformer/model/../../model.joblib",
        "transformer/model/./config.json",
        "transformer\\model\\config.json",
        "C:/Windows/win.ini",
        "transformer/checkpoints/config.json",
        "transformer/model/pytorch_model.bin",
        "transformer/model/modeling_custom.py",
    ],
)
def test_unsafe_manifest_paths_are_rejected(tmp_path: Path, unsafe_path: str) -> None:
    artifact, manifest = _transformer_artifact(tmp_path)
    manifest["transformer_files"][unsafe_path] = {"bytes": 0, "sha256": "0" * 64}

    with pytest.raises(ArtifactError):
        verify_transformer_payload(artifact, manifest)


def test_duplicate_normalized_paths_are_rejected(tmp_path: Path) -> None:
    artifact, manifest = _transformer_artifact(tmp_path)
    record = manifest["transformer_files"]["transformer/model/config.json"]
    manifest["transformer_files"]["transformer/model/CONFIG.json"] = record

    with pytest.raises(ArtifactError, match="duplicate normalized"):
        verify_transformer_payload(artifact, manifest)


def test_symlinks_are_rejected(tmp_path: Path) -> None:
    artifact, manifest = _transformer_artifact(tmp_path)
    target = tmp_path / "outside.json"
    target.write_bytes(b"{}")
    try:
        os.symlink(target, artifact / "transformer" / "model" / "linked.json")
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this platform")

    with pytest.raises(ArtifactError, match="link or special file"):
        verify_transformer_payload(artifact, manifest)
    with pytest.raises(ArtifactError, match="link or special file"):
        build_transformer_file_manifest(artifact)


def _write_shard_files(artifact: Path, weight_map: dict[str, str]) -> None:
    model_dir = artifact / "transformer" / "model"
    (model_dir / "model.safetensors").rename(model_dir / "model-00001-of-00001.safetensors")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}),
        encoding="utf-8",
    )


def test_sharded_index_inside_verified_payload_is_accepted(tmp_path: Path) -> None:
    artifact, manifest = _transformer_artifact(tmp_path)
    _write_shard_files(artifact, {"layer.weight": "model-00001-of-00001.safetensors"})
    manifest["transformer_files"] = build_transformer_file_manifest(artifact)

    verify_transformer_payload(artifact, manifest)


@pytest.mark.parametrize(
    "shard",
    [
        "../../../../etc/passwd",
        "/etc/passwd",
        r"C:\Windows\win.ini",
        "subdir/model-00001-of-00001.safetensors",
        "missing-shard.safetensors",
    ],
)
def test_sharded_index_cannot_escape_the_verified_payload(tmp_path: Path, shard: str) -> None:
    artifact, manifest = _transformer_artifact(tmp_path)
    _write_shard_files(artifact, {"layer.weight": shard})

    with pytest.raises(ArtifactError, match="shard"):
        build_transformer_file_manifest(artifact)

    # A manifest that correctly hashes a crafted index is still refused at load time.
    manifest["transformer_files"] = {
        path.relative_to(artifact).as_posix(): dict(
            zip(("sha256", "bytes"), hash_regular_file(path), strict=True)
        )
        for path in sorted((artifact / "transformer" / "model").iterdir())
    }
    with pytest.raises(ArtifactError, match="shard"):
        verify_transformer_payload(artifact, manifest)


class _RecordingLoader:
    calls: list[tuple[str, Path, dict]] = []

    def __init__(self, name: str) -> None:
        self.name = name

    def from_pretrained(self, path: Path, **options: object) -> object:
        _RecordingLoader.calls.append((self.name, path, options))
        model = types.SimpleNamespace(to=lambda device: None, eval=lambda: None)
        return model


def _install_fake_transformer_stack(monkeypatch: pytest.MonkeyPatch) -> None:
    _RecordingLoader.calls = []
    fake_torch = types.ModuleType("torch")
    fake_torch.device = lambda name: name
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = _RecordingLoader("tokenizer")
    fake_transformers.AutoModelForSequenceClassification = _RecordingLoader("model")
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def test_predictor_loads_transformers_only_after_verification(
    config_factory: Callable[..., ProjectConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = config_factory()
    artifact, _manifest = _transformer_artifact(config.artifacts_dir)
    bind_latest(config, artifact, action="promote")
    _install_fake_transformer_stack(monkeypatch)

    predictor = Predictor.load(config)

    assert predictor.backend == "transformer"
    assert [name for name, _path, _options in _RecordingLoader.calls] == ["tokenizer", "model"]
    for _name, path, options in _RecordingLoader.calls:
        assert path == artifact / "transformer" / "model"
        assert options["local_files_only"] is True
        assert options["trust_remote_code"] is False
    assert _RecordingLoader.calls[1][2]["use_safetensors"] is True

    (artifact / "transformer" / "model" / "model.safetensors").write_bytes(b"tampered")
    _RecordingLoader.calls = []
    with pytest.raises(ArtifactError, match="checksum does not match"):
        Predictor.load(config)
    assert _RecordingLoader.calls == []


def test_ensemble_requires_both_payload_bindings(
    config_factory: Callable[..., ProjectConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = config_factory()
    artifact, manifest = _transformer_artifact(config.artifacts_dir, "run-ensemble")
    manifest["backend"] = "ensemble"
    manifest["model_sha256"] = "0" * 64
    (artifact / "model.joblib").write_bytes(b"not the recorded classical payload")
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    bind_latest(config, artifact, action="promote")
    _install_fake_transformer_stack(monkeypatch)

    with pytest.raises(ArtifactError, match="Classical model checksum"):
        Predictor.load(config)
    assert _RecordingLoader.calls == []


@pytest.mark.slow
def test_real_transformer_round_trip_loads_from_verified_local_files(
    config_factory: Callable[..., ProjectConfig],
) -> None:
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = config_factory()
    artifact = config.artifacts_dir / "run-real"
    model_dir = artifact / "transformer" / "model"
    model_dir.mkdir(parents=True)
    vocabulary = model_dir / "vocab.txt"
    vocabulary.write_text(
        "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "verify", "account", "team"]),
        encoding="utf-8",
    )
    tokenizer = transformers.BertTokenizerFast(vocab_file=str(vocabulary))
    model = transformers.DistilBertForSequenceClassification(
        transformers.DistilBertConfig(
            vocab_size=8,
            dim=16,
            hidden_dim=32,
            n_layers=1,
            n_heads=2,
            max_position_embeddings=64,
            num_labels=2,
        )
    )
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)
    manifest = {
        "schema_version": 2,
        "model_version": "run-real",
        "backend": "transformer",
        "threshold": 0.5,
        "transformer_max_length": 32,
        "transformer_files": build_transformer_file_manifest(artifact),
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    bind_latest(config, artifact, action="promote")

    prediction = Predictor.load(config).predict_text("verify account team")

    assert 0.0 <= prediction.probability <= 1.0
