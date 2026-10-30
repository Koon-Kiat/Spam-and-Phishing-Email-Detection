from __future__ import annotations

import json
import os
import random
import struct
from pathlib import Path

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from spamandphishingdetection import privacy
from spamandphishingdetection.preprocessing import prepare_text
from spamandphishingdetection.privacy import (
    MAX_RECORDED_FINDINGS,
    PrivacyScanError,
    scan_model_artifact,
)

RAW_SENTENCE = "Please review the attached invoice and confirm your payment today"


def _write(root: Path, relative: str, payload: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _safetensors(metadata: dict[str, str], tensor_bytes: bytes) -> bytes:
    header = {
        "__metadata__": metadata,
        "classifier.weight": {
            "dtype": "F32",
            "shape": [2, 768],
            "data_offsets": [0, 1234567890],
        },
    }
    encoded = json.dumps(header).encode()
    return struct.pack("<Q", len(encoded)) + encoded + tensor_bytes


def _classical_payload(root: Path) -> None:
    texts = [
        prepare_text(text, 1_000)[0]
        for text in (
            "URGENT verify your account at http://login.example-bank.com/reset?token=abc now",
            "Contact jane.doe@example.com or call +1 (555) 123-4567 about the invoice",
            "Team lunch agenda for Friday and the approved project notes",
            "Quarterly planning meeting moved to room four",
        )
    ]
    model = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(ngram_range=(1, 2))),
                        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))),
                    ]
                ),
            ),
            ("classifier", LogisticRegression()),
        ]
    ).fit(texts, [1, 1, 0, 0])
    joblib.dump(model, root / "model.joblib")


def _transformer_payload(root: Path) -> None:
    model_dir = "transformer/model"
    _write(root, f"{model_dir}/config.json", b'{"model_type": "distilbert", "dim": 768}')
    _write(
        root,
        f"{model_dir}/tokenizer_config.json",
        b'{"model_max_length": 1000000000000000019884624838656, "do_lower_case": true}',
    )
    vocabulary = ["[PAD]", "[UNK]", "the", "verify", "account", "invoice", "##ing", "1990"]
    _write(
        root,
        f"{model_dir}/tokenizer.json",
        json.dumps({"model": {"vocab": {word: i for i, word in enumerate(vocabulary)}}}).encode(),
    )
    _write(root, f"{model_dir}/vocab.txt", "\n".join(vocabulary).encode())
    _write(
        root,
        f"{model_dir}/model.safetensors",
        _safetensors({"format": "pt"}, random.Random(0).randbytes(256 * 1024)),
    )


@pytest.mark.parametrize("layout", ["classical", "transformer", "ensemble"])
def test_legitimate_artifact_layouts_pass(tmp_path: Path, layout: str) -> None:
    if layout in {"classical", "ensemble"}:
        _classical_payload(tmp_path)
    if layout in {"transformer", "ensemble"}:
        _transformer_payload(tmp_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"recovery": {"reason": RAW_SENTENCE}}),
        encoding="utf-8",
    )

    result = scan_model_artifact(tmp_path)

    assert result["status"] == "passed"
    assert set(result["finding_counts"].values()) == {0}
    assert result["findings"] == []


@pytest.mark.parametrize(
    ("category", "content"),
    [
        ("natural_language", RAW_SENTENCE),
        ("url", "visit https://login.example-bank.com/verify"),
        ("url", "visit www.example-bank.com today"),
        ("phone_number", "call +1 (555) 123-4567"),
        ("street_address", "deliver to 221 Baker Street"),
        ("credential_assignment", "password=hunter22"),
        ("credential_assignment", '"api_key": "sk_live_abcdef"'),
        ("iban", "GB82WEST12345698765432"),
        ("payment_card", "4111 1111 1111 1111"),
        ("email_address", "jane.doe@example.com"),
        ("long_numeric_identifier", "account 123456789012"),
    ],
)
def test_raw_message_content_is_detected(tmp_path: Path, category: str, content: str) -> None:
    joblib.dump({"cached": content}, tmp_path / "model.joblib")

    with pytest.raises(PrivacyScanError) as raised:
        scan_model_artifact(tmp_path)

    assert raised.value.result["finding_counts"][category] >= 1
    assert raised.value.result["status"] == "failed"


@pytest.mark.parametrize(
    ("relative", "payload"),
    [
        ("transformer/model/vocab.txt", RAW_SENTENCE.encode()),
        ("transformer/model/config.json", json.dumps({"note": RAW_SENTENCE}).encode()),
        (
            "transformer/model/model.safetensors",
            _safetensors({"source": RAW_SENTENCE}, b"\x00" * 64),
        ),
        ("transformer/model/model.safetensors", _safetensors({}, RAW_SENTENCE.encode())),
    ],
)
def test_text_json_and_safetensors_content_is_detected(
    tmp_path: Path,
    relative: str,
    payload: bytes,
) -> None:
    _write(tmp_path, relative, payload)

    with pytest.raises(PrivacyScanError) as raised:
        scan_model_artifact(tmp_path)

    assert raised.value.result["finding_counts"]["natural_language"] == 1


def test_long_printable_runs_fail_only_in_binary_payloads(tmp_path: Path) -> None:
    _write(tmp_path, "transformer/model/vocab.txt", b"a" * 1_000)
    assert scan_model_artifact(tmp_path)["status"] == "passed"

    _write(tmp_path, "model.joblib", b"\x00" + b"A" * 300 + b"\x00")
    with pytest.raises(PrivacyScanError) as raised:
        scan_model_artifact(tmp_path, chunk_bytes=64)

    result = raised.value.result
    assert result["finding_counts"]["binary_printable_run"] == 1
    assert result["longest_printable_run"] == 300
    assert result["findings"] == [
        {"category": "binary_printable_run", "offset": 1, "path": "model.joblib"}
    ]


@pytest.mark.parametrize(
    ("category", "secret"),
    [
        ("url", b"https://login.example-bank.com"),
        ("email_address", b"jane.doe@example.com"),
        ("natural_language", RAW_SENTENCE.encode()),
    ],
)
def test_matches_split_across_chunk_boundaries_are_found_once(
    tmp_path: Path,
    category: str,
    secret: bytes,
) -> None:
    _write(tmp_path, "model.joblib", b"\x00" * 37 + secret + b"\x00" * 41)

    for chunk_bytes in (1, 7, 16, 33, 64, 4096):
        with pytest.raises(PrivacyScanError) as raised:
            scan_model_artifact(tmp_path, chunk_bytes=chunk_bytes)
        result = raised.value.result
        assert result["finding_counts"][category] == 1, chunk_bytes
        located = [item for item in result["findings"] if item["category"] == category]
        assert located == [{"category": category, "offset": 37, "path": "model.joblib"}]


def test_large_sparse_files_are_scanned_in_bounded_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    size = 32 * 1024 * 1024
    secret = b"https://login.example-bank.com"
    path = tmp_path / "model.joblib"
    with path.open("wb") as handle:
        handle.truncate(size - len(secret) - 10)
        handle.seek(size - len(secret) - 10)
        handle.write(secret + b"\x00" * 10)

    read_sizes: list[int] = []
    real_open = privacy.open_regular_file

    class RecordingHandle:
        def __init__(self, handle):
            self.handle = handle

        def read(self, size: int = -1) -> bytes:
            read_sizes.append(size)
            return self.handle.read(size)

        def __enter__(self):
            return self

        def __exit__(self, *_exc: object) -> None:
            self.handle.close()

    monkeypatch.setattr(privacy, "open_regular_file", lambda p: RecordingHandle(real_open(p)))
    chunk_bytes = 64 * 1024

    with pytest.raises(PrivacyScanError) as raised:
        scan_model_artifact(tmp_path, chunk_bytes=chunk_bytes)

    result = raised.value.result
    assert result["bytes_scanned"] == size
    assert result["findings"] == [
        {"category": "url", "offset": size - len(secret) - 10, "path": "model.joblib"}
    ]
    assert read_sizes and all(0 < size <= chunk_bytes for size in read_sizes)


def test_failure_evidence_is_bounded_and_redacted(tmp_path: Path) -> None:
    secrets = [f"person{index}@example.com" for index in range(80)]
    joblib.dump({"emails": secrets, "note": RAW_SENTENCE}, tmp_path / "model.joblib")

    with pytest.raises(PrivacyScanError) as raised:
        scan_model_artifact(tmp_path)

    result = raised.value.result
    rendered = json.dumps(result) + str(raised.value)
    assert result["finding_counts"]["email_address"] == 80
    assert len(result["findings"]) == MAX_RECORDED_FINDINGS
    assert result["findings_truncated"] is True
    assert "email_address=80" in str(raised.value)
    assert "example.com" not in rendered
    assert "invoice" not in rendered
    assert all(set(item) == {"category", "offset", "path"} for item in result["findings"])


def test_hidden_files_are_scanned_not_skipped(tmp_path: Path) -> None:
    _write(tmp_path, ".cache/notes.txt", b"contact jane.doe@example.com")

    with pytest.raises(PrivacyScanError) as raised:
        scan_model_artifact(tmp_path)

    assert raised.value.result["hidden_files_scanned"] == 1
    assert raised.value.result["findings"][0]["path"] == ".cache/notes.txt"


def test_malformed_structured_files_fail_closed(tmp_path: Path) -> None:
    _write(tmp_path, "transformer/model/config.json", b"{not json")
    _write(tmp_path, "transformer/model/model.safetensors", struct.pack("<Q", 10**12) + b"{}")

    with pytest.raises(PrivacyScanError) as raised:
        scan_model_artifact(tmp_path)

    assert raised.value.result["finding_counts"]["malformed_structured_file"] == 2


def test_symlinks_are_reported_without_being_followed(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    _write(outside, "secret.txt", b"contact jane.doe@example.com")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    try:
        os.symlink(outside / "secret.txt", artifact / "model.joblib")
        os.symlink(outside, artifact / "transformer", target_is_directory=True)
    except OSError:
        pytest.skip("Creating symlinks is not permitted on this platform")

    with pytest.raises(PrivacyScanError) as raised:
        scan_model_artifact(artifact)

    result = raised.value.result
    assert result["finding_counts"]["link"] == 2
    assert result["finding_counts"]["email_address"] == 0


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="Named pipes require POSIX")
def test_named_pipes_are_rejected_without_blocking(tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "model.joblib")

    with pytest.raises(PrivacyScanError) as raised:
        scan_model_artifact(tmp_path)

    assert raised.value.result["finding_counts"]["special_file"] == 1
