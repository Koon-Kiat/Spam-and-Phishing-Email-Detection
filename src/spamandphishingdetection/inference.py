"""Artifact loading and consistent email prediction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any

import numpy as np
from bs4 import BeautifulSoup

from .config import ProjectConfig
from .integrity import (
    ArtifactError,
    load_classical_model,
    load_trusted_manifest,
    resolve_artifact_dir,
    validate_manifest_fields,
)
from .preprocessing import prepare_text
from .training import _head_tail_token_ids, configure_runtime_cache

__all__ = [
    "ArtifactError",
    "Prediction",
    "Predictor",
    "extract_text_from_eml",
    "resolve_artifact_dir",
]


@dataclass(frozen=True)
class Prediction:
    """Stable prediction response used by the CLI and API."""

    label: str
    is_phishing: bool
    probability: float
    threshold: float
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _part_text(part: Any) -> str:
    try:
        content = part.get_content()
    except LookupError, UnicodeError:
        payload = part.get_payload(decode=True) or b""
        content = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
    return content if isinstance(content, str) else str(content)


def extract_text_from_eml(message: bytes) -> str:
    """Extract subject and body text from an RFC 822 message."""

    if not message.strip():
        return ""
    parsed = BytesParser(policy=policy.default).parsebytes(message)
    subject = str(parsed.get("subject", "")).strip()
    plain_parts: list[str] = []
    html_parts: list[str] = []

    parts = parsed.walk() if parsed.is_multipart() else [parsed]
    for part in parts:
        if part.is_multipart() or part.get_content_disposition() == "attachment":
            continue
        content_type = part.get_content_type()
        if content_type == "text/plain":
            plain_parts.append(_part_text(part))
        elif content_type == "text/html":
            html_parts.append(_part_text(part))

    if plain_parts:
        body = "\n".join(plain_parts)
    elif html_parts:
        body = "\n".join(
            BeautifulSoup(html, "html.parser").get_text(" ", strip=True) for html in html_parts
        )
    else:
        body = _part_text(parsed)
    return f"{subject}\n\n{body}" if subject else body


class Predictor:
    """Load one classical, transformer, or ensemble artifact and predict consistently."""

    def __init__(
        self,
        config: ProjectConfig,
        artifact_dir: Path,
        manifest: dict[str, Any],
        *,
        manifest_sha256: str | None = None,
    ) -> None:
        validate_manifest_fields(manifest)
        self.config = config
        self.artifact_dir = artifact_dir
        self.manifest = manifest
        self.manifest_sha256 = manifest_sha256
        self.backend = str(manifest["backend"])
        self.threshold = float(manifest["threshold"])
        self.model_version = str(manifest["model_version"])
        self.max_text_chars = int(manifest.get("max_text_chars", config.max_text_chars))
        self.classical_model = None
        self.transformer_model = None
        self.tokenizer = None
        self.device = None

        if self.backend in {"classical", "ensemble"}:
            self.classical_model = load_classical_model(artifact_dir, manifest)

        if self.backend in {"transformer", "ensemble"}:
            self._load_transformer()

    @classmethod
    def load(
        cls,
        config: ProjectConfig,
        artifact: str | Path | None = None,
        *,
        expected_manifest_sha256: str | None = None,
    ) -> Predictor:
        """Load a model only after its manifest and payload pass integrity checks."""

        trusted = load_trusted_manifest(
            config,
            artifact,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        try:
            return cls(
                config,
                trusted.artifact_dir,
                trusted.manifest,
                manifest_sha256=trusted.sha256,
            )
        except ArtifactError:
            raise
        except (KeyError, TypeError, ValueError, OSError) as error:
            raise ArtifactError("Model artifact could not be loaded") from error

    def _load_transformer(self) -> None:
        configure_runtime_cache(self.config)
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = self.artifact_dir / "transformer" / "model"
        if not model_dir.is_dir():
            raise ArtifactError(f"Transformer model not found: {model_dir}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.transformer_model = AutoModelForSequenceClassification.from_pretrained(
                model_dir
            )
        except (OSError, ValueError) as error:
            raise ArtifactError("Transformer model could not be loaded") from error
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer_model.to(self.device)
        self.transformer_model.eval()

    def _classical_probability(self, text: str) -> float:
        probabilities = np.asarray(self.classical_model.predict_proba([text]), dtype=float)
        return float(probabilities[0, 1])

    def _transformer_probability(self, text: str) -> float:
        import torch

        max_length = int(
            self.manifest.get(
                "transformer_max_length",
                self.config.transformer_max_length,
            )
        )
        input_ids = _head_tail_token_ids(self.tokenizer, text, max_length)
        inputs = self.tokenizer.pad(
            [{"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}],
            return_tensors="pt",
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        with torch.no_grad():
            logits = self.transformer_model(**inputs).logits
            probability = torch.softmax(logits, dim=-1)[0, 1].item()
        return float(probability)

    def predict_text(self, text: str) -> Prediction:
        prepared, _ = prepare_text(text, self.max_text_chars)
        if not prepared:
            raise ValueError("Email text is empty")

        if self.backend == "classical":
            probability = self._classical_probability(prepared)
        elif self.backend == "transformer":
            probability = self._transformer_probability(prepared)
        else:
            ensemble = self.manifest.get("ensemble") or {}
            weight = float(ensemble.get("classical_weight", 0.5))
            probability = weight * self._classical_probability(prepared) + (
                1.0 - weight
            ) * self._transformer_probability(prepared)

        is_phishing = probability >= self.threshold
        return Prediction(
            label="phishing" if is_phishing else "safe",
            is_phishing=is_phishing,
            probability=probability,
            threshold=self.threshold,
            model_version=self.model_version,
        )

    def predict_eml(self, message: bytes) -> Prediction:
        return self.predict_text(extract_text_from_eml(message))
