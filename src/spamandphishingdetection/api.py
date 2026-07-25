"""Flask prediction API."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

from .config import DEFAULT_CONFIG_PATH, ProjectConfig, load_config
from .inference import ArtifactError, Predictor, extract_text_from_eml

LOGGER = logging.getLogger(__name__)


def create_app(
    config: ProjectConfig | None = None,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    artifact: str | Path | None = None,
    static_folder: str | Path | None = None,
    template_folder: str | Path | None = None,
) -> Flask:
    """Create a Flask application and eagerly validate its model artifact."""

    project_config = config or load_config(config_path)
    app = Flask(
        __name__,
        static_folder=str(static_folder) if static_folder else None,
        template_folder=str(template_folder) if template_folder else None,
    )
    app.config["MAX_CONTENT_LENGTH"] = project_config.max_request_bytes
    predictor: Predictor | None = None
    load_error: str | None = None
    try:
        predictor = Predictor.load(project_config, artifact)
    except ArtifactError as error:
        load_error = str(error)
        LOGGER.warning("Prediction model is unavailable: %s", error)

    def require_predictor() -> Predictor:
        if predictor is None:
            raise ArtifactError(load_error or "Prediction model is unavailable")
        return predictor

    def request_text() -> str:
        if request.is_json:
            payload: dict[str, Any] | None = request.get_json(silent=True)
            if not isinstance(payload, dict):
                raise ValueError("JSON object required")
            text = str(payload.get("text", "")).strip()
            subject = str(payload.get("subject", "")).strip()
            return f"{subject}\n\n{text}" if subject else text
        return extract_text_from_eml(request.get_data(cache=False))

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error: RequestEntityTooLarge) -> tuple[Response, int]:
        return jsonify({"error": "Request body exceeds the configured size limit"}), 413

    @app.get("/health")
    def health() -> tuple[Response, int]:
        if predictor is None:
            return (
                jsonify(
                    {
                        "status": "unavailable",
                        "model_loaded": False,
                        "error": load_error,
                    }
                ),
                503,
            )
        return (
            jsonify(
                {
                    "status": "ok",
                    "model_loaded": True,
                    "model_version": predictor.model_version,
                }
            ),
            200,
        )

    @app.post("/api/v1/predict")
    def predict() -> tuple[Response, int]:
        try:
            model = require_predictor()
            text = request_text()
            if not text.strip():
                return jsonify({"error": "Email text is required"}), 400
            result = model.predict_text(text)
            return jsonify(result.to_dict()), 200
        except ArtifactError as error:
            return jsonify({"error": str(error)}), 503
        except (TypeError, ValueError) as error:
            return jsonify({"error": str(error)}), 400

    return app
