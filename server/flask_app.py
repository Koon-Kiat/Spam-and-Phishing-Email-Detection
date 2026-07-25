"""Browser host for the email risk-analysis interface."""

from __future__ import annotations

from pathlib import Path

from flask import render_template, send_from_directory

from spamandphishingdetection.api import create_app
from spamandphishingdetection.config import ProjectConfig, load_config

SERVER_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = SERVER_ROOT / "static"


def create_web_app(
    config: ProjectConfig | None = None,
    *,
    artifact: str | Path | None = None,
):
    """Create the browser interface and versioned prediction API."""

    resolved_config = config or load_config()
    application = create_app(
        resolved_config,
        artifact=artifact,
        static_folder=STATIC_ROOT,
        template_folder=SERVER_ROOT / "templates",
    )

    @application.get("/")
    def index():
        return render_template(
            "index.html",
            max_request_bytes=resolved_config.max_request_bytes,
        )

    @application.get("/favicon.ico")
    def favicon():
        return send_from_directory(STATIC_ROOT, "favicon.ico")

    return application


project_config = load_config()
app = create_web_app(project_config)


if __name__ == "__main__":
    app.run(
        host=project_config.host,
        port=project_config.port,
        debug=False,
    )
