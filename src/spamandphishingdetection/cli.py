"""Command-line interface for auditing, training, prediction, and serving."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

from .api import create_app
from .config import DEFAULT_CONFIG_PATH, load_config
from .datasets import load_and_audit_datasets, write_audit_report
from .generalization import generate_generalization_report
from .inference import Predictor
from .monitoring import monitor_batch
from .provenance import fetch_data
from .release import package_release, promote_artifact
from .reporting import DEFAULT_LEARNING_FRACTIONS, generate_diagnostic_report
from .training import run_training


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to the repository-relative JSON configuration.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spam-phishing",
        description="Train and serve the spam/phishing email classifier.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable detailed logging.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser(
        "fetch-data",
        help="Download exact, hash-verified dataset versions.",
    )
    _add_config_argument(fetch_parser)
    fetch_parser.add_argument(
        "--accept-licenses",
        action="store_true",
        help="Confirm that you reviewed and accept each source dataset license.",
    )
    fetch_parser.add_argument(
        "--external",
        action="store_true",
        help="Also fetch the blind SpaPhish v5 external corpus.",
    )

    audit_parser = subparsers.add_parser("audit", help="Audit and clean both datasets.")
    _add_config_argument(audit_parser)

    train_parser = subparsers.add_parser("train", help="Benchmark models and save the winner.")
    _add_config_argument(train_parser)
    train_parser.add_argument(
        "--skip-transformer",
        action="store_true",
        help="Run only classical candidates.",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict one EML file.")
    _add_config_argument(predict_parser)
    predict_parser.add_argument("--file", required=True, help="Path to an RFC 822 EML file.")
    predict_parser.add_argument(
        "--artifact",
        help="Explicit artifact directory; defaults to artifacts/latest.json.",
    )

    report_parser = subparsers.add_parser(
        "report",
        help="Generate learning curves and evaluation charts.",
    )
    _add_config_argument(report_parser)
    report_parser.add_argument(
        "--artifact",
        help="Explicit artifact directory; defaults to artifacts/latest.json.",
    )

    generalization_parser = subparsers.add_parser(
        "generalization-report",
        help="Generate grouped, calibration, drift, robustness, and external evidence.",
    )
    _add_config_argument(generalization_parser)
    generalization_parser.add_argument("--artifact")
    generalization_parser.add_argument("--external-path")
    generalization_parser.add_argument(
        "--skip-external",
        action="store_true",
        help="Generate internal diagnostics only; release evidence requires external evaluation.",
    )

    monitor_parser = subparsers.add_parser(
        "monitor",
        help="Measure content-free batch score, slice, and performance drift.",
    )
    _add_config_argument(monitor_parser)
    monitor_parser.add_argument("--input", required=True, help="Input JSONL batch.")
    monitor_parser.add_argument("--artifact")
    monitor_parser.add_argument("--output", help="Optional JSON report path.")

    promote_parser = subparsers.add_parser(
        "promote",
        help="Atomically promote a validated artifact or roll back.",
    )
    _add_config_argument(promote_parser)
    promotion_target = promote_parser.add_mutually_exclusive_group(required=True)
    promotion_target.add_argument("--artifact", help="Run ID or artifact path.")
    promotion_target.add_argument("--rollback", action="store_true")

    package_parser = subparsers.add_parser(
        "package-release",
        help="Build a verified model-only GitHub Release archive.",
    )
    _add_config_argument(package_parser)
    package_parser.add_argument("--artifact", required=True)
    package_parser.add_argument("--version", default="v1.0.0")
    package_parser.add_argument("--output", default="release")
    report_parser.add_argument(
        "--output",
        default="docs/images",
        help="Directory for generated SVG charts.",
    )
    report_parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=list(DEFAULT_LEARNING_FRACTIONS),
        help="Training fractions used for the learning curve.",
    )

    serve_parser = subparsers.add_parser("serve", help="Run the Flask prediction API.")
    _add_config_argument(serve_parser)
    serve_parser.add_argument(
        "--artifact",
        help="Explicit artifact directory; defaults to artifacts/latest.json.",
    )
    serve_parser.add_argument("--host", help="Override the configured host.")
    serve_parser.add_argument("--port", type=int, help="Override the configured port.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)

    if args.command == "fetch-data":
        outcomes = fetch_data(
            config.project_root,
            accept_licenses=args.accept_licenses,
            include_external=args.external,
        )
        print(json.dumps({"datasets": outcomes}, indent=2))
        return 0

    if args.command == "audit":
        audited = load_and_audit_datasets(config)
        report_path = write_audit_report(audited, config)
        print(json.dumps({"report": str(report_path), **audited.audit["clean"]}, indent=2))
        return 0

    if args.command == "train":
        result = run_training(
            config,
            include_transformer=not args.skip_transformer,
        )
        print(
            json.dumps(
                {
                    "artifact": str(result.artifact_dir),
                    "report": str(result.report_dir),
                    "selected_model": result.selected_model,
                    "test_metrics": result.test_metrics,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "predict":
        predictor = Predictor.load(config, args.artifact)
        message = Path(args.file).read_bytes()
        print(json.dumps(predictor.predict_eml(message).to_dict(), indent=2))
        return 0

    if args.command == "report":
        result = generate_diagnostic_report(
            config,
            artifact=args.artifact,
            output_dir=args.output,
            fractions=tuple(args.fractions),
        )
        print(
            json.dumps(
                {
                    "run_id": result.run_id,
                    "learning_curve": str(result.learning_curve_csv),
                    "charts": [str(path) for path in result.charts],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "generalization-report":
        path = generate_generalization_report(
            config,
            artifact=args.artifact,
            external_path=args.external_path,
            include_external=not args.skip_external,
        )
        print(json.dumps({"report": str(path)}, indent=2))
        return 0

    if args.command == "monitor":
        report = monitor_batch(
            config,
            Path(args.input),
            artifact=args.artifact,
        )
        rendered = json.dumps(report, indent=2)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
        return 0

    if args.command == "promote":
        result = promote_artifact(
            config,
            artifact=args.artifact,
            rollback=args.rollback,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "package-release":
        result = package_release(
            config,
            artifact=args.artifact,
            version=args.version,
            output_dir=args.output,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "serve":
        app = create_app(config, artifact=args.artifact)
        app.run(
            host=args.host or config.host,
            port=args.port or config.port,
            debug=False,
        )
        return 0
    raise RuntimeError(f"Unhandled command: {args.command}")
