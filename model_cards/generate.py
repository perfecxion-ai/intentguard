"""Generate model card README.md from template and metadata.

Usage:
    python -m model_cards.generate --vertical finance --output dist/finance/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

TEMPLATE_PATH = Path(__file__).parent / "template.md"


def load_metadata(dist_dir: Path) -> dict:
    """Load export metadata and gating report."""
    metadata = {}

    export_meta = dist_dir / "export_metadata.json"
    if export_meta.exists():
        metadata.update(json.loads(export_meta.read_text()))

    model_path = dist_dir / "model.onnx"
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        metadata["model_size"] = f"{size_mb:.1f}MB"
    else:
        metadata["model_size"] = "N/A"

    return metadata


def load_gating_report(vertical: str) -> dict:
    """Load gating report metrics."""
    report_path = Path(f"evaluation/gating_report_{vertical}.json")
    if not report_path.exists():
        return {}

    report = json.loads(report_path.read_text())
    return {
        "accuracy": report.get("overall_accuracy", "N/A"),
        "adversarial_accuracy": report.get("adversarial_accuracy", "N/A"),
        "p99_latency": report.get("p99_latency_ms", "N/A"),
    }


def load_policy(vertical: str) -> dict:
    """Load policy for display name and core topics."""
    policy_path = Path(f"policies/{vertical}.json")
    if not policy_path.exists():
        return {}

    policy = json.loads(policy_path.read_text())
    core_topics = policy.get("scope", {}).get("core_topics", [])
    return {
        "display_name": policy.get("display_name", vertical.title()),
        "core_topics": ", ".join(core_topics),
    }


def render_template(vertical: str, version: str, dist_dir: Path) -> str:
    """Render model card template with actual metrics."""
    template = TEMPLATE_PATH.read_text()

    metadata = load_metadata(dist_dir)
    gating = load_gating_report(vertical)
    policy_info = load_policy(vertical)

    values = {
        "vertical": vertical,
        "version": version,
        "display_name": policy_info.get("display_name", vertical.title()),
        "core_topics": policy_info.get("core_topics", ""),
        "accuracy": gating.get("accuracy", "N/A"),
        "adversarial_accuracy": gating.get("adversarial_accuracy", "N/A"),
        "p99_latency": gating.get("p99_latency", "N/A"),
        "model_size": metadata.get("model_size", "N/A"),
    }

    result = template
    for key, value in values.items():
        result = result.replace("{{ " + key + " }}", str(value))

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate model card README.md")
    parser.add_argument("--vertical", required=True)
    parser.add_argument("--version", default="1.0")
    parser.add_argument("--output", required=True, help="Output directory (e.g., dist/finance/)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    readme = render_template(args.vertical, args.version, output_dir)
    readme_path = output_dir / "README.md"
    readme_path.write_text(readme)
    logger.info("Model card written to %s", readme_path)


if __name__ == "__main__":
    main()
