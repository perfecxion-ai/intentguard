"""Publish model artifacts to HuggingFace Hub.

Usage:
    python scripts/publish_huggingface.py \
        --vertical finance \
        --version 1.0 \
        --repo-prefix perfecXion/intentguard
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_model_card(vertical: str, version: str, dist_dir: Path):
    """Generate model card before upload."""
    subprocess.run(
        [
            sys.executable, "-m", "model_cards.generate",
            "--vertical", vertical,
            "--version", version,
            "--output", str(dist_dir),
        ],
        check=True,
    )


def publish(vertical: str, version: str, repo_prefix: str, dist_dir: Path):
    """Upload model artifacts to HuggingFace."""
    from huggingface_hub import HfApi

    repo_id = f"{repo_prefix}-{vertical}"
    api = HfApi()

    logger.info("Creating/updating repo: %s", repo_id)
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")

    logger.info("Uploading %s to %s", dist_dir, repo_id)
    api.upload_folder(
        folder_path=str(dist_dir),
        repo_id=repo_id,
        commit_message=f"Release v{version}",
    )

    logger.info("Published: https://huggingface.co/%s", repo_id)


def main():
    parser = argparse.ArgumentParser(description="Publish model to HuggingFace Hub")
    parser.add_argument("--vertical", required=True)
    parser.add_argument("--version", default="1.0")
    parser.add_argument("--repo-prefix", default="perfecXion/intentguard",
                        help="HuggingFace repo prefix (default: perfecXion/intentguard)")
    parser.add_argument("--dist-dir", default=None,
                        help="Path to dist directory (default: dist/{vertical}/)")
    parser.add_argument("--skip-card", action="store_true",
                        help="Skip model card generation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dist_dir = Path(args.dist_dir) if args.dist_dir else Path(f"dist/{args.vertical}")
    if not dist_dir.exists():
        logger.error("Dist directory not found: %s", dist_dir)
        sys.exit(1)

    if not args.skip_card:
        logger.info("Generating model card...")
        generate_model_card(args.vertical, args.version, dist_dir)

    publish(args.vertical, args.version, args.repo_prefix, dist_dir)


if __name__ == "__main__":
    main()
