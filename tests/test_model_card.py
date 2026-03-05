"""Tests for model card generation."""

import tempfile
from pathlib import Path

from model_cards.generate import render_template


class TestModelCardGeneration:
    def test_renders_valid_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = render_template("finance", "1.0", Path(tmpdir))
            assert "# IntentGuard" in result
            assert "finance" in result
            assert "Financial Services" in result

    def test_contains_yaml_frontmatter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = render_template("finance", "1.0", Path(tmpdir))
            assert result.startswith("---")
            assert "library_name: onnx" in result
            assert "pipeline_tag: text-classification" in result

    def test_contains_usage_section(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = render_template("finance", "1.0", Path(tmpdir))
            assert "## Usage" in result
            assert "onnxruntime" in result
            assert "docker" in result.lower()

    def test_replaces_vertical_placeholder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = render_template("healthcare", "2.0", Path(tmpdir))
            assert "healthcare" in result
            assert "{{ vertical }}" not in result

    def test_replaces_version_placeholder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = render_template("legal", "3.0", Path(tmpdir))
            assert "3.0" in result
            assert "{{ version }}" not in result

    def test_handles_missing_metadata(self):
        """Should render with N/A when no metadata files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = render_template("finance", "1.0", Path(tmpdir))
            assert "N/A" in result  # metrics not available
