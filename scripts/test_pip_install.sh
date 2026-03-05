#!/usr/bin/env bash
#
# Test that the package installs cleanly in a fresh venv.
#
# Usage:
#   ./scripts/test_pip_install.sh

set -euo pipefail

VENV_DIR=$(mktemp -d)/ig-test-venv

echo "Creating fresh venv at ${VENV_DIR}..."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

echo "Installing intentguard..."
pip install -e ".[dev]" --quiet

echo "Verifying import..."
python -c "
from intentguard.schema import ClassifyRequest, ClassifyResponse, Decision
from intentguard.policy import Policy, PolicySpec
from intentguard.normalize import normalize, has_encoding_tricks
from intentguard.config import Settings, load_settings
print('All imports successful')
"

echo "Running tests..."
pytest tests/ -v --tb=short

echo "Cleaning up..."
deactivate
rm -rf "${VENV_DIR}"

echo ""
echo "PASS: Package installs and tests pass in a clean environment."
