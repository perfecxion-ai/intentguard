#!/usr/bin/env bash
#
# Package a vertical for customer delivery.
#
# Usage:
#   ./package.sh --vertical finance --version 1.0.0
#
# Outputs:
#   dist/intentguard-finance-1.0.0.tar.gz
#   Contains: Docker image, policy.json, deployment README, gating report

set -euo pipefail

VERTICAL=""
VERSION=""
DIST_DIR="dist"
DOCKER_REPO="perfecxion/intentguard"

usage() {
    echo "Usage: $0 --vertical <name> --version <semver>"
    echo ""
    echo "Options:"
    echo "  --vertical   Vertical name (e.g., finance, healthcare, legal)"
    echo "  --version    Semantic version (e.g., 1.0.0)"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --vertical) VERTICAL="$2"; shift 2 ;;
        --version)  VERSION="$2"; shift 2 ;;
        *)          usage ;;
    esac
done

if [[ -z "$VERTICAL" || -z "$VERSION" ]]; then
    usage
fi

TAG="${DOCKER_REPO}:${VERTICAL}-${VERSION}"
PACKAGE_DIR="${DIST_DIR}/intentguard-${VERTICAL}-${VERSION}"
MODEL_DIR="${DIST_DIR}/${VERTICAL}"
POLICY_FILE="policies/${VERTICAL}.json"
GATING_REPORT="evaluation/gating_report_${VERTICAL}.json"

echo "=== Packaging IntentGuard ==="
echo "  Vertical: ${VERTICAL}"
echo "  Version:  ${VERSION}"
echo "  Tag:      ${TAG}"
echo ""

# Check prerequisites
if [[ ! -f "$POLICY_FILE" ]]; then
    echo "ERROR: Policy file not found: ${POLICY_FILE}"
    exit 1
fi

if [[ ! -f "${MODEL_DIR}/model.onnx" ]]; then
    echo "ERROR: ONNX model not found: ${MODEL_DIR}/model.onnx"
    echo "Run the training pipeline first: make -f training/Makefile train-vertical VERTICAL=${VERTICAL}"
    exit 1
fi

if [[ ! -d "${MODEL_DIR}/tokenizer" ]]; then
    echo "ERROR: Tokenizer not found: ${MODEL_DIR}/tokenizer"
    exit 1
fi

# Create package directory
rm -rf "${PACKAGE_DIR}"
mkdir -p "${PACKAGE_DIR}"

# Copy model artifacts
echo "Copying model artifacts..."
mkdir -p "${PACKAGE_DIR}/models"
cp "${MODEL_DIR}/model.onnx" "${PACKAGE_DIR}/models/"
cp -r "${MODEL_DIR}/tokenizer" "${PACKAGE_DIR}/models/"
if [[ -f "${MODEL_DIR}/calibration_params.json" ]]; then
    cp "${MODEL_DIR}/calibration_params.json" "${PACKAGE_DIR}/models/"
fi

# Copy policy
echo "Copying policy..."
mkdir -p "${PACKAGE_DIR}/policies"
cp "${POLICY_FILE}" "${PACKAGE_DIR}/policies/"

# Copy gating report if available
if [[ -f "$GATING_REPORT" ]]; then
    echo "Copying gating report..."
    cp "${GATING_REPORT}" "${PACKAGE_DIR}/"
fi

# Build Docker image
echo "Building Docker image: ${TAG}..."
docker build \
    --build-arg VERTICAL="${VERTICAL}" \
    -t "${TAG}" \
    -t "${DOCKER_REPO}:${VERTICAL}-latest" \
    .

# Save Docker image
echo "Saving Docker image..."
docker save "${TAG}" | gzip > "${PACKAGE_DIR}/intentguard-${VERTICAL}-${VERSION}.docker.tar.gz"

# Generate deployment README
cat > "${PACKAGE_DIR}/DEPLOYMENT.md" << DEPLOY_EOF
# IntentGuard ${VERTICAL} v${VERSION}

## Quick Start

    docker load < intentguard-${VERTICAL}-${VERSION}.docker.tar.gz
    docker run -p 8080:8080 ${TAG}

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| PORT | 8080 | Server port |
| POLICY_PATH | policies/${VERTICAL}.json | Policy config path |
| MODEL_PATH | models/model.onnx | ONNX model path |
| TOKENIZER_PATH | models/tokenizer | Tokenizer path |
| CALIBRATION_PATH | models/calibration_params.json | Calibration params |
| DOWNSTREAM_URL | (none) | LLM URL for proxy mode |
| DOWNSTREAM_API_KEY | (none) | API key for downstream LLM |
| LOG_QUERY_TEXT | false | Log query text (privacy) |
| DEBUG | false | Show probabilities in responses |
| ORT_INTRA_OP_THREADS | 4 | ONNX Runtime intra-op threads |
| ORT_INTER_OP_THREADS | 1 | ONNX Runtime inter-op threads |

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /v1/classify | POST | Classify a message (sidecar mode) |
| /v1/chat/completions | POST | OpenAI-compatible proxy mode |
| /v1/models | GET | List models |
| /health | GET | Health check |

## Sidecar Mode (Recommended)

    curl -X POST http://localhost:8080/v1/classify \\
      -H "Content-Type: application/json" \\
      -d '{"messages": [{"role": "user", "content": "What are mortgage rates?"}]}'

## Proxy Mode

    docker run -p 8080:8080 \\
      -e DOWNSTREAM_URL=https://api.openai.com/v1/chat/completions \\
      -e DOWNSTREAM_API_KEY=sk-... \\
      ${TAG}

## Hardware Requirements

- CPU: 4+ cores (x86_64 recommended)
- RAM: 2GB minimum
- Disk: 500MB for container image
- Latency target: p99 < 50ms on 4-core CPU

## Policy Customization

Edit policies/${VERTICAL}.json to change:
- Refusal messages (no retrain needed)
- Confidence thresholds (no retrain needed)
- Topic scope (requires retrain)
DEPLOY_EOF

# Create tarball
echo "Creating package tarball..."
cd "${DIST_DIR}"
tar -czf "intentguard-${VERTICAL}-${VERSION}.tar.gz" "intentguard-${VERTICAL}-${VERSION}/"
cd ..

# Summary
TARBALL="${DIST_DIR}/intentguard-${VERTICAL}-${VERSION}.tar.gz"
SIZE=$(du -sh "${TARBALL}" | cut -f1)

echo ""
echo "=== Package Complete ==="
echo "  Tarball: ${TARBALL} (${SIZE})"
echo "  Docker:  ${TAG}"
echo "  Contents:"
ls -la "${PACKAGE_DIR}/"
echo ""
echo "To test:"
echo "  docker run -p 8080:8080 ${TAG}"
echo "  curl http://localhost:8080/health"
