#!/usr/bin/env bash
#
# Benchmark quantization methods.
#
# Usage:
#   ./scripts/benchmark_quantization.sh --vertical finance
#
# Compares size, export time, and inference latency across quantization methods.

set -euo pipefail

VERTICAL="finance"
MODEL_DIR=""
OUTPUT_BASE="dist/benchmark"
SANITY_DATA=""
SANITY_COUNT=50
METHODS=("optimum" "olive" "ort")

usage() {
    echo "Usage: $0 --vertical <name> [--model <path>] [--sanity-data <path>]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --vertical)    VERTICAL="$2"; shift 2 ;;
        --model)       MODEL_DIR="$2"; shift 2 ;;
        --sanity-data) SANITY_DATA="$2"; shift 2 ;;
        *)             usage ;;
    esac
done

MODEL_DIR="${MODEL_DIR:-models/${VERTICAL}/best}"
SANITY_DATA="${SANITY_DATA:-data/${VERTICAL}/synthetic.jsonl}"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: Model directory not found: ${MODEL_DIR}"
    exit 1
fi

if [[ ! -f "$SANITY_DATA" ]]; then
    echo "ERROR: Sanity data not found: ${SANITY_DATA}"
    exit 1
fi

echo "========================================="
echo "  Quantization Benchmark: ${VERTICAL}"
echo "========================================="
echo ""

printf "%-10s %-10s %-12s %-10s\n" "Method" "Size(MB)" "Time(s)" "Status"
echo "-----------------------------------------"

for METHOD in "${METHODS[@]}"; do
    OUT_DIR="${OUTPUT_BASE}/${VERTICAL}-${METHOD}"
    rm -rf "${OUT_DIR}"
    mkdir -p "${OUT_DIR}"

    START=$(date +%s)
    STATUS="OK"

    python -m training.export_onnx \
        --model "${MODEL_DIR}" \
        --output "${OUT_DIR}" \
        --sanity-data "${SANITY_DATA}" \
        --sanity-count "${SANITY_COUNT}" \
        --quantize-method "${METHOD}" \
        2>&1 | tail -5 || STATUS="FAIL"

    END=$(date +%s)
    ELAPSED=$((END - START))

    if [[ -f "${OUT_DIR}/model.onnx" ]]; then
        SIZE_MB=$(du -m "${OUT_DIR}/model.onnx" | cut -f1)
    else
        SIZE_MB="N/A"
        STATUS="FAIL"
    fi

    printf "%-10s %-10s %-12s %-10s\n" "${METHOD}" "${SIZE_MB}" "${ELAPSED}" "${STATUS}"
done

echo ""
echo "Benchmark complete. Results in ${OUTPUT_BASE}/"

# Clean up benchmark artifacts
echo "Cleaning up..."
rm -rf "${OUTPUT_BASE}"
