#!/usr/bin/env bash
#
# Test Docker container for each vertical.
#
# Usage:
#   ./scripts/test_docker.sh
#
# Tests all endpoints, measures image size and startup time,
# prints a pass/fail summary table.

set -euo pipefail

DOCKER_REPO="perfecxion/intentguard"
VERTICALS=("finance" "healthcare" "legal")
PORT=8081
MAX_STARTUP_SECONDS=5
MAX_IMAGE_MB=500
PASS_COUNT=0
FAIL_COUNT=0
RESULTS=()

log()  { echo "[test_docker] $*"; }
pass() { PASS_COUNT=$((PASS_COUNT + 1)); RESULTS+=("PASS  $1"); log "PASS: $1"; }
fail() { FAIL_COUNT=$((FAIL_COUNT + 1)); RESULTS+=("FAIL  $1"); log "FAIL: $1"; }

wait_for_health() {
    local url="http://localhost:${PORT}/health"
    local max_wait=$1
    local start
    start=$(date +%s)

    while true; do
        if curl -sf "$url" > /dev/null 2>&1; then
            return 0
        fi
        local elapsed=$(( $(date +%s) - start ))
        if (( elapsed >= max_wait )); then
            return 1
        fi
        sleep 0.5
    done
}

test_endpoint() {
    local method=$1 url=$2 label=$3 expected_status=${4:-200}
    shift 4 || true
    local extra_args=("$@")

    local status
    status=$(curl -s -o /dev/null -w "%{http_code}" "${extra_args[@]}" "$url")
    if [[ "$status" == "$expected_status" ]]; then
        pass "$label (HTTP $status)"
    else
        fail "$label (expected $expected_status, got $status)"
    fi
}

for VERTICAL in "${VERTICALS[@]}"; do
    TAG="${DOCKER_REPO}:${VERTICAL}-test"
    CONTAINER_NAME="ig-test-${VERTICAL}"

    log "========================================="
    log "Testing vertical: ${VERTICAL}"
    log "========================================="

    # Build image
    log "Building image: ${TAG}..."
    docker build --build-arg "VERTICAL=${VERTICAL}" -t "${TAG}" -q . || {
        fail "${VERTICAL}: docker build"
        continue
    }
    pass "${VERTICAL}: docker build"

    # Check image size
    IMAGE_SIZE_MB=$(docker image inspect "${TAG}" --format='{{.Size}}' | awk '{printf "%.0f", $1/1024/1024}')
    log "Image size: ${IMAGE_SIZE_MB}MB (max ${MAX_IMAGE_MB}MB)"
    if (( IMAGE_SIZE_MB <= MAX_IMAGE_MB )); then
        pass "${VERTICAL}: image size ${IMAGE_SIZE_MB}MB <= ${MAX_IMAGE_MB}MB"
    else
        fail "${VERTICAL}: image size ${IMAGE_SIZE_MB}MB > ${MAX_IMAGE_MB}MB"
    fi

    # Start container and measure startup time
    log "Starting container..."
    docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
    START_TIME=$(date +%s%N)
    docker run -d --name "${CONTAINER_NAME}" -p "${PORT}:8080" "${TAG}" > /dev/null

    if wait_for_health "$((MAX_STARTUP_SECONDS + 5))"; then
        END_TIME=$(date +%s%N)
        STARTUP_MS=$(( (END_TIME - START_TIME) / 1000000 ))
        STARTUP_S=$(awk "BEGIN {printf \"%.1f\", ${STARTUP_MS}/1000}")
        log "Startup time: ${STARTUP_S}s (max ${MAX_STARTUP_SECONDS}s)"
        if (( STARTUP_MS <= MAX_STARTUP_SECONDS * 1000 )); then
            pass "${VERTICAL}: startup ${STARTUP_S}s <= ${MAX_STARTUP_SECONDS}s"
        else
            fail "${VERTICAL}: startup ${STARTUP_S}s > ${MAX_STARTUP_SECONDS}s"
        fi
    else
        fail "${VERTICAL}: container did not become healthy"
        docker logs "${CONTAINER_NAME}" 2>&1 | tail -20
        docker rm -f "${CONTAINER_NAME}" 2>/dev/null || true
        continue
    fi

    BASE="http://localhost:${PORT}"

    # Test /health
    test_endpoint GET "${BASE}/health" "${VERTICAL}: GET /health" 200

    # Test /v1/models
    test_endpoint GET "${BASE}/v1/models" "${VERTICAL}: GET /v1/models" 200

    # Test /v1/classify
    CLASSIFY_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${BASE}/v1/classify" \
        -H "Content-Type: application/json" \
        -d '{"messages":[{"role":"user","content":"test query"}]}')
    if [[ "$CLASSIFY_STATUS" == "200" ]]; then
        pass "${VERTICAL}: POST /v1/classify"
    else
        fail "${VERTICAL}: POST /v1/classify (got $CLASSIFY_STATUS)"
    fi

    # Test /v1/feedback
    FEEDBACK_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${BASE}/v1/feedback" \
        -H "Content-Type: application/json" \
        -d '{"query":"test","expected_decision":"allow","actual_decision":"deny"}')
    if [[ "$FEEDBACK_STATUS" == "200" ]]; then
        pass "${VERTICAL}: POST /v1/feedback"
    else
        fail "${VERTICAL}: POST /v1/feedback (got $FEEDBACK_STATUS)"
    fi

    # Test /metrics
    test_endpoint GET "${BASE}/metrics" "${VERTICAL}: GET /metrics" 200

    # Stop and remove container
    docker rm -f "${CONTAINER_NAME}" > /dev/null 2>&1 || true
    log ""
done

# Summary table
echo ""
echo "========================================="
echo "         TEST SUMMARY"
echo "========================================="
printf "%-6s %s\n" "Result" "Test"
echo "-----------------------------------------"
for r in "${RESULTS[@]}"; do
    printf "%-6s %s\n" "${r%% *}" "${r#* }"
done
echo "-----------------------------------------"
echo "Passed: ${PASS_COUNT}  Failed: ${FAIL_COUNT}  Total: $((PASS_COUNT + FAIL_COUNT))"
echo "========================================="

if (( FAIL_COUNT > 0 )); then
    exit 1
fi
