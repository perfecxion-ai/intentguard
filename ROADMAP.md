# IntentGuard Roadmap

## Phase 3: Ship-Blocking (Days 1-3)

### 3.1 Docker Container Test
- Restart Docker Desktop clean (`docker system prune -af`)
- Rebuild image with all three verticals
- Run container, hit all endpoints, verify responses match local testing
- Test health check works for orchestrators (k8s readiness probe)
- Measure container startup time (target < 5s)
- Measure image size (target < 500MB)

### 3.2 INT8 Quantization
- Try Optimum `ORTModelForSequenceClassification.from_pretrained(export=True)` path
- If DeBERTa-v3 still fails: benchmark ModernBERT-base as replacement
  - Train on same data, compare accuracy vs DeBERTa-v3-xsmall
  - ModernBERT has clean ONNX export and 2-3x faster inference
- If ModernBERT works: retrain all three verticals, re-run gating
- Target: model < 80MB, p99 < 30ms

## Phase 4: Product Readiness (Days 4-7)

### 4.1 Shadow Mode
- Add `mode` query param to `/v1/classify`: `enforce` (default) or `shadow`
- Shadow mode: classify normally but always return `decision: allow`
- Include real classification in `X-Classification-Shadow` header
- Lets customers validate on production traffic before enforcement
- Add tests

### 4.2 Feedback Endpoint
- `POST /v1/feedback` with body: `{ "query": "...", "expected_decision": "allow", "actual_decision": "deny" }`
- Log to structured JSON (same stdout pattern as classification logs)
- Optional: write to a feedback JSONL file for batch export
- This feeds the retrain loop without requiring customer code changes
- Add tests

### 4.3 Prometheus Metrics
- Add `/metrics` endpoint using `prometheus_client`
- Counters: `intentguard_requests_total{decision,vertical}`
- Histograms: `intentguard_latency_seconds{vertical}`
- Gauges: `intentguard_model_loaded`, `intentguard_policy_version`
- Add `prometheus_client` to production deps
- Add tests

### 4.4 Calibration Fix
- Recalibrate healthcare and legal on full synthetic dataset (not just seed)
- Target ECE < 0.03 for all verticals
- Re-export and update dist/

## Phase 5: CI/CD and Publishing (Days 8-10)

### 5.1 GitHub Actions CI
- `.github/workflows/ci.yml`:
  - Trigger on push and PR
  - Lint (ruff)
  - Test (pytest)
  - Type check (optional: mypy)
- `.github/workflows/release.yml`:
  - Trigger on tag push
  - Build Docker image
  - Push to Docker Hub or GHCR

### 5.2 HuggingFace Model Publishing
- Push models to HuggingFace collection `perfecXion/intentguard`:
  - `perfecXion/intentguard-finance-v1`
  - `perfecXion/intentguard-healthcare-v1`
  - `perfecXion/intentguard-legal-v1`
- Each model card includes: vertical description, accuracy metrics, gating report, usage example
- Update Dockerfile to pull from HuggingFace on build (optional)

### 5.3 PyPI / Package Publishing
- Verify `pip install intentguard` works from the repo
- Optional: publish to PyPI for customers who want to run without Docker

## Phase 6: Production Validation (Days 11-14)

### 6.1 Production Traffic Simulation
- Build a script that replays realistic query distributions:
  - 70% clean on-topic
  - 15% clean off-topic
  - 10% borderline/ambiguous
  - 5% adversarial
- Run against all three verticals
- Compare accuracy against adversarial suite results
- Measure latency under sustained load (100 concurrent, 5 minutes)

### 6.2 Load Testing
- Use `wrk` or `locust` to hit `/v1/classify` with concurrent requests
- Measure: throughput (req/s), p50/p95/p99 latency, error rate
- Test at 10, 50, 100, 200 concurrent connections
- Verify graceful degradation (no crashes, latency stays bounded)

### 6.3 Pilot Customer Onboarding
- Write onboarding guide: from Docker pull to first classification
- Document common integration patterns:
  - LangChain middleware
  - LiteLLM proxy chain
  - Direct HTTP from application code
- Collect first production query logs (with customer consent)
- Validate model accuracy on real traffic

## Phase 7: Embeddings Router — v2 Architecture (Days 15-20)

### 7.1 Design
- Single container serving multiple verticals
- Stage 1: vertical router (which vertical does this query belong to?)
- Stage 2: per-vertical classifier (ALLOW/DENY/ABSTAIN within vertical)
- Share embedding computation between stages
- Policy pack mapping: intent -> allowed tools + guardrails

### 7.2 Build Router
- Evaluate embedding models: BGE-small, E5-small, Model2Vec
- Train vertical router on combined data from all verticals
- Benchmark: single-container multi-vertical vs per-vertical containers
- Target: < 25ms total for route + classify

### 7.3 Policy Pack Layer
- Define policy pack schema: tools, guardrails, thresholds per intent
- Implement lookup: (vertical, decision) -> PolicyPack
- Add to `/v1/classify` response: `policy_pack` field
- Customer can use this to gate tool access downstream
