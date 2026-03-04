# IntentGuard

Vertical intent classifier for LLM chatbot guardrails. Enforces topic boundaries so chatbots stay on-topic for their industry vertical.

Ships as a container with an OpenAI-compatible REST API. Each customer gets a model fine-tuned for their vertical.

## Verticals

| Vertical | Accuracy | LBR | OPR | Status |
|----------|----------|-----|-----|--------|
| Finance | 98.3% | 0.37% | 0.00% | Ship |
| Healthcare | 97.7% | 0.00% | 0.00% | Ship |
| Legal | 95.3% | 0.41% | 0.50% | Ship |

LBR = legitimate-block rate (lower is better). OPR = off-topic-pass rate.

## How it works

IntentGuard sits between the user and the LLM. Every incoming message gets classified as:

- **ALLOW** — on-topic, forward to the LLM
- **DENY** — off-topic, return a polite refusal with topic suggestions
- **ABSTAIN** — unclear, ask the user to clarify

The three-way classification avoids false blocks on borderline queries. Instead of guessing, it asks.

## Quick start

```bash
docker run -p 8080:8080 perfecxion/intentguard:finance-latest
```

### Sidecar mode (recommended)

Call `/v1/classify` before sending to your LLM:

```bash
curl -X POST http://localhost:8080/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What are current mortgage rates?"}]}'
```

```json
{
  "decision": "allow",
  "confidence": 0.95,
  "vertical": "finance"
}
```

### Proxy mode

Point your app at IntentGuard instead of your LLM. Set `DOWNSTREAM_URL` and it forwards allowed requests automatically:

```bash
docker run -p 8080:8080 \
  -e DOWNSTREAM_URL=https://api.openai.com/v1/chat/completions \
  -e DOWNSTREAM_API_KEY=sk-... \
  perfecxion/intentguard:finance-latest
```

## Architecture

```
User App --> IntentGuard --> LLM
             (classify)     (only if ALLOW)
```

- Model: DeBERTa-v3-xsmall (22M params), ONNX Runtime on CPU
- Latency: p99 < 50ms on 4-core x86_64
- Decision: margin-based thresholds with temperature-calibrated probabilities
- Input normalization: NFKC unicode, zero-width stripping, encoding trick detection

## Configuration

Each vertical ships with a `policy.json` that defines:

- Which topics are in-scope
- Conditional rules (e.g., healthcare is allowed when related to financial planning)
- Hard exclusions
- Refusal and clarification messages
- Confidence thresholds and margins

Threshold and message changes take effect on restart. Topic scope changes require retraining.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/classify` | POST | Classify a message (sidecar mode) |
| `/v1/chat/completions` | POST | OpenAI-compatible proxy mode |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8080 | Server port |
| `POLICY_PATH` | policies/finance.json | Policy config |
| `MODEL_PATH` | dist/finance/model.onnx | ONNX model |
| `TOKENIZER_PATH` | dist/finance/tokenizer | Tokenizer directory |
| `CALIBRATION_PATH` | dist/finance/calibration_params.json | Calibration params |
| `DOWNSTREAM_URL` | (none) | LLM URL for proxy mode |
| `DOWNSTREAM_API_KEY` | (none) | API key for downstream LLM |
| `LOG_QUERY_TEXT` | false | Log query text (privacy) |
| `DEBUG` | false | Show probabilities in responses |

## Development

```bash
pip install -e ".[dev]"
pytest
uvicorn intentguard.server:app --reload --port 8080
```

### Training a vertical

```bash
pip install -e ".[train]"

# Generate training data
python -m training.data_generation.synthetic_generator \
  --policy policies/finance.json \
  --output data/finance/synthetic.jsonl \
  --provider openai --count 2000

# Train
python -m training.fine_tune \
  --data data/finance \
  --config training/train_config.yaml \
  --output models/finance

# Calibrate
python -m training.calibrate \
  --model models/finance/best \
  --data data/finance/seed.jsonl \
  --output models/finance/calibration_params.json

# Export to ONNX
python -m training.export_onnx \
  --model models/finance/best \
  --output dist/finance \
  --sanity-data data/finance/seed.jsonl

# Evaluate
python -m evaluation.gates \
  --model dist/finance/model.onnx \
  --tokenizer dist/finance/tokenizer \
  --policy policies/finance.json \
  --data evaluation/test_sets/finance_adversarial.reviewed.jsonl
```

## Project structure

```
intentguard/
├── src/intentguard/       # Shipped in container
│   ├── server.py          # FastAPI server
│   ├── classifier.py      # ONNX inference + decision logic
│   ├── policy.py          # Policy loader
│   ├── schema.py          # API models
│   ├── normalize.py       # Input normalization
│   └── config.py          # Settings
├── training/              # Internal tooling
│   ├── data_generation/   # Synthetic data + hard negatives
│   ├── fine_tune.py       # HuggingFace Trainer
│   ├── calibrate.py       # Temperature scaling
│   └── export_onnx.py     # ONNX export + sanity gate
├── evaluation/            # Test suites + gating
│   ├── gates.py           # Ship/no-ship evaluation
│   ├── adversarial_suite.py
│   └── latency_benchmark.py
├── policies/              # Per-vertical configs
├── Dockerfile
└── package.sh             # Customer delivery packaging
```

## License

Apache 2.0
