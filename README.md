# IntentGuard

Vertical intent classifier for LLM chatbot guardrails. Enforces topic boundaries so chatbots stay on-topic for their industry vertical.

Ships as a container with an OpenAI-compatible REST API. Each customer gets a model fine-tuned for their vertical (finance, healthcare, legal, etc.).

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
  "confidence": 0.94,
  "vertical": "finance"
}
```

### Proxy mode

Point your app at IntentGuard instead of your LLM. Set `DOWNSTREAM_URL` and it forwards allowed requests automatically:

```bash
docker run -p 8080:8080 \
  -e DOWNSTREAM_URL=https://api.openai.com/v1/chat/completions \
  perfecxion/intentguard:finance-latest
```

## Configuration

Each vertical ships with a `policy.json` that defines:

- Which topics are in-scope
- Conditional rules (e.g., healthcare is allowed when related to financial planning)
- Refusal and clarification messages
- Confidence thresholds

Threshold and message changes take effect on restart — no retraining needed.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run the server locally
uvicorn intentguard.server:app --reload --port 8080
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/classify` | POST | Classify a message (sidecar mode) |
| `/v1/chat/completions` | POST | OpenAI-compatible proxy mode |
| `/v1/models` | GET | List available models |
| `/health` | GET | Health check |

## License

Apache 2.0
