---
library_name: onnx
pipeline_tag: text-classification
license: apache-2.0
language:
  - en
tags:
  - intentguard
  - guardrails
  - llm-safety
  - content-moderation
  - {{ vertical }}
model-index:
  - name: intentguard-{{ vertical }}
    results:
      - task:
          type: text-classification
          name: Intent Classification
        metrics:
          - name: Accuracy
            type: accuracy
            value: {{ accuracy }}
          - name: Adversarial Accuracy
            type: accuracy
            value: {{ adversarial_accuracy }}
---

# IntentGuard — {{ display_name }} ({{ vertical }})

Vertical intent classifier for LLM chatbot guardrails. Classifies user messages
as **allow**, **deny**, or **abstain** based on whether they fall within the
{{ vertical }} domain.

## Model Details

- **Architecture:** DeBERTa-v3-xsmall fine-tuned for 3-way classification
- **Format:** ONNX (INT8 quantized)
- **Version:** {{ version }}
- **Vertical:** {{ vertical }} ({{ display_name }})
- **Publisher:** [perfecXion.ai](https://perfecxion.ai)

## Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | {{ accuracy }} |
| Adversarial Accuracy | {{ adversarial_accuracy }} |
| p99 Latency (CPU) | {{ p99_latency }} |
| Model Size | {{ model_size }} |

## Usage

### Python (ONNX Runtime)

```python
import onnxruntime as ort
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("perfecXion/intentguard-{{ vertical }}")
session = ort.InferenceSession("model.onnx")

inputs = tokenizer("What are mortgage rates?", return_tensors="np")
logits = session.run(None, dict(inputs))[0]
```

### Docker

```bash
docker pull ghcr.io/perfecxion/intentguard:{{ vertical }}-{{ version }}
docker run -p 8080:8080 ghcr.io/perfecxion/intentguard:{{ vertical }}-{{ version }}

curl -X POST http://localhost:8080/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What are mortgage rates?"}]}'
```

### pip

```bash
pip install intentguard
```

## Core Topics

{{ core_topics }}

## License

Apache 2.0
