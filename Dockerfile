FROM python:3.11-slim AS base

WORKDIR /app

# Copy project files for install
COPY pyproject.toml README.md ./
COPY src/ src/

# Install package and dependencies
RUN pip install --no-cache-dir . && rm -rf /root/.cache

# Copy policies and model artifacts
COPY policies/ policies/
COPY dist/ dist/

# Default paths — override per vertical
ENV POLICY_PATH=policies/finance.json
ENV MODEL_PATH=dist/finance/model.onnx
ENV TOKENIZER_PATH=dist/finance/tokenizer
ENV CALIBRATION_PATH=dist/finance/calibration_params.json
ENV HOST=0.0.0.0
ENV PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/health').raise_for_status()"

CMD ["uvicorn", "intentguard.server:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
