FROM python:3.11-slim AS base

WORKDIR /app

# Install production dependencies only
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir . && rm -rf /root/.cache

# Copy source and policy
COPY src/ src/
COPY policies/ policies/

# Default to finance vertical
ENV POLICY_PATH=policies/finance.json
ENV HOST=0.0.0.0
ENV PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s \
    CMD python -c "import httpx; httpx.get('http://localhost:8080/health').raise_for_status()"

CMD ["uvicorn", "intentguard.server:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
