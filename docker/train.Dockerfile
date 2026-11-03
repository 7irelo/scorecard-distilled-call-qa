# Training image for a Linux GPU host. Pin the base image digest before a real run.
#   docker build -f docker/train.Dockerfile -t scorecard-train .
#   docker run --gpus all --env-file .env -v "$PWD/data:/app/data" -v "$PWD/runs:/app/runs" \
#     scorecard-train train run configs/train/baseline.yaml
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/app/runs/hf-cache
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install --quiet uv==0.11.23

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY configs ./configs
COPY docs/RUBRIC.md ./docs/RUBRIC.md
RUN uv sync --frozen --no-dev --extra train && rm -rf /root/.cache

ENTRYPOINT ["/app/.venv/bin/scorecard"]
