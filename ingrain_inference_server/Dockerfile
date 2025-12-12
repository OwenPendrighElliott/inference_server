########################################
# Stage 1: Builder
########################################
FROM python:3.13-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app

RUN mkdir -p /app/model_cache/
ENV HF_HOME=/app/model_cache/

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=/app/uv.lock \
    --mount=type=bind,source=pyproject.toml,target=/app/pyproject.toml \
    uv sync --locked --no-install-project --no-dev

COPY . /app
COPY ./hello.sh /app/hello.sh

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

#################################
# Stage 2: Runtime
#################################
FROM python:3.13-slim-bookworm AS runtime

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

ENV PATH="/app/.venv/bin:${PATH}"
ENV HF_HOME=/app/model_cache
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

EXPOSE 8686

RUN mkdir -p /app/model_repository

RUN chmod +x /app/start.sh

CMD ["bash", "/app/start.sh"]
