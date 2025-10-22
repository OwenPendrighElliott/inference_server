# -------- Stage 1: Build / dependency install --------
FROM python:3.13-slim-trixie AS builder

RUN apt-get update && apt-get install -y curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy lock + project config
COPY uv.lock pyproject.toml ./

# Cache mount for uv
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project

# Copy full code
COPY . /app

# Install project + deps
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

# -------- Stage 2: Runtime image --------
FROM python:3.13-slim-trixie AS runtime

WORKDIR /app

# Copy only the environment (if uv created .venv) & code
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

# Set PATH to the venv
ENV PATH="/app/.venv/bin:$PATH"

RUN mkdir -p /app/model_cache/
ENV HF_HOME=/app/model_cache/

EXPOSE 8686

CMD ["bash","/app/start.sh"]
