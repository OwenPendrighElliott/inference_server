# -------- Stage 1: Build / dependency install --------
FROM python:3.11-slim-trixie AS builder

# Install build tools
RUN apt-get update \
    && apt-get install -y curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Copy only lockfile & pyproject for dependency install layer
COPY uv.lock pyproject.toml ./

# Install dependencies (without installing project code) — good for caching
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev

# Copy application code
COPY . /app

# Install project + dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

# -------- Stage 2: Runtime image --------
FROM python:3.11-slim-trixie AS runtime

WORKDIR /app

# Copy only the environment (if uv created .venv) & code
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

# Ensure PATH includes the venv
ENV PATH="/app/.venv/bin:$PATH"
ENV HF_HOME=/app/model_cache

RUN mkdir -p /app/model_cache/
ENV HF_HOME=/app/model_cache/

# Expose port
EXPOSE 8687

# Prepare model repository
RUN mkdir -p /app/model_repository \
    && chmod +x /app/start.sh

CMD ["bash", "/app/start.sh"]
