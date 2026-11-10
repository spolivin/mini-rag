FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models

WORKDIR /app

# Dependencies first, so this layer is cached until pyproject.toml / uv.lock change.
# The project itself is not a package (package = false), so only dependencies are installed.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY . .

# Run as a non-root user. /models holds the Hugging Face model cache and is mounted as a
# named volume in docker-compose.yml, so models are downloaded once, not on every start.
RUN useradd --create-home app && mkdir /models && chown app /models
USER app

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
CMD ["fastapi", "run", "--host", "0.0.0.0", "--port", "8000"]
