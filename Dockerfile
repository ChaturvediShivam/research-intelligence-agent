# Multi-stage: dependencies resolve in the builder, only the venv ships.
FROM python:3.13-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Lockfile first so a source-only change does not re-resolve dependencies.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra retrieval --extra postgres --extra mcp

COPY app ./app
RUN uv sync --frozen --extra retrieval --extra postgres --extra mcp


FROM python:3.13-slim AS runtime

# Run unprivileged: nothing in this service needs root.
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv
COPY --from=builder --chown=appuser:appuser /app/app /app/app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    LOG_JSON=true \
    PORT=8000

USER appuser
EXPOSE 8000

# Liveness only — /ready depends on configuration and would fail the container
# for a missing key, which is an operator problem, not a liveness problem.
# Reads $PORT so the check follows the port the server actually bound.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["sh", "-c", "python -c \"import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health', timeout=4).status==200 else 1)\""]

# Shell form so $PORT expands. Render and most PaaS inject PORT and expect the
# process to bind it; an exec-form CMD cannot expand a variable, so the port
# was previously pinned to 8000 and the injected value silently ignored.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port \"${PORT:-8000}\""]
