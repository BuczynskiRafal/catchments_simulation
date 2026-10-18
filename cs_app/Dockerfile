FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.10.9 /uv /uvx /bin/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1 \
    HOME=/home/appuser \
    PATH="/code/.venv/bin:$PATH"

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /code

# Copy dependency metadata and package sources so uv can resolve local project deps.
# LICENSE is required: pyproject.toml lists it in license-files.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src src

# Copy static files needed by collectstatic
COPY cs_app/static static

# Install dependencies (frozen from lock file)
RUN uv sync --frozen --no-dev --group web

# Copy the rest of the application code
COPY cs_app .

# Collect static files and prepare runtime writable directories for non-root user
RUN SECRET_KEY=build-only-secret DATABASE_URL=sqlite:////tmp/db.sqlite3 \
    python manage.py collectstatic --noinput && \
    rm -f /tmp/db.sqlite3 && \
    mkdir -p /code/media/uploaded_files /code/.cache/django && \
    chown -R appuser:appuser /code

USER appuser

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; req = urllib.request.Request('http://127.0.0.1:8000/healthz', headers={'Host': 'localhost', 'X-Forwarded-Proto': 'https'}); urllib.request.urlopen(req, timeout=3)" || exit 1

EXPOSE 8000

CMD ["sh", "-c", "python manage.py migrate --noinput && exec gunicorn cs_app.wsgi:application --bind 0.0.0.0:${PORT:-8000} --workers ${WEB_CONCURRENCY:-2} --threads ${GUNICORN_THREADS:-1} --timeout ${GUNICORN_TIMEOUT:-120} --access-logfile - --error-logfile -"]
