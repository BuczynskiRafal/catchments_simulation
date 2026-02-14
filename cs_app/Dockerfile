FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

# Copy dependency metadata and package sources so uv can resolve local project deps
COPY pyproject.toml uv.lock README.md ./
COPY src src

# Copy static files needed by collectstatic
COPY cs_app/static static

# Install dependencies (frozen from lock file)
RUN uv sync --frozen --no-dev --extra web

# Copy the rest of the application code
COPY cs_app .

# Collect static files
RUN SECRET_KEY=build-only-secret DATABASE_URL=sqlite:////tmp/db.sqlite3 \
  uv run python manage.py collectstatic --noinput

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/', timeout=3)" || exit 1

EXPOSE 8000

CMD ["uv", "run", "gunicorn", "cs_app.wsgi:application", "--bind", "0.0.0.0:8000"]
