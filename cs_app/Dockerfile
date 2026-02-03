FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

# Copy project files needed for dependency resolution
COPY ../pyproject.toml ../uv.lock ./

# Copy static files
COPY static static

# Install dependencies (frozen from lock file)
RUN uv sync --frozen --no-dev --extra django

# Copy the rest of the application code
COPY . .

# Collect static files
RUN uv run python manage.py collectstatic --noinput

EXPOSE 8000

CMD ["uv", "run", "gunicorn", "cs_app.wsgi:application", "--bind", "0.0.0.0:8000"]
