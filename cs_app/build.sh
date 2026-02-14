#!/usr/bin/env bash

set -o errexit  # exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"
uv sync --frozen --no-dev --extra web

cd "${SCRIPT_DIR}"
uv run python manage.py collectstatic --no-input
uv run python manage.py migrate
