# Catchment Simulation - Django Web Application

This is the Django web application for the Catchment Simulation project. It provides a web interface for stormwater analysis and SWMM file processing.

## Quick Start

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/)

### Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/BuczynskiRafal/catchments_simulation.git
   cd catchments_simulation
   ```

2. **Create virtual environment with uv**:
   ```bash
   uv venv --python 3.12  # any Python 3.10+ is supported
   source .venv/bin/activate  # On macOS/Linux
   # or .venv\Scripts\activate on Windows
   ```

3. **Install dependencies for app development**:
   ```bash
   uv sync --frozen --extra dev --extra web
   ```

4. **Optional: install docs toolchain**:
   ```bash
   uv sync --frozen --extra docs --extra web
   ```

5. **macOS only - Fix code signature issues** (if you encounter `SIGKILL` errors):
   ```bash
   find .venv -name "*.so" -o -name "*.dylib" | xargs codesign --force --sign -
   ```

### Configuration

1. **Environment variables**: The app uses a `.env` file in `cs_app/` directory. Start from `.env.example` and adjust values:
   ```
   cp cs_app/.env.example cs_app/.env
   ```

### Running the Application

1. **Navigate to the app directory**:
   ```bash
   cd cs_app
   ```

2. **Run database migrations**:
   ```bash
   uv run python manage.py migrate
   ```

3. **Create a superuser** (optional, for admin access):
   ```bash
   uv run python manage.py createsuperuser
   ```

4. **Start the development server**:
   ```bash
   uv run python manage.py runserver
   ```

5. **Open your browser** at http://127.0.0.1:8000/

### Running Tests

```bash
cd cs_app
uv run pytest --ds=cs_app.test_settings
```

## Project Structure

```
cs_app/
├── cs_app/           # Django project settings
│   ├── settings.py   # Main settings
│   ├── urls.py       # URL configuration
│   └── wsgi.py       # WSGI entry point
├── main/             # Main application (analysis, calculations)
├── register/         # User registration app
├── static/           # Static files (CSS, JS)
├── data/             # Static chart data in JSON format
├── swmm_model/       # ANN runtime artifact (weights.npz)
├── manage.py         # Django management script
```

Note: simulation and timeseries result files are generated in-memory at download time and are not persisted in `MEDIA`.

## Architecture Notes

- Domain-level simulation validation schemas live in `src/catchment_simulation/schemas.py`.
- Django web-only payload schemas (for example, contact form data) live in `cs_app/main/schemas.py`.
- ANN weights are preloaded at app startup via `main.apps.MainConfig.ready()` and reused from a module-level predictor cache.

## Deployment

For production deployment (e.g., on Render), ensure you have:
- `DATABASE_URL` pointing to PostgreSQL
- `DEBUG=False`
- Proper `SECRET_KEY`
- `psycopg2-binary` installed for PostgreSQL support
- Upload limits aligned across layers:
  - Django app limits (`INP_UPLOAD_MAX_BYTES`, `INP_UPLOAD_MAX_BODY_BYTES`) are defined in settings.
  - Reverse proxy request-body limit should be greater than or equal to `INP_UPLOAD_MAX_BODY_BYTES`.

Example Nginx setting:

```nginx
client_max_body_size 11m;
```

Proxy-level limits are the first line of defense; Django limits are the second.

Threat model notes:
- This hardening primarily protects Django worker memory (no full-file `read()` during upload validation).
- Spoofed or missing `Content-Length` is additionally constrained by server-side streamed body limiting.
- Disk/CPU pressure from many concurrent large uploads still requires infrastructure controls (proxy limits,
  rate limiting, and worker sizing).

See `render.yaml` for Render-specific configuration.
