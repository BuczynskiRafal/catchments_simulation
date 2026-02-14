# Catchment Simulation - Django Web Application

This is the Django web application for the Catchment Simulation project. It provides a web interface for stormwater analysis and SWMM file processing.

## Quick Start

### Prerequisites

- Python 3.9 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/BuczynskiRafal/catchments_simulation.git
   cd catchments_simulation
   ```

2. **Create virtual environment with uv**:
   ```bash
   uv venv --python 3.12
   source .venv/bin/activate  # On macOS/Linux
   # or .venv\Scripts\activate on Windows
   ```

3. **Install the main package and dev dependencies**:
   ```bash
   uv pip install -e ".[dev]"
   ```

4. **Install Django app dependencies**:
   ```bash
   uv pip install Django==3.2 Werkzeug==2.0.0 crispy-bootstrap4==2022.1 \
       python-dotenv==1.0.0 django-crispy-forms==2.0 django-import-export==3.2.0 \
       django-storages==1.13.2 plotly==5.18.0 dj-database-url==2.0.0 \
       whitenoise==6.4.0 gunicorn==20.1.0 pytest-django email-validator
   ```

5. **macOS only - Fix code signature issues** (if you encounter `SIGKILL` errors):
   ```bash
   find .venv -name "*.so" -o -name "*.dylib" | xargs codesign --force --sign -
   ```

### Configuration

1. **Environment variables**: The app uses a `.env` file in `cs_app/` directory. Default configuration for local development:
   ```
   DATABASE_URL=sqlite:///db.sqlite3
   DEBUG=True
   SECRET_KEY=your-secret-key-here
   ```

### Running the Application

1. **Navigate to the app directory**:
   ```bash
   cd cs_app
   ```

2. **Run database migrations**:
   ```bash
   python manage.py migrate
   ```

3. **Create a superuser** (optional, for admin access):
   ```bash
   python manage.py createsuperuser
   ```

4. **Start the development server**:
   ```bash
   python manage.py runserver
   ```

5. **Open your browser** at http://127.0.0.1:8000/

### Running Tests

```bash
cd cs_app
pytest --ds=cs_app.test_settings
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
└── requirements.txt  # App-specific dependencies
```

Note: simulation and timeseries result files are generated in-memory at download time and are not persisted in `MEDIA`.

## Deployment

For production deployment (e.g., on Render), ensure you have:
- `DATABASE_URL` pointing to PostgreSQL
- `DEBUG=False`
- Proper `SECRET_KEY`
- `psycopg2-binary` installed for PostgreSQL support

See `render.yaml` for Render-specific configuration.
