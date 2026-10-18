import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path

import pytest

from main import predictor
from main.apps import MainConfig


@pytest.fixture(autouse=True)
def _reset_predictor_state(monkeypatch):
    monkeypatch.setattr(predictor, "_model", None)
    monkeypatch.delenv("CATCHMENT_SIMULATION_WEIGHTS_PATH", raising=False)


def test_get_model_is_thread_safe(monkeypatch):
    created: list[str] = []
    start_barrier = threading.Barrier(8)

    class FakeModel:
        def __init__(self, weights_path: str):
            time.sleep(0.02)
            created.append(weights_path)

    def _worker() -> object:
        start_barrier.wait(timeout=1)
        return predictor._get_model()

    monkeypatch.setattr(predictor, "SimpleMLPModel", FakeModel)
    monkeypatch.setenv("CATCHMENT_SIMULATION_WEIGHTS_PATH", "/tmp/weights.npz")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: _worker(), range(8)))

    assert len(created) == 1


def test_preload_model_is_idempotent(monkeypatch):
    created: list[str] = []

    class FakeModel:
        def __init__(self, weights_path: str):
            created.append(weights_path)

    monkeypatch.setattr(predictor, "SimpleMLPModel", FakeModel)

    first_loaded_now = predictor.preload_model()
    first_instance = predictor._model
    second_loaded_now = predictor.preload_model()

    assert len(created) == 1
    assert first_loaded_now is True
    assert second_loaded_now is False
    assert predictor._model is first_instance


def test_get_weights_path_prefers_env_over_default(monkeypatch):
    env_path = "/tmp/test-weights.npz"
    monkeypatch.setenv("CATCHMENT_SIMULATION_WEIGHTS_PATH", env_path)
    assert predictor._get_weights_path() == env_path


def test_main_config_ready_calls_preload_model_when_enabled(monkeypatch):
    calls: list[str] = []

    def fake_preload() -> None:
        calls.append("called")

    monkeypatch.setattr("main.apps._should_preload_model", lambda: True)
    monkeypatch.setattr("main.predictor.preload_model", fake_preload)

    app_config = MainConfig("main", import_module("main"))
    app_config.ready()

    assert calls == ["called"]


def test_main_config_ready_skips_preload_when_disabled(monkeypatch):
    calls: list[str] = []

    def fake_preload() -> None:
        calls.append("called")

    monkeypatch.setattr("main.apps._should_preload_model", lambda: False)
    monkeypatch.setattr("main.predictor.preload_model", fake_preload)

    app_config = MainConfig("main", import_module("main"))
    app_config.ready()

    assert calls == []


def test_main_config_ready_raises_preload_failure(monkeypatch, caplog):
    def failing_preload() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("main.apps._should_preload_model", lambda: True)
    monkeypatch.setattr("main.predictor.preload_model", failing_preload)

    app_config = MainConfig("main", import_module("main"))
    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="boom"):
        app_config.ready()

    assert "Failed to preload ANN predictor model during app startup" in caplog.text


def test_should_preload_model_for_manage_py_runserver(monkeypatch):
    monkeypatch.delenv("PRELOAD_MODEL", raising=False)
    monkeypatch.setattr(sys, "argv", ["manage.py", "runserver"])
    from main.apps import _should_preload_model

    assert _should_preload_model() is True


def test_should_preload_model_skips_manage_py_migrate(monkeypatch):
    monkeypatch.delenv("PRELOAD_MODEL", raising=False)
    monkeypatch.setattr(sys, "argv", ["manage.py", "migrate"])
    from main.apps import _should_preload_model

    assert _should_preload_model() is False


def test_should_preload_model_honors_env_override(monkeypatch):
    monkeypatch.setenv("PRELOAD_MODEL", "1")
    monkeypatch.setattr(sys, "argv", ["manage.py", "migrate"])
    from main.apps import _should_preload_model

    assert _should_preload_model() is True


def test_should_preload_model_skips_worker_process(monkeypatch):
    monkeypatch.delenv("PRELOAD_MODEL", raising=False)
    monkeypatch.setattr(sys, "argv", ["celery", "-A", "cs_app", "worker"])
    from main.apps import _should_preload_model

    assert _should_preload_model() is False


def test_env_bool_invalid_value_logs_warning(monkeypatch, caplog):
    monkeypatch.setenv("PRELOAD_MODEL", "maybe")
    monkeypatch.setattr(sys, "argv", ["manage.py", "runserver"])
    from main.apps import _should_preload_model

    with caplog.at_level("WARNING"):
        result = _should_preload_model()

    assert result is True
    assert "Ignoring invalid PRELOAD_MODEL='maybe'" in caplog.text


def test_views_import_with_pyswmm_without_multi_simulation_error():
    """pyswmm 1.5, which a plain install of swmmio 0.8 gets, has no MultiSimulationError."""
    code = (
        "import pyswmm.errors\n"
        "if hasattr(pyswmm.errors, 'MultiSimulationError'):\n"
        "    del pyswmm.errors.MultiSimulationError\n"
        "import django\n"
        "django.setup()\n"
        "import main.views\n"
        "assert main.views._ENGINE_BUSY_ERRORS == ()\n"
    )
    app_dir = Path(__file__).resolve().parents[2]
    env = {**os.environ, "DJANGO_SETTINGS_MODULE": "cs_app.test_settings"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=app_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_refused_open_is_reraised_without_closing_the_engine(monkeypatch):
    """With pyswmm 2, a second simulation is refused; closing the engine would end the first."""
    errors = pytest.importorskip("pyswmm.errors")
    if not hasattr(errors, "MultiSimulationError"):
        pytest.skip("pyswmm < 2.0 does not refuse a second simulation")
    from main import views

    def refuse(*args, **kwargs):
        raise errors.MultiSimulationError("Multi-Simulation Error.")

    def no_close():
        raise AssertionError("the running simulation's engine must stay open")

    monkeypatch.setattr(views, "Simulation", refuse)
    monkeypatch.setattr(views, "PySWMM", no_close)
    with pytest.raises(errors.MultiSimulationError):
        views._run_swmm_in_place("model.inp")


@pytest.mark.django_db
def test_healthz_returns_ok(client):
    from django.urls import reverse

    response = client.get(reverse("main:healthz"))
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.django_db
def test_healthz_rejects_post(client):
    from django.urls import reverse

    response = client.post(reverse("main:healthz"))
    assert response.status_code == 405


@pytest.mark.django_db
def test_healthz_returns_503_on_database_error(client, monkeypatch):
    from django.db import connection
    from django.urls import reverse

    def broken_connection():
        raise RuntimeError("DB unreachable")

    monkeypatch.setattr(connection, "ensure_connection", broken_connection)
    response = client.get(reverse("main:healthz"))
    assert response.status_code == 503
    assert response.json() == {"status": "error", "detail": "database"}


def test_cleanup_uploads_command_removes_expired_files(tmp_path, settings):
    from django.core.management import CommandError, call_command

    settings.MEDIA_ROOT = str(tmp_path)
    user_dir = tmp_path / "uploaded_files" / "42"
    user_dir.mkdir(parents=True)
    old_file = user_dir / "old.inp"
    fresh_file = user_dir / "fresh.inp"
    old_file.write_text("[TITLE]\nOld\n")
    fresh_file.write_text("[TITLE]\nFresh\n")

    old_mtime = time.time() - 7200
    os.utime(old_file, (old_mtime, old_mtime))

    call_command("cleanup_uploads", max_age_hours=1.0, dry_run=True)
    assert old_file.exists()

    call_command("cleanup_uploads", max_age_hours=1.0)
    assert not old_file.exists()
    assert fresh_file.exists()

    os.utime(fresh_file, (old_mtime, old_mtime))
    call_command("cleanup_uploads", max_age_hours=1.0)
    assert not fresh_file.exists()
    assert not user_dir.exists()

    with pytest.raises(CommandError):
        call_command("cleanup_uploads", max_age_hours=-1.0)


def test_production_settings_csrf_trusted_origins_and_proxy_ssl():
    code = (
        "import os\n"
        "os.environ['SECRET_KEY'] = 'prod-secret'\n"
        "os.environ['DEBUG'] = 'False'\n"
        "os.environ['DATABASE_URL'] = 'sqlite:///:memory:'\n"
        "os.environ['RENDER_EXTERNAL_HOSTNAME'] = 'app.onrender.com'\n"
        "os.environ['CSRF_TRUSTED_ORIGINS'] = 'https://custom.example.com'\n"
        "os.environ.pop('USE_X_FORWARDED_PROTO', None)\n"
        "from cs_app import settings as prod_settings\n"
        "assert 'https://app.onrender.com' in prod_settings.CSRF_TRUSTED_ORIGINS\n"
        "assert 'https://custom.example.com' in prod_settings.CSRF_TRUSTED_ORIGINS\n"
        "assert prod_settings.SECURE_PROXY_SSL_HEADER == ('HTTP_X_FORWARDED_PROTO', 'https')\n"
    )
    app_dir = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=app_dir,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
