import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module

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
    with caplog.at_level("ERROR"):
        with pytest.raises(RuntimeError, match="boom"):
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
