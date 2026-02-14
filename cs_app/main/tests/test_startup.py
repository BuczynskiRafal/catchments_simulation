import sys
import time
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module

from main import predictor
from main.apps import MainConfig


def test_get_model_is_thread_safe(monkeypatch):
    created: list[str] = []

    class FakeModel:
        def __init__(self, weights_path: str):
            time.sleep(0.01)
            created.append(weights_path)

    monkeypatch.setattr(predictor, "SimpleMLPModel", FakeModel)
    monkeypatch.setattr(predictor, "_model", None)
    monkeypatch.setenv("CATCHMENT_SIMULATION_WEIGHTS_PATH", "/tmp/weights.npz")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: predictor._get_model(), range(20)))

    assert len(created) == 1


def test_preload_model_is_idempotent(monkeypatch):
    created: list[str] = []

    class FakeModel:
        def __init__(self, weights_path: str):
            created.append(weights_path)

    monkeypatch.setattr(predictor, "SimpleMLPModel", FakeModel)
    monkeypatch.setattr(predictor, "_model", None)

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
        try:
            app_config.ready()
        except RuntimeError:
            pass
        else:
            raise AssertionError("ready() should re-raise preload failures")

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
