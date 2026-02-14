import logging
import os
import sys

from django.apps import AppConfig

logger = logging.getLogger(__name__)

TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})
FALSY_VALUES = frozenset({"0", "false", "no", "off"})


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    normalized = raw.strip().lower()
    if normalized in TRUTHY_VALUES:
        return True
    if normalized in FALSY_VALUES:
        return False
    logger.warning(
        "Ignoring invalid %s=%r; expected one of %s/%s",
        name,
        raw,
        TRUTHY_VALUES,
        FALSY_VALUES,
    )
    return None


def _should_preload_model() -> bool:
    """Return True for web server processes, False for management commands."""
    override = _env_bool("PRELOAD_MODEL")
    if override is not None:
        return override

    if not sys.argv:
        return True

    argv0 = os.path.basename(sys.argv[0])
    if argv0 == "manage.py":
        command = sys.argv[1] if len(sys.argv) > 1 else ""
        return command in {"runserver", "runserver_plus"}

    if "pytest" in argv0:
        return False

    return True


class MainConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "main"

    def ready(self) -> None:
        """Preload runtime artifacts to reduce first-request latency."""
        if not _should_preload_model():
            return

        from main.predictor import preload_model

        try:
            preload_model()
        except Exception:
            logger.exception("Failed to preload ANN predictor model during app startup")
            raise
