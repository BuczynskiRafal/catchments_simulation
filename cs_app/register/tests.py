"""Tests for the session and CSRF cookie settings."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Production settings: cookie flags
# ---------------------------------------------------------------------------

_READ_COOKIE_FLAGS = """
import json, dotenv
dotenv.load_dotenv = lambda *args, **kwargs: False  # only the env given below counts
from cs_app import settings
print(json.dumps([settings.SESSION_COOKIE_SECURE, settings.CSRF_COOKIE_SECURE]))
"""


def _cookie_flags(**env: str) -> list[bool]:
    """Import cs_app.settings (not the test settings) in a clean process."""
    clean_env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"DEBUG", "SESSION_COOKIE_SECURE", "CSRF_COOKIE_SECURE"}
    }
    clean_env.update(SECRET_KEY="test", DATABASE_URL="sqlite:///:memory:", **env)
    result = subprocess.run(
        [sys.executable, "-c", _READ_COOKIE_FLAGS],
        cwd=Path(__file__).resolve().parent.parent,
        env=clean_env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    ("env", "expected"),
    [
        # Plain-HTTP dev server: Secure cookies would be dropped by Safari -> CSRF 403.
        ({"DEBUG": "True"}, [False, False]),
        # Production fails closed.
        ({}, [True, True]),
        ({"DEBUG": "False"}, [True, True]),
        # Explicit settings always win.
        (
            {"DEBUG": "True", "SESSION_COOKIE_SECURE": "True", "CSRF_COOKIE_SECURE": "True"},
            [True, True],
        ),
        (
            {"DEBUG": "False", "SESSION_COOKIE_SECURE": "False", "CSRF_COOKIE_SECURE": "False"},
            [False, False],
        ),
    ],
)
def test_cookies_are_secure_unless_debug(env, expected):
    assert _cookie_flags(**env) == expected
