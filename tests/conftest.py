"""Shared fixtures: every test runs in its own directory and must leave no SWMM files behind."""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from catchment_simulation import example_model_path


@pytest.fixture(autouse=True)
def isolated_filesystem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run the test with the CWD and the temp dir inside ``tmp_path``.

    A stray write to the CWD or to the temp-dir fallback then lands where the
    teardown can see it, and any working copy or SWMM report/output left
    behind fails the test.
    """
    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))
    yield
    leftovers = sorted(
        path.relative_to(tmp_path).as_posix()
        for path in tmp_path.rglob("*")
        if ".cs-" in path.name or path.suffix in {".rpt", ".out"}
    )
    assert not leftovers, f"files left behind: {leftovers}"


@pytest.fixture
def model_path(tmp_path: Path) -> Path:
    """A private copy of the bundled example model (subcatchment ``S1``, CMS units)."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    return Path(shutil.copyfile(example_model_path(), model_dir / "example.inp"))
