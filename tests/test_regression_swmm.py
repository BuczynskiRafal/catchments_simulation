"""Regression tests: sweep results must not change unexpectedly between versions.

The reference data lives in ``tests/fixtures/regression_swmm.json``; see
``tests/regenerate_regression_fixtures.py`` for when and how to rebuild it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from catchment_simulation import FeaturesSimulation

REFERENCE: dict[str, Any] = json.loads(
    (Path(__file__).parent / "fixtures" / "regression_swmm.json").read_text()
)
TOLERANCE = 1e-5


@pytest.mark.parametrize("case", sorted(REFERENCE))
def test_sweep_matches_reference(case: str, model_path: Path) -> None:
    reference = REFERENCE[case]
    sim = FeaturesSimulation("S1", model_path)

    frame = getattr(sim, reference["method"])(**reference["params"])

    assert list(frame.columns) == list(reference["expected"])
    for column, expected in reference["expected"].items():
        assert frame[column].tolist() == pytest.approx(expected, abs=TOLERANCE), column
