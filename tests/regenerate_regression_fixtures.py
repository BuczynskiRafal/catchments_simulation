"""Rebuild ``tests/fixtures/regression_swmm.json`` from the current code.

Run it only when the result format changes on purpose, e.g. a column is
renamed or added. Never run it to make a failing regression test pass: a
moved number means the simulation changed, which needs an explanation, not a
new reference. The script enforces that by comparing every value the old and
the new fixture share and refusing to write if any moved by more than 1e-9
relative (``--force`` overrides that, for a deliberate, documented change).

Usage::

    uv run python tests/regenerate_regression_fixtures.py [--force]
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from catchment_simulation import FeaturesSimulation, example_model_path

FIXTURE = Path(__file__).parent / "fixtures" / "regression_swmm.json"

CASES: dict[str, tuple[str, dict[str, float]]] = {
    "area": ("simulate_area", {"start": 1, "stop": 5, "step": 1}),
    "percent_impervious": ("simulate_percent_impervious", {"start": 0, "stop": 50, "step": 10}),
    "n_imperv": ("simulate_n_imperv", {}),
}


def build() -> dict[str, Any]:
    """Run every case on a fresh instance and return the fixture content."""
    fixture: dict[str, Any] = {}
    with tempfile.TemporaryDirectory() as tmp:
        model = shutil.copyfile(example_model_path(), Path(tmp) / "example.inp")
        for name, (method, params) in CASES.items():
            frame = getattr(FeaturesSimulation("S1", model), method)(**params)
            fixture[name] = {
                "method": method,
                "params": params,
                "expected": frame.to_dict(orient="list"),
            }
    return fixture


def moved_values(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    """Describe every value present in both fixtures that differs by more than 1e-9 relative."""
    moved = []
    for case, reference in old.items():
        for column, expected in reference["expected"].items():
            actual = new.get(case, {}).get("expected", {}).get(column)
            if actual is None:
                continue
            if len(actual) != len(expected):
                moved.append(f"{case}.{column}: {len(expected)} rows -> {len(actual)} rows")
                continue
            for index, (before, after) in enumerate(zip(expected, actual, strict=True)):
                if not math.isclose(before, after, rel_tol=1e-9):
                    moved.append(f"{case}.{column}[{index}]: {before!r} -> {after!r}")
    return moved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="write even if numbers moved")
    args = parser.parse_args()

    new = build()
    if FIXTURE.exists():
        moved = moved_values(json.loads(FIXTURE.read_text()), new)
        if moved and not args.force:
            print("Numbers moved; not writing the fixture:", *moved, sep="\n  ")
            return 1
    text = json.dumps(new, indent=2)
    # One line per column and per parameter set keeps the file short and its diffs readable.
    text = re.sub(
        r"([\[{])\s+([^\[\]{}]*?)\s+([\]}])",
        lambda m: f"{m[1]}{' '.join(m[2].split())}{m[3]}",
        text,
    )
    FIXTURE.write_text(text + "\n")
    print(f"Wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
