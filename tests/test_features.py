"""Tests for the feature registry, value validation and build_range."""

from __future__ import annotations

import math
import re
from itertools import pairwise
from typing import get_args

import hypothesis.strategies as st
import pytest
from hypothesis import given

from catchment_simulation import (
    Feature,
    InvalidParameterError,
    SubareaFeature,
    SubcatchmentFeature,
    UnknownFeatureError,
    build_range,
)
from catchment_simulation.features import (
    FEATURES,
    MAX_RUNS,
    validate_feature,
    validate_values,
)

# --- registry and validation ---------------------------------------------------------------


def test_registry_covers_exactly_the_feature_literals() -> None:
    assert set(get_args(Feature)) == set(FEATURES)
    assert set(get_args(SubcatchmentFeature)) == {
        name for name, spec in FEATURES.items() if spec.section == "[SUBCATCHMENTS]"
    }
    assert set(get_args(SubareaFeature)) == {
        name for name, spec in FEATURES.items() if spec.section == "[SUBAREAS]"
    }
    assert {spec.attribute for spec in FEATURES.values()} == {"subcatchments", "subareas"}


@pytest.mark.parametrize(
    ("feature", "inside", "outside", "interval"),
    [
        ("Area", [1e-9, 1e6], [0, -1], "(0, inf)"),
        ("PercImperv", [0, 100], [-0.1, 100.1], "[0, 100]"),
        ("Width", [0.5], [0], "(0, inf)"),
        ("PercSlope", [0.01, 100], [0, 100.5], "(0, 100]"),
        ("CurbLength", [0, 10], [-1], "[0, inf)"),
        ("N-Imperv", [0.011], [0], "(0, inf)"),
        ("N-Perv", [0.8], [0], "(0, inf)"),
        ("S-Imperv", [0, 7.62], [-0.1], "[0, inf)"),
        ("S-Perv", [0], [-1], "[0, inf)"),
        ("PctZero", [0, 100], [101], "[0, 100]"),
    ],
)
def test_validate_values_enforces_physical_bounds(
    feature: Feature, inside: list[float], outside: list[float], interval: str
) -> None:
    assert validate_values(feature, inside) == [float(v) for v in inside]
    assert FEATURES[feature].interval == interval
    for value in outside:
        with pytest.raises(
            InvalidParameterError, match=rf"{feature} value .* {re.escape(interval)}"
        ):
            validate_values(feature, [*inside, value])


@pytest.mark.parametrize("value", [math.nan, math.inf, "wide"])
def test_validate_values_rejects_non_finite_and_non_numbers(value: object) -> None:
    with pytest.raises(InvalidParameterError, match="Width"):
        validate_values("Width", [10, value])  # type: ignore[list-item]


def test_validate_values_rejects_empty_input() -> None:
    with pytest.raises(InvalidParameterError, match="No Area values"):
        validate_values("Area", iter([]))


def test_validate_feature() -> None:
    assert validate_feature("PctZero") == "PctZero"
    with pytest.raises(UnknownFeatureError, match="'pctzero'.*PctZero") as excinfo:
        validate_feature("pctzero")
    assert excinfo.value.allowed == tuple(FEATURES)
    assert isinstance(excinfo.value, ValueError)


# --- build_range: pinned examples ----------------------------------------------------------


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((60, 100, 50), [60.0]),
        ((1, 2.9, 1), [1.0, 2.0]),
        ((0, 0.3, 0.1), [0.0, 0.1, 0.2, 0.3]),
        ((0.1, 0.3, 0.1), [0.1, 0.2, 0.3]),
        ((5, 5, 1), [5.0]),
        ((0, 100, 10), [float(v) for v in range(0, 101, 10)]),
        ((1, 1.01, 0.005), [1.0, 1.005, 1.01]),
    ],
)
def test_build_range_examples(args: tuple[float, float, float], expected: list[float]) -> None:
    values = build_range(*args)
    assert values == expected
    assert all(type(value) is float for value in values)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ((-1, 10, 1), "start"),
        ((0, 10, 0), "step"),
        ((0, 10, -1), "step"),
        ((100, 10, 1), r"start \(100.0\) must be <= stop \(10.0\)"),
        ((0, math.inf, 1), "finite"),
        ((0, 10, math.nan), "step"),
        ((1e17, 1e17 + 64, 1), "too small"),
    ],
)
def test_build_range_rejects_invalid_ranges(args: tuple[float, float, float], match: str) -> None:
    with pytest.raises(InvalidParameterError, match=match) as excinfo:
        build_range(*args)
    assert isinstance(excinfo.value, ValueError)


def test_build_range_enforces_max_runs() -> None:
    assert len(build_range(1, MAX_RUNS, 1)) == MAX_RUNS
    with pytest.raises(InvalidParameterError, match="more than max_runs=1000"):
        build_range(0, MAX_RUNS, 1)
    with pytest.raises(InvalidParameterError, match="max_runs=3"):
        build_range(0, 3, 1, max_runs=3)
    with pytest.raises(InvalidParameterError, match="max_runs"):
        build_range(0, 1e6, 1)
    with pytest.raises(InvalidParameterError, match="max_runs"):
        build_range(0, 1e308, 1e-300)


# --- build_range: properties ---------------------------------------------------------------

finite = {"allow_nan": False, "allow_infinity": False}


@st.composite
def ranges(draw: st.DrawFn) -> tuple[float, float, float]:
    start = draw(st.floats(min_value=0, max_value=1000, **finite))
    step = draw(st.floats(min_value=1e-3, max_value=100, **finite))
    stop = draw(st.floats(min_value=start, max_value=start + step * (MAX_RUNS - 1), **finite))
    return start, stop, step


@given(ranges())
def test_build_range_properties(args: tuple[float, float, float]) -> None:
    start, stop, step = args
    values = build_range(start, stop, step)

    assert values[0] == pytest.approx(start, abs=1e-10)
    assert all(start <= value <= stop for value in values)
    assert all(later > earlier for earlier, later in pairwise(values))
    assert all(
        later - earlier == pytest.approx(step, rel=1e-9, abs=1e-9)
        for earlier, later in pairwise(values)
    )
    # Complete: the next grid point would pass stop.
    assert values[-1] + step > stop - 1e-9 * step
    assert len(values) == math.floor((stop - start) / step + 1e-9) + 1


@given(ranges(), st.integers(min_value=1, max_value=MAX_RUNS))
def test_build_range_max_runs_property(args: tuple[float, float, float], max_runs: int) -> None:
    expected = math.floor((args[1] - args[0]) / args[2] + 1e-9) + 1
    if expected > max_runs:
        with pytest.raises(InvalidParameterError, match="max_runs"):
            build_range(*args, max_runs=max_runs)
    else:
        assert len(build_range(*args, max_runs=max_runs)) == expected
