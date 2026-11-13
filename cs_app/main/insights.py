"""
What the sweep pages can tell the user about their model before a run.

* ``read_subcatchment_values``: the model's own value of every sweepable feature
  for one subcatchment, read with swmmio.
* ``suggest_range``: a sweep range around such a value that the forms accept.
* ``lab_query``: the query string that opens the interactive model on the "How it
  works" page with a subcatchment's values, converted to SI units.

Everything here is a pure function of its arguments except the swmmio read.
"""

from __future__ import annotations

import math
from urllib.parse import urlencode

import swmmio

from catchment_simulation.features import FEATURES
from main.forms import SWEEP_MAX_VALUE, SWEEP_MIN_STEP
from main.guide import PARAMETER_GUIDE
from main.units import _is_finite_number, _unit_labels, _unit_system

# A suggestion spans about half to one and a half times the model's value in this many steps.
SUGGESTED_STEPS = 10
# Percentages bounded by [0, 100] are always suggested over their whole range.
FULL_RANGE_FEATURES = frozenset({"PercImperv", "PctZero"})
# Range-based features without a usable model value (missing, zero or negative).
DEFAULT_RANGES: dict[str, tuple[float, float, float]] = {
    "Area": (1.0, 10.0, 1.0),
    "Width": (10.0, 100.0, 10.0),
    "PercSlope": (1.0, 10.0, 1.0),
    "CurbLength": (0.0, 100.0, 10.0),
}
SUGGESTED_RANGE_FEATURES = ("PercSlope", "Area", "Width", "PercImperv", "PctZero", "CurbLength")
_NICE_MANTISSAS = (1.0, 2.0, 2.5, 5.0, 10.0)
# Floating-point noise absorbed when snapping to the step grid.
_GRID_EPSILON = 1e-9

# Which unit each feature is measured in: a key of main.units._unit_labels, "%" or "".
FEATURE_DIMENSIONS = {
    "Area": "area",
    "PercImperv": "%",
    "Width": "length",
    "PercSlope": "%",
    "CurbLength": "length",
    "N-Imperv": "",
    "N-Perv": "",
    "S-Imperv": "storage",
    "S-Perv": "storage",
    "PctZero": "%",
}
# US customary -> SI for the interactive model, which works in ha, m and mm.
_US_TO_SI = {"area": 0.40468564224, "length": 0.3048, "storage": 25.4}
# Rain depth x area -> volume: mm x ha -> m³ (SI) and in x acre -> ft³ (US).
RAIN_VOLUME_FACTORS = {"SI": 10.0, "US": 3630.0}


def feature_units(flow_units: str | None) -> dict[str, str]:
    """The unit of every sweepable feature for a model with *flow_units*, e.g. ``{"Area": "ha"}``."""
    labels = _unit_labels(flow_units)
    return {
        feature: labels.get(dimension, dimension)
        for feature, dimension in FEATURE_DIMENSIONS.items()
    }


def read_subcatchment_values(inp_path: str, catchment: str) -> dict[str, float | None] | None:
    """
    The value of every feature of ``FEATURES`` for *catchment* in the model at *inp_path*.

    A value is None when its section has no row for the subcatchment or the cell is
    not a finite number. Returns None when the model has no such subcatchment.
    """
    inp = swmmio.Model(inp_path).inp
    frames = {}
    for spec in FEATURES.values():
        if spec.attribute not in frames:
            frame = getattr(inp, spec.attribute)
            frame.index = frame.index.astype(str)
            frames[spec.attribute] = frame
    if catchment not in frames["subcatchments"].index:
        return None

    values: dict[str, float | None] = {}
    for feature, spec in FEATURES.items():
        frame = frames[spec.attribute]
        value = None
        if catchment in frame.index and feature in frame.columns:
            cell = frame.at[catchment, feature]
            # A duplicated ID gives a Series; SWMM itself would reject the model.
            if not hasattr(cell, "__len__"):
                try:
                    value = float(cell)
                except (TypeError, ValueError):
                    value = None
        values[feature] = value if value is not None and math.isfinite(value) else None
    return values


def _nice_step(raw: float) -> float:
    """The number of the form 1, 2, 2.5 or 5 x 10^k closest to *raw* on a log scale."""
    exponent = math.floor(math.log10(raw))
    candidates = [mantissa * 10.0**exponent for mantissa in _NICE_MANTISSAS]
    return min(candidates, key=lambda candidate: abs(math.log(candidate / raw)))


def _clean(value: float) -> float:
    """Round away float noise, so 0.30000000000000004 becomes 0.3."""
    return float(round(value, 10))


def suggest_range(feature: str, value: float | None) -> tuple[float, float, float]:
    """
    A sweep range (start, stop, step) around the model's *value* of a range-based feature.

    About half to one and a half times the value in ten "nice" steps, kept inside the
    feature's physical range and the forms' bounds, so the forms accept it as it is.
    """
    spec = FEATURES[feature]  # type: ignore[index]
    if feature in FULL_RANGE_FEATURES:
        return (0.0, 100.0, 10.0)
    if not _is_finite_number(value) or value <= 0:
        return DEFAULT_RANGES[feature]

    upper = min(spec.maximum, SWEEP_MAX_VALUE)
    # Ten steps always fit below the upper bound (100 or 10000, so the cap is a nice number).
    step = min(_nice_step(max(value / SUGGESTED_STEPS, SWEEP_MIN_STEP)), upper / SUGGESTED_STEPS)
    start = math.floor(0.5 * value / step + _GRID_EPSILON) * step
    if start + step > upper:
        # A value at or above the forms' bound: take the top ten steps they allow.
        start = math.floor(upper / step + _GRID_EPSILON) * step - SUGGESTED_STEPS * step
    if start <= spec.minimum and not spec.min_inclusive:
        start = spec.minimum + step
    start = max(start, spec.minimum)
    steps = min(SUGGESTED_STEPS, math.floor((upper - start) / step + _GRID_EPSILON))
    return (_clean(start), _clean(start + steps * step), _clean(step))


def suggested_ranges(values: dict[str, float | None]) -> dict[str, list[float]]:
    """``suggest_range`` for every range-based feature, as JSON-ready lists."""
    return {
        feature: list(suggest_range(feature, values.get(feature)))
        for feature in SUGGESTED_RANGE_FEATURES
    }


def lab_query(
    values: dict[str, float | None], flow_units: str | None, vary: str | None = None
) -> str | None:
    """
    The query string of the interactive model for these feature *values*, in SI units.

    Keys are the ``lab_key`` of each feature in ``PARAMETER_GUIDE``; features without
    one or without a value are left out, and so is *vary* when it has no ``lab_key``.
    None when the model's unit system is unknown, since the values cannot be converted.
    """
    system = _unit_system(flow_units)
    if system not in RAIN_VOLUME_FACTORS:
        return None
    params = {}
    for feature, guide in PARAMETER_GUIDE.items():
        value = values.get(feature)
        if guide.lab_key is None or not _is_finite_number(value):
            continue
        if system == "US":
            value *= _US_TO_SI.get(FEATURE_DIMENSIONS[feature], 1.0)
        params[guide.lab_key] = f"{value:.10g}"
    vary_guide = PARAMETER_GUIDE.get(vary or "")
    if vary_guide is not None and vary_guide.lab_key is not None:
        params["vary"] = vary_guide.feature
    return urlencode(params)
