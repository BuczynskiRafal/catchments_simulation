"""Sweepable subcatchment features, their physical bounds and range construction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Literal, cast

from pydantic import ValidationError

from .exceptions import InvalidParameterError, UnknownFeatureError
from .schemas import SimulationParams

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

SubcatchmentFeature = Literal["Area", "PercImperv", "Width", "PercSlope", "CurbLength"]
"""Columns of the ``[SUBCATCHMENTS]`` section that can be swept."""

SubareaFeature = Literal["N-Imperv", "N-Perv", "S-Imperv", "S-Perv", "PctZero"]
"""Columns of the ``[SUBAREAS]`` section that can be swept."""

Feature = Literal[SubcatchmentFeature, SubareaFeature]
"""Any sweepable feature, named as the column in the SWMM input file."""

MAX_RUNS = 1000
"""Default upper limit on the number of SWMM runs one range may request."""


@dataclass(frozen=True)
class FeatureSpec:
    """Where a feature lives in the input file and which values are physical.

    Attributes
    ----------
    section : str
        INP section header, e.g. ``"[SUBCATCHMENTS]"``.
    attribute : str
        Name of the matching ``swmmio`` ``inp`` attribute.
    minimum, maximum : float
        Bounds of the physical range; ``maximum`` is inclusive.
    min_inclusive : bool
        Whether ``minimum`` itself is allowed.
    """

    section: str
    attribute: str
    minimum: float
    maximum: float = math.inf
    min_inclusive: bool = True

    def contains(self, value: float) -> bool:
        """Return whether *value* lies in the physical range."""
        above = value >= self.minimum if self.min_inclusive else value > self.minimum
        return above and value <= self.maximum

    @property
    def interval(self) -> str:
        """The physical range in interval notation, e.g. ``"(0, 100]"``."""
        left = "[" if self.min_inclusive else "("
        right = ")" if math.isinf(self.maximum) else "]"
        return f"{left}{self.minimum:g}, {self.maximum:g}{right}"


_SUBCATCHMENTS = ("[SUBCATCHMENTS]", "subcatchments")
_SUBAREAS = ("[SUBAREAS]", "subareas")

FEATURES: Mapping[Feature, FeatureSpec] = {
    "Area": FeatureSpec(*_SUBCATCHMENTS, minimum=0, min_inclusive=False),
    "PercImperv": FeatureSpec(*_SUBCATCHMENTS, minimum=0, maximum=100),
    "Width": FeatureSpec(*_SUBCATCHMENTS, minimum=0, min_inclusive=False),
    "PercSlope": FeatureSpec(*_SUBCATCHMENTS, minimum=0, maximum=100, min_inclusive=False),
    "CurbLength": FeatureSpec(*_SUBCATCHMENTS, minimum=0),
    "N-Imperv": FeatureSpec(*_SUBAREAS, minimum=0, min_inclusive=False),
    "N-Perv": FeatureSpec(*_SUBAREAS, minimum=0, min_inclusive=False),
    "S-Imperv": FeatureSpec(*_SUBAREAS, minimum=0),
    "S-Perv": FeatureSpec(*_SUBAREAS, minimum=0),
    "PctZero": FeatureSpec(*_SUBAREAS, minimum=0, maximum=100),
}
"""Registry of every sweepable feature. Bounds match :class:`~.schemas.SubcatchmentParams`."""


def validate_feature(feature: str) -> Feature:
    """Return *feature* if it names a sweepable feature.

    Raises
    ------
    UnknownFeatureError
        If *feature* is not a key of :data:`FEATURES` (names are case-sensitive).
    """
    if feature not in FEATURES:
        raise UnknownFeatureError(feature, FEATURES)
    return cast("Feature", feature)


def validate_values(feature: Feature, values: Iterable[float]) -> list[float]:
    """Return *values* as floats after checking them against the feature's physical range.

    Parameters
    ----------
    feature : Feature
        A validated feature name.
    values : iterable of float
        The values to sweep, in the model's units.

    Returns
    -------
    list of float
        The values as plain Python floats, in the given order.

    Raises
    ------
    InvalidParameterError
        If *values* is empty, or a value is not a finite number or lies
        outside the feature's physical range.
    """
    spec = FEATURES[feature]
    try:
        checked = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise InvalidParameterError(f"{feature} values must be numbers: {exc}") from exc
    if not checked:
        raise InvalidParameterError(f"No {feature} values to simulate.")
    for value in checked:
        if not math.isfinite(value) or not spec.contains(value):
            raise InvalidParameterError(
                f"{feature} value {value:g} is outside the physical range {spec.interval}."
            )
    return checked


def build_range(start: float, stop: float, step: float, *, max_runs: int = MAX_RUNS) -> list[float]:
    """Build the inclusive range ``start, start + step, ...`` that never exceeds *stop*.

    *stop* is included when it lies on the grid within 1e-9 of a step, which
    absorbs floating-point noise. Values are rounded to 10 decimal places, so
    ``build_range(0, 0.3, 0.1)`` gives ``[0.0, 0.1, 0.2, 0.3]`` and the values
    can be used as dictionary keys. Every value lies in ``[start, stop]``.

    Parameters
    ----------
    start : float
        First value, >= 0.
    stop : float
        Upper limit, >= start.
    step : float
        Spacing, > 0.
    max_runs : int, optional
        Largest number of values accepted, defaults to :data:`MAX_RUNS`.

    Returns
    -------
    list of float
        Strictly increasing plain Python floats.

    Raises
    ------
    InvalidParameterError
        If the arguments are not finite, violate :class:`~.schemas.SimulationParams`,
        give more than *max_runs* values, or a step too small to give distinct values.
    """
    try:
        params = SimulationParams(start=start, stop=stop, step=step)
    except ValidationError as exc:
        messages = "; ".join(
            f"{'.'.join(map(str, error['loc']))}: {error['msg']}" if error["loc"] else error["msg"]
            for error in exc.errors()
        )
        raise InvalidParameterError(f"Invalid range: {messages}") from exc
    start, stop, step = params.start, params.stop, params.step
    if not all(map(math.isfinite, (start, stop, step))):
        raise InvalidParameterError("start, stop and step must be finite.")

    steps = (stop - start) / step + 1e-9
    # Compared before flooring: a huge span over a tiny step overflows to inf.
    if steps >= max_runs:
        raise InvalidParameterError(
            f"The range gives more than max_runs={max_runs} runs; use a larger step."
        )
    count = math.floor(steps) + 1
    # Clamping keeps the guarantee when rounding or the tolerance pushes a value past a bound.
    values = [min(max(round(start + i * step, 10), start), stop) for i in range(count)]
    if any(later <= earlier for earlier, later in pairwise(values)):
        raise InvalidParameterError(
            f"step={step:g} is too small to give distinct values between {start:g} and {stop:g}."
        )
    return values
