"""Exceptions raised by catchment_simulation.

Invalid input raises subclasses of :class:`ValueError` and a failed SWMM run
raises a subclass of :class:`RuntimeError`, so callers that catch the built-in
types keep working.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class CatchmentSimulationError(Exception):
    """Base class for every error raised by this package."""


class InvalidParameterError(CatchmentSimulationError, ValueError):
    """A simulation parameter is invalid.

    Raised for a bad ``start``/``stop``/``step`` range, a value outside the
    physical range of a feature, too many runs, or an unknown option name.
    """


class UnknownFeatureError(InvalidParameterError):
    """The feature name is not one of the sweepable subcatchment features.

    Attributes
    ----------
    feature : str
        The rejected name.
    allowed : tuple of str
        The accepted feature names.
    """

    def __init__(self, feature: str, allowed: Iterable[str]) -> None:
        self.feature = feature
        self.allowed = tuple(allowed)
        super().__init__(
            f"Unknown feature {feature!r}. Allowed features: {', '.join(self.allowed)}"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), (self.feature, self.allowed)


class SubcatchmentNotFoundError(InvalidParameterError):
    """The subcatchment ID does not exist in the model.

    Attributes
    ----------
    subcatchment_id : str
        The requested ID.
    available : tuple of str
        The subcatchment IDs defined in the model.
    """

    def __init__(self, subcatchment_id: str, available: Iterable[str]) -> None:
        self.subcatchment_id = subcatchment_id
        self.available = tuple(available)
        super().__init__(
            f"Subcatchment {subcatchment_id!r} not found in model. "
            f"Available subcatchments: {list(self.available)}"
        )

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), (self.subcatchment_id, self.available)


class SimulationError(CatchmentSimulationError, RuntimeError):
    """SWMM failed to run the model.

    Attributes
    ----------
    report_errors : tuple of str
        The ``ERROR ...`` lines SWMM wrote to the run's report file; empty
        when the report holds none.
    """

    def __init__(self, message: str, report_errors: Iterable[str] = ()) -> None:
        self.report_errors = tuple(report_errors)
        self.message = message
        details = "".join(f"\n  {line}" for line in self.report_errors)
        super().__init__(message + details)

    def __reduce__(self) -> tuple[Any, ...]:
        return type(self), (self.message, self.report_errors)
