"""The example SWMM model shipped with the package."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


def example_model_path() -> Path:
    """Return the path of the bundled example model.

    The model has one subcatchment, ``S1`` (5 ha, 25 % impervious), draining to
    a two-conduit network, with 12 hours of hourly rainfall in CMS flow units.
    The package must be installed as files on disk (the default for pip and uv).

    Returns
    -------
    pathlib.Path
        Path of ``catchment_simulation/data/example.inp``.

    Examples
    --------
    >>> from catchment_simulation import FeaturesSimulation, example_model_path
    >>> sim = FeaturesSimulation("S1", example_model_path())
    """
    return Path(str(files("catchment_simulation") / "data" / "example.inp"))
