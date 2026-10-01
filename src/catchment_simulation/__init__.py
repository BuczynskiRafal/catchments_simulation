"""Sensitivity analysis of SWMM subcatchment features."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .analysis import runoff_volume, time_to_peak
from .catchment_features_simulation import FeaturesSimulation
from .examples import example_model_path
from .exceptions import (
    CatchmentSimulationError,
    InvalidParameterError,
    SimulationError,
    SubcatchmentNotFoundError,
    UnknownFeatureError,
)
from .features import Feature, SubareaFeature, SubcatchmentFeature, build_range
from .schemas import SimulationMethodParams, SimulationParams, SubcatchmentParams

__all__ = [
    "CatchmentSimulationError",
    "Feature",
    "FeaturesSimulation",
    "InvalidParameterError",
    "SimulationError",
    "SimulationMethodParams",
    "SimulationParams",
    "SubareaFeature",
    "SubcatchmentFeature",
    "SubcatchmentNotFoundError",
    "SubcatchmentParams",
    "UnknownFeatureError",
    "build_range",
    "example_model_path",
    "runoff_volume",
    "time_to_peak",
]

try:
    __version__ = version("catchment_simulation")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0+unknown"
