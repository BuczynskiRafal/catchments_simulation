"""Package for simulating subcatchment features with Storm Water Management Model."""

from importlib.metadata import version

from .analysis import runoff_volume, time_to_peak
from .catchment_features_simulation import FeaturesSimulation
from .schemas import SimulationMethodParams, SimulationParams, SubcatchmentParams

__all__ = [
    "FeaturesSimulation",
    "SimulationParams",
    "SubcatchmentParams",
    "SimulationMethodParams",
    "time_to_peak",
    "runoff_volume",
]
__version__ = version("catchment_simulation")
