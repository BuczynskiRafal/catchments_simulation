"""Pydantic schemas for data validation in the catchment simulation package.

Provides strict runtime validation of simulation parameters and subcatchment data,
ensuring physically meaningful values before any computation begins.
"""

from typing import Literal

from pydantic import BaseModel, Field, PositiveFloat, model_validator


class SimulationParams(BaseModel):
    """Validated parameters for a simulation range sweep.

    Enforces that start <= stop and step > 0, replacing manual ``if`` checks
    with a declarative schema that raises ``ValidationError`` on bad input.
    """

    start: float = Field(ge=0, description="Start of the parameter range (>= 0)")
    stop: float = Field(ge=0, description="End of the parameter range (>= 0)")
    step: float = Field(gt=0, description="Step size (> 0)")

    @model_validator(mode="after")
    def start_le_stop(self) -> "SimulationParams":
        if self.start > self.stop:
            raise ValueError(f"start ({self.start}) must be <= stop ({self.stop})")
        return self


class SubcatchmentParams(BaseModel):
    """Physical parameters of a single subcatchment.

    All constraints reflect physically valid ranges used in SWMM modelling.
    """

    area: PositiveFloat = Field(..., description="Catchment area [ha]")
    slope: float = Field(..., gt=0, le=100, description="Terrain slope [%]")
    imperviousness: float = Field(..., ge=0, le=100, description="Percent impervious [%]")
    width: PositiveFloat = Field(..., description="Characteristic width [m]")
    n_imperv: float = Field(..., gt=0, description="Manning's n for impervious area [-]")
    n_perv: float = Field(..., gt=0, description="Manning's n for pervious area [-]")
    s_imperv: float = Field(..., ge=0, description="Depression storage, impervious [mm]")
    s_perv: float = Field(..., ge=0, description="Depression storage, pervious [mm]")
    pct_zero: float = Field(
        ...,
        ge=0,
        le=100,
        description="Percent of impervious area with zero depression storage [%]",
    )


class SimulationMethodParams(BaseModel):
    """Parameters submitted from the web form for a single simulation run."""

    method_name: Literal[
        "simulate_percent_slope",
        "simulate_area",
        "simulate_width",
        "simulate_percent_impervious",
        "simulate_percent_zero_imperv",
    ] = Field(..., description="Name of the simulation method to execute")
    start: int = Field(ge=1, le=100, description="Start value")
    stop: int = Field(ge=1, le=100, description="Stop value")
    step: int = Field(ge=1, le=100, description="Step size")
    catchment_name: str = Field(
        ..., min_length=1, max_length=100, description="Subcatchment identifier"
    )

    @model_validator(mode="after")
    def start_le_stop(self) -> "SimulationMethodParams":
        if self.start > self.stop:
            raise ValueError(f"start ({self.start}) must be <= stop ({self.stop})")
        return self
