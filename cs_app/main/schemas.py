"""Pydantic schemas for the Django web application.

These schemas provide runtime validation for data flowing between
the views, services, and external systems (email, file uploads).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

from catchment_simulation.schemas import SimulationMethodParams

SimulationOption = Literal[
    "simulate_percent_slope",
    "simulate_area",
    "simulate_width",
    "simulate_percent_impervious",
    "simulate_percent_zero_imperv",
    "simulate_curb_length",
    "simulate_n_imperv",
    "simulate_n_perv",
    "simulate_s_imperv",
    "simulate_s_perv",
]

TimeseriesFeature = Literal[
    "PercSlope",
    "Area",
    "Width",
    "PercImperv",
    "CurbLength",
]


class ContactMessage(BaseModel):
    """Validated payload for the contact form email service.

    Replaces the untyped ``dict[str, Any]`` that was passed to ``send_message()``.
    """

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid", frozen=True)

    email: EmailStr = Field(..., description="Sender email address")
    title: str = Field(..., min_length=1, max_length=200, description="Message subject")
    content: str = Field(..., min_length=1, max_length=5000, description="Message body")
    send_to_me: bool = Field(default=False, description="CC a copy to the sender")


class SimulationRequest(BaseModel):
    """Request schema for the ``POST /api/simulation/`` parameter sweep endpoint."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid", frozen=True)

    option: SimulationOption = Field(..., description="Simulation method name")
    catchment_name: str = Field(
        ..., min_length=1, max_length=100, description="Subcatchment identifier"
    )
    start: float | None = Field(
        default=None, ge=0, allow_inf_nan=False, description="Sweep range start value"
    )
    stop: float | None = Field(
        default=None, ge=0, allow_inf_nan=False, description="Sweep range stop value"
    )
    step: float | None = Field(
        default=None, gt=0, allow_inf_nan=False, description="Sweep range step size"
    )

    @field_validator("start", "stop", "step", mode="before")
    @classmethod
    def reject_bool_numbers(cls, value: object) -> object:
        """Reject boolean values for numeric sweep range fields."""
        if isinstance(value, bool):
            raise ValueError("Enter a number.")
        return value

    @model_validator(mode="after")
    def validate_sweep_range(self) -> "SimulationRequest":
        """Require start, stop, and step (with start <= stop) for range-based methods."""
        if self.option not in SimulationMethodParams.PREDEFINED_METHODS:
            if self.start is None or self.stop is None or self.step is None:
                raise ValueError(f"'{self.option}' requires start, stop, and step parameters")
            if self.start > self.stop:
                raise ValueError(f"start ({self.start}) must be <= stop ({self.stop})")
        return self


class TimeseriesRequest(BaseModel):
    """Request schema for the ``POST /api/timeseries/`` analysis endpoint."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid", frozen=True)

    mode: Literal["single", "sweep"] = Field(
        default="single", description="Single hydrograph run or parameter sweep"
    )
    catchment_name: str = Field(
        ..., min_length=1, max_length=100, description="Subcatchment identifier"
    )
    feature: TimeseriesFeature | None = Field(
        default=None, description="SWMM feature to vary in sweep mode"
    )
    start: float | None = Field(
        default=None, ge=0, allow_inf_nan=False, description="Sweep range start value"
    )
    stop: float | None = Field(
        default=None, ge=0, allow_inf_nan=False, description="Sweep range stop value"
    )
    step: float | None = Field(
        default=None, gt=0, allow_inf_nan=False, description="Sweep range step size"
    )

    @field_validator("start", "stop", "step", mode="before")
    @classmethod
    def reject_bool_numbers(cls, value: object) -> object:
        """Reject boolean values for numeric sweep range fields."""
        if isinstance(value, bool):
            raise ValueError("Enter a number.")
        return value

    @model_validator(mode="after")
    def validate_sweep_mode(self) -> "TimeseriesRequest":
        """Require feature, start, stop, and step (with start <= stop) in sweep mode."""
        if self.mode == "sweep":
            if self.feature is None or self.start is None or self.stop is None or self.step is None:
                raise ValueError("Sweep mode requires feature, start, stop, and step parameters")
            if self.start > self.stop:
                raise ValueError(f"start ({self.start}) must be <= stop ({self.stop})")
        return self
