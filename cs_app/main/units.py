"""Hydrological unit system classification, chart axis labels, and hydrograph metrics."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from catchment_simulation.analysis import runoff_volume, time_to_peak
from catchment_simulation.catchment_features_simulation import US_FLOW_UNITS

SI_FLOW_UNITS = frozenset({"CMS", "LPS", "MLD"})

# runoff_volume() integrates flow over seconds, i.e. returns flow unit x s; these
# factors turn that into the volume unit of the model's unit system (m³ or ft³).
_US_GALLON_FT3 = 0.133680556
RUNOFF_VOLUME_FACTORS = {
    "CMS": 1.0,
    "LPS": 1e-3,
    "MLD": 1e3 / 86_400,
    "CFS": 1.0,
    "GPM": _US_GALLON_FT3 / 60,
    "MGD": 1e6 * _US_GALLON_FT3 / 86_400,
}

# SWMM reports a subcatchment's total runoff in 10^6 L (SI models) or 10^6 US gal
# (US models); swmmio names that column TotalRunoffMG either way.
REPORT_RUNOFF_FACTORS = {"SI": 1e3, "US": 1e6 * _US_GALLON_FT3}
# The ANN model was trained exclusively on SI runoff (10^6 L = 10^3 m³); US models
# therefore convert 10^3 m³ -> ft³ rather than 10^6 US gal -> ft³.
_M3_TO_FT3 = 1.0 / (0.3048**3)
ANN_RUNOFF_FACTORS = {"SI": 1e3, "US": 1e3 * _M3_TO_FT3}


def _is_finite_number(value: object) -> bool:
    return (
        isinstance(value, int | float) and not isinstance(value, bool) and bool(np.isfinite(value))
    )


def _normalize_flow_units(flow_units: str | None) -> str:
    """Normalize FLOW_UNITS token for comparisons and labels."""
    return str(flow_units or "").strip().upper()


def _extract_options_flow_units(options: object) -> str:
    """Extract normalized FLOW_UNITS from swmmio Model.inp.options, defaulting to SWMM's CFS."""
    entries: list[tuple[object, object]] = []
    if isinstance(options, pd.DataFrame):
        if not options.columns.empty:
            value_column = "Value" if "Value" in options.columns else options.columns[0]
            entries = list(options[value_column].items())
    elif isinstance(options, dict):
        entries = list(options.items())
    # SWMM keywords are case-insensitive; omitted FLOW_UNITS defaults to CFS in SWMM 5.
    flow_units = {str(key).strip().upper(): value for key, value in entries}.get("FLOW_UNITS")
    normalized = _normalize_flow_units(str(flow_units) if flow_units is not None else None)
    return normalized or "CFS"


def _unit_system(flow_units: str | None) -> str:
    """Classify FLOW_UNITS into SI/US/UNKNOWN buckets."""
    normalized = _normalize_flow_units(flow_units)
    if normalized in SI_FLOW_UNITS:
        return "SI"
    if normalized in US_FLOW_UNITS:
        return "US"
    return "UNKNOWN"


def _unit_labels(flow_units: str | None) -> dict[str, str]:
    """Return display units for chart labels.

    SWMM reports rainfall and infiltration rates per hour but evaporation rates per day.
    """
    normalized = _normalize_flow_units(flow_units)
    system = _unit_system(normalized)
    if system == "SI":
        return {
            "length": "m",
            "area": "ha",
            "storage": "mm",
            "depth_rate": "mm/h",
            "evaporation_rate": "mm/day",
            "volume": "m³",
            "flow_rate": normalized,
        }
    if system == "US":
        return {
            "length": "ft",
            "area": "acre",
            "storage": "in",
            "depth_rate": "in/h",
            "evaporation_rate": "in/day",
            "volume": "ft³",
            "flow_rate": normalized,
        }
    return {
        "length": "model length units",
        "area": "model area units",
        "storage": "model storage units",
        "depth_rate": "model depth/time units",
        "evaporation_rate": "model evaporation rate units",
        "volume": "model volume units",
        "flow_rate": normalized or "model flow units",
    }


def _build_simulation_axis_labels(
    feature_name: str, y_columns: list[str], flow_units: str | None
) -> tuple[str, dict[str, str]]:
    """Build X and Y labels for simulation charts."""
    units = _unit_labels(flow_units)
    x_templates = {
        "PercSlope": "Percent Slope [%]",
        "Area": "Area [{area}]",
        "Width": "Width [{length}]",
        "PercImperv": "Impervious Area [%]",
        "PctZero": "Zero-Impervious Area [%]",
        "CurbLength": "Curb Length [{length}]",
        "N-Imperv": "Manning n (Impervious) [-]",
        "N-Perv": "Manning n (Pervious) [-]",
        "S-Imperv": "Depression Storage (Impervious) [{storage}]",
        "S-Perv": "Depression Storage (Pervious) [{storage}]",
    }
    y_templates = {
        "runoff": "Total Runoff Volume [{volume}]",
        "peak_runoff_rate": "Peak Runoff Rate [{flow_rate}]",
        "infiltration": "Total Infiltration Volume [{volume}]",
        "evaporation": "Total Evaporation Volume [{volume}]",
    }

    x_template = x_templates.get(feature_name)
    x_label = x_template.format(**units) if x_template else feature_name or "Parameter"
    y_labels = {}
    for column in y_columns:
        y_template = y_templates.get(column)
        y_labels[column] = y_template.format(**units) if y_template else column
    return x_label, y_labels


def _build_timeseries_axis_labels(
    columns: list[str], flow_units: str | None
) -> tuple[str, dict[str, str]]:
    """Build X and Y labels for timeseries charts."""
    units = _unit_labels(flow_units)
    y_templates = {
        "rainfall": "Rainfall Intensity [{depth_rate}]",
        "runoff": "Runoff Rate [{flow_rate}]",
        "infiltration_loss": "Infiltration Loss [{depth_rate}]",
        "evaporation_loss": "Evaporation Loss [{evaporation_rate}]",
        "runon": "Runon Rate [{flow_rate}]",
    }
    y_labels = {}
    for column in columns:
        y_template = y_templates.get(column)
        y_labels[column] = y_template.format(**units) if y_template else column
    return "Time", y_labels


def _report_runoff_unit(flow_units: str | None) -> tuple[float, str]:
    """Return the factor from the report's total runoff to the display volume unit, and that unit."""
    system = _unit_system(flow_units)
    if system in REPORT_RUNOFF_FACTORS:
        return REPORT_RUNOFF_FACTORS[system], _unit_labels(flow_units)["volume"]
    return 1.0, "10⁶ L or gal"


def _ann_runoff_factor(flow_units: str | None) -> float:
    """Return the factor from the ANN output (10^6 L = 10^3 m³) to the model's display volume unit."""
    system = _unit_system(flow_units)
    return ANN_RUNOFF_FACTORS.get(system, 1.0)


def _runoff_volume_unit(flow_units: str | None) -> tuple[float, str]:
    """Return the factor from runoff_volume() to the display volume unit, and that unit."""
    normalized = _normalize_flow_units(flow_units)
    units = _unit_labels(normalized)
    if normalized in RUNOFF_VOLUME_FACTORS:
        return RUNOFF_VOLUME_FACTORS[normalized], units["volume"]
    return 1.0, f"{units['flow_rate']} × s"


def _format_duration(delta: pd.Timedelta) -> str:
    """Format a duration as e.g. "1 d 8 h 55 min", leaving out zero parts."""
    remaining = round(delta.total_seconds())
    parts = []
    for size, unit in ((86_400, "d"), (3_600, "h"), (60, "min"), (1, "s")):
        amount, remaining = divmod(remaining, size)
        if amount:
            parts.append(f"{amount} {unit}")
    return " ".join(parts) or "0 min"


def _format_parameter_value(value: float) -> str:
    """Shortest text for a swept value, without float noise: 0.0 -> "0", 0.30000000000000004 -> "0.3"."""
    return f"{value:.10g}"


def _timeseries_records(ts_df: pd.DataFrame) -> list[dict]:
    """JSON-safe rows of one run: the datetime index as text, NaN as null."""
    frame = ts_df.reset_index()
    frame["datetime"] = frame["datetime"].astype(str)
    return json.loads(frame.to_json(orient="records"))


def _hydrograph_metrics(ts_df: pd.DataFrame, volume_factor: float) -> dict:
    """Peak, time to peak and volume of one run's runoff; None where a value is undefined."""
    runoff = ts_df["runoff"]
    try:
        delay: pd.Timedelta | None = time_to_peak(ts_df, column="runoff")
    except ValueError:  # no runoff at all, so there is no peak to time
        delay = None
    try:
        volume = runoff_volume(ts_df, column="runoff") * volume_factor
    except ValueError:  # fewer than two timesteps
        volume = None
    peak = float(runoff.max()) if not runoff.empty else None
    has_peak = delay is not None and not pd.isna(delay)
    peak_idx = runoff.idxmax() if has_peak else None
    return {
        "peak": peak if _is_finite_number(peak) else None,
        "peak_at": peak_idx.strftime("%Y-%m-%d %H:%M") if hasattr(peak_idx, "strftime") else None,
        "time_to_peak": _format_duration(delay) if delay is not None and has_peak else None,
        "time_to_peak_seconds": delay.total_seconds() if delay is not None and has_peak else None,
        "volume": volume if _is_finite_number(volume) else None,
    }


def _hydrograph_summary(records: list[dict], volume_factor: float) -> dict:
    """Headline numbers of the example event, computed like a timeseries run's metrics."""
    frame = pd.DataFrame(records)
    frame.index = pd.to_datetime(frame.pop("datetime"))
    return {
        "peak_rainfall": float(frame["rainfall"].max()),
        **_hydrograph_metrics(frame, volume_factor),
    }
