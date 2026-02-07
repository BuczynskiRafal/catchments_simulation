"""Analytical helpers for SWMM subcatchment hydrograph data.

Functions in this module operate on ``pd.DataFrame`` objects returned by
:meth:`FeaturesSimulation.calculate_timeseries` and are intentionally
decoupled from the simulation class itself.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd


def time_to_peak(
    timeseries: pd.DataFrame,
    column: str = "runoff",
) -> pd.Timedelta:
    """Calculate the elapsed time from simulation start to the peak value.

    Parameters
    ----------
    timeseries : pd.DataFrame
        DataFrame with a ``DatetimeIndex`` as returned by
        :meth:`FeaturesSimulation.calculate_timeseries`.
    column : str, optional
        Column to analyse, defaults to ``"runoff"``.

    Returns
    -------
    pd.Timedelta
        Time elapsed from the first timestamp to the peak.

    Raises
    ------
    ValueError
        If *timeseries* is empty, *column* is not present, or all values
        in the column are zero (no meaningful peak).
    """
    if timeseries.empty:
        raise ValueError("Cannot compute time to peak on an empty timeseries.")
    if column not in timeseries.columns:
        raise ValueError(
            f"Column '{column}' not found. " f"Available columns: {list(timeseries.columns)}"
        )
    if timeseries[column].max() == 0:
        raise ValueError(f"All values in column '{column}' are zero — no meaningful peak exists.")
    peak_time = timeseries[column].idxmax()
    return peak_time - timeseries.index[0]


def runoff_volume(
    timeseries: pd.DataFrame,
    column: str = "runoff",
    start: pd.Timestamp | datetime | None = None,
    end: pd.Timestamp | datetime | None = None,
) -> float:
    """Compute total volume by integrating a flow-rate column over time.

    Uses the trapezoidal rule.  The result is in *flow-unit x seconds*
    (e.g. if the model uses CMS the volume is in cubic metres).

    .. note::
        Slicing with *start* / *end* uses ``pd.Series.loc`` which is
        **inclusive on both endpoints** for ``DatetimeIndex``.

    Parameters
    ----------
    timeseries : pd.DataFrame
        DataFrame with a ``DatetimeIndex`` as returned by
        :meth:`FeaturesSimulation.calculate_timeseries`.
    column : str, optional
        Column to integrate, defaults to ``"runoff"``.
    start : pd.Timestamp | datetime | None, optional
        Start of the integration window (inclusive).
        Defaults to the first timestamp.
    end : pd.Timestamp | datetime | None, optional
        End of the integration window (inclusive).
        Defaults to the last timestamp.

    Returns
    -------
    float
        Total volume over the selected interval.

    Raises
    ------
    ValueError
        If *column* is not present or the (sliced) timeseries has fewer
        than 2 data points.
    """
    if column not in timeseries.columns:
        raise ValueError(
            f"Column '{column}' not found. " f"Available columns: {list(timeseries.columns)}"
        )
    ts = timeseries[column]
    if start is not None or end is not None:
        ts = ts.loc[start:end]  # type: ignore[misc]
    if len(ts) < 2:
        raise ValueError("At least 2 data points are required for volume integration.")
    dt_seconds = (ts.index[1:] - ts.index[:-1]).total_seconds().values
    avg_flow = (ts.values[:-1] + ts.values[1:]) / 2
    return float(np.sum(avg_flow * dt_seconds))
