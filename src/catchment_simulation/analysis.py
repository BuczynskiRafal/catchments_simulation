"""Analytical helpers for SWMM subcatchment hydrograph data.

Functions in this module operate on ``pd.DataFrame`` objects returned by
:meth:`FeaturesSimulation.calculate_timeseries` and are intentionally
decoupled from the simulation class itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from datetime import datetime


def time_to_peak(
    timeseries: pd.DataFrame,
    column: str = "runoff",
    *,
    start: pd.Timestamp | datetime | None = None,
) -> pd.Timedelta:
    """Calculate the time from a reference instant to the peak value.

    Parameters
    ----------
    timeseries : pd.DataFrame
        DataFrame with a ``DatetimeIndex`` as returned by
        :meth:`FeaturesSimulation.calculate_timeseries`.
    column : str, optional
        Column to analyse, defaults to ``"runoff"``.
    start : pd.Timestamp | datetime | None, optional
        Reference instant, e.g. the simulation's ``START_DATE``/``START_TIME``.
        Defaults to the first timestamp of *timeseries*. A hydrograph from
        :meth:`FeaturesSimulation.calculate_timeseries` is stamped at the end
        of each routing step, so its first timestamp is one step after the
        simulation start.

    Returns
    -------
    pd.Timedelta
        Time elapsed from *start* to the first occurrence of the peak.

    Raises
    ------
    ValueError
        If *timeseries* is empty or has no ``DatetimeIndex``, *column* is not
        present, the column is all NaN or all zero (no meaningful peak), or
        *start* is later than the first timestamp.
    """
    if timeseries.empty:
        raise ValueError("Cannot compute time to peak on an empty timeseries.")
    index = timeseries.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(f"The timeseries needs a DatetimeIndex, not {type(index).__name__}.")
    if column not in timeseries.columns:
        raise ValueError(
            f"Column '{column}' not found. Available columns: {list(timeseries.columns)}"
        )
    values = timeseries[column]
    if values.isna().all():
        raise ValueError(f"All values in column '{column}' are NaN — no meaningful peak exists.")
    if values.max() == 0:
        raise ValueError(f"All values in column '{column}' are zero — no meaningful peak exists.")
    reference = index[0] if start is None else pd.Timestamp(start)
    if reference > index[0]:
        raise ValueError(f"start ({reference}) is later than the first timestamp ({index[0]}).")
    # argmax skips NaN and returns the position of the first maximum.
    peak_time = index[int(values.argmax())]
    return peak_time - reference


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
        If the index is not a monotonically increasing ``DatetimeIndex``,
        *column* is not present, or the (sliced) timeseries has fewer than
        2 data points.
    """
    if not isinstance(timeseries.index, pd.DatetimeIndex):
        raise ValueError(
            f"The timeseries needs a DatetimeIndex, not {type(timeseries.index).__name__}."
        )
    if not timeseries.index.is_monotonic_increasing:
        raise ValueError("The timeseries index must be sorted in increasing order.")
    if column not in timeseries.columns:
        raise ValueError(
            f"Column '{column}' not found. Available columns: {list(timeseries.columns)}"
        )
    ts = timeseries[column]
    if start is not None or end is not None:
        ts = ts.loc[slice(start, end)]
    if len(ts) < 2:
        raise ValueError("At least 2 data points are required for volume integration.")
    # The slice keeps the DatetimeIndex; the stubs only know it as an Index.
    times = pd.DatetimeIndex(ts.index)
    dt_seconds = (times[1:] - times[:-1]).total_seconds().to_numpy()
    flow = ts.to_numpy(dtype=float)
    avg_flow = (flow[:-1] + flow[1:]) / 2
    return float(np.sum(avg_flow * dt_seconds))
