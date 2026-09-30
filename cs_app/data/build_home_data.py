"""Regenerate the example data shown on the home page from ``example.inp``.

Every chart on the home page is computed from the bundled example model
(subcatchment ``S1``), so the numbers match what users get from the code
snippets next to them. Run after changing ``example.inp`` or the package:

    cd cs_app && ../.venv/bin/python data/build_home_data.py

Writes, next to this file:
    df_slope.json, df_area.json, df_width.json   parameter sweeps (total runoff volume)
    example_hydrograph.json                      ``calculate_timeseries()`` records
"""

from __future__ import annotations

import json
from pathlib import Path

from catchment_simulation.catchment_features_simulation import FeaturesSimulation

DATA_DIR = Path(__file__).resolve().parent
MODEL = DATA_DIR / "example.inp"
SUBCATCHMENT_ID = "S1"
SIGNIFICANT_DIGITS = 6

# Output file -> (x key in the records, sweep method, start, stop, step).
# Keep the ranges in sync with the snippets in main/templates/main/main_view.html.
SWEEPS = {
    "df_slope.json": ("slope", "simulate_percent_slope", 1, 100, 1),
    "df_area.json": ("area", "simulate_area", 1, 100, 1),
    "df_width.json": ("width", "simulate_width", 10, 1000, 10),
}


def _round(value: float) -> float:
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def _write(name: str, records: list[dict]) -> None:
    (DATA_DIR / name).write_text(json.dumps(records, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    for name, (x_key, method, start, stop, step) in SWEEPS.items():
        with FeaturesSimulation(subcatchment_id=SUBCATCHMENT_ID, raw_file=str(MODEL)) as model:
            frame = getattr(model, method)(start=start, stop=stop, step=step)
        feature_column = frame.columns[-1]  # the swept parameter is the last column
        _write(
            name,
            [
                {x_key: _round(row[feature_column]), "runoff": _round(row["runoff"])}
                for _, row in frame.iterrows()
            ],
        )

    with FeaturesSimulation(subcatchment_id=SUBCATCHMENT_ID, raw_file=str(MODEL)) as model:
        timeseries = model.calculate_timeseries()
    _write(
        "example_hydrograph.json",
        [
            {"datetime": moment.isoformat(), **{key: _round(value) for key, value in row.items()}}
            for moment, row in timeseries.iterrows()
        ],
    )


if __name__ == "__main__":
    main()
