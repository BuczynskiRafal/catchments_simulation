"""Regenerate the example data shown on the home page from ``example.inp``.

Every chart on the home page is computed from the bundled example model
(subcatchment ``S1``), so the numbers match what users get from the code
snippets next to them. Run after changing ``example.inp`` or the package:

    uv run python cs_app/data/build_home_data.py

Writes, next to this file:
    df_slope.json, df_area.json, df_width.json   parameter sweeps (total runoff volume)
    example_hydrograph.json                      ``calculate_timeseries()`` records
"""

from __future__ import annotations

import json
from pathlib import Path

from catchment_simulation import FeaturesSimulation, build_range

DATA_DIR = Path(__file__).resolve().parent
MODEL = DATA_DIR / "example.inp"
SUBCATCHMENT_ID = "S1"
SIGNIFICANT_DIGITS = 6

# Output file -> (x key in the records, swept feature, start, stop, step).
# Keep the ranges in sync with the snippets in main/templates/main/main_view.html.
SWEEPS = {
    "df_slope.json": ("slope", "PercSlope", 1, 100, 1),
    "df_area.json": ("area", "Area", 1, 100, 1),
    "df_width.json": ("width", "Width", 10, 1000, 10),
}


def _round(value: float) -> float:
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def _write(name: str, records: list[dict]) -> None:
    (DATA_DIR / name).write_text(json.dumps(records, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    model = FeaturesSimulation(subcatchment_id=SUBCATCHMENT_ID, raw_file=MODEL)
    for name, (x_key, feature, start, stop, step) in SWEEPS.items():
        # Every call starts from the model on disk, so one instance serves every sweep.
        frame = model.simulate(feature, build_range(start, stop, step))
        _write(
            name,
            [
                {x_key: _round(row[feature]), "runoff": _round(row["runoff"])}
                for _, row in frame.iterrows()
            ],
        )

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
