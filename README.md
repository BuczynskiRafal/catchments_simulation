# catchment-simulation

[![PyPI version](https://img.shields.io/pypi/v/catchment-simulation.svg)](https://pypi.org/project/catchment-simulation/)
[![Python versions](https://img.shields.io/pypi/pyversions/catchment-simulation.svg)](https://pypi.org/project/catchment-simulation/)
[![CI](https://github.com/BuczynskiRafal/catchments_simulation/actions/workflows/python-package.yml/badge.svg?branch=main)](https://github.com/BuczynskiRafal/catchments_simulation/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/BuczynskiRafal/catchments_simulation/branch/main/graph/badge.svg?token=40S5AYWXK6)](https://codecov.io/gh/BuczynskiRafal/catchments_simulation)
[![Documentation Status](https://readthedocs.org/projects/catchments-simulation/badge/?version=latest)](https://catchments-simulation.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/BuczynskiRafal/catchments_simulation/blob/main/LICENSE)

Sensitivity sweeps of subcatchment features in [EPA SWMM](https://www.epa.gov/water-research/storm-water-management-model-swmm)
models. `catchment-simulation` runs a SWMM model many times, each time with one
feature of one subcatchment (area, imperviousness, width, slope, Manning's n,
depression storage, ...) set to a different value, and returns that
subcatchment's runoff statistics or hydrographs as pandas objects. It answers
questions such as "how much does peak runoff change if this catchment is 20 %
more impervious?" without editing `.inp` files by hand. SWMM runs through
[pyswmm](https://github.com/pyswmm/pyswmm) and the input file is edited with
[swmmio](https://github.com/pyswmm/swmmio).

## Installation

```bash
pip install catchment-simulation
```

or, in a [uv](https://docs.astral.sh/uv/) project:

```bash
uv add catchment-simulation
```

Python 3.10 or newer is required. The SWMM engine comes with the
`swmm-toolkit` wheel that pyswmm depends on; nothing else needs installing.

## Quickstart

The package ships a small example model (one subcatchment, `S1`, 5 ha, 25 %
impervious, 12 hours of rain, CMS flow units), so this runs as-is:

```python
from catchment_simulation import FeaturesSimulation, example_model_path

sim = FeaturesSimulation("S1", example_model_path())

results = sim.simulate_percent_impervious(start=0, stop=100, step=25)
print(results[["PercImperv", "runoff", "peak_runoff_rate", "infiltration"]].round(2))
```

```text
   PercImperv   runoff  peak_runoff_rate  infiltration
0         0.0  2580.37              0.16        555.89
1        25.0  3360.36              0.19        406.71
2        50.0  4021.30              0.23        261.11
3        75.0  4520.01              0.26        122.05
4       100.0  4779.30              0.28          0.00
```

Use your own model by passing its path (a `str` or `pathlib.Path`) and a
subcatchment ID: `FeaturesSimulation("S12", "models/catchment.inp")`.

Each result is a `pandas.DataFrame` with one row per run: the swept feature
first, named as its column in the SWMM input file, then the subcatchment's
statistics.

| Column             | Meaning                                    | SI models  | US models |
|--------------------|--------------------------------------------|------------|-----------|
| `runoff`           | total runoff volume                        | m³         | ft³       |
| `peak_runoff_rate` | peak runoff rate                           | flow units | flow units|
| `infiltration`     | total infiltration volume                  | m³         | ft³       |
| `evaporation`      | total evaporation volume                   | m³         | ft³       |
| `precipitation`    | total precipitation depth                  | mm         | in        |
| `runon`            | total runon volume from upstream areas     | m³         | ft³       |

`sim.calculate()` returns the same statistics for the unmodified model as a
`dict`.

## Sweep methods

| Method                                                 | Feature column           | Unit (SI / US)  | Default values                       |
|--------------------------------------------------------|--------------------------|-----------------|--------------------------------------|
| `simulate_area(start, stop, step)`                     | `Area`                   | ha / ac         | 1 to 10, step 1                      |
| `simulate_percent_impervious(start, stop, step)`       | `PercImperv`             | %               | 0 to 100, step 10                    |
| `simulate_width(start, stop, step)`                    | `Width`                  | m / ft          | 10 to 100, step 10                   |
| `simulate_percent_slope(start, stop, step)`            | `PercSlope`              | %               | 10 to 100, step 10                   |
| `simulate_curb_length(start, stop, step)`              | `CurbLength`             | m / ft          | 0 to 100, step 10                    |
| `simulate_percent_zero_imperv(start, stop, step)`      | `PctZero`                | %               | 0 to 100, step 10                    |
| `simulate_n_imperv()`, `simulate_manning_n("Imperv")`  | `N-Imperv`               | Manning's n     | 15 values, 0.011 to 0.8 ¹            |
| `simulate_n_perv()`, `simulate_manning_n("Perv")`      | `N-Perv`                 | Manning's n     | 15 values, 0.011 to 0.8 ¹            |
| `simulate_s_imperv()`, `simulate_destore("Imperv")`    | `S-Imperv`               | mm / in         | 1.27, 2.54, 5.08, 7.62 mm ²          |
| `simulate_s_perv()`, `simulate_destore("Perv")`        | `S-Perv`                 | mm / in         | 1.27, 2.54, 5.08, 7.62 mm ²          |
| `simulate_subcatchment(feature, start, stop, step)`    | any of the above         | as above        | 0 to 100, step 10 ³                  |

¹ `FeaturesSimulation.MANNING_N` maps each value to its surface, from
McCuen et al. (1996). ² 0.05, 0.1, 0.2 and 0.3 in for US models (ASCE, 1992).
³ Pass a `start` inside the feature's physical range: 0 is rejected for
`Area`, `Width`, `PercSlope`, `N-Imperv` and `N-Perv`.

A range is `start, start + step, ...` and never goes past `stop`; `stop` is
included when it falls on the step grid. Every value is checked against the
feature's physical range (for example `PercImperv` in [0, 100], `Width` > 0,
`PercSlope` in (0, 100]) before SWMM runs, and a range may give at most 1000
runs. `catchment_simulation.build_range(start, stop, step)` builds the same
values for your own use.

## Arbitrary values

`simulate(feature, values)` runs any sequence of values, so non-linear or
hand-picked designs need no range:

```python
from catchment_simulation import FeaturesSimulation, example_model_path

sim = FeaturesSimulation("S1", example_model_path())

widths = sim.simulate("Width", [5, 20, 80, 320])
roughness = sim.simulate("N-Perv", [0.05, 0.15, 0.4])
print(widths[["Width", "runoff", "peak_runoff_rate"]].round(3))
```

```text
   Width    runoff  peak_runoff_rate
0    5.0  1442.822             0.082
1   20.0  2097.928             0.115
2   80.0  3190.788             0.182
3  320.0  4013.534             0.239
```

## Hydrographs and analysis

`calculate_timeseries()` returns the subcatchment's hydrograph for the
unmodified model, and `simulate_timeseries(feature, values)` or
`simulate_subcatchment_timeseries(feature, start, stop, step)` return one
hydrograph per value as a `dict`. Each hydrograph is a `DataFrame` indexed by
time (the end of each routing step) with the columns `rainfall` and
`infiltration_loss` (mm/h or in/h), `evaporation_loss` (mm/day or in/day),
`runoff` and `runon` (flow units).

`runoff_volume` integrates a flow column over time and `time_to_peak`
measures the time to its maximum:

```python
import pandas as pd

from catchment_simulation import (
    FeaturesSimulation,
    example_model_path,
    runoff_volume,
    time_to_peak,
)

sim = FeaturesSimulation("S1", example_model_path())
hydrographs = sim.simulate_subcatchment_timeseries("PercImperv", start=0, stop=100, step=50)

start = pd.Timestamp("2022-06-17 00:00")  # START_DATE and START_TIME of the model
for imperviousness, hydrograph in hydrographs.items():
    volume = runoff_volume(hydrograph)  # flow units x seconds: m³ for CMS
    peak = time_to_peak(hydrograph, start=start)
    print(f"{imperviousness:5.1f} %  {volume:7.1f} m³  peak after {peak}")
```

```text
  0.0 %   2569.9 m³  peak after 0 days 09:56:00
 50.0 %   4013.2 m³  peak after 0 days 08:56:00
100.0 %   4773.0 m³  peak after 0 days 08:56:00
```

Without `start=`, `time_to_peak` counts from the first timestamp of the
hydrograph, which is one routing step after the simulation start. Both
functions take a `column` argument and `runoff_volume` takes an optional
`start`/`end` window (inclusive).

## Units

Values are read and written in the units of the model, which SWMM derives from
its `FLOW_UNITS` option (`sim.flow_units`):

- SI: `CMS`, `LPS`, `MLD` — hectares, metres, millimetres, cubic metres.
- US: `CFS`, `GPM`, `MGD` — acres, feet, inches, cubic feet.

A model without `FLOW_UNITS` uses SWMM's default, `CFS`. The depression storage
sweeps pick millimetre or inch values to match.

## Errors

Every exception raised by the package derives from `CatchmentSimulationError`:

```text
CatchmentSimulationError
├── InvalidParameterError (also a ValueError)   bad range, value out of bounds, too many runs
│   ├── UnknownFeatureError                     feature name not sweepable (.feature, .allowed)
│   └── SubcatchmentNotFoundError               ID not in the model (.subcatchment_id, .available)
└── SimulationError (also a RuntimeError)       SWMM failed (.report_errors: ERROR lines of the .rpt)
```

Invalid input is rejected before the first SWMM run:

```python
from catchment_simulation import FeaturesSimulation, InvalidParameterError, example_model_path

sim = FeaturesSimulation("S1", example_model_path())
try:
    sim.simulate_percent_impervious(start=60, stop=120, step=20)
except InvalidParameterError as error:
    print(error)
```

```text
PercImperv value 120 is outside the physical range [0, 100].
```

A missing model file raises `FileNotFoundError`.

## How runs work

A `FeaturesSimulation` reads the model file once, when it is created: changes
to the file afterwards are not seen, so create a new instance to run them.
Every call writes that model to a uniquely named file next to the original, so
relative `FILE` references (rain gauges, time series) keep resolving, runs
SWMM on the copy and deletes the copy with its `.rpt` and `.out` files before
returning. The original file is never modified and every call starts from the
model as it was read, so results do not depend on earlier calls. If the
model's directory is not writable, the copy goes to a new private temporary
directory, where relative `FILE` references cannot resolve: the run fails with
a `SimulationError` that says so.

SWMM runs one simulation per process at a time, so a `FeaturesSimulation` is
not thread-safe. To run sweeps in parallel, use processes:

```python
from concurrent.futures import ProcessPoolExecutor

from catchment_simulation import FeaturesSimulation, example_model_path


def sweep(feature):
    sim = FeaturesSimulation("S1", example_model_path())
    return sim.simulate_subcatchment(feature, start=10, stop=100, step=30)


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=2) as pool:
        for frame in pool.map(sweep, ["Width", "PercSlope"]):
            print(frame.iloc[:, :3].round(3).to_string(index=False))
```

## Documentation

- Full documentation and API reference: <https://catchments-simulation.readthedocs.io>
- Release notes: [CHANGELOG.md](https://github.com/BuczynskiRafal/catchments_simulation/blob/main/CHANGELOG.md)
- Bugs and feature requests: [issue tracker](https://github.com/BuczynskiRafal/catchments_simulation/issues)
- Contributing: [CONTRIBUTING.md](https://github.com/BuczynskiRafal/catchments_simulation/blob/main/CONTRIBUTING.md)

The repository also holds a Django web application built on this package
(upload a model, run sweeps, plot and download results, compare SWMM with a
neural network). It is not part of the PyPI package; see
[cs_app/README.md](https://github.com/BuczynskiRafal/catchments_simulation/blob/main/cs_app/README.md).

## Citing

If you use `catchment-simulation` in research, please cite it. The repository
contains a [CITATION.cff](https://github.com/BuczynskiRafal/catchments_simulation/blob/main/CITATION.cff)
file, and GitHub's "Cite this repository" button gives APA and BibTeX formats.

## License

MIT, see [LICENSE](https://github.com/BuczynskiRafal/catchments_simulation/blob/main/LICENSE).
