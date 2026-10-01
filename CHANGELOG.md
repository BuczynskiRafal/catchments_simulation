# Changelog

All notable changes to the `catchment-simulation` package are documented in
this file. The Django web application in `cs_app/` is not released and is not
covered here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
While the major version is 0, a minor release (0.x.0) may contain breaking
changes; they are listed under **Changed** and **Removed**.

## [0.1.0] - Unreleased

This release fixes results that could be silently wrong in 0.0.7. Single
sweeps on a fresh `FeaturesSimulation` give the same numbers as before; what
changes is the behaviour on repeated use, at range boundaries and on invalid
input, plus the result column names. Read **Changed** before upgrading.

### Fixed

- A second sweep of the same section on one instance (for example
  `simulate_area()` then `simulate_width()`) ran on the values left by the
  first sweep and returned wrong results. Every sweep now starts from the
  model file on disk.
- The working copy of the model was written to the current working directory
  under a fixed name: models with relative `FILE` references (rain gauge or
  time series files) failed with SWMM `ERROR 361`, two instances on
  same-named models in different directories overwrote each other's files,
  and `.inp`/`.rpt`/`.out` files were left behind. Each call now copies the
  model to a uniquely named file next to the original (or to the system temp
  directory when that directory is read-only) and deletes the copy, its report
  and its output before returning. Nothing is written to the working
  directory.
- A typo in a parameter, such as `simulate_destore("imperv")`, was silently
  ignored and returned a constant result. It now raises
  `InvalidParameterError`, and unknown feature names raise
  `UnknownFeatureError`.
- Ranges overshot `stop` (`simulate_area(1, 2.9, 1)` ran Area 3.0,
  `simulate_percent_impervious(60, 100, 50)` ran 110 %) and produced
  floating-point keys such as `0.30000000000000004`. Ranges now never pass
  `stop` and hold values rounded to 10 decimal places.
- Values outside a feature's physical range (`Width` 0, `PercSlope` 0,
  `PercImperv` 110, ...) were passed to SWMM. They now raise
  `InvalidParameterError` before the first run.
- `pathlib.Path` model paths raised `AttributeError`; any `os.PathLike` is
  accepted now.
- Depression storage sweeps wrote millimetre values into US-unit models
  (25.4 times too large). Models with `CFS`, `GPM` or `MGD` flow units are now
  swept in inches.
- Sweeping a `[SUBAREAS]` feature (`N-Imperv`, `N-Perv`, `S-Imperv`,
  `S-Perv`, `PctZero`) of a subcatchment that has no `[SUBAREAS]` row wrote
  an incomplete row into the model and failed inside SWMM with `ERROR 203`.
  It now raises `InvalidParameterError` before the first run. (SWMM itself
  runs such a subcatchment with zero runoff and no warning.)
- `DEPRESSION_STORAGE_VALUES` held the millimetre conversion with
  floating-point noise (`7.619999999999999`); the values are rounded to
  0.01 mm now: 1.27, 2.54, 5.08 and 7.62. Results are unchanged.
- `time_to_peak` did not reject an all-NaN column, and `runoff_volume`
  integrated data with an unsorted or non-datetime index. Both raise
  `ValueError` now.

### Changed

- **Result columns.** The swept feature is the first column and uses its
  SWMM input file name: `PctZero` (was `Zero-Imperv`), `S-Imperv` and `S-Perv`
  (were `Destore-Imperv` and `Destore-Perv`). `RESULT_KEYS` lists six
  statistics: `runoff`, `peak_runoff_rate`, `infiltration`, `evaporation`,
  `precipitation` and `runon`.
- `calculate()` and `calculate_timeseries()` always run the unmodified model.
  Before, a call after a sweep ran the last swept value.
- `simulate_width()` and `simulate_percent_slope()` start at 10 by default
  (was 0, which is not a physical value).
- Invalid ranges raise `InvalidParameterError` instead of pydantic's
  `ValidationError`. `InvalidParameterError` is a `ValueError`, so code that
  catches `ValueError` keeps working.
- SWMM failures raise `SimulationError` (a `RuntimeError`) whose
  `report_errors` attribute holds the `ERROR` lines of the run's report file,
  instead of a bare `Exception` from pyswmm.
- `simulate_subcatchment()` and `simulate_subcatchment_timeseries()` accept
  any of the ten sweepable features.
- `time_to_peak` documents that it counts from the first timestamp of the
  hydrograph, which is one routing step after the simulation start.
- A `with` block is no longer needed: no working file outlives a call.
  `close()` and the context manager remain.
- Raised dependency floors to versions that install and pass the test suite:
  `pandas>=2.1.1`, `numpy>=1.26`, `pyswmm>=1.5.0`, `pydantic>=2.1`.
- Works with pyswmm 1.5 and 2.x. A plain `pip install` gets pyswmm 1.5,
  because swmmio 0.8 requires `pyswmm<2`. pyswmm 1.5 cannot detect a second
  simulation running in the same process, so overlapping runs give undefined
  results there; pyswmm 2 refuses them with an error.
- The license is declared as the SPDX expression `MIT`.
- Releases are published from GitHub Actions with PyPI Trusted Publishing;
  release candidates go to TestPyPI.

### Added

- `FeaturesSimulation.simulate(feature, values)` and
  `simulate_timeseries(feature, values)` sweep any sequence of values,
  including non-linear ones.
- The `precipitation` and `runon` statistics in `calculate()` and in sweep
  results.
- Exception hierarchy in `catchment_simulation.exceptions`:
  `CatchmentSimulationError`, `InvalidParameterError`, `UnknownFeatureError`,
  `SubcatchmentNotFoundError` and `SimulationError`.
- `catchment_simulation.features`: the `Feature`, `SubcatchmentFeature` and
  `SubareaFeature` literal types, the `FEATURES` registry with physical
  bounds, `MAX_RUNS` and `build_range()`.
- `example_model_path()` and a bundled example model, so examples run right
  after installation.
- `FeaturesSimulation.flow_units`, `MANNING_N` (Manning's n per surface type)
  and `DEPRESSION_STORAGE_VALUES_IN`.
- `time_to_peak(..., start=...)` measures from a given instant, such as the
  simulation start.
- `catchment_simulation.__version__`.
- Support for Python 3.14; the test suite runs on Python 3.10 to 3.14.
- The source distribution includes the tests and their fixtures.

### Deprecated

- `FeaturesSimulation.copy_file()`, `get_section()` and the `model` property
  emit `DeprecationWarning` and will be removed in 0.2.0. Use
  `swmmio.Model(path)` directly instead of `get_section()` and `model`.

### Removed

- The `FeaturesSimulation.file` attribute (the path of the working copy).
- Package extras. The development, documentation and web dependencies are
  [PEP 735](https://peps.python.org/pep-0735/) dependency groups of the
  repository (`uv sync --group web`), so
  `pip install catchment-simulation[django]` (0.0.7) or `[web]` no longer
  installs anything extra. The Django application was never part of the
  package.

## [0.0.7] - 2026-02-08

### Added

- Context manager support and cleanup of temporary files (#23).
- Validation of the subcatchment ID when the simulation is created (#36).
- Per-timestep hydrographs: `calculate_timeseries()` and
  `simulate_subcatchment_timeseries()` (#35).
- `analysis.time_to_peak()` and `analysis.runoff_volume()` (#38).
- Pydantic schemas for simulation parameters (#30, #37).
- Type information (`py.typed`) (#24).

### Changed

- Packaging moved from `setup.py` to `pyproject.toml` (#24) and the package
  to a `src/` layout (#28).
- Python 3.10 or newer is required.
- Fixed pandas `FutureWarning`s (#25).

## [0.0.6] - 2022-11-13

Releases 0.0.1 to 0.0.6 were published to PyPI on 2022-11-12 and 2022-11-13
without git tags. They introduced `FeaturesSimulation` with sweeps of area,
imperviousness, slope, width, curb length, Manning's n, depression storage
and the percentage of impervious area without depression storage.

[0.1.0]: https://github.com/BuczynskiRafal/catchments_simulation/compare/v0.0.7...HEAD
[0.0.7]: https://github.com/BuczynskiRafal/catchments_simulation/releases/tag/v0.0.7
[0.0.6]: https://pypi.org/project/catchment-simulation/0.0.6/
