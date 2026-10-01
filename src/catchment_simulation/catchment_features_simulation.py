"""Sensitivity sweeps of SWMM subcatchment features.

:class:`FeaturesSimulation` runs a SWMM model repeatedly, each time with one
feature of one subcatchment set to a different value, and collects that
subcatchment's runoff statistics or hydrographs.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeVar, get_args
from uuid import uuid4

import pandas as pd
import swmmio
from pyswmm import Simulation, Subcatchments
from pyswmm.swmm5 import PySWMM
from swmmio.utils.modify_model import replace_inp_section

from .exceptions import InvalidParameterError, SimulationError, SubcatchmentNotFoundError
from .features import FEATURES, Feature, build_range, validate_feature, validate_values

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from types import TracebackType

try:
    from pyswmm.errors import MultiSimulationError

    _ENGINE_BUSY_ERRORS: tuple[type[Exception], ...] = (MultiSimulationError,)
except ImportError:
    # pyswmm < 2.0, which swmmio 0.8.x requires, has no MultiSimulationError: it never
    # refuses to open a second simulation, which silently takes over the engine instead.
    _ENGINE_BUSY_ERRORS = ()

T = TypeVar("T")

Surface = Literal["Imperv", "Perv"]

US_FLOW_UNITS = frozenset({"CFS", "GPM", "MGD"})
"""``FLOW_UNITS`` values for which SWMM uses US customary units."""


class FeaturesSimulation:
    """Sweep the features of one subcatchment of a SWMM model.

    Every public call copies the model to a uniquely named working file next to
    the source, so relative ``FILE`` references in the model keep resolving,
    runs SWMM on the copy and deletes the copy, its report and its output
    before returning. The source file is never modified and every call starts
    from it, so a result never depends on earlier calls. If the source
    directory is not writable, the copy goes to the system temp directory,
    where relative ``FILE`` references cannot resolve; SWMM's error then
    surfaces as :class:`SimulationError`.

    SWMM runs one simulation per process at a time: an instance is not
    thread-safe, and parallel sweeps need separate processes. pyswmm 2 refuses
    to start a second simulation, which surfaces as :class:`SimulationError`;
    pyswmm 1.5, which swmmio 0.8 requires, cannot detect one, and overlapping
    runs then give undefined results.

    Units follow the model's ``FLOW_UNITS`` (see :attr:`flow_units`). With US
    units (CFS, GPM, MGD) areas are in acres, widths and curb lengths in feet,
    depths in inches and volumes in cubic feet; with SI units (CMS, LPS, MLD)
    they are in hectares, metres, millimetres and cubic metres.

    Parameters
    ----------
    subcatchment_id : str
        ID of the subcatchment to vary and report on.
    raw_file : str or os.PathLike
        Path of the SWMM input (``.inp``) file.

    Attributes
    ----------
    subcatchment_id : str
        The subcatchment ID.
    raw_file : str
        The model path as given.
    flow_units : str
        The model's ``FLOW_UNITS``; ``"CFS"``, SWMM's default, when unset.

    Raises
    ------
    FileNotFoundError
        If *raw_file* does not exist.
    SubcatchmentNotFoundError
        If the model has no subcatchment *subcatchment_id*.

    Examples
    --------
    >>> from catchment_simulation import FeaturesSimulation, example_model_path
    >>> sim = FeaturesSimulation("S1", example_model_path())
    >>> sim.simulate_percent_impervious(0, 100, 25).columns.tolist()[:3]
    ['PercImperv', 'runoff', 'peak_runoff_rate']
    """

    SWMM_SIDE_EXTENSIONS: ClassVar[tuple[str, ...]] = (".inp", ".rpt", ".out")

    RESULT_KEYS: ClassVar[tuple[str, ...]] = (
        "runoff",
        "peak_runoff_rate",
        "infiltration",
        "evaporation",
        "precipitation",
        "runon",
    )
    TIMESERIES_KEYS: ClassVar[tuple[str, ...]] = (
        "rainfall",
        "runoff",
        "infiltration_loss",
        "evaporation_loss",
        "runon",
    )

    # Source: McCuen, R. et al. (1996), Hydrology, FHWA-SA-96-067, Federal Highway Administration, Washington, DC.
    MANNING_N: ClassVar[dict[str, float]] = {
        "smooth asphalt": 0.011,
        "smooth concrete": 0.012,
        "ordinary concrete lining": 0.013,
        "good wood, brick with cement mortar": 0.014,
        "vitrified clay, cast iron": 0.015,
        "corrugated metal pipe, cement rubble surface": 0.024,
        "fallow soil (no residue)": 0.05,
        "cultivated soil, residue cover <= 20%": 0.06,
        "natural range": 0.13,
        "short prairie grass": 0.15,
        "cultivated soil, residue cover > 20%": 0.17,
        "dense grass": 0.24,
        "woods, light underbrush": 0.4,
        "Bermuda grass": 0.41,
        "woods, dense underbrush": 0.8,
    }
    MANNING_N_VALUES: ClassVar[tuple[float, ...]] = tuple(MANNING_N.values())

    # Source: ASCE (1992), Design & Construction of Urban Stormwater Management Systems, New York, NY.
    DEPRESSION_STORAGE_VALUES_IN: ClassVar[tuple[float, ...]] = (0.05, 0.1, 0.2, 0.3)
    DEPRESSION_STORAGE_VALUES: ClassVar[tuple[float, ...]] = tuple(
        round(val * 25.4, 2) for val in DEPRESSION_STORAGE_VALUES_IN
    )
    """Depression storage depths in mm: :attr:`DEPRESSION_STORAGE_VALUES_IN` rounded to 0.01 mm."""

    def __init__(self, subcatchment_id: str, raw_file: str | os.PathLike[str]) -> None:
        self.subcatchment_id = subcatchment_id
        self.raw_file = os.fspath(raw_file)
        self._source = Path(raw_file).resolve(strict=True)
        self._copies: list[Path] = []
        inp = swmmio.Model(str(self._source)).inp
        self._pristine: dict[str, pd.DataFrame] = {
            attribute: getattr(inp, attribute).copy()
            for attribute in {spec.attribute for spec in FEATURES.values()}
        }
        options = inp.options["Value"]
        self.flow_units = str(options.get("FLOW_UNITS", "CFS")).upper()
        available = [str(name) for name in self._pristine["subcatchments"].index]
        if subcatchment_id not in available:
            raise SubcatchmentNotFoundError(subcatchment_id, available)

    def __enter__(self) -> FeaturesSimulation:
        """Return the instance; leaving the block calls :meth:`close`."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Call :meth:`close`; exceptions propagate."""
        self.close()

    def close(self) -> None:
        """Delete the files made by the deprecated :meth:`copy_file`.

        Working files never outlive a call, so neither ``close()`` nor a
        ``with`` block is required. Calling it again does nothing, and the
        instance stays usable.
        """
        for copy in self._copies:
            _remove_swmm_files(copy)
        self._copies.clear()

    # --- running SWMM -----------------------------------------------------------------------

    @contextmanager
    def _working_copy(self) -> Iterator[Path]:
        """Yield a fresh copy of the source model; delete it and its SWMM outputs afterwards."""
        work = self._source.with_name(f"{self._source.stem}.cs-{uuid4().hex[:8]}.inp")
        try:
            shutil.copyfile(self._source, work)
        except OSError:
            _remove_swmm_files(work)
            work = Path(tempfile.gettempdir()) / work.name
            shutil.copyfile(self._source, work)
        try:
            yield work
        finally:
            _remove_swmm_files(work)

    def _run(
        self, inp: Path, on_step: Callable[[Any, Any], None] | None = None
    ) -> dict[str, float]:
        """Run SWMM on *inp* and return the subcatchment's statistics.

        *on_step* is called with the simulation and the subcatchment after
        every routing step.
        """
        report = inp.with_suffix(".rpt")
        try:
            with _open_simulation(inp, report) as sim:
                subcatchment = Subcatchments(sim)[self.subcatchment_id]
                for _ in sim:
                    if on_step is not None:
                        on_step(sim, subcatchment)
                statistics: dict[str, float] = subcatchment.statistics
        except Exception as exc:
            errors = _report_errors(report)
            message = f"SWMM failed to run {self._source.name}"
            if not errors:
                message += f": {str(exc).strip() or type(exc).__name__}"
            raise SimulationError(message, errors) from exc
        return statistics

    def _timeseries(self, inp: Path) -> pd.DataFrame:
        times: list[Any] = []
        rows: list[list[float]] = []

        def record(sim: Any, subcatchment: Any) -> None:
            times.append(sim.current_time)
            rows.append([getattr(subcatchment, key) for key in self.TIMESERIES_KEYS])

        self._run(inp, record)
        return pd.DataFrame(
            rows,
            columns=list(self.TIMESERIES_KEYS),
            index=pd.DatetimeIndex(times, name="datetime"),
        )

    def _sweep(
        self, feature: str, values: Iterable[float], run: Callable[[Path], T]
    ) -> tuple[Feature, list[float], list[T]]:
        """Run *run* once per value with *feature* of the subcatchment set to that value.

        Every value is validated before the first run, and every sweep starts
        from the source model.
        """
        name = validate_feature(feature)
        checked = validate_values(name, values)
        spec = FEATURES[name]
        section = self._pristine[spec.attribute].copy()
        if self.subcatchment_id not in section.index:
            # Setting the value would add a row of blanks, which SWMM rejects. Left alone,
            # SWMM runs such a subcatchment with zero runoff and no warning.
            raise InvalidParameterError(
                f"Subcatchment {self.subcatchment_id!r} has no row in the {spec.section} "
                f"section of the model, so {name} cannot be simulated."
            )
        section[name] = section[name].astype(float)
        results = []
        with self._working_copy() as work:
            for value in checked:
                section.loc[self.subcatchment_id, name] = value
                replace_inp_section(str(work), spec.section, section)
                results.append(run(work))
        return name, checked, results

    # --- single runs ------------------------------------------------------------------------

    def calculate(self) -> dict[str, float]:
        """Run the unmodified model and return the subcatchment's statistics.

        Before 0.1.0 a call after a sweep ran the last swept value; it now
        always runs the model as it is on disk.

        Returns
        -------
        dict of str to float
            ``runoff``, ``infiltration``, ``evaporation`` and ``runon`` are
            total volumes (m³ or ft³), ``peak_runoff_rate`` is in the model's
            flow units and ``precipitation`` is the total depth (mm or in).

        Raises
        ------
        SimulationError
            If SWMM fails to run the model.
        """
        with self._working_copy() as work:
            return self._run(work)

    def calculate_timeseries(self) -> pd.DataFrame:
        """Run the unmodified model and return the subcatchment's hydrograph.

        Returns
        -------
        pandas.DataFrame
            One row per routing step, indexed by a ``DatetimeIndex`` named
            ``"datetime"`` (the end of each step), with the columns of
            :attr:`TIMESERIES_KEYS`: ``rainfall`` and ``infiltration_loss`` in
            mm/h (in/h), ``evaporation_loss`` in mm/day (in/day), ``runoff``
            and ``runon`` in the model's flow units.

        Raises
        ------
        SimulationError
            If SWMM fails to run the model.
        """
        with self._working_copy() as work:
            return self._timeseries(work)

    # --- sweeps -----------------------------------------------------------------------------

    def simulate(self, feature: Feature, values: Iterable[float]) -> pd.DataFrame:
        """Run the model once per value of *feature* and collect the statistics.

        Parameters
        ----------
        feature : Feature
            The SWMM column to vary, e.g. ``"Width"`` or ``"N-Imperv"``.
        values : iterable of float
            The values to run, in the model's units (see the class docstring).

        Returns
        -------
        pandas.DataFrame
            One row per value, with the *feature* column first and then the
            :attr:`RESULT_KEYS` columns, in the units of :meth:`calculate`.

        Raises
        ------
        UnknownFeatureError
            If *feature* is not a sweepable feature.
        InvalidParameterError
            If *values* is empty, a value is outside the feature's physical
            range, or the subcatchment has no row in the feature's section
            (e.g. ``[SUBAREAS]``); raised before any run.
        SimulationError
            If SWMM fails to run the model.
        """
        name, checked, statistics = self._sweep(feature, values, self._run)
        frame = pd.DataFrame(
            [[stats[key] for key in self.RESULT_KEYS] for stats in statistics],
            columns=list(self.RESULT_KEYS),
        )
        frame.insert(0, name, checked)
        return frame

    def simulate_timeseries(
        self, feature: Feature, values: Iterable[float]
    ) -> dict[float, pd.DataFrame]:
        """Run the model once per value of *feature* and collect the hydrographs.

        Parameters
        ----------
        feature : Feature
            The SWMM column to vary.
        values : iterable of float
            The values to run, in the model's units.

        Returns
        -------
        dict of float to pandas.DataFrame
            Each value mapped to its hydrograph, shaped as in
            :meth:`calculate_timeseries`.

        Raises
        ------
        UnknownFeatureError
            If *feature* is not a sweepable feature.
        InvalidParameterError
            If *values* is empty, a value is outside the feature's physical
            range, or the subcatchment has no row in the feature's section
            (e.g. ``[SUBAREAS]``); raised before any run.
        SimulationError
            If SWMM fails to run the model.
        """
        _, checked, series = self._sweep(feature, values, self._timeseries)
        return dict(zip(checked, series, strict=True))

    def simulate_subcatchment(
        self, feature: Feature, start: float = 0, stop: float = 100, step: float = 10
    ) -> pd.DataFrame:
        """Sweep *feature* over ``start, start + step, ...`` up to *stop*.

        Parameters
        ----------
        feature : Feature
            The SWMM column to vary.
        start : float, optional
            First value, defaults to 0.
        stop : float, optional
            Largest value; included when it lies on the step grid. Defaults to 100.
        step : float, optional
            Spacing, defaults to 10.

        Returns
        -------
        pandas.DataFrame
            As :meth:`simulate`.

        Raises
        ------
        InvalidParameterError
            If the range is invalid (see :func:`~catchment_simulation.build_range`),
            leaves the feature's physical range, or the subcatchment has no row
            in the feature's section; raised before any run.
        UnknownFeatureError
            If *feature* is not a sweepable feature.
        SimulationError
            If SWMM fails to run the model.
        """
        return self.simulate(feature, build_range(start, stop, step))

    def simulate_subcatchment_timeseries(
        self, feature: Feature, start: float = 0, stop: float = 100, step: float = 10
    ) -> dict[float, pd.DataFrame]:
        """Sweep *feature* over ``start, start + step, ...`` up to *stop*, collecting hydrographs.

        Parameters
        ----------
        feature : Feature
            The SWMM column to vary.
        start, stop, step : float, optional
            The range, in the feature's units, defaults 0, 100, 10; see
            :meth:`simulate_subcatchment`.

        Returns
        -------
        dict of float to pandas.DataFrame
            Each range value mapped to its hydrograph, shaped as in
            :meth:`calculate_timeseries`. The keys are rounded to 10 decimal
            places, so ``results[0.3]`` works.

        Raises
        ------
        InvalidParameterError
            If the range is invalid, leaves the feature's physical range, or the
            subcatchment has no row in the feature's section; raised before any run.
        UnknownFeatureError
            If *feature* is not a sweepable feature.
        SimulationError
            If SWMM fails to run the model.
        """
        return self.simulate_timeseries(feature, build_range(start, stop, step))

    def simulate_area(self, start: float = 1, stop: float = 10, step: float = 1) -> pd.DataFrame:
        """Sweep the subcatchment area.

        Parameters
        ----------
        start, stop, step : float, optional
            Range of areas in hectares (acres for US units), defaults 1, 10, 1. *stop* is
            included when it lies on the step grid.

        Returns
        -------
        pandas.DataFrame
            The ``"Area"`` column, then :attr:`RESULT_KEYS`; see :meth:`simulate`.

        Raises
        ------
        InvalidParameterError
            If the range is invalid or includes an area <= 0; raised before any run.
        SimulationError
            If SWMM fails to run the model.
        """
        return self.simulate_subcatchment("Area", start, stop, step)

    def simulate_percent_impervious(
        self, start: float = 0, stop: float = 100, step: float = 10
    ) -> pd.DataFrame:
        """Sweep the percentage of the subcatchment that is impervious.

        Parameters
        ----------
        start, stop, step : float, optional
            Range of percentages in [0, 100], defaults 0, 100, 10. *stop* is
            included when it lies on the step grid.

        Returns
        -------
        pandas.DataFrame
            The ``"PercImperv"`` column, then :attr:`RESULT_KEYS`; see :meth:`simulate`.

        Raises
        ------
        InvalidParameterError
            If the range is invalid or includes a value outside [0, 100]; raised before any run.
        SimulationError
            If SWMM fails to run the model.
        """
        return self.simulate_subcatchment("PercImperv", start, stop, step)

    def simulate_percent_slope(
        self, start: float = 10, stop: float = 100, step: float = 10
    ) -> pd.DataFrame:
        """Sweep the average surface slope.

        Parameters
        ----------
        start, stop, step : float, optional
            Range of slopes in percent, in (0, 100], defaults 10, 100, 10. *stop* is
            included when it lies on the step grid.

        Returns
        -------
        pandas.DataFrame
            The ``"PercSlope"`` column, then :attr:`RESULT_KEYS`; see :meth:`simulate`.

        Raises
        ------
        InvalidParameterError
            If the range is invalid or includes a value outside (0, 100]; raised before any run.
        SimulationError
            If SWMM fails to run the model.
        """
        return self.simulate_subcatchment("PercSlope", start, stop, step)

    def simulate_width(
        self, start: float = 10, stop: float = 100, step: float = 10
    ) -> pd.DataFrame:
        """Sweep the characteristic width of the overland flow path.

        Parameters
        ----------
        start, stop, step : float, optional
            Range of widths in metres (feet for US units), defaults 10, 100, 10. *stop* is
            included when it lies on the step grid.

        Returns
        -------
        pandas.DataFrame
            The ``"Width"`` column, then :attr:`RESULT_KEYS`; see :meth:`simulate`.

        Raises
        ------
        InvalidParameterError
            If the range is invalid or includes a width <= 0; raised before any run.
        SimulationError
            If SWMM fails to run the model.
        """
        return self.simulate_subcatchment("Width", start, stop, step)

    def simulate_curb_length(
        self, start: float = 0, stop: float = 100, step: float = 10
    ) -> pd.DataFrame:
        """Sweep the total curb length, which only affects pollutant build-up.

        Parameters
        ----------
        start, stop, step : float, optional
            Range of lengths in metres (feet for US units), defaults 0, 100, 10. *stop* is
            included when it lies on the step grid.

        Returns
        -------
        pandas.DataFrame
            The ``"CurbLength"`` column, then :attr:`RESULT_KEYS`; see :meth:`simulate`.

        Raises
        ------
        InvalidParameterError
            If the range is invalid or includes a negative length; raised before any run.
        SimulationError
            If SWMM fails to run the model.
        """
        return self.simulate_subcatchment("CurbLength", start, stop, step)

    def simulate_percent_zero_imperv(
        self, start: float = 0, stop: float = 100, step: float = 10
    ) -> pd.DataFrame:
        """Sweep the percentage of impervious area without depression storage.

        Parameters
        ----------
        start, stop, step : float, optional
            Range of percentages in [0, 100], defaults 0, 100, 10. *stop* is
            included when it lies on the step grid.

        Returns
        -------
        pandas.DataFrame
            The ``"PctZero"`` column, then :attr:`RESULT_KEYS`; see :meth:`simulate`.

        Raises
        ------
        InvalidParameterError
            If the range is invalid, includes a value outside [0, 100], or the
            subcatchment has no ``[SUBAREAS]`` row; raised before any run.
        SimulationError
            If SWMM fails to run the model.
        """
        return self.simulate_subcatchment("PctZero", start, stop, step)

    def simulate_manning_n(self, param: Surface) -> pd.DataFrame:
        """Sweep Manning's n of one surface over :attr:`MANNING_N_VALUES`.

        Parameters
        ----------
        param : {"Imperv", "Perv"}
            The surface whose roughness to vary.

        Returns
        -------
        pandas.DataFrame
            As :meth:`simulate`, with the feature column ``"N-Imperv"`` or
            ``"N-Perv"`` (dimensionless n).

        Raises
        ------
        InvalidParameterError
            If *param* is not ``"Imperv"`` or ``"Perv"``, or the subcatchment
            has no ``[SUBAREAS]`` row; raised before any run.
        SimulationError
            If SWMM fails to run the model.
        """
        return self.simulate(_surface_feature("N", param), self.MANNING_N_VALUES)

    def simulate_n_imperv(self) -> pd.DataFrame:
        """Sweep Manning's n of the impervious surface.

        Returns
        -------
        pandas.DataFrame
            The ``"N-Imperv"`` column (:attr:`MANNING_N_VALUES`), then :attr:`RESULT_KEYS`;
            see :meth:`simulate_manning_n`.

        Raises
        ------
        InvalidParameterError
            If the subcatchment has no ``[SUBAREAS]`` row; raised before any run.
        SimulationError
            If SWMM fails to run the model.
        """
        return self.simulate_manning_n("Imperv")

    def simulate_n_perv(self) -> pd.DataFrame:
        """Sweep Manning's n of the pervious surface.

        Returns
        -------
        pandas.DataFrame
            The ``"N-Perv"`` column (:attr:`MANNING_N_VALUES`), then :attr:`RESULT_KEYS`;
            see :meth:`simulate_manning_n`.

        Raises
        ------
        InvalidParameterError
            If the subcatchment has no ``[SUBAREAS]`` row; raised before any run.
        SimulationError
            If SWMM fails to run the model.
        """
        return self.simulate_manning_n("Perv")

    def simulate_destore(self, param: Surface) -> pd.DataFrame:
        """Sweep the depression storage depth of one surface.

        The depths are :attr:`DEPRESSION_STORAGE_VALUES` (mm) for SI models and
        :attr:`DEPRESSION_STORAGE_VALUES_IN` (inches) for US models.

        Parameters
        ----------
        param : {"Imperv", "Perv"}
            The surface whose depression storage to vary.

        Returns
        -------
        pandas.DataFrame
            As :meth:`simulate`, with the feature column ``"S-Imperv"`` or
            ``"S-Perv"``.

        Raises
        ------
        InvalidParameterError
            If *param* is not ``"Imperv"`` or ``"Perv"``, or the subcatchment
            has no ``[SUBAREAS]`` row; raised before any run.
        SimulationError
            If SWMM fails to run the model.
        """
        feature = _surface_feature("S", param)
        if self.flow_units in US_FLOW_UNITS:
            return self.simulate(feature, self.DEPRESSION_STORAGE_VALUES_IN)
        return self.simulate(feature, self.DEPRESSION_STORAGE_VALUES)

    def simulate_s_imperv(self) -> pd.DataFrame:
        """Sweep the depression storage depth of the impervious surface.

        Returns
        -------
        pandas.DataFrame
            The ``"S-Imperv"`` column (mm, or inches for US units), then :attr:`RESULT_KEYS`;
            see :meth:`simulate_destore`.

        Raises
        ------
        InvalidParameterError
            If the subcatchment has no ``[SUBAREAS]`` row; raised before any run.
        SimulationError
            If SWMM fails to run the model.
        """
        return self.simulate_destore("Imperv")

    def simulate_s_perv(self) -> pd.DataFrame:
        """Sweep the depression storage depth of the pervious surface.

        Returns
        -------
        pandas.DataFrame
            The ``"S-Perv"`` column (mm, or inches for US units), then :attr:`RESULT_KEYS`;
            see :meth:`simulate_destore`.

        Raises
        ------
        InvalidParameterError
            If the subcatchment has no ``[SUBAREAS]`` row; raised before any run.
        SimulationError
            If SWMM fails to run the model.
        """
        return self.simulate_destore("Perv")

    # --- deprecated -------------------------------------------------------------------------

    def copy_file(self, copy: str | os.PathLike[str] | None = None, suffix: str = "copy") -> str:
        """Copy a model to ``<name>_<suffix>.inp`` next to it.

        .. deprecated:: 0.1.0
            Working copies are managed internally; this will be removed in 0.2.0.

        Parameters
        ----------
        copy : str or os.PathLike, optional
            The model to copy, defaults to the instance's model.
        suffix : str, optional
            Suffix of the copy's name, defaults to ``"copy"``.

        Returns
        -------
        str
            Path of the copy, which :meth:`close` deletes.
        """
        _deprecated("copy_file()", "every call now makes and removes its own working copy")
        source = Path(copy) if copy is not None else self._source
        target = source.with_name(f"{source.stem}_{suffix}.inp")
        shutil.copyfile(source, target)
        self._copies.append(target)
        return str(target)

    def get_section(self, section: str = "subcatchments") -> pd.DataFrame:
        """Return a section of the model parsed by swmmio.

        .. deprecated:: 0.1.0
            Use ``swmmio.Model(path).inp`` directly; this will be removed in 0.2.0.

        Parameters
        ----------
        section : str, optional
            Name of a ``swmmio`` ``inp`` attribute, defaults to ``"subcatchments"``.

        Returns
        -------
        pandas.DataFrame
            The section, parsed from the source file.
        """
        _deprecated("get_section()", "use swmmio.Model(path).inp instead")
        section_frame: pd.DataFrame = getattr(swmmio.Model(str(self._source)).inp, section)
        return section_frame

    @property
    def model(self) -> swmmio.Model:
        """A freshly parsed ``swmmio.Model`` of the source file.

        .. deprecated:: 0.1.0
            Use ``swmmio.Model(path)`` directly; this will be removed in 0.2.0.
        """
        _deprecated("model", "use swmmio.Model(path) instead")
        return swmmio.Model(str(self._source))


def _deprecated(name: str, instead: str) -> None:
    warnings.warn(
        f"FeaturesSimulation.{name} is deprecated and will be removed in 0.2.0; {instead}.",
        DeprecationWarning,
        stacklevel=3,
    )


def _surface_feature(prefix: Literal["N", "S"], param: str) -> Feature:
    surfaces = get_args(Surface)
    if param not in surfaces:
        raise InvalidParameterError(
            f"Unknown surface {param!r}. Allowed values: {', '.join(surfaces)}"
        )
    return validate_feature(f"{prefix}-{param}")


def _open_simulation(inp: Path, report: Path) -> Simulation:
    try:
        return Simulation(str(inp), str(report), str(inp.with_suffix(".out")))
    except _ENGINE_BUSY_ERRORS:
        # pyswmm >= 2.0 refused to open because another simulation is running; closing
        # the engine would end that one.
        raise
    except Exception:
        # A failed swmm_open leaves the engine half-open with the report unflushed.
        # Closing it writes the report and frees its files.
        PySWMM().swmm_close()
        raise


def _report_errors(report: Path) -> list[str]:
    try:
        lines = report.read_text(errors="replace").splitlines()
    except OSError:
        return []
    return [line.strip() for line in lines if line.strip().startswith("ERROR")]


def _remove_swmm_files(inp: Path) -> None:
    for extension in FeaturesSimulation.SWMM_SIDE_EXTENSIONS:
        inp.with_suffix(extension).unlink(missing_ok=True)
