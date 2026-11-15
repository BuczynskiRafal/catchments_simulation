"""
The engine check: CS.runoffModel, run in the browser, against SWMM itself.

CS.runoffModel claims to solve the same equations with the same procedure as
the SWMM engine. Here it runs the example storm with SWMM's 5-minute step for
subcatchment S1 of the bundled model and for values of each of the nine runoff
parameters, and its runoff volume, peak runoff rate and infiltration are
compared with catchment_simulation (pyswmm). The script is injected into an
empty page, so the check needs neither a page of the site nor its server.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from django.contrib.staticfiles import finders

from catchment_simulation import FeaturesSimulation, example_model_path

if TYPE_CHECKING:
    from playwright.sync_api import Page

pytestmark = pytest.mark.e2e

RELATIVE_TOLERANCE = 1e-3  # 0.1 %
TIME_STEP = 300  # s, the step of the example model
# Values swept per feature, in the model's units, and the matching CS.runoffModel parameter.
SWEEPS = {
    "Area": ("area", [1, 20]),
    "PercImperv": ("imperviousness", [0, 60, 100]),
    "Width": ("width", [20, 500]),
    "PercSlope": ("slope", [1, 50]),
    "N-Imperv": ("nImperv", [0.005, 0.024]),
    "N-Perv": ("nPerv", [0.15, 0.4]),
    "S-Imperv": ("storageImperv", [0, 2.54]),
    "S-Perv": ("storagePerv", [0, 5.08]),
    "PctZero": ("zeroImperv", [0, 100]),
}
OUTPUTS = {"runoff": "runoff", "peak_runoff_rate": "peak", "infiltration": "infiltration"}

CASES = [("defaults", None, None)] + [
    (f"{feature}={value:g}", feature, value)
    for feature, (_, values) in SWEEPS.items()
    for value in values
]


@pytest.fixture(scope="module")
def swmm() -> dict:
    """SWMM's results for every case, keyed like CASES."""
    results = {}
    with FeaturesSimulation(subcatchment_id="S1", raw_file=example_model_path()) as model:
        results["defaults"] = model.calculate()
        for feature, (_, values) in SWEEPS.items():
            frame = model.simulate(feature, values)
            for row in frame.to_dict(orient="records"):
                results[f"{feature}={row[feature]:g}"] = row
    return results


@pytest.fixture
def model_page(page: Page) -> Page:
    """An empty page with the model script injected from the source tree."""
    page.add_script_tag(path=finders.find("main/js/model/runoff-model.js"))
    return page


def _browser_run(page: Page, feature: str | None, value: float | None) -> dict:
    overrides = {} if feature is None else {SWEEPS[feature][0]: value}
    result = page.evaluate(
        """([overrides, timeStep]) => {
            const model = CS.runoffModel;
            const storm = model.STORMS.example;
            return model.simulate(Object.assign({}, model.DEFAULTS, overrides,
                { rainfall: storm.rainfall, duration: storm.duration, timeStep }));
        }""",
        [overrides, TIME_STEP],
    )
    return {
        "runoff": result["totals"]["runoff"],
        "peak": result["peak"]["runoff"],
        "infiltration": result["totals"]["infiltration"],
        "totals": result["totals"],
        "continuity": result["continuity"],
    }


@pytest.mark.parametrize(("case", "feature", "value"), CASES, ids=[case for case, _, _ in CASES])
def test_model_matches_swmm(model_page: Page, swmm: dict, case: str, feature, value) -> None:
    reference = swmm[case]
    ours = _browser_run(model_page, feature, value)
    for column, key in OUTPUTS.items():
        expected = reference[column]
        if expected == 0:
            assert ours[key] == pytest.approx(0, abs=1e-6), (case, column)
        else:
            assert ours[key] == pytest.approx(expected, rel=RELATIVE_TOLERANCE), (case, column)


@pytest.mark.parametrize(("case", "feature", "value"), CASES, ids=[case for case, _, _ in CASES])
def test_water_balance_closes(model_page: Page, case: str, feature, value) -> None:
    ours = _browser_run(model_page, feature, value)
    totals = ours["totals"]
    remainder = (
        totals["rainfall"]
        - totals["runoff"]
        - totals["infiltration"]
        - totals["evaporation"]
        - totals["storage"]
    )
    assert remainder / totals["rainfall"] * 100 == pytest.approx(ours["continuity"], abs=1e-9)
    assert abs(ours["continuity"]) < 1
