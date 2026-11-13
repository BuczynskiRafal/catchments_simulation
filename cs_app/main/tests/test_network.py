"""The network as data: its inputs per subcatchment, its payload, and the calculations page that draws it."""

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import swmmio
from django.conf import settings
from django.urls import reverse

from main import predictor
from main.predictor import (
    NO_UNIT,
    REQUIRED_SUBCATCHMENT_COLUMNS,
    network_payload,
    predict_runoff,
    prediction_inputs,
)
from main.views import _summarise_comparison, _user_upload_dir

EXAMPLE_MODEL = Path(settings.BASE_DIR) / "data" / "example.inp"
# S1 of the bundled example model, in the order the network reads it.
EXAMPLE_INPUTS = [25.0, 100.0, 10.0, 0.01, 1.0, 0.05, 0.05, 25.0]


def _fake_model(flow_units: str, **columns) -> SimpleNamespace:
    """A swmmio-like model with one subcatchment and the given FLOW_UNITS."""
    row = dict(zip(REQUIRED_SUBCATCHMENT_COLUMNS, EXAMPLE_INPUTS, strict=True)) | columns
    dataframe = pd.DataFrame([row], index=["S1"])
    options = pd.DataFrame(index=["FLOW_UNITS"], data={"Value": [flow_units]})
    return SimpleNamespace(
        subcatchments=SimpleNamespace(dataframe=dataframe), inp=SimpleNamespace(options=options)
    )


# ---------------------------------------------------------------------------
# prediction_inputs
# ---------------------------------------------------------------------------


def test_prediction_inputs_of_si_model_are_read_as_they_are():
    inputs = prediction_inputs(swmmio.Model(str(EXAMPLE_MODEL)))

    assert inputs.dtype == np.float32
    assert inputs.shape == (1, 8)
    np.testing.assert_allclose(inputs[0], EXAMPLE_INPUTS, rtol=1e-6)


def test_prediction_inputs_of_us_model_convert_width_and_storage_to_si():
    model = _fake_model("CFS", Width=100.0, **{"S-Imperv": 0.05, "S-Perv": 0.1})

    inputs = prediction_inputs(model)[0]

    assert inputs[1] == pytest.approx(30.48)  # ft -> m
    assert inputs[5] == pytest.approx(1.27)  # in -> mm
    assert inputs[6] == pytest.approx(2.54)
    # Percentages and Manning's n have no unit to convert.
    assert inputs[[0, 2, 3, 4, 7]] == pytest.approx([25.0, 10.0, 0.01, 1.0, 25.0])


def test_prediction_inputs_report_missing_columns():
    model = _fake_model("CMS")
    model.subcatchments.dataframe = model.subcatchments.dataframe.drop(columns=["Width"])

    with pytest.raises(KeyError, match="Width"):
        prediction_inputs(model)


def test_predict_runoff_runs_the_network_on_prediction_inputs():
    model = _fake_model("CFS", Width=300.0)

    expected = predictor._get_model().predict(prediction_inputs(model)).flatten()

    np.testing.assert_array_equal(predict_runoff(model), expected)


# ---------------------------------------------------------------------------
# network_payload
# ---------------------------------------------------------------------------


def test_network_payload_describes_the_loaded_network():
    payload = network_payload()
    model = predictor._get_model()

    assert [feature["key"] for feature in payload["features"]] == list(
        REQUIRED_SUBCATCHMENT_COLUMNS
    )
    assert [feature["unit"] for feature in payload["features"]] == [
        "%",
        "m",
        "%",
        NO_UNIT,
        NO_UNIT,
        "mm",
        "mm",
        "%",
    ]
    assert payload["features"][0]["label"] == "Imperviousness"
    assert all(feature["short"] for feature in payload["features"])
    assert payload["mean"] == model.norm_mean.tolist()
    assert payload["std"] == np.sqrt(model.norm_variance + 1e-7).tolist()
    assert [np.shape(layer["weights"]) for layer in payload["layers"]] == [
        (8, 8),
        (8, 8),
        (8, 1),
    ]
    assert [len(layer["bias"]) for layer in payload["layers"]] == [8, 8, 1]
    assert payload["output"] == {"unit": "10³ m³"}
    bounds = {feature["key"]: (feature["min"], feature["max"]) for feature in payload["features"]}
    assert bounds == {
        "PercImperv": (0.0, 100.0),
        "Width": (0.0, None),
        "PercSlope": (0.0, None),
        "N-Imperv": (0.0, None),
        "N-Perv": (0.0, None),
        "S-Imperv": (0.0, None),
        "S-Perv": (0.0, None),
        "PctZero": (0.0, 100.0),
    }


def test_network_payload_is_json_and_reproduces_the_prediction():
    payload = json.loads(json.dumps(network_payload()))
    x = np.array(EXAMPLE_INPUTS)

    # The forward pass of network.js, in float64.
    activation = (x - np.array(payload["mean"])) / np.array(payload["std"])
    for layer in payload["layers"]:
        activation = np.maximum(0, activation @ np.array(layer["weights"]) + layer["bias"])

    expected = predict_runoff(swmmio.Model(str(EXAMPLE_MODEL)))[0]
    assert activation[0] == pytest.approx(expected, rel=1e-5)


def test_network_payload_is_built_once():
    assert network_payload() is network_payload()


# ---------------------------------------------------------------------------
# Comparison rows and the calculations page
# ---------------------------------------------------------------------------


def _comparison(with_inputs: bool) -> pd.DataFrame:
    data = {"Name": ["S1", "S2"], "SWMM_Runoff": [3360.0, 100.0], "ANN_Runoff": [2465.94, 120.0]}
    if with_inputs:
        inputs = np.array([EXAMPLE_INPUTS, EXAMPLE_INPUTS], dtype=np.float32)
        data["Inputs"] = inputs.tolist()
    return pd.DataFrame(data)


def test_rows_carry_the_network_inputs_without_float32_noise():
    summary = _summarise_comparison(_comparison(with_inputs=True), "m³")

    for row in summary["rows"]:
        assert row["inputs"] == EXAMPLE_INPUTS
        assert all(type(value) is float for value in row["inputs"])


def test_rows_of_a_comparison_without_inputs_have_none():
    summary = _summarise_comparison(_comparison(with_inputs=False), "m³")

    assert all("inputs" not in row for row in summary["rows"])


@pytest.mark.slow
def test_compare_swmm_and_ann_keeps_the_inputs_of_every_subcatchment(tmp_path):
    from main.views import _compare_swmm_and_ann

    model_path = tmp_path / "example.inp"
    model_path.write_bytes(EXAMPLE_MODEL.read_bytes())

    summary = _summarise_comparison(*_compare_swmm_and_ann(str(model_path)))

    [row] = summary["rows"]
    assert row["inputs"] == EXAMPLE_INPUTS


@pytest.fixture
def calc_client(client, user, monkeypatch):
    """Logged-in client whose comparison run returns ``client.comparison``."""
    monkeypatch.setattr("main.views._compare_swmm_and_ann", lambda _path: (client.comparison, "m³"))
    client.force_login(user)
    session = client.session
    session["uploaded_file_path"] = os.path.join(_user_upload_dir(user.id), "test.inp")
    session.save()
    return client


def _embedded_json(html: str, element_id: str):
    match = re.search(rf'<script id="{element_id}" type="application/json">(.*?)</script>', html)
    return json.loads(match.group(1)) if match else None


@pytest.mark.django_db
def test_page_embeds_the_network_before_any_run(client):
    html = client.get(reverse("main:calculations")).content.decode()

    assert _embedded_json(html, "calculations-network") == json.loads(json.dumps(network_payload()))
    assert 'id="calc-network-title"' in html
    assert f'href="{reverse("main:how_it_works")}#network"' in html
    assert "The equations behind the network" in html


@pytest.mark.django_db
def test_failed_run_page_still_embeds_the_network(client, user):
    """The no-JS error path renders the full page, diagram data included."""
    client.force_login(user)

    html = client.post(reverse("main:calculations"), follow=True).content.decode()

    assert "Please upload a file first." in html
    assert _embedded_json(html, "calculations-network") is not None


@pytest.mark.django_db
def test_page_renders_without_the_diagram_when_the_weights_cannot_be_loaded(client, monkeypatch):
    def unavailable():
        raise FileNotFoundError("weights.npz")

    monkeypatch.setattr("main.views.network_payload", unavailable)

    response = client.get(reverse("main:calculations"))

    assert response.status_code == 200
    html = response.content.decode()
    assert _embedded_json(html, "calculations-network") is None
    assert 'id="run-calculations-button"' in html
    # A note in place of the diagram; the explanation and the equations link stay.
    assert (
        '<p class="calc-net__unavailable" id="calc-network-unavailable">'
        "The network diagram is not available right now.</p>"
    ) in html
    for absent in (
        "calc-network-stage",
        "calc-network-diagram",
    ):
        assert f'id="{absent}"' not in html
    assert 'class="calc-net__intro"' in html
    assert "The equations behind the network" in html


@pytest.mark.django_db
def test_comparison_too_large_to_trace_is_kept_without_the_inputs(calc_client, monkeypatch):
    calc_client.comparison = _comparison(with_inputs=True)
    with_inputs = len(json.dumps(_summarise_comparison(calc_client.comparison, "m³")))
    # Room for the comparison itself, but not for the inputs of its rows.
    monkeypatch.setattr("main.views.MAX_RESULT_CACHE_BYTES", with_inputs - 60)

    response = calc_client.post(
        reverse("main:calculations"), HTTP_X_REQUESTED_WITH="XMLHttpRequest"
    )

    assert response.status_code == 200
    html = response.content.decode()
    rows = _embedded_json(html, "calculations-chart-data")
    assert rows
    assert all("inputs" not in row for row in rows)
