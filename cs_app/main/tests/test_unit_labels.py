"""The sweep parameters are labelled in the units the package reads them in.

catchment_simulation takes Area in acres and Width and Curb Length in feet for a model
with US ``FLOW_UNITS`` (CFS, GPM, MGD, and CFS when unset), and in hectares and metres
for SI ones; the option labels and range errors must say so.
"""

import json
from pathlib import Path

import pytest
from django.urls import reverse

from catchment_simulation import example_model_path
from main.forms import SimulationForm, TimeseriesForm, parameter_labels
from main.views import _user_upload_dir

US_UNITS = {"area": "acre", "length": "ft"}
SI_LABELS = {"Area": "Area (ha)", "Width": "Width (m)", "CurbLength": "Curb Length (m)"}
US_LABELS = {"Area": "Area (acre)", "Width": "Width (ft)", "CurbLength": "Curb Length (ft)"}
SWEEP_METHODS = {
    "Area": "simulate_area",
    "Width": "simulate_width",
    "CurbLength": "simulate_curb_length",
}


def _option_labels(form: SimulationForm) -> dict[str, str]:
    return {value: label for _, group in form.fields["option"].choices for value, label in group}


def _feature_labels(form: TimeseriesForm) -> dict[str, str]:
    return dict(form.fields["feature"].choices)


# --- forms ----------------------------------------------------------------------------------


@pytest.mark.parametrize(("units", "labels"), [(None, SI_LABELS), (US_UNITS, US_LABELS)])
def test_forms_label_sweep_parameters_in_the_given_units(units, labels):
    options = _option_labels(SimulationForm(units=units))
    features = _feature_labels(TimeseriesForm(units=units))

    for feature, label in labels.items():
        assert options[SWEEP_METHODS[feature]] == label
        assert features[feature] == label
    # Unitless parameters keep their labels.
    assert options["simulate_percent_slope"] == features["PercSlope"] == "Slope (%)"
    assert options["simulate_n_imperv"] == "Manning's n - Impervious"


def test_range_errors_name_the_models_units():
    simulation = SimulationForm(
        data={
            "option": "simulate_area",
            "start": 0,
            "stop": 5,
            "step": 1,
            "catchment_name": "S1",
        },
        catchment_choices=[("S1", "S1")],
        units=US_UNITS,
    )
    timeseries = TimeseriesForm(
        data={
            "mode": "sweep",
            "feature": "Width",
            "start": 0,
            "stop": 10,
            "step": 5,
            "catchment_name": "S1",
        },
        catchment_choices=[("S1", "S1")],
        units=US_UNITS,
    )

    assert not simulation.is_valid()
    assert simulation.errors["start"] == ["Area (acre) must be greater than 0."]
    assert not timeseries.is_valid()
    assert timeseries.errors["start"] == ["Width (ft) must be greater than 0."]


def test_parameter_labels_cover_both_forms():
    labels = parameter_labels(US_UNITS)

    assert labels == {
        **_option_labels(SimulationForm(units=US_UNITS)),
        **_feature_labels(TimeseriesForm(units=US_UNITS)),
    }
    assert labels["simulate_area"] == labels["Area"] == "Area (acre)"
    assert parameter_labels()["Area"] == "Area (ha)"


# --- pages ----------------------------------------------------------------------------------


def _session_model(client, user, flow_units_line: str | None) -> Path:
    """Put a copy of the example model, with *flow_units_line* in [OPTIONS], in the session."""
    text = example_model_path().read_text()
    assert "FLOW_UNITS           CMS\n" in text
    text = text.replace(
        "FLOW_UNITS           CMS\n", f"{flow_units_line}\n" if flow_units_line else ""
    )
    path = Path(_user_upload_dir(user.id)) / "model.inp"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    client.force_login(user)
    session = client.session
    session["uploaded_file_path"] = str(path)
    session.save()
    return path


def _page_has_labels(response, labels: dict[str, str]) -> bool:
    content = response.content.decode()
    return all(f">{label}</option>" in content for label in labels.values())


@pytest.mark.django_db
@pytest.mark.parametrize("url_name", ["main:simulation", "main:timeseries"])
@pytest.mark.parametrize(
    ("flow_units_line", "labels"),
    [
        ("FLOW_UNITS CMS", SI_LABELS),
        ("FLOW_UNITS LPS", SI_LABELS),
        ("FLOW_UNITS CFS", US_LABELS),
        ("FLOW_UNITS MGD", US_LABELS),
        # SWMM keywords are case-insensitive, and SWMM's default is CFS.
        ("flow_units cfs", US_LABELS),
        ("Flow_Units Cms", SI_LABELS),
        (None, US_LABELS),
    ],
)
def test_pages_label_sweep_parameters_in_the_models_units(
    client, user, url_name, flow_units_line, labels
):
    _session_model(client, user, flow_units_line)

    response = client.get(reverse(url_name))

    assert response.status_code == 200
    assert _page_has_labels(response, labels)


@pytest.mark.django_db
@pytest.mark.parametrize("url_name", ["main:simulation", "main:timeseries"])
def test_pages_without_a_model_use_si_labels(client, user, url_name):
    client.force_login(user)

    response = client.get(reverse(url_name))

    assert _page_has_labels(response, SI_LABELS)


@pytest.mark.django_db
def test_invalid_post_reports_the_range_in_the_models_units(client, user):
    _session_model(client, user, "FLOW_UNITS CFS")

    response = client.post(
        reverse("main:simulation"),
        {"option": "simulate_width", "start": 0, "stop": 10, "step": 5, "catchment_name": "S1"},
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )

    assert response.status_code == 400
    assert response.json()["field_errors"]["start"] == ["Width (ft) must be greater than 0."]


@pytest.mark.django_db
def test_a_new_model_is_labelled_in_its_own_units(client, user):
    """The units are cached with the model, and forgotten with it."""
    path = _session_model(client, user, "FLOW_UNITS CMS")
    assert _page_has_labels(client.get(reverse("main:simulation")), SI_LABELS)

    us_model = path.read_bytes().replace(b"FLOW_UNITS CMS", b"FLOW_UNITS CFS")
    response = client.post(
        reverse("main:upload"),
        {"file": _uploaded("model.inp", us_model)},
        HTTP_X_REQUESTED_WITH="XMLHttpRequest",
    )
    assert response.status_code == 200

    assert _page_has_labels(client.get(reverse("main:simulation")), US_LABELS)


def _uploaded(name: str, content: bytes):
    from django.core.files.uploadedfile import SimpleUploadedFile

    return SimpleUploadedFile(name, content, content_type="text/plain")


# --- the upload zone relabels the options without a reload ----------------------------------


@pytest.mark.django_db
def test_subcatchments_endpoint_returns_the_models_option_labels(client, user):
    _session_model(client, user, "FLOW_UNITS GPM")

    response = client.get(reverse("main:subcatchments"), HTTP_X_REQUESTED_WITH="XMLHttpRequest")

    data = json.loads(response.content)
    assert data["subcatchments"] == ["S1"]
    assert data["option_labels"] == parameter_labels(US_UNITS)


@pytest.mark.django_db
def test_upload_clear_returns_the_si_option_labels(client, user):
    _session_model(client, user, "FLOW_UNITS CFS")

    response = client.post(reverse("main:upload_clear"), HTTP_X_REQUESTED_WITH="XMLHttpRequest")

    assert json.loads(response.content)["option_labels"] == parameter_labels()
