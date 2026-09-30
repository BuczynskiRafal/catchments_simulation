"""Home page data, content pages (contact, profile) and the account pages."""

import json
import logging
import os
import uuid

import pytest
from django.conf import settings
from django.contrib.auth.models import User
from django.urls import reverse

from catchment_simulation.catchment_features_simulation import FeaturesSimulation
from main import views
from main.forms import ContactForm, UserProfileForm
from main.models import UserProfile

# Random per run, like the e2e credentials, so no password is written in the source.
SIGNUP_PASSWORD = f"T!{uuid.uuid4().hex}"


@pytest.fixture()
def fresh_home_cache():
    views._load_home_data_cached.cache_clear()
    yield
    views._load_home_data_cached.cache_clear()


# ---------------------------------------------------------------------------
# Home page data
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_home_context_carries_charts_labels_and_numbers(client, fresh_home_cache):
    home = client.get(reverse("main:main_view")).context["home"]

    hydrograph = home["chart_data"]["hydrograph"]
    assert set(hydrograph["records"][0]) == {"datetime", "rainfall", "runoff"}
    assert hydrograph["yLabels"] == {
        "rainfall": "Rainfall Intensity [mm/h]",
        "runoff": "Runoff Rate [CMS]",
    }
    slope = home["chart_data"]["sweeps"]["slope"]
    assert (slope["xLabel"], slope["yLabel"]) == ("Percent Slope [%]", "Total Runoff Volume [m³]")
    assert slope["xRange"] == [1, 100]
    assert home["units"]["volume"] == "m³"
    summary = home["hydrograph_summary"]
    assert summary["peak"] == max(row["runoff"] for row in hydrograph["records"])
    assert summary["time_to_peak"] == "8 h 55 min"
    # The example model is in CMS, so the integrated flow is already in m³.
    assert summary["volume"] == pytest.approx(3350.98, abs=0.01)


@pytest.mark.django_db
def test_home_renders_chart_json_and_metrics(client, fresh_home_cache):
    html = client.get(reverse("main:main_view")).content.decode()

    assert 'id="chart-data"' in html
    assert 'id="plot-hydrograph"' in html
    assert 'aria-describedby="plot-width-summary"' in html
    assert "3,351" in html  # runoff volume of the example storm, grouped


def test_home_data_matches_bundled_model():
    """The precomputed charts must come from data/example.inp (regenerate with build_home_data.py)."""
    model_path = os.path.join(settings.BASE_DIR, "data", "example.inp")
    with FeaturesSimulation(subcatchment_id="S1", raw_file=model_path) as model:
        baseline = model.calculate()  # S1 as stored: slope 10 %, area 5 ha, width 100 m

    data = views._build_home_data()
    sweeps = data["chart_data"]["sweeps"]
    for key, value in (("slope", 10), ("area", 5), ("width", 100)):
        row = next(r for r in sweeps[key]["records"] if r[sweeps[key]["xField"]] == value)
        assert row["runoff"] == pytest.approx(baseline["runoff"], rel=1e-5), key
    assert data["hydrograph_summary"]["peak"] == pytest.approx(
        baseline["peak_runoff_rate"], rel=1e-5
    )


@pytest.mark.django_db
def test_home_renders_without_example_data(client, fresh_home_cache, monkeypatch):
    def broken():
        raise ValueError("corrupt data file")

    monkeypatch.setattr(views, "_build_home_data", broken)

    response = client.get(reverse("main:main_view"))

    assert response.status_code == 200
    assert response.context["home"] is None
    html = response.content.decode()
    assert "The example data is not available" in html
    assert 'id="plot-hydrograph"' not in html
    assert 'id="plot-slope"' not in html


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), True, "12", None], ids=repr)
def test_chart_json_rejects_non_finite_or_non_numeric(tmp_path, monkeypatch, bad_value):
    (tmp_path / "sweep.json").write_text(json.dumps([{"slope": 1, "runoff": bad_value}]))
    monkeypatch.setattr(views, "HOME_DATA_DIR", str(tmp_path))

    with pytest.raises(ValueError):
        views._load_chart_json("sweep.json", "slope", "runoff")


def test_chart_json_keeps_only_plotted_keys(tmp_path, monkeypatch):
    (tmp_path / "sweep.json").write_text(json.dumps([{"Unnamed: 0": 0, "slope": 1, "runoff": 2.5}]))
    monkeypatch.setattr(views, "HOME_DATA_DIR", str(tmp_path))

    assert views._load_chart_json("sweep.json", "slope", "runoff") == [{"slope": 1, "runoff": 2.5}]


def test_hydrograph_json_rejects_rows_without_datetime(tmp_path, monkeypatch):
    rows = [{"datetime": None, "rainfall": 0.0, "runoff": 0.0}]
    (tmp_path / views.HOME_HYDROGRAPH_FILE).write_text(json.dumps(rows))
    monkeypatch.setattr(views, "HOME_DATA_DIR", str(tmp_path))

    with pytest.raises(ValueError):
        views._load_hydrograph_json()


# ---------------------------------------------------------------------------
# Contact
# ---------------------------------------------------------------------------


def test_contact_form_is_english_and_matches_schema_limits():
    form = ContactForm()

    assert [form[name].label for name in ("email", "title", "content", "send_to_me")] == [
        "Email",
        "Subject",
        "Message",
        "Send me a copy",
    ]
    assert form.helper.inputs[0].value == "Send message"
    assert form.fields["title"].max_length == 200
    assert form.fields["content"].max_length == 5000


@pytest.mark.django_db
def test_contact_overlong_subject_is_a_form_error_not_a_crash(client):
    response = client.post(
        reverse("main:contact"),
        data={"email": "a@example.com", "title": "x" * 201, "content": "Hi"},
    )

    assert response.status_code == 200
    assert "title" in response.context["form"].errors


@pytest.mark.django_db
def test_contact_page_heading_and_title(client):
    html = client.get(reverse("main:contact")).content.decode()

    assert "<title>Contact — Catchment Simulation</title>" in html
    assert '<h1 class="cs-page-header__title">Contact</h1>' in html


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_profile_form_never_lists_other_accounts(client, user):
    User.objects.create_user(username="someone-else", password="x")
    client.force_login(user)

    html = client.get(reverse("main:userprofile", args=[user.id])).content.decode()

    assert "someone-else" not in html
    assert 'id="id_user"' not in html
    assert 'value="Save profile"' in html
    assert "<title>Profile — Catchment Simulation</title>" in html


@pytest.mark.django_db
def test_profile_post_cannot_reassign_the_account(client, user):
    other = User.objects.create_user(username="other", password="x")
    client.force_login(user)

    client.post(
        reverse("main:userprofile", args=[user.id]), data={"user": other.id, "bio": "Mine."}
    )

    assert UserProfile.objects.get(bio="Mine.").user == user
    assert not UserProfile.objects.filter(user=other).exists()


@pytest.mark.django_db
def test_profile_form_owner_comes_from_instance_or_initial(user):
    assert UserProfileForm(initial={"user": user, "bio": ""}).owner == user
    profile = UserProfile.objects.create(user=user, bio="Hi")
    assert UserProfileForm(instance=profile).owner == user


@pytest.mark.django_db
def test_non_owner_sees_bio_as_text_without_email(client, user):
    owner = User.objects.create_user(username="owner", email="owner@example.com", password="x")
    UserProfile.objects.create(user=owner, bio="Hydrologist.")
    client.force_login(user)

    html = client.get(reverse("main:userprofile", args=[owner.id])).content.decode()

    assert '<p class="profile-card__bio">Hydrologist.</p>' in html
    assert 'id="id_bio"' not in html
    assert "owner@example.com" not in html


# ---------------------------------------------------------------------------
# Log in / Create account
# ---------------------------------------------------------------------------


@pytest.mark.django_db
def test_login_page_keeps_next_and_links_to_register(client):
    html = client.get(reverse("login") + "?next=/timeseries").content.decode()

    assert "<title>Log in — Catchment Simulation</title>" in html
    assert '<input type="hidden" name="next" value="/timeseries">' in html
    assert f'href="{reverse("register:register")}">Create an account</a>' in html


@pytest.mark.django_db
def test_register_success_logs_in_and_welcomes(client):
    response = client.post(
        reverse("register:register"),
        data={
            "username": "newbie",
            "email": "newbie@example.com",
            "first_name": "New",
            "last_name": "Bie",
            "password1": SIGNUP_PASSWORD,
            "password2": SIGNUP_PASSWORD,
        },
        follow=True,
    )

    assert response.redirect_chain == [(reverse("main:main_view"), 302)]
    assert response.context["user"].username == "newbie"
    assert b"Welcome, New. Your account is ready." in response.content


@pytest.mark.django_db
def test_register_invalid_logs_field_names_but_not_values(client, caplog, capsys):
    password1, password2 = f"T!{uuid.uuid4().hex}", f"T!{uuid.uuid4().hex}"
    with caplog.at_level(logging.INFO, logger="register.views"):
        response = client.post(
            reverse("register:register"),
            data={"username": "x", "password1": password1, "password2": password2},
        )

    assert response.status_code == 200
    assert "Create account" in response.content.decode()
    assert "password2" in caplog.text
    assert password1 not in caplog.text and password2 not in caplog.text
    assert capsys.readouterr().out == ""
