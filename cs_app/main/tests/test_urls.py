from django.urls import resolve, reverse


def test_main_view_url():
    """Tests the URL of the main view."""
    path = reverse("main:main_view")
    assert resolve(path).view_name == "main:main_view"


def test_upload_url():
    """This function tests the URL endpoint for uploading files."""
    path = reverse("main:upload")
    assert resolve(path).view_name == "main:upload"


def test_upload_sample_url():
    """Tests the URL endpoint for sample upload."""
    path = reverse("main:upload_sample")
    assert resolve(path).view_name == "main:upload_sample"


def test_user_profile_url():
    """Tests the URL of the user profile."""
    path = reverse("main:userprofile", kwargs={"user_id": 1})
    assert resolve(path).view_name == "main:userprofile"


def test_contact_url():
    """Tests the URL of the contact page."""
    path = reverse("main:contact")
    assert resolve(path).view_name == "main:contact"


def test_about_url():
    """Tests the URL of the about page."""
    path = reverse("main:about")
    assert resolve(path).view_name == "main:about"


def test_calculations_url():
    """Tests the URL of the calculations page."""
    path = reverse("main:calculations")
    assert resolve(path).view_name == "main:calculations"


def test_simulation_view_url():
    """Tests the URL of the simulation page."""
    path = reverse("main:simulation")
    assert resolve(path).view_name == "main:simulation"


def test_timeseries_view_url():
    """Tests the URL of the timeseries page."""
    path = reverse("main:timeseries")
    assert resolve(path).view_name == "main:timeseries"


def test_download_simulation_results_url():
    """Tests the URL of the simulation download endpoint."""
    path = reverse("main:download_simulation_results")
    assert resolve(path).view_name == "main:download_simulation_results"


def test_download_timeseries_results_url():
    """Tests the URL of the timeseries download endpoint."""
    path = reverse("main:download_timeseries_results")
    assert resolve(path).view_name == "main:download_timeseries_results"


def test_download_timeseries_csv_url():
    """Tests the URL of the timeseries CSV download endpoint."""
    path = reverse("main:download_timeseries_csv")
    assert resolve(path).view_name == "main:download_timeseries_csv"


def test_subcatchments_url():
    """Tests the URL of the subcatchments endpoint."""
    path = reverse("main:subcatchments")
    assert resolve(path).view_name == "main:subcatchments"
