"""
This module defines the URL patterns for the main app, including the main view,
user profile, contact form, about page, simulation view, timeseries view,
and file upload and download.

Django uses the urlpatterns list to match the requested URL
with the corresponding view function.
"""

from django.urls import path

from .api import (
    api_calculations,
    api_simulation,
    api_status,
    api_subcatchments,
    api_timeseries,
    api_upload,
    api_upload_clear,
    api_upload_sample,
)
from .views import (
    about,
    calculations,
    contact,
    download_simulation_results,
    download_timeseries_csv,
    download_timeseries_results,
    healthz,
    main_view,
    simulation_view,
    subcatchments,
    timeseries_view,
    upload,
    upload_clear,
    upload_sample,
    upload_status,
    user_profile,
)

app_name = "main"


urlpatterns = [
    path("", main_view, name="main_view"),
    path("healthz", healthz, name="healthz"),
    path("upload/", upload, name="upload"),
    path("upload/sample/", upload_sample, name="upload_sample"),
    path("upload/clear/", upload_clear, name="upload_clear"),
    path("upload/status/", upload_status, name="upload_status"),
    path("user/<int:user_id>/profile", user_profile, name="userprofile"),
    path("contact", contact, name="contact"),
    path("about", about, name="about"),
    path("calculations", calculations, name="calculations"),
    path("simulation", simulation_view, name="simulation"),
    path(
        "simulation/download/",
        download_simulation_results,
        name="download_simulation_results",
    ),
    path("timeseries", timeseries_view, name="timeseries"),
    path(
        "timeseries/download/",
        download_timeseries_results,
        name="download_timeseries_results",
    ),
    path(
        "timeseries/download/csv/",
        download_timeseries_csv,
        name="download_timeseries_csv",
    ),
    path("subcatchments/", subcatchments, name="subcatchments"),
    path("api/status/", api_status, name="api_status"),
    path("api/upload/", api_upload, name="api_upload"),
    path("api/upload/sample/", api_upload_sample, name="api_upload_sample"),
    path("api/upload/clear/", api_upload_clear, name="api_upload_clear"),
    path("api/subcatchments/", api_subcatchments, name="api_subcatchments"),
    path("api/simulation/", api_simulation, name="api_simulation"),
    path("api/timeseries/", api_timeseries, name="api_timeseries"),
    path("api/calculations/", api_calculations, name="api_calculations"),
]
