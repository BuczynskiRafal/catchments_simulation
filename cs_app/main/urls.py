"""
This module defines the URL patterns for the main app, including the main view,
user profile, contact form, about page, simulation view, timeseries view,
and file upload and download.

Django uses the urlpatterns list to match the requested URL
with the corresponding view function.
"""

from django.urls import path

from .views import (
    about,
    calculations,
    contact,
    main_view,
    simulation_view,
    subcatchments,
    timeseries_view,
    upload,
    upload_clear,
    upload_status,
    user_profile,
)

app_name = "main"


urlpatterns = [
    path("", main_view, name="main_view"),
    path("upload/", upload, name="upload"),
    path("upload/clear/", upload_clear, name="upload_clear"),
    path("upload/status/", upload_status, name="upload_status"),
    path("user/<int:user_id>/profile", user_profile, name="userprofile"),
    path("contact", contact, name="contact"),
    path("about", about, name="about"),
    path("calculations", calculations, name="calculations"),
    path("simulation", simulation_view, name="simulation"),
    path("timeseries", timeseries_view, name="timeseries"),
    path("subcatchments/", subcatchments, name="subcatchments"),
]
