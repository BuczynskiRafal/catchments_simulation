from django.urls import path

from .views import register

app_name = "register"


urlpatterns = [
    path("register/", register, name="register"),
]
