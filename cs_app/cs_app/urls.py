from django.contrib import admin
from django.contrib.auth.views import LoginView, LogoutView
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.urls import include, path
from django.views.decorators.http import require_POST
from register.forms import LoginForm

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("main.urls")),
    path("", include("register.urls")),
    path("accounts/login/", LoginView.as_view(authentication_form=LoginForm), name="login"),
    path("accounts/logout/", require_POST(LogoutView.as_view()), name="logout"),
]

urlpatterns += staticfiles_urlpatterns()
