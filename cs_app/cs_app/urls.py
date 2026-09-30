from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.contrib.auth.views import LoginView
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.urls import include, path
from register.forms import LoginForm

urlpatterns = (
    [
        path("admin/", admin.site.urls),
        path("", include("main.urls")),
        path("", include("register.urls")),
        # Ahead of django.contrib.auth.urls so it wins the "login" name and path.
        path("accounts/login/", LoginView.as_view(authentication_form=LoginForm), name="login"),
        path("accounts/", include("django.contrib.auth.urls")),
    ]
    + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
)

urlpatterns += staticfiles_urlpatterns()
