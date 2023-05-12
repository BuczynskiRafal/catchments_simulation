from django.contrib import admin
from django.urls import path, include
from django.conf.urls.static import static
from django.conf import settings
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = (
    [
        path("admin/", admin.site.urls),
        path("", include("main.urls")),
        path("", include("register.urls")),
        path("accounts/", include("django.contrib.auth.urls")),
    ]
    + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
)

urlpatterns += staticfiles_urlpatterns()
