from django.urls import path
from .views import main_view, about, contact, user_profile, simulation_view, upload, download_result, calculations

app_name = "main"


urlpatterns = [
    path('', main_view, name='main_view'),
    path('upload/', upload, name='upload'),
    path('user/<int:user_id>/profile', user_profile, name='userprofile'),
    path('contact', contact, name='contact'),
    path('about', about, name='about'),
    path('calculations', calculations, name='calculations'),
    path('simulation', simulation_view, name='simulation'),
]
