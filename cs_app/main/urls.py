from django.urls import path
from .views import main_view, about, contact, user_profile, simulation_view, upload, download_result

app_name = "main"


urlpatterns = [
    path('', main_view, name='main_view'),
    path('about', about, name='about'),
    path('simulation', simulation_view, name='simulation'),
    path('contact', contact, name='contact'),
    path('user/<int:user_id>/profile', user_profile, name='userprofile'),
    path('upload/', upload, name='upload'),
]
