import os
import pytest
from django.urls import reverse
from django.test import RequestFactory, Client
from django.contrib.auth.models import AnonymousUser, User
from main.views import simulation_view, download_result, calculations
from django.contrib.sessions.middleware import SessionMiddleware
from django.contrib.messages.middleware import MessageMiddleware


@pytest.mark.django_db
def test_main_view(client):
    response = client.get(reverse("main:main_view"))
    assert response.status_code == 200
    assert "main/main_view.html" in [template.name for template in response.templates]


@pytest.mark.django_db
def test_about_view(client):
    response = client.get(reverse('main:about'))
    assert response.status_code == 200
    assert 'main/about.html' in [template.name for template in response.templates]


@pytest.mark.django_db
def test_contact_view_GET(client):
    response = client.get(reverse('main:contact'))
    assert response.status_code == 200
    assert 'main/contact.html' in [template.name for template in response.templates]


@pytest.mark.django_db
def test_contact_view_POST_no_data(client):
    response = client.post(reverse('main:contact'))
    assert response.status_code == 200


@pytest.mark.django_db
def test_user_profile_view(client, create_user, test_password):
    client.login(username='testuser', password=test_password)
    response = client.get(reverse('main:userprofile', args=[create_user.id]))
    assert response.status_code == 200
    assert 'main/userprofile.html' in [template.name for template in response.templates]

# @pytest.mark.django_db
# def test_upload_file(client, uploaded_file):
#     response = client.post(reverse('main:upload'), {'file': uploaded_file})
#     assert response.status_code == 200
#     assert response.json() == {"message": "File was sent."}

#     assert 'uploaded_file_path' in client.session  # Check if file path is stored in session

#     assert os.path.exists(client.session['uploaded_file_path'])  # Check if file was saved on server
#     os.remove(client.session['uploaded_file_path'])  # Clean up after test

# @pytest.mark.django_db
# def test_upload_file_wrong_extension(client, wrong_extension_file):
#     response = client.post('/upload', {'file': wrong_extension_file})
#     assert response.status_code == 200
#     assert response.json() == {"error": "Invalid file type. Please upload a .inp file."}

# @pytest.mark.django_db
# def test_upload_file_no_file(client):
#     response = client.post('/path/to/upload', {})
#     assert response.status_code == 200
#     assert response.json() == {"error": "Error occurred while sending file."}


@pytest.mark.django_db
def test_simulation_view(user):
    factory = RequestFactory()
    request = factory.post(
        'simulation',
        {
            "option": "slope",
            "start": "1",
            "stop": "100",
            "step": "1",
            "catchment_name": "S1",
        }
    )
    middleware = SessionMiddleware(lambda req: None)
    middleware.process_request(request)
    request.session.save()

    request.user = user
    response = simulation_view(request)

    assert response.status_code == 200

@pytest.mark.django_db
def test_simulation_view_GET(user):
    factory = RequestFactory()
    request = factory.get('simulation')
    middleware = SessionMiddleware(lambda req: None)
    middleware.process_request(request)
    request.session.save()
    request.user = user

    response = simulation_view(request)

    assert response.status_code == 200

@pytest.mark.django_db
def test_download_result():
    factory = RequestFactory()
    request = factory.get('simulation')

    response = download_result(request)

    assert response.status_code == 404

# @pytest.mark.django_db
# def test_calculations():
#     factory = RequestFactory()
#     request = factory.post('calculations')
#     middleware = SessionMiddleware(lambda req: None)
#     middleware.process_request(request)
#     request.session.save()
#     request.user = AnonymousUser()

#     response = calculations(request)

#     assert response.status_code == 200

@pytest.mark.django_db
def test_calculations_GET():
    factory = RequestFactory()
    request = factory.get('calculations')
    request.user = AnonymousUser()

    response = calculations(request)

    assert response.status_code == 200
