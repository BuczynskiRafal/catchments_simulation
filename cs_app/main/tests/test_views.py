import pytest
from django.urls import reverse
from django.test import RequestFactory
from django.contrib.auth.models import AnonymousUser
from main.views import simulation_view, download_result, calculations

from django.contrib.sessions.middleware import SessionMiddleware
from django.contrib.messages.middleware import MessageMiddleware


@pytest.mark.django_db
def test_main_view(client):
    """
    Test the main view by sending a GET request to the URL
    and checking that the response status code is 200.
    Also check that the correct template is used to render the response.
    """
    response = client.get(reverse("main:main_view"))
    assert response.status_code == 200
    assert "main/main_view.html" in [template.name for template in response.templates]


@pytest.mark.django_db
def test_about_view(client):
    """
    Test the about view using Django's test client.
    """
    response = client.get(reverse("main:about"))
    assert response.status_code == 200
    assert "main/about.html" in [template.name for template in response.templates]


@pytest.mark.django_db
def test_contact_view_get(client):
    """
    Test the functionality of the contact view by checking the response status
    code and the name of the template used for rendering the response.
    """
    response = client.get(reverse("main:contact"))
    assert response.status_code == 200
    assert "main/contact.html" in [template.name for template in response.templates]


@pytest.mark.django_db
def test_contact_view_post_no_data(client):
    """
    Test that the contact view returns a 200 response when 
    submitting no data via POST request.
    """
    response = client.post(reverse("main:contact"))
    assert response.status_code == 200


@pytest.mark.django_db
def test_user_profile_view(client, create_user, test_password):
    """
    Test the user profile view by logging in with a client,
    getting the user's profile, and checking that the response
    status code is 200 and the correct template was used.
    """
    client.login(username="testuser", password=test_password)
    response = client.get(reverse("main:userprofile", args=[create_user.id]))
    assert response.status_code == 200
    assert "main/userprofile.html" in [template.name for template in response.templates]


@pytest.mark.django_db
def test_simulation_view(user):
    """
    Test the simulation view using a Django test client.
    """
    factory = RequestFactory()
    request = factory.post(
        "simulation",
        {
            "option": "slope",
            "start": "1",
            "stop": "100",
            "step": "1",
            "catchment_name": "S1",
        },
    )
    middleware = SessionMiddleware(lambda req: None)
    middleware.process_request(request)
    request.session.save()

    request.user = user
    response = simulation_view(request)

    assert response.status_code == 200


@pytest.mark.django_db
def test_simulation_view_get(user):
    """
    Test the simulation view's GET response by creating a new request via
    RequestFactory with middleware to handle sessions.
    The user is set to the provided user fixture.
    The response status code is asserted to be 200.
    """
    factory = RequestFactory()
    request = factory.get("simulation")
    middleware = SessionMiddleware(lambda req: None)
    middleware.process_request(request)
    request.session.save()
    request.user = user

    response = simulation_view(request)

    assert response.status_code == 200


# @pytest.mark.django_db
# def test_download_result():
#     """
#     Test the download_result view function by simulating a request to the 
#     view and checking that the response
#     has status code 200. Uses the Django test client.
#     """
#     factory = RequestFactory()
#     request = factory.get("simulation")

#     response = download_result(request)

#     assert response.status_code == 200


@pytest.mark.django_db
def test_calculations():
    """
    Test calculations function.
    Uses Django test client to create a request to 'calculations' view.
    Asserts that the response status code is 200.
    """
    factory = RequestFactory()
    request = factory.post("calculations")

    session_middleware = SessionMiddleware(lambda req: None)
    session_middleware.process_request(request)
    request.session.save()

    message_middleware = MessageMiddleware(lambda req: None)
    message_middleware.process_request(request)

    request.user = AnonymousUser()

    response = calculations(request)

    assert response.status_code == 200


@pytest.mark.django_db
def test_calculations_get():
    """
    Test calculations get.
    This function tests the calculations view function with a GET request.
    It creates a request factory, sets the request
    method to GET, and sets the user attribute to AnonymousUser. It then calls
    the calculations view function with the
    created request object and asserts that the response status code is 200.

    """
    factory = RequestFactory()
    request = factory.get("calculations")
    request.user = AnonymousUser()

    response = calculations(request)

    assert response.status_code == 200
