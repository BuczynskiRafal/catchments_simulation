"""
This module contains views and helper functions for rendering and managing
the main view, about page, contact form, and user profiles, as well as
functions for creating interactive plots.
"""

import datetime
import logging
import os
from functools import wraps

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import swmmio
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.core.files.storage import default_storage
from django.http import (
    FileResponse,
    HttpRequest,
    HttpResponse,
    HttpResponseNotFound,
    HttpResponseRedirect,
    JsonResponse,
)
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from plotly.subplots import make_subplots
from pyswmm import Simulation

from catchment_simulation.catchment_features_simulation import FeaturesSimulation
from catchment_simulation.schemas import SimulationMethodParams
from main.forms import ContactForm, SimulationForm, UserProfileForm
from main.predictor import predict_runoff
from main.schemas import ContactMessage
from main.services import send_message

logger = logging.getLogger(__name__)


def ajax_login_required(view_func):
    """
    Decorator that checks authentication for AJAX requests.

    For AJAX requests, returns a 401 JSON response with a login URL.
    For regular requests, redirects to the login page.
    """

    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.user.is_authenticated:
            login_url = settings.LOGIN_URL
            is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
            if is_ajax:
                return JsonResponse(
                    {"error": "Authentication required.", "login_url": login_url},
                    status=401,
                )
            return redirect(f"{login_url}?next={request.path}")
        return view_func(request, *args, **kwargs)

    return wrapper


def plot(
    x: str,
    y: str | list[str] | None = "runoff",
    path: str | None = None,
    df: pd.DataFrame | None = None,
    xaxes: bool | None = False,
    start: int | None = 0,
    stop: int | None = 100,
    title: str | None = None,
    rename_labels: bool | None = False,
    x_name: str | None = None,
    y_name: str | None = None,
) -> str:
    """
    Create an interactive plot using Plotly.

    Parameters
    ----------
    x : str
        Column name for x-axis.
    y : str, optional
        Column name for y-axis (default is "runoff").
    path : str, optional
        Path to the input file (default is None).
    df : pd.DataFrame, optional
        Input dataframe (default is None).
    xaxes : bool, optional
        Whether to show x-axis range or not (default is False).
    start : int, optional
        Start value for x-axis range (default is 0).
    stop : int, optional
        Stop value for x-axis range (default is 100).
    title : str, optional
        Title for the plot (default is None).
    rename_labels : bool, optional
        Whether to rename axis labels (default is False).
    x_name : str, optional
        New name for x-axis label (default is None).
    y_name : str, optional
        New name for y-axis label (default is None).

    Returns
    -------
    str
        The generated plot as an HTML string.
    """
    if path is not None:
        df = pd.read_excel(path)

    if isinstance(y, list) and len(y) > 1:
        import math

        n = len(y)
        cols = 2
        rows = math.ceil(n / cols)
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=y,
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )
        for i, col in enumerate(y):
            r = i // cols + 1
            c = i % cols + 1
            fig.add_trace(
                go.Scatter(x=df[x], y=df[col], mode="lines", name=col),
                row=r,
                col=c,
            )
        fig.update_layout(
            height=350 * rows,
            title=dict(text=title, x=0.5, xanchor="center"),
            showlegend=False,
        )
        if xaxes:
            fig.update_xaxes(range=[start, stop])
    else:
        fig = px.line(df, x, y, title=title)
        if xaxes:
            fig.update_xaxes(range=[start, stop])
        if rename_labels:
            fig.update_xaxes(title_text=x_name)
            fig.update_yaxes(title_text=y_name)
        fig.update_layout(title=dict(text=title, x=0.5, xanchor="center"))

    plot_div = fig.to_html(full_html=False)
    return plot_div


def main_view(request: HttpRequest) -> HttpResponse:
    """
    Render the main view with interactive plots.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP request.

    Returns
    -------
    HttpResponse
        The HTTP response with the rendered main view template.
    """
    context = {
        "plot_slope": plot(
            path=os.path.join(settings.BASE_DIR, "data", "df_slope.xlsx"),
            x="slope",
            xaxes=True,
            title="Dependence of runoff on subcatchment slope.",
            rename_labels=True,
            x_name="Percent Slope [-]",
            y_name="Runoff [m3]",
        ),
        "plot_area": plot(
            path=os.path.join(settings.BASE_DIR, "data", "df_area.xlsx"),
            x="area",
            xaxes=False,
            title="Dependence of runoff on subcatchment area.",
            rename_labels=True,
            x_name="Area [ha]",
            y_name="Runoff [m3]",
        ),
        "plot_width": plot(
            path=os.path.join(settings.BASE_DIR, "data", "df_width.xlsx"),
            x="width",
            xaxes=True,
            stop=1000,
            title="Dependence of runoff on subcatchment width.",
            rename_labels=True,
            x_name="Width [m]",
            y_name="Runoff [m3]",
        ),
    }
    return render(request, "main/main_view.html", context)


def about(request: HttpRequest) -> HttpResponse:
    """
    Render the about page.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP request.

    Returns
    -------
    HttpResponse
        The HTTP response with the rendered about page template.
    """
    return render(request, "main/about.html")


def contact(request: HttpRequest) -> HttpResponse:
    """
    Render the contact form and handle form submission.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP request.

    Returns
    -------
    HttpResponse
        The HTTP response with the rendered contact form template.
    """
    if request.method == "POST":
        form = ContactForm(data=request.POST)
        if form.is_valid():
            message = ContactMessage.model_validate(form.cleaned_data)
            send_message(message)
            return HttpResponseRedirect(reverse("contact"))
    else:
        form = ContactForm()
    return render(request, "main/contact.html", {"form": form})


def user_profile(request: HttpRequest, user_id: int) -> HttpResponse:
    """
    Render the user profile page and handle form submission.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP request.
    user_id : int
        The user ID for the profile.

    Returns
    -------
    HttpResponse
        The HTTP response with the rendered user profile template.
    """
    user = get_object_or_404(get_user_model(), id=user_id)
    if request.method == "POST":
        try:
            profile = user.userprofile
            form = UserProfileForm(request.POST, instance=profile)
        except AttributeError:
            logger.debug(f"User {user_id} has no profile, creating new form for POST")
            form = UserProfileForm(request.POST, initial={"user": user, "bio": ""})
        if form.is_valid():
            form.save()
    else:
        try:
            profile = user.userprofile
            form = UserProfileForm(instance=profile)
        except AttributeError:
            logger.debug(f"User {user_id} has no profile, creating empty form")
            form = UserProfileForm(initial={"user": user, "bio": ""})
        if request.user != user:
            for field in form.fields:
                form.fields[field].disabled = True
            form.helper.inputs = []
    return render(request, "main/userprofile.html", {"form": form})


MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB


def _sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks."""
    import re

    # Remove path separators and other dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", filename)
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip(". ")
    # Limit length
    return sanitized[:100] if sanitized else "uploaded_file"


def _validate_inp_file_content(file_content: bytes) -> bool:
    """
    Validate that the file content appears to be a valid SWMM .inp file.

    SWMM .inp files typically start with section headers like [TITLE], [OPTIONS], etc.
    Handles BOM markers and various line endings (CRLF/LF).
    """
    try:
        if file_content.startswith(b"\xef\xbb\xbf"):
            file_content = file_content[3:]
        elif file_content.startswith(b"\xff\xfe") or file_content.startswith(b"\xfe\xff"):
            file_content = file_content[2:]

        try:
            content_str = file_content.decode("utf-8")
        except UnicodeDecodeError:
            content_str = file_content.decode("latin-1")

        content_upper = content_str.replace("\r\n", "\n").replace("\r", "\n").upper()

        # Check for common SWMM section headers
        swmm_sections = ["[TITLE]", "[OPTIONS]", "[RAINGAGES]", "[SUBCATCHMENTS]", "[SUBAREAS]"]
        return any(section in content_upper for section in swmm_sections)
    except Exception:
        return False


@ajax_login_required
def upload(request: HttpRequest) -> JsonResponse:
    """
    Upload a .inp file to the server.

    Requires user authentication. For AJAX requests (like Dropzone.js),
    returns a 401 JSON response if not authenticated.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP request.

    Returns
    -------
    JsonResponse
        JSON response containing a success message if the file was uploaded successfully, or an error message if not.
    """
    if request.method == "POST":
        if "file" not in request.FILES:
            return JsonResponse({"error": "No file provided."}, status=400)

        uploaded_file = request.FILES["file"]
        filename, file_extension = os.path.splitext(uploaded_file.name)

        # Check file extension
        if file_extension.lower() != ".inp":
            return JsonResponse(
                {"error": "Invalid file type. Please upload a .inp file."},
                status=400,
            )

        # Check file size
        if uploaded_file.size > MAX_UPLOAD_SIZE:
            return JsonResponse(
                {
                    "error": f"File too large. Maximum size is {MAX_UPLOAD_SIZE // (1024 * 1024)} MB."
                },
                status=400,
            )

        # Read file content for validation
        file_content = uploaded_file.read()
        uploaded_file.seek(0)  # Reset file pointer

        # Validate file content
        if not _validate_inp_file_content(file_content):
            logger.warning(f"Invalid .inp file content uploaded: {uploaded_file.name}")
            return JsonResponse(
                {
                    "error": "Invalid file content. The file does not appear to be a valid SWMM .inp file."
                },
                status=400,
            )

        # Sanitize filename
        safe_filename = _sanitize_filename(filename)
        file_path = os.path.join("uploaded_files", safe_filename + file_extension)

        # Ensure upload directory exists
        os.makedirs("uploaded_files", exist_ok=True)

        with open(file_path, "wb+") as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        request.session["uploaded_file_path"] = file_path
        logger.info(f"File uploaded successfully: {file_path}")

        return JsonResponse({"message": "File was sent."})

    return JsonResponse({"error": "Error occurred while sending file."}, status=400)


def get_feature_name(method_name: str) -> str:
    """
    Get the feature name based on the method name.

    Parameters
    ----------
    method_name : str
        The name of the method.

    Returns
    -------
    str
        The feature name associated with the method name.
    """

    feature_map = {
        "simulate_percent_slope": "PercSlope",
        "simulate_area": "Area",
        "simulate_width": "Width",
        "simulate_percent_impervious": "PercImperv",
        "simulate_percent_zero_imperv": "Zero-Imperv",
    }
    return feature_map.get(method_name, "")


def save_output_file(request: HttpRequest, df: pd.DataFrame, output_file_name: str) -> None:
    """
    Save the output file to the server.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP request.
    df : pd.DataFrame
        The dataframe containing the simulation results.
    output_file_name : str
        The name of the output file.
    """
    output_file_path = "output_files/simulation_result.xlsx"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    df.to_excel(output_file_path, index=False)

    with open(output_file_path, "rb") as file:
        default_storage.save(output_file_name, file)

    output_file_url = default_storage.url(output_file_name)
    request.session["output_file_url"] = output_file_url
    request.session["output_file_name"] = output_file_name


def get_session_variables(request: HttpRequest) -> dict:
    """
    Get the session variables.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP request.

    Returns
    -------
    dict
        A dictionary containing the session variables.
    """
    return {
        "show_download_button": request.session.get("show_download_button", False),
        "user_plot": request.session.get("user_plot", None),
        "results_columns": request.session.get("results_columns", []),
        "results_data": request.session.get("results_data", []),
        "feature_name": request.session.get("feature_name", ""),
        "output_file_name": request.session.get("output_file_name", None),
        "output_file_url": request.session.get("output_file_url", None),
    }


def clear_session_variables(request: HttpRequest) -> None:
    """
    Clear the session variables.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP request.
    """
    for variable in [
        "show_download_button",
        "user_plot",
        "results_columns",
        "results_data",
        "feature_name",
        "output_file_name",
        "output_file_url",
        "uploaded_file_path",
    ]:
        if variable in request.session:
            del request.session[variable]


@login_required
def simulation_view(request: HttpRequest) -> HttpResponse:
    """
    Render the simulation view.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP request.

    Returns
    -------
    HttpResponse
        The HTTP response with the rendered simulation template.
    """
    show_download_button = False
    user_plot = None
    output_file_name = None
    session_data = {}

    if request.method == "POST":
        form = SimulationForm(request.POST)
        if form.is_valid():
            params = SimulationMethodParams(
                method_name=form.cleaned_data["option"],
                start=form.cleaned_data["start"],
                stop=form.cleaned_data["stop"],
                step=form.cleaned_data["step"],
                catchment_name=form.cleaned_data["catchment_name"],
            )

            uploaded_file_path = request.session.get(
                "uploaded_file_path",
                os.path.abspath("catchment_simulation/example.inp"),
            )
            model = FeaturesSimulation(
                subcatchment_id=params.catchment_name, raw_file=uploaded_file_path
            )
            feature_name = get_feature_name(params.method_name)

            method = getattr(model, params.method_name)
            df = method(start=params.start, stop=params.stop, step=params.step)
            # Reorder: feature first, then the rest
            other_cols = [c for c in df.columns if c != feature_name]
            df = df[[feature_name] + other_cols]

            show_download_button = True

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file_name = f"{request.user.username}_simulation_result_{timestamp}.xlsx"

            user_plot = plot(
                df=df,
                x=feature_name,
                y=other_cols,
                xaxes=False,
                title=f"Dependence of runoff on subcatchment {feature_name}.",
            )

            request.session["show_download_button"] = show_download_button
            request.session["user_plot"] = user_plot
            request.session["results_columns"] = df.columns.tolist()
            request.session["results_data"] = df.values.tolist()
            request.session["feature_name"] = feature_name

            save_output_file(request, df, output_file_name)
            return redirect("main:simulation")
    else:
        form = SimulationForm()
        session_data = get_session_variables(request)

    return render(
        request,
        "main/simulation.html",
        {"form": form, **session_data},
    )


def download_result(request: HttpRequest) -> HttpResponse:
    """
    Download the simulation result file.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP request.

    Returns
    -------
    HttpResponse
        The HTTP response containing the file download or a 404 response if the file is not found.
    """
    output_file_path = "output_files/simulation_result.xlsx"

    if os.path.exists(output_file_path):
        response = FileResponse(
            open(output_file_path, "rb"),
            content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        response["Content-Disposition"] = "attachment; filename=simulation_result.xlsx"
        return response
    else:
        return HttpResponseNotFound("File not found.")


def calculations(request: HttpRequest) -> HttpResponse:
    """
    Perform calculations on the uploaded file.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP request.

    Returns
    -------
    HttpResponse
        The HTTP response with the rendered calculations template.
    """
    df = None
    if request.method == "POST":
        uploaded_file_path = request.session.get("uploaded_file_path", None)

        if not uploaded_file_path:
            messages.error(request, "Please upload a file first.")
        else:
            try:
                swmmio_model = swmmio.Model(uploaded_file_path)

                with Simulation(uploaded_file_path) as sim:
                    for _ in sim:
                        pass
                ann_predictions = predict_runoff(swmmio_model).transpose()
                df = pd.DataFrame(
                    data={
                        "Name": swmmio_model.subcatchments.dataframe.index,
                        "SWMM_Runoff_m3": swmmio_model.subcatchments.dataframe[
                            "TotalRunoffMG"
                        ].values,
                        "ANN_Runoff_m3": np.round(ann_predictions, 2),
                    },
                )

            except Exception as e:
                logger.error(e)
                messages.error(
                    request,
                    f"Error occurred while performing calculations: {str(e)}",
                )

    df_is_empty = df is None or df.empty

    return render(request, "main/calculations.html", {"df": df, "df_is_empty": df_is_empty})
