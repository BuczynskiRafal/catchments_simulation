"""
This module contains views and helper functions for rendering and managing
the main view, about page, contact form, and user profiles, as well as
functions for creating interactive plots.
"""

import datetime
import logging
import math
import os
import uuid
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
    HttpRequest,
    HttpResponse,
    HttpResponseRedirect,
    JsonResponse,
)
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from plotly.subplots import make_subplots
from pyswmm import Simulation

from catchment_simulation.analysis import runoff_volume, time_to_peak
from catchment_simulation.catchment_features_simulation import FeaturesSimulation
from catchment_simulation.schemas import SimulationMethodParams
from main.forms import ContactForm, SimulationForm, TimeseriesForm, UserProfileForm
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


@ajax_login_required
def upload_clear(request: HttpRequest) -> JsonResponse:
    """
    Clear the uploaded file from the session and disk.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP request.

    Returns
    -------
    JsonResponse
        JSON response confirming the upload was cleared.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed."}, status=405)

    file_path = request.session.pop("uploaded_file_path", None)
    if file_path and os.path.exists(file_path):
        os.remove(file_path)

    return JsonResponse({"message": "Upload cleared."})


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
        "simulate_curb_length": "CurbLength",
        "simulate_n_imperv": "N-Imperv",
        "simulate_n_perv": "N-Perv",
        "simulate_s_imperv": "Destore-Imperv",
        "simulate_s_perv": "Destore-Perv",
    }
    return feature_map.get(method_name, "")


def save_output_file(
    request: HttpRequest,
    df: pd.DataFrame,
    output_file_name: str,
    session_prefix: str = "",
) -> None:
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
    session_prefix : str
        Prefix for session keys (e.g. "ts_" for timeseries).
    """
    output_file_path = f"output_files/result_{uuid.uuid4().hex[:8]}.xlsx"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    df.to_excel(output_file_path, index=False)

    with open(output_file_path, "rb") as file:
        default_storage.save(output_file_name, file)

    os.remove(output_file_path)

    output_file_url = default_storage.url(output_file_name)
    request.session[f"{session_prefix}output_file_url"] = output_file_url
    request.session[f"{session_prefix}output_file_name"] = output_file_name


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
        "ts_user_plot",
        "ts_time_to_peak",
        "ts_runoff_volume",
        "ts_show_results",
        "ts_output_file_url",
        "ts_output_file_name",
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
            option = form.cleaned_data["option"]
            is_predefined = option in SimulationForm.PREDEFINED_METHODS

            params = SimulationMethodParams(
                method_name=option,
                start=form.cleaned_data.get("start") if not is_predefined else None,
                stop=form.cleaned_data.get("stop") if not is_predefined else None,
                step=form.cleaned_data.get("step") if not is_predefined else None,
                catchment_name=form.cleaned_data["catchment_name"],
            )

            uploaded_file_path = request.session.get(
                "uploaded_file_path",
                os.path.abspath("catchment_simulation/example.inp"),
            )

            try:
                model = FeaturesSimulation(
                    subcatchment_id=params.catchment_name, raw_file=uploaded_file_path
                )
                feature_name = get_feature_name(params.method_name)

                method = getattr(model, params.method_name)
                if is_predefined:
                    df = method()
                else:
                    df = method(start=params.start, stop=params.stop, step=params.step)
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

            except Exception as e:
                logger.error("Simulation failed: %s", e)
                messages.error(request, f"Error running simulation: {e}")
    else:
        form = SimulationForm()
        session_data = get_session_variables(request)

    return render(
        request,
        "main/simulation.html",
        {"form": form, **session_data},
    )


def _build_sweep_timeseries_plot(
    results: dict[float, pd.DataFrame], feature: str, catchment_name: str
) -> str:
    """Build an overlay Plotly chart for timeseries parameter sweep results."""
    columns = list(FeaturesSimulation.TIMESERIES_KEYS)
    n = len(columns)
    cols = 2
    rows = math.ceil(n / cols)
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=columns,
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    for param_val, ts_df in results.items():
        ts_df_reset = ts_df.reset_index()
        for i, col in enumerate(columns):
            r = i // cols + 1
            c = i % cols + 1
            fig.add_trace(
                go.Scatter(
                    x=ts_df_reset["datetime"],
                    y=ts_df_reset[col],
                    mode="lines",
                    name=f"{feature}={param_val}",
                    legendgroup=str(param_val),
                    showlegend=(i == 0),
                ),
                row=r,
                col=c,
            )
    fig.update_layout(
        height=350 * rows,
        title=dict(
            text=f"Timeseries sweep: {feature} for {catchment_name}",
            x=0.5,
            xanchor="center",
        ),
    )
    return fig.to_html(full_html=False)


def _save_sweep_output(
    request: HttpRequest, results: dict[float, pd.DataFrame], output_file_name: str
) -> None:
    """Save timeseries sweep results to a multi-sheet Excel file."""
    output_file_path = f"output_files/result_{uuid.uuid4().hex[:8]}.xlsx"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with pd.ExcelWriter(output_file_path, engine="openpyxl") as writer:
        for param_val, ts_df in results.items():
            sheet_name = f"val_{param_val:.2f}"[:31]
            ts_df.reset_index().to_excel(writer, sheet_name=sheet_name, index=False)
    with open(output_file_path, "rb") as file:
        default_storage.save(output_file_name, file)
    os.remove(output_file_path)
    output_file_url = default_storage.url(output_file_name)
    request.session["ts_output_file_url"] = output_file_url
    request.session["ts_output_file_name"] = output_file_name


@login_required
def timeseries_view(request: HttpRequest) -> HttpResponse:
    """
    Render the timeseries analysis view.

    Supports two modes:
    - single: Run a single simulation and display per-timestep data with
      analytical metrics (time to peak, runoff volume).
    - sweep: Vary a subcatchment parameter over a range and overlay the
      resulting hydrographs.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP request.

    Returns
    -------
    HttpResponse
        The HTTP response with the rendered timeseries template.
    """
    session_data = {}

    if request.method == "POST":
        form = TimeseriesForm(request.POST)
        if form.is_valid():
            mode = form.cleaned_data["mode"]
            catchment_name = form.cleaned_data["catchment_name"]

            uploaded_file_path = request.session.get(
                "uploaded_file_path",
                os.path.abspath("catchment_simulation/example.inp"),
            )

            try:
                model = FeaturesSimulation(
                    subcatchment_id=catchment_name, raw_file=uploaded_file_path
                )
            except Exception as e:
                logger.error("Failed to initialise FeaturesSimulation: %s", e)
                messages.error(request, f"Error initialising model: {e}")
                return render(request, "main/timeseries.html", {"form": form})

            try:
                if mode == "single":
                    ts_df = model.calculate_timeseries()

                    # Compute analytical metrics
                    try:
                        ttp = time_to_peak(ts_df, column="runoff")
                        ttp_str = str(ttp)
                    except ValueError:
                        ttp_str = "N/A"

                    try:
                        vol = runoff_volume(ts_df, column="runoff")
                        vol_str = f"{vol:.4f}"
                    except ValueError:
                        vol_str = "N/A"

                    # Build Plotly chart with subplots for each timeseries column
                    ts_columns = list(FeaturesSimulation.TIMESERIES_KEYS)
                    ts_df_reset = ts_df.reset_index()
                    ts_df_reset["datetime"] = ts_df_reset["datetime"].astype(str)

                    user_plot = plot(
                        df=ts_df_reset,
                        x="datetime",
                        y=ts_columns,
                        xaxes=False,
                        title=f"Timeseries for subcatchment {catchment_name}",
                    )

                    # Save for download
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file_name = f"{request.user.username}_timeseries_{timestamp}.xlsx"
                    save_output_file(
                        request, ts_df.reset_index(), output_file_name, session_prefix="ts_"
                    )

                    request.session["ts_user_plot"] = user_plot
                    request.session["ts_time_to_peak"] = ttp_str
                    request.session["ts_runoff_volume"] = vol_str
                    request.session["ts_show_results"] = True

                    return redirect("main:timeseries")

                elif mode == "sweep":
                    feature = form.cleaned_data["feature"]
                    start = form.cleaned_data["start"]
                    stop = form.cleaned_data["stop"]
                    step = form.cleaned_data["step"]

                    results = model.simulate_subcatchment_timeseries(
                        feature=feature, start=start, stop=stop, step=step
                    )

                    user_plot = _build_sweep_timeseries_plot(results, feature, catchment_name)

                    # Save multi-sheet Excel
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file_name = f"{request.user.username}_ts_sweep_{timestamp}.xlsx"
                    _save_sweep_output(request, results, output_file_name)

                    request.session["ts_user_plot"] = user_plot
                    request.session["ts_show_results"] = True
                    request.session["ts_time_to_peak"] = None
                    request.session["ts_runoff_volume"] = None

                    return redirect("main:timeseries")

            except Exception as e:
                logger.error("Timeseries analysis failed: %s", e)
                messages.error(request, f"Error running analysis: {e}")
    else:
        form = TimeseriesForm()
        session_data = {
            "ts_user_plot": request.session.get("ts_user_plot"),
            "ts_time_to_peak": request.session.get("ts_time_to_peak"),
            "ts_runoff_volume": request.session.get("ts_runoff_volume"),
            "ts_show_results": request.session.get("ts_show_results", False),
            "output_file_url": request.session.get("ts_output_file_url"),
            "output_file_name": request.session.get("ts_output_file_name"),
        }

    return render(request, "main/timeseries.html", {"form": form, **session_data})


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
