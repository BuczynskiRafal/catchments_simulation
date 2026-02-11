"""
This module contains views and helper functions for rendering and managing
the main view, about page, contact form, and user profiles.
"""

import datetime
import json
import logging
import os
import uuid
from functools import lru_cache, wraps

import numpy as np
import pandas as pd
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
from django.views.decorators.http import require_GET
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


@lru_cache(maxsize=1)
def _load_static_chart_data() -> dict:
    """Load and cache static chart data from Excel files (read once at startup)."""
    data_dir = os.path.join(settings.BASE_DIR, "data")
    return {
        "slope": json.loads(
            pd.read_excel(os.path.join(data_dir, "df_slope.xlsx")).to_json(orient="records")
        ),
        "area": json.loads(
            pd.read_excel(os.path.join(data_dir, "df_area.xlsx")).to_json(orient="records")
        ),
        "width": json.loads(
            pd.read_excel(os.path.join(data_dir, "df_width.xlsx")).to_json(orient="records")
        ),
    }


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
    context = {"chart_data": _load_static_chart_data()}
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

        # Sanitize filename and scope to user
        safe_filename = _sanitize_filename(filename)
        user_dir = os.path.join("uploaded_files", str(request.user.id))
        file_path = os.path.join(user_dir, safe_filename + file_extension)

        # Remove previous uploaded file from disk
        old_path = request.session.get("uploaded_file_path")
        if old_path and old_path != file_path and os.path.exists(old_path):
            os.remove(old_path)

        # Ensure upload directory exists
        os.makedirs(user_dir, exist_ok=True)

        with open(file_path, "wb+") as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        request.session["uploaded_file_path"] = file_path
        logger.info(f"File uploaded successfully: {file_path}")

        return JsonResponse({"message": "File was sent."})

    return JsonResponse({"error": "Error occurred while sending file."}, status=400)


@require_GET
@ajax_login_required
def upload_status(request: HttpRequest) -> JsonResponse:
    """
    Return the current uploaded file status from the session.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP request.

    Returns
    -------
    JsonResponse
        JSON with ``has_file``, ``filename``, and ``size`` when a file is
        present, or ``has_file: false`` otherwise.
    """
    file_path = request.session.get("uploaded_file_path")
    if file_path:
        try:
            size = os.path.getsize(file_path)
            return JsonResponse(
                {
                    "has_file": True,
                    "filename": os.path.basename(file_path),
                    "size": size,
                }
            )
        except OSError:
            # File disappeared between session set and now – clean up
            request.session.pop("uploaded_file_path", None)
    return JsonResponse({"has_file": False})


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


def _delete_stored_file(request: HttpRequest, session_key: str) -> None:
    """Delete a file from default_storage using the name stored in a session key."""
    name = request.session.get(session_key)
    if name:
        try:
            default_storage.delete(name)
        except Exception:
            logger.warning("Failed to delete stored file %s", name, exc_info=True)


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

    try:
        df.to_excel(output_file_path, index=False)
        with open(output_file_path, "rb") as file:
            saved_name = default_storage.save(output_file_name, file)
    finally:
        if os.path.exists(output_file_path):
            os.remove(output_file_path)

    _delete_stored_file(request, f"{session_prefix}output_file_name")
    output_file_url = default_storage.url(saved_name)
    request.session[f"{session_prefix}output_file_url"] = output_file_url
    request.session[f"{session_prefix}output_file_name"] = saved_name


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
        "chart_config": request.session.get("chart_config", None),
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
    _delete_stored_file(request, "output_file_name")
    _delete_stored_file(request, "ts_output_file_name")

    for variable in [
        "show_download_button",
        "chart_config",
        "results_columns",
        "results_data",
        "feature_name",
        "output_file_name",
        "output_file_url",
        "ts_chart_config",
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
                with FeaturesSimulation(
                    subcatchment_id=params.catchment_name, raw_file=uploaded_file_path
                ) as model:
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

                request.session["show_download_button"] = show_download_button
                request.session["chart_config"] = {
                    "data": json.loads(df.to_json(orient="records")),
                    "x": feature_name,
                    "y": other_cols,
                    "title": f"Dependence of runoff on subcatchment {feature_name}.",
                }
                request.session["results_columns"] = df.columns.tolist()
                request.session["results_data"] = df.values.tolist()
                request.session["feature_name"] = feature_name

                save_output_file(request, df, output_file_name)
                return redirect("main:simulation")

            except Exception as e:
                logger.error("Simulation failed: %s", e)
                messages.error(request, "An error occurred while running the simulation.")
    else:
        form = SimulationForm()
        session_data = get_session_variables(request)

    return render(
        request,
        "main/simulation.html",
        {"form": form, **session_data},
    )


def _save_sweep_output(
    request: HttpRequest, results: dict[float, pd.DataFrame], output_file_name: str
) -> None:
    """Save timeseries sweep results to a multi-sheet Excel file."""
    output_file_path = f"output_files/result_{uuid.uuid4().hex[:8]}.xlsx"
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    try:
        with pd.ExcelWriter(output_file_path, engine="openpyxl") as writer:
            for param_val, ts_df in results.items():
                sheet_name = f"val_{param_val:.2f}"[:31]
                ts_df.reset_index().to_excel(writer, sheet_name=sheet_name, index=False)
        with open(output_file_path, "rb") as file:
            saved_name = default_storage.save(output_file_name, file)
    finally:
        if os.path.exists(output_file_path):
            os.remove(output_file_path)

    _delete_stored_file(request, "ts_output_file_name")
    output_file_url = default_storage.url(saved_name)
    request.session["ts_output_file_url"] = output_file_url
    request.session["ts_output_file_name"] = saved_name


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
                with FeaturesSimulation(
                    subcatchment_id=catchment_name, raw_file=uploaded_file_path
                ) as model:
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

                        # Serialize timeseries data as JSON for frontend rendering
                        ts_columns = list(FeaturesSimulation.TIMESERIES_KEYS)
                        ts_df_reset = ts_df.reset_index()
                        ts_df_reset["datetime"] = ts_df_reset["datetime"].astype(str)

                        # Save for download
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_file_name = f"{request.user.username}_timeseries_{timestamp}.xlsx"
                        save_output_file(
                            request, ts_df.reset_index(), output_file_name, session_prefix="ts_"
                        )

                        request.session["ts_chart_config"] = {
                            "mode": "single",
                            "data": json.loads(ts_df_reset.to_json(orient="records")),
                            "columns": ts_columns,
                            "title": f"Timeseries for subcatchment {catchment_name}",
                        }
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

                        ts_columns = list(FeaturesSimulation.TIMESERIES_KEYS)
                        sweep_data = {}
                        for param_val, ts_df in results.items():
                            ts_df_reset = ts_df.reset_index()
                            ts_df_reset["datetime"] = ts_df_reset["datetime"].astype(str)
                            sweep_data[str(param_val)] = ts_df_reset.to_dict(orient="records")

                        # Save multi-sheet Excel
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_file_name = f"{request.user.username}_ts_sweep_{timestamp}.xlsx"
                        _save_sweep_output(request, results, output_file_name)

                        request.session["ts_chart_config"] = {
                            "mode": "sweep",
                            "data": sweep_data,
                            "columns": ts_columns,
                            "title": f"Timeseries sweep: {feature} for {catchment_name}",
                            "feature": feature,
                            "catchment": catchment_name,
                        }
                        request.session["ts_show_results"] = True
                        request.session["ts_time_to_peak"] = None
                        request.session["ts_runoff_volume"] = None

                        return redirect("main:timeseries")

            except Exception as e:
                logger.error("Timeseries analysis failed: %s", e)
                messages.error(request, "An error occurred while running the analysis.")
    else:
        form = TimeseriesForm()
        session_data = {
            "ts_chart_config": request.session.get("ts_chart_config"),
            "ts_time_to_peak": request.session.get("ts_time_to_peak"),
            "ts_runoff_volume": request.session.get("ts_runoff_volume"),
            "ts_show_results": request.session.get("ts_show_results", False),
            "output_file_url": request.session.get("ts_output_file_url"),
            "output_file_name": request.session.get("ts_output_file_name"),
        }

    return render(request, "main/timeseries.html", {"form": form, **session_data})


def _cleanup_swmm_side_files(inp_path: str) -> None:
    """Remove .rpt and .out files generated by SWMM alongside a .inp file."""
    base = os.path.splitext(inp_path)[0]
    for ext in FeaturesSimulation.SWMM_SIDE_EXTENSIONS:
        if ext == ".inp":
            continue
        try:
            os.remove(base + ext)
        except OSError:
            pass


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
                _cleanup_swmm_side_files(uploaded_file_path)

            except Exception as e:
                logger.error(e)
                messages.error(
                    request,
                    "An error occurred while performing calculations.",
                )

    df_is_empty = df is None or df.empty

    return render(request, "main/calculations.html", {"df": df, "df_is_empty": df_is_empty})
