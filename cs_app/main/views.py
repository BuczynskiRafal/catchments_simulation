"""
This module contains views and helper functions for rendering and managing
the main view, about page, contact form, and user profiles.
"""

import datetime
import json
import logging
import os
import re
import uuid
from copy import deepcopy
from functools import lru_cache, wraps
from io import BytesIO

import numpy as np
import pandas as pd
import swmmio
from django import forms
from django.conf import settings
from django.contrib import messages
from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from django.core.exceptions import ValidationError
from django.http import (
    FileResponse,
    HttpRequest,
    HttpResponse,
    HttpResponseRedirect,
    JsonResponse,
)
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_GET, require_POST
from pyswmm import Simulation

from catchment_simulation.analysis import runoff_volume, time_to_peak
from catchment_simulation.catchment_features_simulation import FeaturesSimulation
from catchment_simulation.schemas import SimulationMethodParams
from main.forms import ContactForm, SimulationForm, TimeseriesForm, UserProfileForm
from main.predictor import predict_runoff
from main.schemas import ContactMessage
from main.services import send_message

logger = logging.getLogger(__name__)
SIM_FORM_STATE_SESSION_KEY = "sim_form_state"
TS_FORM_STATE_SESSION_KEY = "ts_form_state"
SIM_FORM_STATE_FIELDS = ("option", "start", "stop", "step", "catchment_name")
TS_FORM_STATE_FIELDS = ("mode", "feature", "start", "stop", "step", "catchment_name")
EXCEL_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
SIM_RESULT_TOKEN_SESSION_KEY = "sim_result_token"
TS_RESULT_TOKEN_SESSION_KEY = "ts_result_token"
RESULT_CACHE_TTL_SECONDS = 30 * 60
MAX_RESULT_CACHE_BYTES = 2 * 1024 * 1024


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


def _load_chart_json(filename: str, x_key: str, y_key: str) -> list[dict]:
    """Load and validate chart data JSON structure."""
    data_dir = os.path.join(settings.BASE_DIR, "data")
    path = os.path.join(data_dir, filename)
    with open(path, encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError(f"{filename} must contain a JSON list")

    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            raise ValueError(f"{filename} row {idx} is not an object")
        if x_key not in row or y_key not in row:
            raise ValueError(f"{filename} row {idx} missing required keys")
        if not isinstance(row[x_key], int | float) or not isinstance(row[y_key], int | float):
            raise ValueError(f"{filename} row {idx} contains non-numeric values")

    return payload


@lru_cache(maxsize=1)
def _load_static_chart_data_cached() -> dict:
    """Load static chart data once per process."""
    try:
        return {
            "slope": _load_chart_json("df_slope.json", "slope", "runoff"),
            "area": _load_chart_json("df_area.json", "area", "runoff"),
            "width": _load_chart_json("df_width.json", "width", "runoff"),
        }
    except Exception:
        logger.exception("Failed to load static chart data from JSON files")
        return {"slope": [], "area": [], "width": []}


def _load_static_chart_data() -> dict:
    """Return a defensive copy so cache data cannot be mutated by callers."""
    return deepcopy(_load_static_chart_data_cached())


def _result_cache_key(scope: str, user_id: int, token: str) -> str:
    return f"result:{scope}:{user_id}:{token}"


def _delete_cached_result(scope: str, user_id: int, token: str | None) -> None:
    if token:
        cache.delete(_result_cache_key(scope, user_id, token))


def _store_cached_result(scope: str, user_id: int, payload: dict) -> str:
    """Store result payload in cache and return opaque token."""
    serialized = json.dumps(payload, separators=(",", ":"))
    payload_size = len(serialized.encode("utf-8"))
    if payload_size > MAX_RESULT_CACHE_BYTES:
        raise ValueError("Result payload too large")

    token = uuid.uuid4().hex
    cache.set(
        _result_cache_key(scope, user_id, token), serialized, timeout=RESULT_CACHE_TTL_SECONDS
    )
    return token


def _load_cached_result(scope: str, user_id: int, token: str | None) -> dict | None:
    if not token:
        return None

    serialized = cache.get(_result_cache_key(scope, user_id, token))
    if serialized is None:
        return None

    try:
        payload = json.loads(serialized)
    except (TypeError, json.JSONDecodeError):
        _delete_cached_result(scope, user_id, token)
        return None

    if not isinstance(payload, dict):
        _delete_cached_result(scope, user_id, token)
        return None

    return payload


def _safe_download_filename(name: str | None, fallback: str) -> str:
    """Sanitize user-facing filenames used in Content-Disposition."""
    candidate = os.path.basename(name or "").strip()
    if not candidate:
        candidate = fallback
    candidate = re.sub(r'[\r\n"]+', "_", candidate)
    if not candidate.lower().endswith(".xlsx"):
        candidate = f"{candidate}.xlsx"
    return candidate[:150]


def _normalize_sheet_name(raw_name: str, used_names: set[str]) -> str:
    """Create Excel-safe unique sheet names (<=31 chars)."""
    sanitized = "".join("_" if char in r"[]:*?/\\" else char for char in (raw_name or "sheet"))
    sanitized = sanitized.strip() or "sheet"
    base = sanitized[:31]
    candidate = base
    suffix = 1
    while candidate.lower() in used_names:
        suffix_text = f"_{suffix}"
        candidate = f"{base[: 31 - len(suffix_text)]}{suffix_text}"
        suffix += 1
    used_names.add(candidate.lower())
    return candidate


def _is_valid_result_token(token: str | None) -> bool:
    return bool(token and re.fullmatch(r"[0-9a-f]{32}", token))


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
        request.session.pop(SIM_FORM_STATE_SESSION_KEY, None)
        request.session.pop(TS_FORM_STATE_SESSION_KEY, None)
        # Invalidate cached subcatchment IDs so they are re-read from new file
        request.session.pop("_subcatchment_ids", None)
        request.session.pop("_subcatchment_ids_file", None)
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

    request.session.pop(SIM_FORM_STATE_SESSION_KEY, None)
    request.session.pop(TS_FORM_STATE_SESSION_KEY, None)
    # Clear cached subcatchment IDs
    request.session.pop("_subcatchment_ids", None)
    request.session.pop("_subcatchment_ids_file", None)

    return JsonResponse({"message": "Upload cleared."})


def _get_subcatchment_ids(request: HttpRequest) -> list[str]:
    """
    Return subcatchment IDs from the uploaded INP file, using session cache.

    The result is cached in the session under ``_subcatchment_ids`` and
    ``_subcatchment_ids_file``.  The cache is invalidated when the uploaded
    file path changes.
    """
    file_path = request.session.get("uploaded_file_path")
    cached_file = request.session.get("_subcatchment_ids_file")
    if file_path and file_path == cached_file:
        cached_ids = request.session.get("_subcatchment_ids")
        if cached_ids is not None:
            return cached_ids

    if not file_path or not os.path.exists(file_path):
        request.session.pop("_subcatchment_ids", None)
        request.session.pop("_subcatchment_ids_file", None)
        return []

    try:
        model = swmmio.Model(file_path)
        ids = list(model.inp.subcatchments.index)
    except Exception:
        logger.warning("Failed to read subcatchments from %s", file_path, exc_info=True)
        ids = []

    request.session["_subcatchment_ids"] = ids
    request.session["_subcatchment_ids_file"] = file_path
    return ids


def _get_catchment_choices(request: HttpRequest) -> list[tuple[str, str]]:
    """
    Extract subcatchment IDs from the uploaded INP file stored in the session.

    Returns a list of (id, id) tuples suitable for a Select widget, or a
    placeholder when no file is available.
    """
    ids = _get_subcatchment_ids(request)
    if ids:
        return [("", "--- Select catchment ---")] + [(sid, sid) for sid in ids]
    return [("", "--- Upload a file first ---")]


@require_GET
@ajax_login_required
def subcatchments(request: HttpRequest) -> JsonResponse:
    """
    Return the list of subcatchment IDs from the uploaded INP file.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP request.

    Returns
    -------
    JsonResponse
        JSON with ``subcatchments`` list.
    """
    return JsonResponse({"subcatchments": _get_subcatchment_ids(request)})


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


def _excel_attachment_response(
    output_file_name: str, sheets: dict[str, pd.DataFrame]
) -> HttpResponse:
    """Return an in-memory Excel attachment from sheet->DataFrame mapping."""
    buffer = BytesIO()
    used_names: set[str] = set()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            normalized_sheet_name = _normalize_sheet_name(sheet_name, used_names)
            df.to_excel(writer, sheet_name=normalized_sheet_name, index=False)
    buffer.seek(0)
    safe_name = _safe_download_filename(output_file_name, "results.xlsx")
    return FileResponse(
        buffer,
        as_attachment=True,
        filename=safe_name,
        content_type=EXCEL_CONTENT_TYPE,
    )


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
    token = request.session.get(SIM_RESULT_TOKEN_SESSION_KEY)
    payload = _load_cached_result("sim", request.user.id, token)
    if not payload:
        if token:
            request.session.pop(SIM_RESULT_TOKEN_SESSION_KEY, None)
        return {
            "show_download_button": False,
            "chart_config": None,
            "results_columns": [],
            "results_data": [],
            "feature_name": "",
            "output_file_name": None,
            "download_token": None,
        }

    return {
        "show_download_button": True,
        "chart_config": payload.get("chart_config"),
        "results_columns": payload.get("results_columns", []),
        "results_data": payload.get("results_data", []),
        "feature_name": payload.get("feature_name", ""),
        "output_file_name": payload.get("output_file_name"),
        "download_token": token,
    }


def clear_session_variables(request: HttpRequest) -> None:
    """
    Clear the session variables.

    Parameters
    ----------
    request : HttpRequest
        The incoming HTTP request.
    """
    user = getattr(request, "user", None)
    user_id = user.id if getattr(user, "is_authenticated", False) else None
    if user_id is not None:
        _delete_cached_result("sim", user_id, request.session.get(SIM_RESULT_TOKEN_SESSION_KEY))
        _delete_cached_result("ts", user_id, request.session.get(TS_RESULT_TOKEN_SESSION_KEY))

    for variable in [
        "show_download_button",
        "chart_config",
        "results_columns",
        "results_data",
        "feature_name",
        "output_file_name",
        "ts_chart_config",
        "ts_time_to_peak",
        "ts_runoff_volume",
        "ts_show_results",
        "ts_output_file_name",
        SIM_RESULT_TOKEN_SESSION_KEY,
        TS_RESULT_TOKEN_SESSION_KEY,
        SIM_FORM_STATE_SESSION_KEY,
        TS_FORM_STATE_SESSION_KEY,
    ]:
        if variable in request.session:
            del request.session[variable]


def _save_form_state(
    request: HttpRequest,
    session_key: str,
    cleaned_data: dict,
    fields: tuple[str, ...],
) -> None:
    """
    Persist selected form values in session for subsequent GET requests.

    Values that are ``None`` are skipped so the form can fall back to field defaults.
    """
    state = {}
    for field_name in fields:
        value = cleaned_data.get(field_name)
        if value is not None:
            state[field_name] = value
    request.session[session_key] = state


def _get_form_initial(
    request: HttpRequest,
    session_key: str,
    catchment_choices: list[tuple[str, str]],
    form_class: type[forms.Form],
    fields: tuple[str, ...],
) -> dict:
    """
    Return sanitized initial values restored from session.

    Dynamic ``catchment_name`` is validated against current catchment choices.
    Other fields are validated/coerced using Django field ``to_python`` and
    static choice sets where applicable.
    """
    state = request.session.get(session_key)
    if not isinstance(state, dict):
        return {}

    initial = {}
    for field_name in fields:
        if field_name not in state:
            continue

        field = form_class.base_fields.get(field_name)
        if field is None:
            continue

        try:
            coerced_value = field.to_python(state[field_name])
        except (ValidationError, TypeError, ValueError):
            continue

        if coerced_value is None:
            continue

        if isinstance(field, forms.ChoiceField) and field_name != "catchment_name":
            if not field.valid_value(coerced_value):
                continue

        initial[field_name] = coerced_value

    valid_catchments = {value for value, _ in catchment_choices if value}
    catchment_name = initial.get("catchment_name")
    if catchment_name and catchment_name not in valid_catchments:
        initial["catchment_name"] = ""

    if initial != state:
        request.session[session_key] = initial

    return initial


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
    session_data = {}

    if request.method == "POST":
        catchment_choices = _get_catchment_choices(request)
        form = SimulationForm(request.POST, catchment_choices=catchment_choices)
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

                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file_name = f"{request.user.username}_simulation_result_{timestamp}.xlsx"
                chart_config = {
                    "data": json.loads(df.to_json(orient="records")),
                    "x": feature_name,
                    "y": other_cols,
                    "title": f"Dependence of runoff on subcatchment {feature_name}.",
                }
                payload = {
                    "chart_config": chart_config,
                    "results_columns": df.columns.tolist(),
                    "results_data": df.values.tolist(),
                    "feature_name": feature_name,
                    "output_file_name": output_file_name,
                }
                old_token = request.session.get(SIM_RESULT_TOKEN_SESSION_KEY)
                token = _store_cached_result("sim", request.user.id, payload)
                _delete_cached_result("sim", request.user.id, old_token)
                request.session[SIM_RESULT_TOKEN_SESSION_KEY] = token
                _save_form_state(
                    request, SIM_FORM_STATE_SESSION_KEY, form.cleaned_data, SIM_FORM_STATE_FIELDS
                )
                return redirect("main:simulation")

            except ValueError:
                messages.error(
                    request,
                    "Result set is too large to keep for download. Narrow the simulation range.",
                )
            except Exception as e:
                logger.error("Simulation failed: %s", e)
                messages.error(request, "An error occurred while running the simulation.")
    else:
        catchment_choices = _get_catchment_choices(request)
        initial = _get_form_initial(
            request,
            SIM_FORM_STATE_SESSION_KEY,
            catchment_choices,
            SimulationForm,
            SIM_FORM_STATE_FIELDS,
        )
        form = SimulationForm(catchment_choices=catchment_choices, initial=initial)
        session_data = get_session_variables(request)

    return render(
        request,
        "main/simulation.html",
        {"form": form, **session_data},
    )


@login_required
@require_POST
def download_simulation_results(request: HttpRequest) -> HttpResponse:
    """Download simulation results as an Excel file generated in memory."""
    token = request.POST.get("token")
    if not _is_valid_result_token(token):
        messages.error(request, "Invalid download token.")
        return redirect("main:simulation")
    payload = _load_cached_result("sim", request.user.id, token)
    if not payload:
        messages.error(request, "No simulation results available to download.")
        return redirect("main:simulation")

    try:
        df = pd.DataFrame(payload["results_data"], columns=payload["results_columns"])
        return _excel_attachment_response(payload.get("output_file_name"), {"results": df})
    except Exception:
        logger.exception("Failed to generate simulation download")
        messages.error(request, "Failed to generate the simulation output file.")
        return redirect("main:simulation")


@login_required
@require_POST
def download_timeseries_results(request: HttpRequest) -> HttpResponse:
    """Download timeseries analysis results as an in-memory Excel file."""
    token = request.POST.get("token")
    if not _is_valid_result_token(token):
        messages.error(request, "Invalid download token.")
        return redirect("main:timeseries")
    payload = _load_cached_result("ts", request.user.id, token)
    if not payload:
        messages.error(request, "No timeseries results available to download.")
        return redirect("main:timeseries")

    mode = payload.get("mode")
    data = payload.get("data")
    try:
        if mode == "single" and isinstance(data, list):
            df = pd.DataFrame(data)
            return _excel_attachment_response(payload.get("output_file_name"), {"timeseries": df})

        if mode == "sweep" and isinstance(data, dict):
            sheets = {}
            for param, rows in data.items():
                sheet_name = f"val_{param}"
                sheets[sheet_name] = pd.DataFrame(rows)
            return _excel_attachment_response(payload.get("output_file_name"), sheets)
    except Exception:
        logger.exception("Failed to generate timeseries download")

    messages.error(request, "Failed to generate the timeseries output file.")
    return redirect("main:timeseries")


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
        catchment_choices = _get_catchment_choices(request)
        form = TimeseriesForm(request.POST, catchment_choices=catchment_choices)
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
                        chart_config = {
                            "mode": "single",
                            "data": json.loads(ts_df_reset.to_json(orient="records")),
                            "columns": ts_columns,
                            "title": f"Timeseries for subcatchment {catchment_name}",
                        }
                        payload = {
                            "mode": "single",
                            "data": chart_config["data"],
                            "output_file_name": output_file_name,
                            "chart_config": chart_config,
                            "ts_time_to_peak": ttp_str,
                            "ts_runoff_volume": vol_str,
                            "ts_show_results": True,
                        }
                        old_token = request.session.get(TS_RESULT_TOKEN_SESSION_KEY)
                        token = _store_cached_result("ts", request.user.id, payload)
                        _delete_cached_result("ts", request.user.id, old_token)
                        request.session[TS_RESULT_TOKEN_SESSION_KEY] = token

                        _save_form_state(
                            request,
                            TS_FORM_STATE_SESSION_KEY,
                            form.cleaned_data,
                            TS_FORM_STATE_FIELDS,
                        )
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
                        chart_config = {
                            "mode": "sweep",
                            "data": sweep_data,
                            "columns": ts_columns,
                            "title": f"Timeseries sweep: {feature} for {catchment_name}",
                            "feature": feature,
                            "catchment": catchment_name,
                        }
                        payload = {
                            "mode": "sweep",
                            "data": sweep_data,
                            "output_file_name": output_file_name,
                            "chart_config": chart_config,
                            "ts_show_results": True,
                            "ts_time_to_peak": None,
                            "ts_runoff_volume": None,
                        }
                        old_token = request.session.get(TS_RESULT_TOKEN_SESSION_KEY)
                        token = _store_cached_result("ts", request.user.id, payload)
                        _delete_cached_result("ts", request.user.id, old_token)
                        request.session[TS_RESULT_TOKEN_SESSION_KEY] = token

                        _save_form_state(
                            request,
                            TS_FORM_STATE_SESSION_KEY,
                            form.cleaned_data,
                            TS_FORM_STATE_FIELDS,
                        )
                        return redirect("main:timeseries")

            except ValueError:
                messages.error(
                    request,
                    "Result set is too large to keep for download. Narrow the timeseries range.",
                )
            except Exception as e:
                logger.error("Timeseries analysis failed: %s", e)
                messages.error(request, "An error occurred while running the analysis.")
    else:
        catchment_choices = _get_catchment_choices(request)
        initial = _get_form_initial(
            request,
            TS_FORM_STATE_SESSION_KEY,
            catchment_choices,
            TimeseriesForm,
            TS_FORM_STATE_FIELDS,
        )
        form = TimeseriesForm(catchment_choices=catchment_choices, initial=initial)
        token = request.session.get(TS_RESULT_TOKEN_SESSION_KEY)
        payload = _load_cached_result("ts", request.user.id, token)
        if token and not payload:
            request.session.pop(TS_RESULT_TOKEN_SESSION_KEY, None)
        session_data = {
            "ts_chart_config": payload.get("chart_config") if payload else None,
            "ts_time_to_peak": payload.get("ts_time_to_peak") if payload else None,
            "ts_runoff_volume": payload.get("ts_runoff_volume") if payload else None,
            "ts_show_results": payload.get("ts_show_results", False) if payload else False,
            "output_file_name": payload.get("output_file_name") if payload else None,
            "download_token": token if payload else None,
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
                with Simulation(uploaded_file_path) as sim:
                    for _ in sim:
                        pass
                # Build the model after SWMM run so report-derived columns are available.
                swmmio_model = swmmio.Model(uploaded_file_path)
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

            except Exception:
                logger.exception("Error while performing calculations.")
                messages.error(
                    request,
                    "An error occurred while performing calculations.",
                )

    df_is_empty = df is None or df.empty

    return render(request, "main/calculations.html", {"df": df, "df_is_empty": df_is_empty})
