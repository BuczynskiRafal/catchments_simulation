"""JSON REST API endpoints for programmatic access to catchment simulations and analysis."""

import json
import os
from functools import wraps
from typing import Any

from django import forms
from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from pydantic import BaseModel

from main.forms import SimulationForm, TimeseriesForm, parameter_labels
from main.schemas import SimulationRequest, TimeseriesRequest
from main.views import (
    AUTH_REQUIRED_MESSAGE,
    CALC_RESULT_TOKEN_SESSION_KEY,
    FORM_INVALID_MESSAGE,
    SIM_RESULT_TOKEN_SESSION_KEY,
    TS_RESULT_TOKEN_SESSION_KEY,
    RunFailure,
    _classify_run_error,
    _forget_model_state,
    _form_units,
    _get_catchment_choices,
    _get_subcatchment_ids,
    _is_in_user_upload_dir,
    _load_cached_result,
    _load_comparison,
    _run_comparison,
    _run_simulation,
    _run_timeseries,
    _session_flow_units,
    upload,
    upload_clear,
    upload_sample,
)

INVALID_JSON_MESSAGE = "Invalid JSON payload."
METHOD_NOT_ALLOWED_MESSAGE = "Method not allowed."
NO_SIMULATION_RESULTS_MESSAGE = "No simulation results available."
NO_TIMESERIES_RESULTS_MESSAGE = "No timeseries results available."
NO_CALCULATIONS_RESULTS_MESSAGE = "No calculation results available."
NUMERIC_FIELDS = ("start", "stop", "step")
STRING_FIELDS = ("option", "catchment_name", "mode", "feature")


def _api_error(
    message: str, status: int, field_errors: dict[str, list[str]] | None = None
) -> JsonResponse:
    """Return a uniform JSON error payload for API consumers."""
    return JsonResponse(
        {
            "error": message,
            "message": message,
            "field_errors": field_errors or {},
        },
        status=status,
    )


def _api_method_not_allowed(allowed: tuple[str, ...]) -> JsonResponse:
    response = _api_error(METHOD_NOT_ALLOWED_MESSAGE, 405)
    response["Allow"] = ", ".join(allowed)
    return response


def _api_form_invalid(form: forms.Form) -> JsonResponse:
    field_errors = {
        field: [error["message"] for error in errors]
        for field, errors in form.errors.get_json_data().items()
    }
    return _api_error(FORM_INVALID_MESSAGE, 400, field_errors)


def _api_run_failure(failure: RunFailure, form: forms.Form | None = None) -> JsonResponse:
    field = failure.field if form is not None and failure.field in form.fields else None
    field_errors = {field: [failure.message]} if field else None
    return _api_error(failure.message, failure.status, field_errors)


def api_login_required(view_func):
    """Require an authenticated session and always respond with 401 JSON when unauthenticated."""

    @wraps(view_func)
    def wrapper(request: HttpRequest, *args: Any, **kwargs: Any) -> HttpResponse:
        if not request.user.is_authenticated:
            return JsonResponse(
                {
                    "error": AUTH_REQUIRED_MESSAGE,
                    "message": AUTH_REQUIRED_MESSAGE,
                    "field_errors": {},
                    "login_url": settings.LOGIN_URL,
                },
                status=401,
            )
        return view_func(request, *args, **kwargs)

    return wrapper


def _parse_request_data(
    request: HttpRequest, schema_cls: type[BaseModel]
) -> tuple[dict[str, Any] | None, JsonResponse | None]:
    """Parse JSON or form-encoded request parameters and validate field types/names."""
    allowed_fields = set(schema_cls.model_fields)
    content_type = (request.content_type or "").lower()
    if content_type.startswith("application/json"):
        try:
            raw_body = request.body.decode(request.encoding or "utf-8")
            parsed = json.loads(raw_body) if raw_body.strip() else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None, _api_error(INVALID_JSON_MESSAGE, 400)
        if not isinstance(parsed, dict):
            return None, _api_error(INVALID_JSON_MESSAGE, 400)

        field_errors: dict[str, list[str]] = {}
        for field in parsed:
            if field not in allowed_fields:
                field_errors[field] = ["Unexpected field."]
        for field in NUMERIC_FIELDS:
            val = parsed.get(field)
            if isinstance(val, bool) or (
                val is not None and not isinstance(val, (int, float, str))
            ):
                field_errors[field] = ["Enter a number."]
        for field in STRING_FIELDS:
            val = parsed.get(field)
            if field in allowed_fields and val is not None and not isinstance(val, str):
                field_errors[field] = ["Enter a valid string."]
        if field_errors:
            return None, _api_error(FORM_INVALID_MESSAGE, 400, field_errors)
        return parsed, None

    form_data = request.POST.dict()
    extra_errors = {
        field: ["Unexpected field."]
        for field in form_data
        if field not in allowed_fields and field != "csrfmiddlewaretoken"
    }
    if extra_errors:
        return None, _api_error(FORM_INVALID_MESSAGE, 400, extra_errors)
    return form_data, None


def _session_model_metadata(request: HttpRequest) -> dict[str, Any]:
    """Return flow units, subcatchment IDs, and unit-aware option labels for the session."""
    return {
        "flow_units": _session_flow_units(request),
        "subcatchments": _get_subcatchment_ids(request),
        "option_labels": parameter_labels(_form_units(request)),
    }


def _load_session_result(
    request: HttpRequest, scope: str, session_key: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Load the session's cached result for *scope*, cleaning up any stale session token."""
    token = request.session.get(session_key)
    payload = _load_cached_result(scope, request.user.id, token)
    if not payload:
        if token:
            request.session.pop(session_key, None)
        return None, None
    return payload, token


def _serialize_simulation_payload(payload: dict[str, Any], token: str | None) -> dict[str, Any]:
    chart_config = payload.get("chart_config") or {}
    return {
        "feature_name": payload.get("feature_name", ""),
        "columns": payload.get("results_columns", []),
        "data": payload.get("results_data", []),
        "records": chart_config.get("data", []),
        "chart_config": chart_config,
        "output_file_name": payload.get("output_file_name"),
        "download_token": token,
    }


def _serialize_timeseries_payload(payload: dict[str, Any], token: str | None) -> dict[str, Any]:
    return {
        "mode": payload.get("mode"),
        "data": payload.get("data"),
        "metrics": payload.get("metrics"),
        "sweep_summary": payload.get("sweep_summary"),
        "units": payload.get("units", {}),
        "chart_config": payload.get("chart_config"),
        "output_file_name": payload.get("output_file_name"),
        "download_token": token,
    }


def _serialize_calculations_payload(
    comparison: dict[str, Any], token: str | None
) -> dict[str, Any]:
    return {
        "unit": comparison.get("unit"),
        "rows": comparison.get("rows", []),
        "metrics": comparison.get("metrics", {}),
        "download_token": token,
    }


@api_login_required
def api_status(request: HttpRequest) -> JsonResponse:
    """Return current session model metadata, flow units, subcatchments, and parameter labels."""
    if request.method != "GET":
        return _api_method_not_allowed(("GET",))

    file_path = request.session.get("uploaded_file_path")
    if file_path:
        if not _is_in_user_upload_dir(file_path, request.user.id):
            request.session.pop("uploaded_file_path", None)
            _forget_model_state(request)
        else:
            try:
                size = os.path.getsize(file_path)
            except OSError:
                request.session.pop("uploaded_file_path", None)
                _forget_model_state(request)
            else:
                return JsonResponse(
                    {
                        "has_file": True,
                        "filename": os.path.basename(file_path),
                        "size": size,
                        **_session_model_metadata(request),
                    }
                )

    return JsonResponse(
        {
            "has_file": False,
            "filename": None,
            "size": None,
            **_session_model_metadata(request),
        }
    )


@api_login_required
def api_subcatchments(request: HttpRequest) -> JsonResponse:
    """Return subcatchment identifiers, flow units, and unit-aware parameter labels."""
    if request.method != "GET":
        return _api_method_not_allowed(("GET",))

    return JsonResponse(_session_model_metadata(request))


@csrf_exempt
@api_login_required
def api_upload(request: HttpRequest) -> HttpResponse:
    """Upload a SWMM .inp model file via multipart/form-data and return JSON."""
    if request.method != "POST":
        return _api_method_not_allowed(("POST",))
    request.META["HTTP_X_REQUESTED_WITH"] = "XMLHttpRequest"
    request.__dict__.pop("headers", None)
    return upload(request)


@api_login_required
def api_upload_sample(request: HttpRequest) -> JsonResponse:
    """Load the bundled sample SWMM .inp model into the current session and return JSON."""
    if request.method != "POST":
        return _api_method_not_allowed(("POST",))
    return upload_sample(request)


@api_login_required
def api_upload_clear(request: HttpRequest) -> JsonResponse:
    """Clear the current session's uploaded SWMM model and cached results."""
    if request.method != "POST":
        return _api_method_not_allowed(("POST",))
    return upload_clear(request)


@api_login_required
def api_simulation(request: HttpRequest) -> JsonResponse:
    """GET the latest parameter-sweep result or POST parameters to run a new sweep."""
    if request.method == "GET":
        payload, token = _load_session_result(request, "sim", SIM_RESULT_TOKEN_SESSION_KEY)
        if not payload:
            return _api_error(NO_SIMULATION_RESULTS_MESSAGE, 404)
        return JsonResponse(_serialize_simulation_payload(payload, token))

    if request.method != "POST":
        return _api_method_not_allowed(("GET", "POST"))

    data, error_response = _parse_request_data(request, SimulationRequest)
    if error_response is not None:
        return error_response

    form = SimulationForm(
        data,
        catchment_choices=_get_catchment_choices(request),
        units=_form_units(request),
    )
    if not form.is_valid():
        return _api_form_invalid(form)

    is_predefined = form.cleaned_data["option"] in SimulationForm.PREDEFINED_METHODS
    SimulationRequest(
        option=form.cleaned_data["option"],
        catchment_name=form.cleaned_data["catchment_name"],
        start=None if is_predefined else form.cleaned_data.get("start"),
        stop=None if is_predefined else form.cleaned_data.get("stop"),
        step=None if is_predefined else form.cleaned_data.get("step"),
    )

    try:
        _run_simulation(request, form)
    except Exception as error:
        failure = _classify_run_error(
            error,
            "Simulation",
            "Result set is too large to keep for download. Narrow the simulation range.",
            "An error occurred while running the simulation.",
        )
        return _api_run_failure(failure, form)

    payload, token = _load_session_result(request, "sim", SIM_RESULT_TOKEN_SESSION_KEY)
    return JsonResponse(_serialize_simulation_payload(payload or {}, token))


@api_login_required
def api_timeseries(request: HttpRequest) -> JsonResponse:
    """GET the latest timeseries analysis result or POST parameters to run a new analysis."""
    if request.method == "GET":
        payload, token = _load_session_result(request, "ts", TS_RESULT_TOKEN_SESSION_KEY)
        if not payload:
            return _api_error(NO_TIMESERIES_RESULTS_MESSAGE, 404)
        return JsonResponse(_serialize_timeseries_payload(payload, token))

    if request.method != "POST":
        return _api_method_not_allowed(("GET", "POST"))

    data, error_response = _parse_request_data(request, TimeseriesRequest)
    if error_response is not None:
        return error_response
    if data is not None and "mode" not in data:
        data = {"mode": "single", **data}

    form = TimeseriesForm(
        data,
        catchment_choices=_get_catchment_choices(request),
        units=_form_units(request),
    )
    if not form.is_valid():
        return _api_form_invalid(form)

    is_sweep = form.cleaned_data["mode"] == "sweep"
    TimeseriesRequest(
        mode=form.cleaned_data["mode"],
        catchment_name=form.cleaned_data["catchment_name"],
        feature=form.cleaned_data.get("feature") if is_sweep else None,
        start=form.cleaned_data.get("start") if is_sweep else None,
        stop=form.cleaned_data.get("stop") if is_sweep else None,
        step=form.cleaned_data.get("step") if is_sweep else None,
    )

    try:
        _run_timeseries(request, form)
    except Exception as error:
        failure = _classify_run_error(
            error,
            "Timeseries analysis",
            "Result set is too large to keep for download. Narrow the timeseries range.",
            "An error occurred while running the analysis.",
        )
        return _api_run_failure(failure, form)

    payload, token = _load_session_result(request, "ts", TS_RESULT_TOKEN_SESSION_KEY)
    return JsonResponse(_serialize_timeseries_payload(payload or {}, token))


@api_login_required
def api_calculations(request: HttpRequest) -> JsonResponse:
    """GET the latest SWMM vs ANN comparison or POST to run a new comparison."""
    if request.method == "GET":
        comparison = _load_comparison(request)
        if not comparison:
            return _api_error(NO_CALCULATIONS_RESULTS_MESSAGE, 404)
        token = request.session.get(CALC_RESULT_TOKEN_SESSION_KEY)
        return JsonResponse(_serialize_calculations_payload(comparison, token))

    if request.method != "POST":
        return _api_method_not_allowed(("GET", "POST"))

    try:
        _run_comparison(request)
    except Exception as error:
        failure = _classify_run_error(
            error,
            "Calculations",
            "The comparison is too large to keep. Use a model with fewer subcatchments.",
            "An error occurred while performing calculations.",
        )
        return _api_run_failure(failure)

    comparison = _load_comparison(request) or {}
    token = request.session.get(CALC_RESULT_TOKEN_SESSION_KEY)
    return JsonResponse(_serialize_calculations_payload(comparison, token))
