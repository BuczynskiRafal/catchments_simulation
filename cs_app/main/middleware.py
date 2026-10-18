"""Middleware enforcing request body size limits before CSRF multipart parsing."""

from django.conf import settings
from django.http import HttpRequest, HttpResponse, JsonResponse


class RequestBodySizeLimitMiddleware:
    """Reject oversized non-upload request bodies before CsrfViewMiddleware parses them."""

    def __init__(self, get_response):
        self.get_response = get_response

    @staticmethod
    def _too_large_response(request: HttpRequest) -> HttpResponse:
        message = "Request body too large."
        if (
            request.headers.get("X-Requested-With") == "XMLHttpRequest"
            or request.path == "/api"
            or request.path.startswith("/api/")
        ):
            return JsonResponse(
                {"error": message, "message": message, "field_errors": {}},
                status=413,
            )
        return HttpResponse(message, status=413, content_type="text/plain; charset=utf-8")

    def __call__(self, request: HttpRequest) -> HttpResponse:
        # The dedicated /upload/ and /api/upload/ views enforce their own INP_UPLOAD_MAX_BODY_BYTES
        # limit after authentication checks; all other endpoints are capped at DATA_UPLOAD_MAX_MEMORY_SIZE.
        is_inp_upload = request.path.rstrip("/") in ("/upload", "/api/upload")
        if request.method in ("POST", "PUT", "PATCH") and not is_inp_upload:
            max_bytes = getattr(settings, "DATA_UPLOAD_MAX_MEMORY_SIZE", None)
            if max_bytes is not None:
                raw_length = request.META.get("CONTENT_LENGTH")
                try:
                    content_length = int(raw_length or 0)
                except (TypeError, ValueError):
                    content_length = 0
                if content_length > max_bytes:
                    return self._too_large_response(request)

                from main import views

                request.upload_handlers.insert(
                    0, views.BodySizeLimitUploadHandler(request, max_bytes)
                )

        response = self.get_response(request)
        if getattr(request, "_upload_body_too_large", False) and not is_inp_upload:
            return self._too_large_response(request)
        return response
