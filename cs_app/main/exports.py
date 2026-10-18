"""Spreadsheet and CSV export helpers for simulation and timeseries results."""

from __future__ import annotations

import os
import re
from io import BytesIO

import pandas as pd
from django.http import FileResponse, HttpResponse

EXCEL_CONTENT_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _safe_download_filename(name: str | None, fallback: str, extension: str = ".xlsx") -> str:
    """Sanitize user-facing filenames used in Content-Disposition."""
    candidate = os.path.basename(name or "").strip()
    if not candidate:
        candidate = fallback
    candidate = re.sub(r'[\r\n"]+', "_", candidate)
    base_name = os.path.splitext(candidate)[0] or os.path.splitext(fallback)[0] or "results"
    if not extension:
        normalized_extension = ""
    else:
        normalized_extension = extension if extension.startswith(".") else f".{extension}"
    safe_name = f"{base_name}{normalized_extension}"
    return safe_name[:150]


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


def _excel_attachment_response(
    output_file_name: str | None, sheets: dict[str, pd.DataFrame]
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


def _timeseries_payload_to_csv_df(payload: dict) -> pd.DataFrame:
    """Convert cached timeseries payload to a single CSV-friendly DataFrame."""
    mode = payload.get("mode")
    data = payload.get("data")

    if mode == "single" and isinstance(data, list):
        return pd.DataFrame(data)

    if mode == "sweep" and isinstance(data, dict):
        frames: list[pd.DataFrame] = []
        for parameter_value, rows in data.items():
            frame = pd.DataFrame(rows)
            frame.insert(0, "parameter_value", parameter_value)
            frames.append(frame)
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame(columns=["parameter_value"])

    raise ValueError("Invalid timeseries payload for CSV export.")
