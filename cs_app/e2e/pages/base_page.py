"""Base page object that all page objects inherit from.

Provides common helpers: navigation, title checks, Bootstrap alert reading,
NavComponent composition, and axe-core accessibility auditing from a locally
vendored axe.min.js.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from playwright.sync_api import Page

from .nav_component import NavComponent

AXE_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "..", "vendor", "axe.min.js")


@dataclass
class AxeViolation:
    """Single axe-core violation entry."""

    id: str
    impact: str
    description: str
    help_url: str
    nodes_count: int


@dataclass
class AxeReport:
    """Aggregated axe-core audit report."""

    violations: list[AxeViolation] = field(default_factory=list)

    @property
    def critical(self) -> list[AxeViolation]:
        return [v for v in self.violations if v.impact == "critical"]

    @property
    def serious(self) -> list[AxeViolation]:
        return [v for v in self.violations if v.impact == "serious"]

    @property
    def moderate(self) -> list[AxeViolation]:
        return [v for v in self.violations if v.impact == "moderate"]

    @property
    def minor(self) -> list[AxeViolation]:
        return [v for v in self.violations if v.impact == "minor"]


class BasePage:
    """Base page object providing shared helpers for all pages."""

    def __init__(self, page: Page, base_url: str) -> None:
        self.page = page
        self.base_url = base_url.rstrip("/")
        self.nav = NavComponent(page)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def navigate(self, path: str = "/") -> None:
        """Go to *path* relative to the live server base URL."""
        url = f"{self.base_url}{path}"
        self.page.goto(url, wait_until="domcontentloaded")

    def get_current_url(self) -> str:
        return self.page.url

    def get_title(self) -> str:
        return self.page.title()

    # ------------------------------------------------------------------
    # Common elements
    # ------------------------------------------------------------------

    def wait_for_content(self) -> None:
        """Wait for the main content wrapper to be visible."""
        self.page.locator(".content-wrapper").wait_for(state="visible")

    def get_alerts(self) -> list[str]:
        """Return text of all Bootstrap alert messages currently visible.

        Uses ``all_inner_texts()`` to avoid race conditions between
        ``.count()`` and ``nth().inner_text()`` calls.
        """
        return self.page.locator(".alert").all_inner_texts()

    def get_heading(self, level: int = 2) -> str | None:
        """Return the text of the first heading at *level*."""
        loc = self.page.locator(f"h{level}").first
        if loc.is_visible():
            return loc.inner_text()
        return None

    # ------------------------------------------------------------------
    # Accessibility
    # ------------------------------------------------------------------

    def run_axe_audit(self) -> AxeReport:
        """Inject vendored axe-core and run a full-page audit.

        Guards against double-injection — if axe is already loaded on the
        page the script tag is not added again.
        """
        if not os.path.isfile(AXE_SCRIPT_PATH):
            raise FileNotFoundError(
                f"axe-core script not found at {AXE_SCRIPT_PATH}. "
                "Run: curl -sL https://cdn.jsdelivr.net/npm/axe-core@4.10.2/axe.min.js "
                "-o cs_app/e2e/vendor/axe.min.js"
            )

        # Only inject if not already present (guard against double injection)
        already_loaded = self.page.evaluate("typeof window.axe !== 'undefined'")
        if not already_loaded:
            self.page.add_script_tag(path=AXE_SCRIPT_PATH)
            self.page.wait_for_function("typeof window.axe !== 'undefined'")

        raw = self.page.evaluate("() => axe.run()")
        violations: list[AxeViolation] = []
        for v in raw.get("violations", []):
            violations.append(
                AxeViolation(
                    id=v["id"],
                    impact=v.get("impact", "unknown"),
                    description=v.get("description", ""),
                    help_url=v.get("helpUrl", ""),
                    nodes_count=len(v.get("nodes", [])),
                )
            )
        return AxeReport(violations=violations)

    def assert_no_critical_a11y_violations(self) -> AxeReport:
        """Run axe audit and fail if critical issues exist."""
        report = self.run_axe_audit()
        if report.critical:
            details = "\n".join(
                f"  [{v.impact}] {v.id}: {v.description} ({v.nodes_count} nodes)"
                for v in report.critical
            )
            raise AssertionError(f"Critical a11y violations on {self.page.url}:\n{details}")
        return report
