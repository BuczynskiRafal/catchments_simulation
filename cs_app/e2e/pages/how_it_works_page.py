"""How it works page object: equations and parameter guide."""

from __future__ import annotations

from .base_page import BasePage


class HowItWorksPage(BasePage):
    """POM for ``/how-it-works``."""

    PATH = "/how-it-works"

    def navigate_to(self) -> None:
        self.navigate(self.PATH)

    # ------------------------------------------------------------------
    # Equations and guide
    # ------------------------------------------------------------------

    def rendered_equation_numbers(self) -> list[str]:
        """Numbers of the display equations KaTeX has typeset, in page order."""
        return self.page.locator(
            ".hiw-equation:has(.hiw-equation__body.is-rendered .katex-display)"
        ).evaluate_all(
            "els => els.map(el => el.querySelector('.hiw-equation__number').textContent)"
        )
