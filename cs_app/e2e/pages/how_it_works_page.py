"""How it works page object: equations, parameter guide and the interactive model (#lab)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base_page import BasePage

if TYPE_CHECKING:
    from playwright.sync_api import Locator


class HowItWorksPage(BasePage):
    """POM for ``/how-it-works``."""

    PATH = "/how-it-works"

    def navigate_to(self, query: str = "") -> None:
        self.navigate(self.PATH + query)
        self.wait_for_lab()

    def wait_for_lab(self) -> None:
        """Wait until the script has set up the interactive model and drawn its first run."""
        self.page.locator("#hiw-lab:not([hidden]) .hiw-chart__runoff[d^='M']").wait_for(
            state="attached"
        )

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

    def try_button(self, feature: str) -> Locator:
        return self.page.locator(f"[data-hiw-try='{feature}']")

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def slider(self, key: str) -> Locator:
        return self.page.locator(f"#lab-{key}")

    def slider_readout(self, key: str) -> Locator:
        return self.page.locator(f"#lab-{key}-value")

    def slider_max(self, key: str) -> Locator:
        return self.page.locator(f"[data-hiw-param='{key}'] .hiw-slider__bound--max")

    def nudge(self, key: str, presses: int, key_name: str = "ArrowRight") -> None:
        """Move a slider from the keyboard."""
        self.slider(key).focus()
        for _ in range(presses):
            self.page.keyboard.press(key_name)

    def choose_storm(self, storm: str) -> None:
        self.page.locator(f"label[for='lab-storm-{storm}']").click()

    @property
    def reset_button(self) -> Locator:
        return self.page.locator("#lab-reset")

    # ------------------------------------------------------------------
    # Stage
    # ------------------------------------------------------------------

    @property
    def play_button(self) -> Locator:
        return self.page.locator("#lab-play")

    @property
    def scrubber(self) -> Locator:
        return self.page.locator("#lab-time")

    @property
    def time_value(self) -> Locator:
        return self.page.locator("#lab-time-value")

    @property
    def hydrograph(self) -> Locator:
        return self.page.locator("#lab-hydrograph")

    def path(self, css_class: str) -> str:
        """Path data of one element of the hydrograph ("" when it draws nothing)."""
        return self.hydrograph.locator(f".{css_class}").get_attribute("d") or ""

    def figure(self, css_class: str) -> Locator:
        """One element of the hydrograph, by its class."""
        return self.hydrograph.locator(f".{css_class}")

    def drawn_sweep_lines(self) -> Locator:
        return self.hydrograph.locator(".hiw-chart__sweep-line[d^='M']")

    def response_points(self) -> Locator:
        return self.page.locator("#lab-response .hiw-chart__point")

    @property
    def controls_column(self) -> Locator:
        """The scrolling, sticky contents of the controls column."""
        return self.page.locator(".hiw-lab__controls-inner")

    @property
    def hydrograph_card(self) -> Locator:
        return self.page.locator(".hiw-hydrograph-card")

    @property
    def dock(self) -> Locator:
        return self.page.locator("#lab-dock")

    def dock_value(self, key: str) -> str:
        return self.dock.locator(f"[data-hiw-dock='{key}']").inner_text()

    def scroll_into_view(self, locator: Locator, block: str = "center") -> None:
        """Scroll without smooth scrolling, so the next measurement sees the final position."""
        locator.evaluate(f"el => el.scrollIntoView({{ block: {block!r}, behavior: 'instant' }})")

    def settle_scroll(self) -> None:
        """Wait until a (smooth) scroll has come to rest: the position holds for a few frames."""
        self.page.evaluate(
            """() => new Promise((resolve) => {
                let last = -1, still = 0;
                (function check() {
                    still = window.scrollY === last ? still + 1 : 0;
                    last = window.scrollY;
                    if (still >= 5) { resolve(); } else { requestAnimationFrame(check); }
                })();
            })"""
        )

    def in_viewport(self, locator: Locator) -> bool:
        """Whether the element lies wholly between the site header and the bottom of the viewport."""
        return locator.evaluate(
            """el => {
                const box = el.getBoundingClientRect();
                const header = document.querySelector('.cs-header').getBoundingClientRect().bottom;
                return box.top >= header - 1 && box.bottom <= window.innerHeight + 1;
            }"""
        )

    def metric_value(self, name: str) -> str:
        return self.page.locator(f"[data-hiw-metric='{name}'] [data-hiw-value]").inner_text()

    def metric_delta(self, name: str) -> Locator:
        return self.page.locator(f"[data-hiw-metric='{name}'] [data-hiw-delta]")

    # ------------------------------------------------------------------
    # Sweep
    # ------------------------------------------------------------------

    @property
    def sweep_parameter(self) -> Locator:
        return self.page.locator("#lab-sweep-parameter")

    @property
    def sweep_button(self) -> Locator:
        return self.page.locator("#lab-sweep-run")

    @property
    def sweep_status(self) -> Locator:
        return self.page.locator("#lab-sweep-status")

    def solve(self, overrides: str = "{}", storm: str = "example", time_step: int = 60) -> dict:
        """Run CS.runoffModel in the page: DEFAULTS overridden by a JS object literal."""
        return self.page.evaluate(
            f"""() => {{
                const model = CS.runoffModel;
                const storm = model.STORMS[{storm!r}];
                const result = model.simulate(Object.assign({{}}, model.DEFAULTS, {overrides},
                    {{ rainfall: storm.rainfall, duration: storm.duration, timeStep: {time_step} }}));
                return {{ peak: result.peak, totals: result.totals, continuity: result.continuity }};
            }}"""
        )
