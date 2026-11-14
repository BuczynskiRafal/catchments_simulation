"""Calculations (SWMM vs ANN comparison) page tests."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest
import swmmio
from django.conf import settings
from main.predictor import predict_runoff, prediction_inputs
from playwright.sync_api import Page, expect

from .pages.calculations_page import CalculationsPage

pytestmark = pytest.mark.e2e

EXAMPLE_MODEL = Path(settings.BASE_DIR) / "data" / "example.inp"
# S2-S5 next to S1 of the example model: %Imperv, Width, %Slope (S3's width of 1,500 m
# and S5's slope of 130 % lie outside the training range of the network).
EXTRA_SUBCATCHMENTS = {
    "S2": (60, 300, 5),
    "S3": (90, 1500, 2),
    "S4": (10, 50, 40),
    "S5": (40, 800, 130),
}
EDGES = 8 * 8 + 8 * 8 + 8
NEURONS = 8 + 8 + 8 + 1

RESULT_COLUMNS = [
    "Name",
    "SWMM Runoff [m³]",
    "ANN Runoff [m³]",
    "Difference [m³]",
    "Difference [%]",
]


def _open(page: Page, live_server) -> CalculationsPage:
    cp = CalculationsPage(page, live_server.url)
    cp.navigate_to()
    return cp


class TestCalculationsPageRendering:
    """Calculations page renders its content sections."""

    def test_page_heading_and_title(self, page: Page, live_server) -> None:
        cp = _open(page, live_server)
        assert cp.get_page_heading() == "SWMM vs neural network runoff"
        assert "Calculations" in cp.get_title()

    def test_nn_heading_visible(self, page: Page, live_server) -> None:
        cp = _open(page, live_server)
        heading = cp.get_nn_heading()
        assert heading is not None
        assert "Neural Network" in heading

    def test_empty_state_heading_before_first_run(self, page: Page, live_server) -> None:
        cp = _open(page, live_server)
        heading = cp.empty_state.get_by_role("heading", level=2)
        expect(heading).to_have_text("No comparison yet")
        # The results heading (the focus target after a run) comes with the results.
        expect(cp.results_heading).to_have_count(0)

    def test_empty_state_before_first_run(self, page: Page, live_server) -> None:
        cp = _open(page, live_server)
        expect(cp.empty_state).to_be_visible()
        expect(cp.empty_state).to_contain_text("No comparison yet")
        expect(cp.results_table).to_have_count(0)

    def test_validity_note_visible(self, page: Page, live_server) -> None:
        cp = _open(page, live_server)
        expect(cp.validity_note).to_be_visible()
        expect(cp.validity_note).to_contain_text("limited precipitation range")

    def test_validity_note_points_to_the_diagram(self, page: Page, live_server) -> None:
        cp = _open(page, live_server)
        link = cp.validity_note.get_by_role("link", name="The network diagram")

        link.click()

        expect(page).to_have_url(re.compile(r"#calc-network-title$"))

    def test_model_details_expand_with_keyboard(self, page: Page, live_server) -> None:
        cp = _open(page, live_server)
        features = cp.model_details("Features")
        expect(features.locator("dl")).to_be_hidden()

        features.locator("summary").focus()
        page.keyboard.press("Enter")

        expect(features).to_have_attribute("open", "")
        expect(features.locator("dt")).to_have_count(8)
        expect(cp.model_details("Architecture")).to_have_count(0)

    def test_upload_zone_present(self, page: Page, live_server) -> None:
        cp = _open(page, live_server)
        assert cp.upload.is_visible()
        expect(page.locator(".upload-zone-wrapper")).to_have_class(
            re.compile("upload-zone--compact")
        )

    def test_run_calculations_button_present(self, page: Page, live_server) -> None:
        cp = _open(page, live_server)
        expect(cp.run_button).to_be_visible()
        expect(cp.run_button).to_have_text("Run Calculations")
        expect(cp.loading_state).to_be_hidden()

    def test_anonymous_user_can_access_page(self, page: Page, live_server) -> None:
        """Calculations page does NOT require @login_required."""
        cp = _open(page, live_server)
        assert "/calculations" in page.url
        expect(cp.empty_state).to_be_visible()

    def test_anonymous_run_redirects_to_login(self, page: Page, live_server) -> None:
        cp = _open(page, live_server)
        cp.run_calculations()
        page.wait_for_url("**/login/**", timeout=10_000)
        assert "next=%2Fcalculations" in page.url

    def test_no_horizontal_scroll_on_phone(self, page: Page, live_server) -> None:
        page.set_viewport_size({"width": 360, "height": 780})
        _open(page, live_server)
        overflow = page.evaluate(
            "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
        )
        assert overflow <= 0


class TestCalculationsAsyncErrors:
    """Server errors are shown in the control panel without a page reload."""

    def test_run_without_model_shows_inline_error(self, auth_page: Page, live_server) -> None:
        cp = _open(auth_page, live_server)
        auth_page.evaluate("window.__csNoReload = true")

        cp.run_calculations()

        expect(cp.form_error).to_have_text("Please upload a file first.")
        expect(cp.form_error).to_have_attribute("role", "alert")
        expect(cp.run_button).to_be_enabled()
        expect(cp.run_button).to_be_focused()
        expect(cp.loading_state).to_be_hidden()
        expect(cp.empty_state).to_be_visible()
        expect(auth_page.locator(".cs-toast--danger")).to_be_visible()
        assert auth_page.evaluate("window.__csNoReload") is True

    def test_error_clears_on_next_run(self, auth_page: Page, live_server) -> None:
        cp = _open(auth_page, live_server)
        cp.run_calculations()
        expect(cp.form_error).to_have_count(1)

        cp.run_calculations()

        # A repeated failure replaces the alert instead of stacking a second one.
        expect(cp.form_error).to_have_count(1)
        expect(cp.run_button).to_be_enabled()


@pytest.mark.slow
class TestCalculationsExecution:
    """Running calculations (requires sample data + SWMM + ANN model)."""

    @pytest.fixture
    def results_page(self, auth_page: Page, live_server) -> CalculationsPage:
        cp = _open(auth_page, live_server)
        cp.upload.click_sample_data()
        auth_page.evaluate("window.__csNoReload = true")
        cp.run_and_wait_for_results()
        return cp

    def test_run_swaps_results_without_reload(self, results_page: CalculationsPage) -> None:
        page = results_page.page
        assert page.evaluate("window.__csNoReload") is True
        expect(results_page.results_heading).to_be_focused()
        expect(results_page.run_button).to_be_enabled()
        expect(results_page.loading_state).to_be_hidden()
        expect(results_page.empty_state).to_have_count(0)
        expect(page.locator("#calculations-results")).to_have_count(1)
        expect(page.locator(".cs-toast--success")).to_contain_text("Comparison finished")

    def test_results_table_columns_and_rows(self, results_page: CalculationsPage) -> None:
        assert results_page.get_results_columns() == RESULT_COLUMNS
        assert results_page.get_results_row_count() > 0
        assert results_page.column_values(0)[0] == "S1"
        # SWMM reports 3.36 x 10^6 L for S1 of the sample model: shown in m³, as on the other tools.
        assert results_page.column_values(1)[0] == "3360.00"
        for value in results_page.column_values(1) + results_page.column_values(2):
            assert re.fullmatch(r"-?[\d,]+\.\d{2}", value)

    def test_metric_tiles(self, results_page: CalculationsPage) -> None:
        assert re.fullmatch(r"[\d,]+\.\d{2}m³", results_page.metric_value("Mean absolute error"))
        assert re.fullmatch(r"[\d,]+\.\d{2}m³", results_page.metric_value("Root mean square error"))
        assert results_page.metric_value("Subcatchments") == str(
            results_page.get_results_row_count()
        )

    def test_charts_render_and_follow_theme(self, results_page: CalculationsPage) -> None:
        page = results_page.page
        for chart in (results_page.parity_chart, results_page.bars_chart):
            expect(chart).to_have_attribute("role", "img")
            expect(chart.locator(".main-svg").first).to_be_attached()
        marker_color = (
            "() => document.getElementById('calculations-parity-chart').data[1].marker.color"
        )
        light_color = page.evaluate(marker_color)

        results_page.nav.set_theme("dark")

        page.wait_for_function(f"({marker_color})() !== {light_color!r}")
        expect(results_page.parity_chart.locator(".main-svg").first).to_be_attached()
        expect(results_page.bars_chart.locator(".main-svg").first).to_be_attached()

    def test_table_sorts_by_difference(self, results_page: CalculationsPage) -> None:
        header = results_page.results_table.locator("th", has_text="Difference [m³]")

        results_page.sort_by("Difference [m³]")
        expect(header).to_have_attribute("aria-sort", "ascending")
        ascending = [float(v.replace(",", "")) for v in results_page.column_values(3)]
        assert ascending == sorted(ascending)

        results_page.sort_by("Difference [m³]")
        expect(header).to_have_attribute("aria-sort", "descending")
        descending = [float(v.replace(",", "")) for v in results_page.column_values(3)]
        assert descending == sorted(ascending, reverse=True)

    def test_results_fit_phone_width(self, results_page: CalculationsPage) -> None:
        page = results_page.page
        page.set_viewport_size({"width": 360, "height": 780})
        # Plotly resizes the charts on a debounced window resize.
        page.wait_for_function(
            "() => document.documentElement.scrollWidth <= document.documentElement.clientWidth",
            timeout=5_000,
        )
        # The wide table scrolls inside its keyboard-focusable region instead.
        expect(results_page.page.locator(".cs-table-wrap")).to_have_attribute("tabindex", "0")

    def test_rerun_replaces_results_in_place(self, results_page: CalculationsPage) -> None:
        results_page.page.evaluate(
            "document.getElementById('calculations-table').dataset.firstRun = '1'"
        )
        results_page.run_calculations()
        expect(results_page.results_table).not_to_have_attribute(
            "data-first-run", "1", timeout=60_000
        )
        expect(results_page.results_table).to_have_count(1)
        expect(results_page.results_heading).to_be_focused()


def _five_subcatchment_model(directory: Path) -> Path:
    """The example model with S2-S5 added next to S1 (same rain gage, outlet and soil)."""
    text = EXAMPLE_MODEL.read_text()
    rows = {"[SUBCATCHMENTS]": [], "[SUBAREAS]": [], "[INFILTRATION]": []}
    for name, (imperv, width, slope) in EXTRA_SUBCATCHMENTS.items():
        rows["[SUBCATCHMENTS]"].append(f"{name} 1 J1 5 {imperv} {width} {slope} 0")
        rows["[SUBAREAS]"].append(f"{name} 0.01 1 0.05 0.05 25 OUTLET")
        rows["[INFILTRATION]"].append(f"{name} 3.5 0.5 0.25")
    lines = text.splitlines()
    for section, added in rows.items():
        # Insert after the section's S1 row.
        start = lines.index(section)
        s1 = next(i for i in range(start, len(lines)) if lines[i].startswith("S1 "))
        lines[s1 + 1 : s1 + 1] = added
    path = directory / "five_subcatchments.inp"
    path.write_text("\n".join(lines) + "\n")
    return path


class TestNetworkDiagram:
    """ "Inside the network" before any run: the weights, and the keyboard readout."""

    def test_diagram_shows_the_weights(self, page: Page, live_server) -> None:
        cp = _open(page, live_server)

        expect(cp.network_diagram).to_have_attribute("role", "img")
        expect(cp.network_diagram).to_have_attribute(
            "aria-label", re.compile(r"^Diagram of the neural network: 8 inputs")
        )
        expect(cp.network_diagram.locator(".calc-net__edge")).to_have_count(EDGES)
        expect(cp.network_diagram.locator(".calc-net__node")).to_have_count(NEURONS)
        # No trace: no neuron is filled by an activation.
        expect(cp.network_diagram.locator(".calc-net__node.is-on")).to_have_count(0)
        expect(cp.network_caption).to_have_text(
            "Line weight shows the strength of each connection; teal adds to the next neuron, "
            "ochre subtracts."
        )
        expect(cp.trace_select).to_be_hidden()
        expect(cp.network_output).to_be_hidden()
        expect(cp.range_items).to_have_count(8)
        # The bars stop at what an input can physically take: a share at 100 %, a slope
        # (rise over run) at two standard deviations above its mean.
        ends = cp.page.locator("#calc-range-list .calc-range__ends > span:last-child")
        expect(ends.nth(0)).to_have_text(re.compile(r"\b100 %$"))
        expect(ends.nth(7)).to_have_text(re.compile(r"\b100 %$"))
        expect(ends.nth(2)).to_have_text(re.compile(r"112\.3 %$"))
        expect(cp.range_summary).to_have_text(
            "Run the comparison to check each subcatchment's inputs against these ranges."
        )
        expect(page.get_by_role("link", name="The equations behind the network")).to_have_attribute(
            "href", "/how-it-works#network"
        )

    def test_arrow_keys_read_each_neuron(self, page: Page, live_server) -> None:
        cp = _open(page, live_server)

        cp.network_stage.focus()
        expect(cp.network_readout).to_contain_text("Imperviousness: standardised with")
        page.keyboard.press("ArrowRight")
        expect(cp.network_readout).to_have_text(re.compile(r"^Layer 1, neuron 1: bias"))
        page.keyboard.press("End")
        expect(cp.network_readout).to_have_text(re.compile(r"^Layer 1, neuron 8: "))
        page.keyboard.press("ArrowRight")
        page.keyboard.press("ArrowRight")
        expect(cp.network_readout).to_have_text(re.compile(r"^Output neuron: bias"))
        expect(cp.network_diagram.locator(".calc-net__node.is-current")).to_have_count(1)

    def test_forward_pass_matches_the_server(self, page: Page, live_server) -> None:
        """network.js in the browser reproduces main.predictor on the sample model."""
        cp = _open(page, live_server)
        model = swmmio.Model(str(EXAMPLE_MODEL))
        inputs = prediction_inputs(model)[0].astype(float).tolist()
        expected = float(predict_runoff(model)[0])

        result = cp.forward(inputs)

        assert [len(layer) for layer in result["layers"]] == [8, 8, 1]
        assert result["output"] == result["layers"][2][0]
        assert abs(result["output"] - expected) / expected < 1e-4

    def test_diagram_fits_a_phone(self, page: Page, live_server) -> None:
        page.set_viewport_size({"width": 390, "height": 844})
        cp = _open(page, live_server)

        box = cp.network_diagram.bounding_box()
        stage = cp.network_stage.bounding_box()
        assert box["width"] <= stage["width"] + 1
        # Short labels at their own size, not a scaled-down desktop drawing.
        expect(cp.network_diagram.locator("text", has_text="Storage, imperv.")).to_have_count(1)
        assert page.evaluate(
            "document.documentElement.scrollWidth <= document.documentElement.clientWidth"
        )


class TestNetworkStates:
    """What reaches a screen reader, and the diagram without weights or without inputs."""

    def test_only_the_keyboard_announces_neurons(self, page: Page, live_server) -> None:
        cp = _open(page, live_server)
        announce = page.locator("#calc-network-announce")
        expect(announce).to_have_attribute("aria-live", "polite")
        expect(cp.network_readout).not_to_have_attribute("aria-live", re.compile(".*"))

        # Pointing updates the visible readout only.
        cp.network_diagram.locator('[data-node="1:0"] .calc-net__dot').hover()
        expect(cp.network_readout).to_have_text(re.compile(r"^Layer 1, neuron 1: "))
        cp.network_diagram.locator('[data-node="2:3"] .calc-net__dot').hover()
        expect(cp.network_readout).to_have_text(re.compile(r"^Layer 2, neuron 4: "))
        page.mouse.move(0, 0)
        expect(cp.network_readout).to_have_text(re.compile(r"^Point at a neuron"))
        expect(announce).to_have_text("")

        # The arrow keys announce; leaving the diagram does not announce the reset.
        cp.network_stage.focus()
        page.keyboard.press("ArrowDown")
        expect(announce).to_have_text(re.compile(r"^Width: standardised with"))
        page.keyboard.press("Escape")
        expect(cp.network_readout).to_have_text(re.compile(r"^Point at a neuron"))
        expect(announce).to_have_text(re.compile(r"^Width: "))

    def test_page_without_the_weights_shows_a_note(
        self, auth_page: Page, live_server, monkeypatch
    ) -> None:
        page = auth_page  # signed in: an anonymous page logs the upload zone's 401

        def unavailable():
            raise FileNotFoundError("weights.npz")

        monkeypatch.setattr("main.views.network_payload", unavailable)
        errors: list[str] = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("console", lambda m: m.type == "error" and errors.append(m.text))
        cp = _open(page, live_server)

        expect(page.locator("#calc-network-unavailable")).to_have_text(
            "The network diagram is not available right now."
        )
        expect(cp.network_stage).to_have_count(0)
        expect(page.locator(".calc-net__intro")).to_be_visible()
        expect(page.get_by_role("link", name="The equations behind the network")).to_be_visible()
        assert errors == []

    def test_rows_without_inputs_show_the_weights_only(
        self, auth_page: Page, live_server, monkeypatch
    ) -> None:
        """A comparison too large to keep the inputs cannot be traced: the weights stay."""
        rows = pd.DataFrame(
            {"Name": ["S1", "S2"], "SWMM_Runoff": [3360.0, 100.0], "ANN_Runoff": [2465.94, 120.0]}
        )
        monkeypatch.setattr("main.views._compare_swmm_and_ann", lambda _path: (rows, "m³"))
        cp = _open(auth_page, live_server)
        cp.upload.click_sample_data()

        cp.run_and_wait_for_results()

        expect(auth_page.locator("#calc-network-notice")).to_be_visible()
        expect(auth_page.locator("#calc-network-notice")).to_have_text(
            "This comparison has too many subcatchments to trace individually."
        )
        expect(cp.trace_select).to_be_hidden()
        expect(cp.results_table.get_by_role("button", name=re.compile("^Trace"))).to_have_count(0)
        expect(cp.network_caption).to_have_text(
            "Line weight shows the strength of each connection; teal adds to the next neuron, "
            "ochre subtracts."
        )
        expect(cp.network_diagram.locator(".calc-net__node.is-on")).to_have_count(0)
        expect(cp.network_output).to_be_hidden()

    def test_the_notice_is_hidden_before_a_run(self, page: Page, live_server) -> None:
        _open(page, live_server)
        expect(page.locator("#calc-network-notice")).to_be_hidden()


@pytest.mark.slow
class TestNetworkTrace:
    """Tracing subcatchments through the network after a run."""

    @pytest.fixture
    def console_errors(self, auth_page: Page) -> list[str]:
        errors: list[str] = []
        auth_page.on("console", lambda m: m.type == "error" and errors.append(m.text))
        auth_page.on("pageerror", lambda error: errors.append(str(error)))
        return errors

    @pytest.fixture
    def sample_page(self, auth_page: Page, live_server, console_errors) -> CalculationsPage:
        cp = _open(auth_page, live_server)
        cp.upload.click_sample_data()
        cp.run_and_wait_for_results()
        return cp

    @pytest.fixture
    def five_page(self, auth_page: Page, live_server, console_errors, tmp_path) -> CalculationsPage:
        cp = _open(auth_page, live_server)
        cp.upload.upload_file(str(_five_subcatchment_model(tmp_path)))
        cp.run_calculations()
        expect(cp.results_table.locator("tbody tr")).to_have_count(5, timeout=cp.RUN_TIMEOUT_MS)
        return cp

    def test_sample_run_traces_its_subcatchment(
        self, sample_page: CalculationsPage, console_errors: list[str]
    ) -> None:
        cp = sample_page
        expect(cp.trace_select).to_be_visible()
        assert cp.trace_select.locator("option").all_inner_texts() == ["S1"]
        expect(cp.network_diagram).to_have_attribute(
            "aria-label", re.compile(r"^Neural network traced for subcatchment S1: ")
        )
        expect(cp.network_diagram.locator(".calc-net__node.is-on").first).to_be_attached()
        # The output readout is the table's ANN value, rounded the same way.
        table_ann = float(cp.column_values(2)[0].replace(",", ""))
        assert cp.traced_ann_value() == table_ann
        expect(cp.network_output).to_contain_text("SWMM 3,360.00 m³ (−26.6 %)")
        expect(cp.range_summary).to_have_text(
            "All eight inputs are within the range the network was trained on."
        )
        expect(cp.range_items.locator(".calc-range__flag")).to_have_count(0)
        assert console_errors == []

    def test_trace_button_switches_the_trace(
        self, five_page: CalculationsPage, console_errors: list[str]
    ) -> None:
        cp = five_page
        assert cp.trace_select.locator("option").all_inner_texts() == ["S1", "S2", "S3", "S4", "S5"]
        # Each Trace button names its subcatchment.
        expect(
            cp.results_table.get_by_role("button", name=re.compile(r"^Trace S3\b"))
        ).to_have_count(1)
        expect(cp.trace_select).to_have_value("0")
        # The Trace column is not data: Copy table and the column list leave it out.
        assert cp.get_results_columns() == RESULT_COLUMNS

        cp.trace_button("S3").click()

        expect(cp.trace_select).to_have_value("2")
        expect(cp.network_heading).to_be_focused()
        expect(cp.network_heading).to_be_in_viewport()
        s3_ann = float(cp.column_values(2)[2].replace(",", ""))
        assert cp.traced_ann_value() == s3_ann
        expect(cp.range_summary).to_have_text(
            "1 of 8 inputs is outside the range the network was trained on (Width): "
            "treat this prediction as indicative."
        )
        width = cp.range_items.nth(1)
        expect(width).to_contain_text("outside the training range")
        expect(width.locator(".calc-range__arrow")).to_have_count(1)

        cp.trace_select.select_option(label="S5")
        expect(cp.range_summary).to_contain_text("(Slope)")
        assert console_errors == []

    def test_trace_button_is_keyboard_reachable(self, five_page: CalculationsPage) -> None:
        cp = five_page
        button = cp.trace_button("S2")
        button.focus()
        expect(button).to_be_focused()

        cp.page.keyboard.press("Enter")

        expect(cp.trace_select).to_have_value("1")
        expect(cp.network_heading).to_be_focused()

    def test_trace_follows_the_dark_theme(self, five_page: CalculationsPage) -> None:
        cp = five_page
        edge = cp.network_diagram.locator(".calc-net__edge--positive").first
        stroke = "(el) => getComputedStyle(el).stroke"
        light = edge.evaluate(stroke)

        cp.nav.set_theme("dark")

        expect(cp.page.locator("html")).to_have_attribute("data-bs-theme", "dark")
        assert edge.evaluate(stroke) != light

    def test_traced_page_fits_a_phone(self, five_page: CalculationsPage) -> None:
        page = five_page.page
        page.set_viewport_size({"width": 390, "height": 844})
        five_page.trace_button("S3").click()
        page.wait_for_function(
            "() => document.documentElement.scrollWidth <= document.documentElement.clientWidth",
            timeout=5_000,
        )

    @pytest.mark.parametrize("scheme", ["light", "dark"])
    def test_no_serious_axe_violations(self, five_page: CalculationsPage, scheme: str) -> None:
        five_page.page.emulate_media(color_scheme=scheme)
        five_page.trace_button("S3").click()

        report = five_page.run_axe_audit()

        assert not report.critical + report.serious, [
            (v.id, v.description) for v in report.critical + report.serious
        ]
