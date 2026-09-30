"""Contract tests for the shared core scripts (static/main/js/core/*).

The run endpoints are mocked with Playwright routing so the client side of the
AJAX contract (DESIGN_SPEC §4) is exercised in isolation from the views.
"""

from __future__ import annotations

import json
import re

import pytest
from playwright.sync_api import Page, Route, expect

pytestmark = pytest.mark.e2e

RUN_URL = "**/__cs_test_run__"
DOWNLOAD_URL = "**/__cs_test_download__"
TIMEOUT_TEXT = "The server did not finish the run."
ERROR_TOAST_TEXT = "The run did not complete."

# Builds a run form on the current page and enhances it with CS.asyncForm.
MOUNT_FORM = """() => {
    const main = document.getElementById("main-content");
    main.insertAdjacentHTML("beforeend", `
        <form id="t-form" action="/__cs_test_run__" method="post">
            <input type="hidden" name="csrfmiddlewaretoken" value="t">
            <label for="id_start">Start</label>
            <input id="id_start" name="start" class="form-control" value="1">
            <button type="submit" id="t-run">Run</button>
            <button type="submit" id="t-download" formaction="/__cs_test_download__">Download</button>
            <div id="t-loading" hidden>Running <span data-cs-elapsed></span></div>
        </form>
        <div id="t-results"></div>`);
    window.__events = [];
    CS.asyncForm({
        form: document.getElementById("t-form"),
        submitButton: document.getElementById("t-run"),
        target: document.getElementById("t-results"),
        loadingEl: document.getElementById("t-loading"),
        successMessage: "Simulation finished",
        onSuccess: (target) => window.__events.push(["success", target.id]),
        onError: (error) => window.__events.push(["error", error.status, error.message]),
    });
}"""


def _mount(page: Page, live_server) -> None:
    page.goto(f"{live_server.url}/about")
    page.evaluate(MOUNT_FORM)


def _wait_for_route(page: Page, pending: list[Route]) -> None:
    """Pump Playwright events until the intercepted request reaches the handler."""
    for _ in range(100):
        if pending:
            return
        page.wait_for_timeout(20)
    raise AssertionError("request was never routed")


def _json(route: Route, status: int, payload: dict) -> None:
    route.fulfill(status=status, content_type="application/json", body=json.dumps(payload))


class TestAsyncFormSuccess:
    def test_busy_state_while_running_then_fragment_swapped_and_focused(
        self, page: Page, live_server
    ) -> None:
        pending: list[Route] = []
        page.route(RUN_URL, lambda route: pending.append(route))
        _mount(page, live_server)

        page.locator("#t-run").click()
        expect(page.locator("#t-run")).to_be_disabled()
        expect(page.locator("#t-run")).to_have_attribute("aria-busy", "true")
        expect(page.locator("#t-results")).to_have_attribute("aria-busy", "true")
        expect(page.locator("#t-loading")).to_be_visible()
        expect(page.locator("#t-loading [data-cs-elapsed]")).to_have_text(re.compile(r"^\d+ s$"))

        _wait_for_route(page, pending)
        request = pending[0].request
        assert request.method == "POST"
        assert request.headers.get("x-requested-with") == "XMLHttpRequest"
        assert 'name="csrfmiddlewaretoken"\r\n\r\nt\r\n' in (request.post_data or "")
        pending[0].fulfill(status=200, content_type="text/html", body="<h3>Results</h3><p>42</p>")

        expect(page.locator("#t-results h3")).to_have_text("Results")
        expect(page.locator("#t-results h3")).to_be_focused()
        expect(page.locator("#t-run")).to_be_enabled()
        expect(page.locator("#t-run")).not_to_have_attribute("aria-busy", "true")
        expect(page.locator("#t-results")).not_to_have_attribute("aria-busy", "true")
        expect(page.locator("#t-loading")).to_be_hidden()
        assert page.evaluate("window.__events") == [["success", "t-results"]]
        expect(page.locator(".toast-container .cs-toast--success")).to_have_text(
            "Simulation finished"
        )

    def test_second_submit_while_running_is_ignored(self, page: Page, live_server) -> None:
        pending: list[Route] = []
        page.route(RUN_URL, lambda route: pending.append(route))
        _mount(page, live_server)

        fetch_calls = page.evaluate(
            """() => {
                let calls = 0;
                const realFetch = window.fetch;
                window.fetch = (...args) => { calls += 1; return realFetch(...args); };
                const form = document.getElementById("t-form");
                form.requestSubmit();
                form.requestSubmit();
                return calls;
            }"""
        )
        assert fetch_calls == 1
        expect(page.locator("#t-run")).to_be_disabled()
        _wait_for_route(page, pending)
        pending[0].fulfill(status=200, content_type="text/html", body="<h3>Done</h3>")
        expect(page.locator("#t-results h3")).to_have_text("Done")


class TestAsyncFormErrors:
    def test_field_and_general_errors_render_inline_and_clear(
        self, page: Page, live_server
    ) -> None:
        page.route(
            RUN_URL,
            lambda route: _json(
                route,
                400,
                {
                    "message": "Please correct the highlighted fields.",
                    "field_errors": {"start": ["Too large."], "__all__": ["Range too long."]},
                },
            ),
        )
        _mount(page, live_server)
        page.locator("#t-run").click()

        field = page.locator("#id_start")
        expect(field).to_have_class(re.compile(r"\bis-invalid\b"))
        expect(field).to_have_attribute("aria-invalid", "true")
        expect(page.locator("#t-form .invalid-feedback")).to_have_text("Too large.")
        expect(field).to_have_attribute("aria-describedby", "id_start-async-error")
        alert = page.locator("#t-form .alert.alert-danger[role='alert']")
        expect(alert).to_contain_text("Please correct the highlighted fields.")
        expect(alert).to_contain_text("Range too long.")
        expect(field).to_be_focused()
        expect(page.locator("#t-run")).to_be_enabled()
        expect(page.locator(".toast-container .cs-toast--danger")).to_contain_text(ERROR_TOAST_TEXT)
        assert page.evaluate("window.__events") == [
            ["error", 400, "Please correct the highlighted fields."]
        ]

        field.fill("2")
        expect(field).not_to_have_class(re.compile(r"\bis-invalid\b"))
        expect(page.locator("#t-form .invalid-feedback")).to_have_count(0)

        page.locator("#t-run").click()
        expect(page.locator("#t-form .alert")).to_have_count(1)

    def test_field_errors_without_message_get_a_fields_summary(
        self, page: Page, live_server
    ) -> None:
        page.route(
            RUN_URL,
            lambda route: _json(route, 400, {"message": "", "field_errors": {"start": ["Bad."]}}),
        )
        _mount(page, live_server)
        page.locator("#t-run").click()

        expect(page.locator("#t-form .alert-danger")).to_have_text(
            "Please correct the highlighted fields."
        )

    @pytest.mark.parametrize(
        "respond",
        [
            lambda route: route.fulfill(
                status=502, content_type="text/html", body="<html>Bad gateway</html>"
            ),
            lambda route: route.abort(),
        ],
        ids=["proxy-502-html", "network-failure"],
    )
    def test_non_json_failures_show_timeout_message(self, page: Page, live_server, respond) -> None:
        page.route(RUN_URL, respond)
        _mount(page, live_server)
        page.locator("#t-run").click()

        expect(page.locator("#t-form .alert-danger")).to_contain_text(TIMEOUT_TEXT)
        expect(page.locator("#t-run")).to_be_enabled()
        expect(page.locator("#t-run")).to_be_focused()
        expect(page.locator("#t-loading")).to_be_hidden()

    @pytest.mark.parametrize(
        ("content_type", "body"),
        [
            ("application/json", json.dumps({"message": "Log in", "field_errors": {}})),
            ("text/html", "<p>Unauthorized</p>"),
        ],
        ids=["json", "html"],
    )
    def test_unauthenticated_redirects_to_login_with_next(
        self, page: Page, live_server, content_type: str, body: str
    ) -> None:
        page.route(
            RUN_URL,
            lambda route: route.fulfill(status=401, content_type=content_type, body=body),
        )
        _mount(page, live_server)
        page.locator("#t-run").click()

        page.wait_for_url(re.compile(r".*/accounts/login/\?next=%2Fabout$"))


class TestAsyncFormReruns:
    def test_failed_rerun_marks_kept_results_and_next_run_clears_previous_feedback(
        self, page: Page, live_server
    ) -> None:
        responses = [
            lambda route: route.fulfill(
                status=200,
                content_type="text/html",
                body="<div data-cs-results><h3>First</h3></div>",
            ),
            lambda route: _json(route, 400, {"message": "Bad range.", "field_errors": {}}),
            lambda route: route.fulfill(
                status=200,
                content_type="text/html",
                body="<div data-cs-results><h3>Second</h3></div>",
            ),
        ]
        page.route(RUN_URL, lambda route: responses.pop(0)(route))
        _mount(page, live_server)
        page.evaluate(
            """() => document.getElementById("t-results").addEventListener(
                "cs:beforeswap", (event) => window.__events.push(["beforeswap", event.target.id]))"""
        )
        success_toasts = page.locator(".toast-container .cs-toast--success")
        danger_toasts = page.locator(".toast-container .cs-toast--danger")
        notice = page.locator("#t-results > .cs-stale-notice:first-child")

        page.locator("#t-run").click()
        expect(page.locator("#t-results h3")).to_have_text("First")
        expect(success_toasts).to_have_count(1)
        expect(notice).to_have_count(0)

        page.locator("#t-run").click()
        expect(notice).to_have_text(re.compile(r"^Showing results from the previous run"))
        expect(page.locator("#t-results h3")).to_have_text("First")
        expect(danger_toasts).to_have_count(1)
        expect(success_toasts).to_have_count(0)

        page.locator("#t-run").click()
        expect(page.locator("#t-results h3")).to_have_text("Second")
        expect(page.locator("#t-results .cs-stale-notice")).to_have_count(0)
        expect(danger_toasts).to_have_count(0)
        expect(success_toasts).to_have_count(1)
        assert [e for e in page.evaluate("window.__events") if e[0] == "beforeswap"] == [
            ["beforeswap", "t-results"],
            ["beforeswap", "t-results"],
        ]

    def test_failed_first_run_adds_no_stale_notice(self, page: Page, live_server) -> None:
        page.route(
            RUN_URL, lambda route: _json(route, 400, {"message": "Bad.", "field_errors": {}})
        )
        _mount(page, live_server)
        page.locator("#t-run").click()

        expect(page.locator("#t-form .alert-danger")).to_have_text("Bad.")
        expect(page.locator(".cs-stale-notice")).to_have_count(0)


class TestAsyncFormNativeSubmitters:
    def test_download_button_submits_natively_without_loading(
        self, page: Page, live_server
    ) -> None:
        run_calls: list[Route] = []
        page.route(RUN_URL, lambda route: run_calls.append(route))
        downloads: list[dict] = []

        def fulfil_download(route: Route) -> None:
            downloads.append(
                {
                    "navigation": route.request.is_navigation_request(),
                    "xhr": route.request.headers.get("x-requested-with"),
                }
            )
            route.fulfill(status=200, content_type="text/html", body="<p>file</p>")

        page.route(DOWNLOAD_URL, fulfil_download)
        _mount(page, live_server)

        page.locator("#t-download").click()
        page.wait_for_url(re.compile(r".*/__cs_test_download__$"))
        assert downloads == [{"navigation": True, "xhr": None}]
        assert run_calls == []


class TestAuthGuard:
    def test_anonymous_action_button_redirects_to_login_with_next(
        self, page: Page, live_server
    ) -> None:
        page.goto(f"{live_server.url}/about?tab=1")
        page.evaluate(
            """() => document.getElementById("main-content").insertAdjacentHTML(
                "beforeend", '<button type="button" data-authenticated="False">Run</button>')"""
        )
        page.get_by_role("button", name="Run", exact=True).click()
        page.wait_for_url(re.compile(r".*/accounts/login/\?next=%2Fabout%3Ftab%3D1$"))


class TestToast:
    def test_toast_renders_text_safely_and_can_be_dismissed(self, page: Page, live_server) -> None:
        page.goto(f"{live_server.url}/about")
        page.evaluate("CS.toast('<img src=x onerror=alert(1)> done', 'success')")

        toast = page.locator(".toast-container .toast.cs-toast--success")
        expect(toast).to_be_visible()
        expect(toast.locator(".toast-body")).to_have_text("<img src=x onerror=alert(1)> done")
        expect(toast.locator("img")).to_have_count(0)

        toast.get_by_role("button", name="Close").click()
        expect(page.locator(".toast-container .toast")).to_have_count(0)

    def test_unknown_level_falls_back_to_info(self, page: Page, live_server) -> None:
        page.goto(f"{live_server.url}/about")
        page.evaluate("CS.toast('hello', 'bogus')")
        expect(page.locator(".toast-container .cs-toast--info")).to_be_visible()


class TestTableAndFormat:
    MOUNT_TABLE = """() => {
        document.getElementById("main-content").insertAdjacentHTML("beforeend", `
            <table id="t-table" class="table cs-table">
                <thead><tr><th>Name</th><th class="cs-num">Runoff</th></tr></thead>
                <tbody>
                    <tr><td>S10</td><td class="cs-num">2</td></tr>
                    <tr><td>S2</td><td class="cs-num">10</td></tr>
                    <tr><td>=SUM(A1), "x"</td><td class="cs-num">-1.5</td></tr>
                </tbody>
            </table>`);
        CS.table.sortable(document.getElementById("t-table"));
    }"""

    def _column(self, page: Page, index: int) -> list[str]:
        return page.locator(f"#t-table tbody tr td:nth-child({index})").all_inner_texts()

    def test_numeric_aware_sorting_toggles_and_sets_aria_sort(
        self, page: Page, live_server
    ) -> None:
        page.goto(f"{live_server.url}/about")
        page.evaluate(self.MOUNT_TABLE)
        runoff_header = page.locator("#t-table th").nth(1)
        sort_button = runoff_header.get_by_role("button", name="Runoff")

        sort_button.click()
        expect(runoff_header).to_have_attribute("aria-sort", "ascending")
        assert self._column(page, 2) == ["-1.5", "2", "10"]

        sort_button.press("Enter")
        expect(runoff_header).to_have_attribute("aria-sort", "descending")
        assert self._column(page, 2) == ["10", "2", "-1.5"]

        name_header = page.locator("#t-table th").nth(0)
        name_header.get_by_role("button", name="Name").click()
        expect(name_header).to_have_attribute("aria-sort", "ascending")
        expect(runoff_header).not_to_have_attribute("aria-sort", re.compile(".*"))
        assert self._column(page, 1) == ['=SUM(A1), "x"', "S2", "S10"]

    def test_sortable_twice_binds_once(self, page: Page, live_server) -> None:
        page.goto(f"{live_server.url}/about")
        page.evaluate(self.MOUNT_TABLE)
        page.evaluate("CS.table.sortable(document.getElementById('t-table'))")
        header = page.locator("#t-table th").nth(1)

        header.get_by_role("button", name="Runoff").click()
        expect(header).to_have_attribute("aria-sort", "ascending")
        expect(header.locator(".cs-sort")).to_have_count(1)

    def test_csv_export_escapes_and_neutralises_formulas(self, page: Page, live_server) -> None:
        page.goto(f"{live_server.url}/about")
        page.evaluate(self.MOUNT_TABLE)
        csv = page.evaluate("CS.table.toCSV(document.getElementById('t-table'))")
        assert csv.split("\r\n") == [
            "Name,Runoff",
            "S10,2",
            "S2,10",
            '"\'=SUM(A1), ""x""",-1.5',
        ]

    def test_number_format(self, page: Page, live_server) -> None:
        page.goto(f"{live_server.url}/about")
        assert page.evaluate("CS.format.number(1234.5, 1)") == "1,234.5"
        assert page.evaluate("CS.format.number('3.14159')") == "3.14"
        assert page.evaluate("CS.format.number(null)") == "—"


class TestClipboard:
    def test_copy_writes_text_and_confirms(self, page: Page, live_server) -> None:
        page.context.grant_permissions(["clipboard-read", "clipboard-write"])
        page.goto(f"{live_server.url}/about")

        assert page.evaluate("CS.clipboard.copy('pip install catchment-simulation')") is True
        assert page.evaluate("navigator.clipboard.readText()") == "pip install catchment-simulation"
        expect(page.locator(".toast-container .cs-toast--success")).to_contain_text("Copied")


class TestChartsLifecycle:
    """Charts about to be swapped out or already detached are purged (no leaked listeners)."""

    DRAW = """async () => {
        const wrap = document.createElement("div");
        wrap.id = "t-charts";
        wrap.innerHTML = '<div id="t-a" style="width:400px"></div><div id="t-b" style="width:400px"></div>';
        document.getElementById("main-content").append(wrap);
        const rows = [{x: 1, y: 2}, {x: 2, y: 3}];
        await CS.charts.line("t-a", rows, "x", "y");
        await CS.charts.line("t-b", rows, "x", "y");
    }"""
    ATTACHED = "id => Boolean(document.getElementById(id)._responsiveChartHandler)"

    def test_before_swap_purges_charts_inside_the_target(self, page: Page, live_server) -> None:
        page.goto(f"{live_server.url}/")
        page.evaluate(self.DRAW)
        assert page.evaluate(self.ATTACHED, "t-a")

        page.evaluate(
            """() => document.getElementById("t-charts").dispatchEvent(
                new CustomEvent("cs:beforeswap", {bubbles: true}))"""
        )
        assert not page.evaluate(self.ATTACHED, "t-a")
        assert not page.evaluate(self.ATTACHED, "t-b")

    def test_detached_chart_is_purged_on_next_render(self, page: Page, live_server) -> None:
        page.goto(f"{live_server.url}/")
        page.evaluate(self.DRAW)
        detached = page.evaluate_handle(
            """() => { const el = document.getElementById("t-a"); el.remove(); return el; }"""
        )
        page.evaluate("""() => CS.charts.line("t-b", [{x: 1, y: 1}], "x", "y")""")

        assert not detached.evaluate("el => Boolean(el._responsiveChartHandler)")
        assert page.evaluate(self.ATTACHED, "t-b")
