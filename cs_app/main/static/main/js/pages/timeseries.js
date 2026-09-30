/*
 * Timeseries page (/timeseries).
 *
 * Run form: CSS shows the sweep range only in sweep mode; here its fieldset is
 * also disabled in single mode, so leftover values there are neither sent nor
 * checked by the browser's constraint validation (an invalid hidden input would
 * silently block the submit). A live "N runs" hint follows the range. Runs go
 * through CS.asyncForm, which swaps the server-rendered results fragment into
 * #timeseries-canvas.
 *
 * Results: the server renders the metrics and the sweep summary table;
 * initResults(root) adds what needs the data — the hydrograph (single) or sweep
 * chart, the timestep table and table sorting — on page load and after every
 * swap. Controls inside the fragment are handled by delegation on the canvas.
 */
(function (CS) {
    "use strict";

    var DESKTOP_QUERY = "(min-width: 992px)";
    var PHONE_QUERY = "(max-width: 575.98px)";
    var EXPORT_UNAVAILABLE = "Chart export is unavailable right now. Reload the page and try again.";
    var CHART_FAILED = "The chart could not be drawn. Download the results to inspect the data.";

    var currentConfig = null;

    // ── Run form ──────────────────────────────────────────────────────

    function initRunForm(form) {
        var sweepFields = form.querySelector("#ts-sweep-fields");
        var hint = form.querySelector("#ts-runs-hint");
        var maxSteps = Number(form.dataset.maxSteps);

        function numberValue(name) {
            var value = form.elements.namedItem(name).value;
            return value.trim() === "" ? NaN : Number(value);
        }

        function updateHint() {
            var result = CS.format.sweepRange(
                numberValue("start"), numberValue("stop"), numberValue("step"), maxSteps
            );
            hint.textContent = result.text;
            hint.classList.toggle("is-warning", result.warning);
        }

        function syncMode() {
            sweepFields.disabled = form.elements.namedItem("mode").value !== "sweep";
        }

        form.addEventListener("change", function (event) {
            if (event.target.name === "mode") {
                syncMode();
            }
        });
        sweepFields.addEventListener("input", updateHint);
        syncMode();
        updateHint();
    }

    // ── Timestep table ────────────────────────────────────────────────

    function headerCell(label, scope, className) {
        var th = document.createElement("th");
        th.scope = scope;
        th.textContent = label;
        th.className = className;
        return th;
    }

    /** null (NaN on the server) stays missing instead of becoming 0 through Number(null). */
    function toNumber(value) {
        return value === null || value === undefined ? NaN : Number(value);
    }

    function fillTimestepTable(table, config) {
        var yLabels = config.yLabels || {};
        var caption = table.createCaption();
        caption.className = "visually-hidden";
        caption.textContent = "Values at each reported timestep";

        var headRow = table.createTHead().insertRow();
        headRow.appendChild(headerCell(config.xLabel || "Time", "col", ""));
        config.columns.forEach(function (column) {
            headRow.appendChild(headerCell(yLabels[column] || column, "col", "cs-num"));
        });

        var body = table.createTBody();
        config.data.forEach(function (record) {
            var row = body.insertRow();
            row.appendChild(headerCell(String(record.datetime), "row", "cs-mono"));
            config.columns.forEach(function (column) {
                var value = toNumber(record[column]);
                var cell = row.insertCell();
                cell.className = "cs-num";
                cell.textContent = CS.format.number(value, 4);
                cell.dataset.sortValue = Number.isFinite(value) ? String(value) : "";
            });
        });
        CS.table.sortable(table);
    }

    function initTimestepTable(root, config) {
        var details = root.querySelector("[data-ts-data]");
        var table = details.querySelector("table");
        details.addEventListener("toggle", function () {
            if (details.open && !table.rows.length) {
                fillTimestepTable(table, config);
            }
        });
        details.hidden = false;
    }

    // ── Charts ────────────────────────────────────────────────────────

    /** Phones stack the legend above the plot, so they get extra height for it (see .ts-chart). */
    function chartHeight() {
        return window.matchMedia(PHONE_QUERY).matches ? 520 : 440;
    }

    function pngFileName(root) {
        return root.querySelector("#download-timeseries-png-button").dataset.filename;
    }

    function showChartError(chartEl, error) {
        console.error("Failed to render the timeseries chart:", error);
        var message = document.createElement("div");
        message.className = "cs-empty-state";
        message.textContent = CHART_FAILED;
        chartEl.replaceChildren(message);
    }

    /** draw() returns the Plotly promise (or null); failures, sync or async, leave a message. */
    function drawChart(chartEl, draw) {
        try {
            var drawing = draw();
            if (drawing) {
                drawing.catch(function (error) {
                    showChartError(chartEl, error);
                });
            }
        } catch (error) {
            showChartError(chartEl, error);
        }
    }

    function renderHydrograph(root, config) {
        var chartEl = root.querySelector("#timeseries-chart");
        drawChart(chartEl, function () {
            return CS.charts.hydrograph(chartEl, config.data, {
                yLabels: config.yLabels,
                xLabel: config.xLabel,
                height: chartHeight(),
                rangeSlider: window.matchMedia(DESKTOP_QUERY).matches,
                ariaLabel: "Rainfall and runoff chart. " + (config.title || ""),
                fileName: pngFileName(root),
            });
        });
    }

    function renderSweep(root, config, column) {
        var chartEl = root.querySelector("#timeseries-chart");
        var label = (config.yLabels || {})[column] || column;
        drawChart(chartEl, function () {
            return CS.charts.sweep(chartEl, config.data, [column], config.parameterLabel, config.catchment, {
                title: "",
                xLabel: config.xLabel,
                yLabels: config.yLabels,
                height: chartHeight(),
                ariaLabel: label + " over time for each " + config.parameterName + " value. " + (config.title || ""),
                fileName: pngFileName(root),
            });
        });
    }

    // ── Results fragment ──────────────────────────────────────────────

    function initResults(root) {
        var configEl = root.querySelector("#ts-chart-config");
        currentConfig = configEl ? JSON.parse(configEl.textContent) : null;
        if (!currentConfig) {
            return;
        }
        root.querySelector("#download-timeseries-png-button").hidden = false;
        if (currentConfig.mode === "sweep") {
            renderSweep(root, currentConfig, root.querySelector("[data-ts-series]").value);
            CS.table.sortable(root.querySelector('[data-ts-table="sweep"]'));
        } else {
            renderHydrograph(root, currentConfig);
            initTimestepTable(root, currentConfig);
        }
    }

    function exportPng(canvas) {
        var exported = CS.charts.downloadPng(canvas.querySelector("#timeseries-chart"), pngFileName(canvas));
        canvas.querySelector("#timeseries-export-feedback").textContent = exported ? "" : EXPORT_UNAVAILABLE;
    }

    /** Controls inside the fragment are replaced on every run, so listen on the canvas. */
    function bindCanvas(canvas) {
        canvas.addEventListener("click", function (event) {
            if (event.target.closest("#download-timeseries-png-button")) {
                exportPng(canvas);
            }
        });
        canvas.addEventListener("change", function (event) {
            if (event.target.matches("[data-ts-series]") && currentConfig) {
                renderSweep(canvas, currentConfig, event.target.value);
            }
        });
    }

    document.addEventListener("DOMContentLoaded", function () {
        var form = document.getElementById("timeseries-form");
        var canvas = document.getElementById("timeseries-canvas");
        if (!form || !canvas) {
            return;
        }
        initRunForm(form);
        bindCanvas(canvas);
        initResults(canvas);
        CS.asyncForm({
            form: form,
            submitButton: document.getElementById("run-timeseries-button"),
            target: canvas,
            loadingEl: document.getElementById("timeseries-loading-state"),
            successMessage: "Timeseries analysis finished",
            onSuccess: initResults,
        });
    });
})(window.CS);
