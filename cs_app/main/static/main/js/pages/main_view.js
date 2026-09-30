/*
 * Home page: the example hyetograph/hydrograph, the example sweep charts and
 * the Copy buttons of the code snippets. Chart data and axis labels come from
 * the server (#chart-data); the table of contents is Bootstrap scrollspy
 * (data attributes in the template).
 */
(function (CS) {
    "use strict";

    var SWEEP_CHART_HEIGHT = 320;
    var HERO_CHART_HEIGHT = 360;

    function readChartData() {
        var element = document.getElementById("chart-data");
        return element ? JSON.parse(element.textContent) : null;
    }

    function renderHydrograph(hydrograph) {
        var element = document.getElementById("plot-hydrograph");
        if (!element || !hydrograph) {
            return;
        }
        // Runoff only: the example model has no runon, and the loss rates would
        // crowd the hero chart.
        CS.charts.hydrograph(element, hydrograph.records, {
            flowFields: ["runoff"],
            yLabels: hydrograph.yLabels,
            xLabel: hydrograph.xLabel,
            height: HERO_CHART_HEIGHT,
            fileName: "example-storm-S1",
        });
    }

    function renderSweeps(sweeps) {
        document.querySelectorAll("[data-home-sweep]").forEach(function (element) {
            var key = element.getAttribute("data-home-sweep");
            var sweep = sweeps && sweeps[key];
            if (!sweep) {
                return;
            }
            CS.charts.line(element, sweep.records, sweep.xField, "runoff", {
                xLabel: sweep.xLabel,
                yLabel: sweep.yLabel,
                xRange: sweep.xRange,
                height: SWEEP_CHART_HEIGHT,
                fileName: "example-" + key + "-sweep",
            });
        });
    }

    function copySnippet(event) {
        var button = event.target.closest(".cs-code__copy");
        if (!button) {
            return;
        }
        var code = button.closest(".cs-code").querySelector("pre code");
        CS.clipboard.copy(code.textContent, "Code copied to the clipboard.");
    }

    document.addEventListener("click", copySnippet);

    document.addEventListener("DOMContentLoaded", function () {
        var data = readChartData();
        if (!data) {
            return;
        }
        renderHydrograph(data.hydrograph);
        renderSweeps(data.sweeps);
    });
})(window.CS);
