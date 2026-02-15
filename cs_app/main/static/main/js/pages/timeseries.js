"use strict";

document.addEventListener("DOMContentLoaded", function () {
    var modeSelect = document.getElementById("id_mode");
    var sweepWrappers = ["feature-wrapper", "start-wrapper", "stop-wrapper", "step-wrapper"];
    var runTimeseriesButton = document.getElementById("run-timeseries-button");
    var timeseriesLoadingState = document.getElementById("timeseries-loading-state");
    var timeseriesForm = runTimeseriesButton ? runTimeseriesButton.closest("form") : null;
    var isTimeseriesSubmitting = false;

    function toggleSweepFields() {
        var isSweep = modeSelect && modeSelect.value === "sweep";
        sweepWrappers.forEach(function (wrapperId) {
            var wrapper = document.getElementById(wrapperId);
            if (wrapper) {
                wrapper.style.display = isSweep ? "" : "none";
            }
        });
    }

    if (modeSelect) {
        modeSelect.addEventListener("change", toggleSweepFields);
        toggleSweepFields();
    }

    function resetTimeseriesSubmitState() {
        isTimeseriesSubmitting = false;
        if (runTimeseriesButton) {
            runTimeseriesButton.disabled = false;
            runTimeseriesButton.removeAttribute("aria-disabled");
        }
        if (timeseriesLoadingState) {
            timeseriesLoadingState.classList.add("d-none");
        }
    }

    if (timeseriesForm && runTimeseriesButton) {
        timeseriesForm.addEventListener("submit", function (event) {
            var submitter = event.submitter || document.activeElement;
            if (submitter && submitter.id && submitter.id !== "run-timeseries-button") {
                return;
            }
            if (isTimeseriesSubmitting) {
                event.preventDefault();
                return;
            }
            isTimeseriesSubmitting = true;
            runTimeseriesButton.disabled = true;
            runTimeseriesButton.setAttribute("aria-disabled", "true");
            if (timeseriesLoadingState) {
                timeseriesLoadingState.classList.remove("d-none");
            }
        });
        window.addEventListener("pageshow", resetTimeseriesSubmitState);
    }

    var configElement = document.getElementById("ts-chart-config");
    if (configElement) {
        try {
            var config = JSON.parse(configElement.textContent);
            if (config.mode === "single") {
                renderLineChart("timeseries-chart", config.data, "datetime", config.columns, {
                    title: config.title,
                });
            } else if (config.mode === "sweep") {
                renderSweepChart(
                    "timeseries-chart",
                    config.data,
                    config.columns,
                    config.feature,
                    config.catchment
                );
            }
        } catch (error) {
            console.error("Failed to render chart:", error);
            var chartElement = document.getElementById("timeseries-chart");
            if (chartElement) {
                chartElement.innerHTML = '<p class="text-danger text-center">Failed to load chart.</p>';
            }
        }
    }

    var pngButton = document.getElementById("download-timeseries-png-button");
    var exportFeedback = document.getElementById("timeseries-export-feedback");
    if (!pngButton) {
        return;
    }

    pngButton.addEventListener("click", function () {
        var rawName = pngButton.getAttribute("data-filename") || "timeseries_chart";
        var pngName = rawName.replace(/\.xlsx$/i, ".png");
        if (exportFeedback) {
            exportFeedback.classList.add("d-none");
            exportFeedback.textContent = "";
        }

        var exported = downloadChartAsPng("timeseries-chart", pngName);
        if (!exported && exportFeedback) {
            exportFeedback.textContent = "Chart export is unavailable right now. Reload the page and try again.";
            exportFeedback.classList.remove("d-none");
        }
    });
});
