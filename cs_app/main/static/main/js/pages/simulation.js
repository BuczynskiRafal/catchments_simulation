"use strict";

document.addEventListener("DOMContentLoaded", function () {
    var optionSelect = document.getElementById("id_option");
    var predefinedMethods = ["simulate_n_imperv", "simulate_n_perv", "simulate_s_imperv", "simulate_s_perv"];

    function toggleRangeFields() {
        if (!optionSelect) {
            return;
        }
        var isPredefined = predefinedMethods.indexOf(optionSelect.value) !== -1;
        ["id_start", "id_stop", "id_step"].forEach(function (fieldId) {
            var field = document.getElementById(fieldId);
            if (!field) {
                return;
            }
            var wrapper = field.closest(".col-md-2");
            if (wrapper) {
                wrapper.style.display = isPredefined ? "none" : "";
            }
        });
    }

    if (optionSelect) {
        optionSelect.addEventListener("change", toggleRangeFields);
        toggleRangeFields();
    }

    var runSimulationButton = document.getElementById("run-simulation-button");
    var simulationLoadingState = document.getElementById("simulation-loading-state");
    var simulationForm = runSimulationButton ? runSimulationButton.closest("form") : null;
    var isSubmitting = false;

    function resetSimulationSubmitState() {
        isSubmitting = false;
        if (runSimulationButton) {
            runSimulationButton.disabled = false;
            runSimulationButton.removeAttribute("aria-disabled");
        }
        if (simulationLoadingState) {
            simulationLoadingState.classList.add("d-none");
        }
    }

    if (simulationForm && runSimulationButton) {
        simulationForm.addEventListener("submit", function (event) {
            var submitter = event.submitter || document.activeElement;
            if (submitter && submitter.id && submitter.id !== "run-simulation-button") {
                return;
            }
            if (isSubmitting) {
                event.preventDefault();
                return;
            }
            isSubmitting = true;
            runSimulationButton.disabled = true;
            runSimulationButton.setAttribute("aria-disabled", "true");
            if (simulationLoadingState) {
                simulationLoadingState.classList.remove("d-none");
            }
        });
        window.addEventListener("pageshow", resetSimulationSubmitState);
    }

    var configElement = document.getElementById("chart-config");
    if (!configElement) {
        return;
    }

    try {
        var config = JSON.parse(configElement.textContent);
        renderLineChart("simulation-chart", config.data, config.x, config.y, {
            title: config.title,
            xLabel: config.xLabel,
            yLabels: config.yLabels,
        });
    } catch (error) {
        console.error("Failed to render chart:", error);
        var chartElement = document.getElementById("simulation-chart");
        if (chartElement) {
            chartElement.innerHTML = '<p class="text-danger text-center">Failed to load chart.</p>';
        }
    }
});
