/*
 * SWMM vs ANN comparison page (/calculations).
 *
 * The run form is enhanced with CS.asyncForm: the server returns the results
 * fragment (partials/_calculations_results.html), which replaces the contents of
 * #calculations-output. initResults(root) wires up whatever that fragment holds and
 * runs both on page load (results rendered by a no-JS POST) and after every swap.
 * All numbers are computed server-side; the charts read the rows embedded as
 * #calculations-chart-data.
 */
(function (CS) {
    "use strict";

    function readRows(root) {
        var payload = root.querySelector("#calculations-chart-data");
        return payload ? JSON.parse(payload.textContent) : null;
    }

    function drawCharts(root, rows) {
        // The volume unit follows the model's flow units (m³ for SI, ft³ for US models).
        var unit = " [" + root.querySelector("[data-calc-unit]").dataset.calcUnit + "]";
        CS.charts.parity(root.querySelector("#calculations-parity-chart"), rows, {
            xLabel: "SWMM runoff" + unit,
            yLabel: "ANN runoff" + unit,
            ariaLabel: "Parity chart of ANN against SWMM runoff per subcatchment, with a 1:1 reference line",
            fileName: "swmm-vs-ann-parity",
        });
        CS.charts.bars(root.querySelector("#calculations-bars-chart"), rows, {
            xLabel: "Subcatchment",
            yLabel: "Runoff" + unit,
            ariaLabel: "Grouped bar chart of SWMM and ANN runoff per subcatchment",
            fileName: "swmm-vs-ann-runoff",
        });
    }

    function initResults(root) {
        var rows = readRows(root);
        if (!rows) {
            return;
        }
        // Unhide before drawing: Plotly sizes each chart from its container.
        root.querySelectorAll("[data-calc-requires-js]").forEach(function (el) {
            el.hidden = false;
        });
        drawCharts(root, rows);
        CS.table.sortable(root.querySelector("#calculations-table"));
    }

    // Delegated, so the button in every swapped-in fragment works.
    document.addEventListener("click", function (event) {
        if (!event.target.closest("#copy-calculations-table-button")) {
            return;
        }
        var table = document.getElementById("calculations-table");
        CS.clipboard.copy(CS.table.toCSV(table), "Comparison table copied as CSV.");
    });

    var output = document.getElementById("calculations-output");
    initResults(output);
    CS.asyncForm({
        form: document.getElementById("calculations-form"),
        submitButton: document.getElementById("run-calculations-button"),
        target: output,
        loadingEl: document.getElementById("calculations-loading-state"),
        successMessage: "Comparison finished.",
        onSuccess: initResults,
    });
})(window.CS);
