"use strict";

document.addEventListener("DOMContentLoaded", function () {
    var chartDataElement = document.getElementById("chart-data");
    if (!chartDataElement) {
        return;
    }

    try {
        var data = JSON.parse(chartDataElement.textContent);

        renderLineChart("plot-slope", data.slope, "slope", "runoff", {
            title: "Dependence of runoff on subcatchment slope.",
            xLabel: "Percent Slope [-]",
            yLabel: "Runoff [m3]",
            xRange: [0, 100],
        });
        renderLineChart("plot-area", data.area, "area", "runoff", {
            title: "Dependence of runoff on subcatchment area.",
            xLabel: "Area [ha]",
            yLabel: "Runoff [m3]",
        });
        renderLineChart("plot-width", data.width, "width", "runoff", {
            title: "Dependence of runoff on subcatchment width.",
            xLabel: "Width [m]",
            yLabel: "Runoff [m3]",
            xRange: [0, 1000],
        });
    } catch (error) {
        console.error("Failed to render charts:", error);
        ["plot-slope", "plot-area", "plot-width"].forEach(function (id) {
            var element = document.getElementById(id);
            if (element) {
                element.innerHTML = '<p class="text-danger text-center">Failed to load chart.</p>';
            }
        });
    }
});
