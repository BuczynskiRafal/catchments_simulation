/*
 * Simulation page (main/simulation.html).
 *
 * Panel: a hint in step 1 while no model is loaded; the range fields only for
 * range-based methods, with a live "N runs" hint that warns before the server's
 * step limit is hit. The run form is enhanced by CS.asyncForm; the results
 * fragment is initialised by initResults() on load and after every swap
 * (key figures, chart, sortable table). Result actions are delegated from the
 * canvas, so they keep working across swaps.
 */
(function (CS) {
    "use strict";

    var RUNOFF = "runoff";
    var PEAK_RATE = "peak_runoff_rate";

    // ── Parameters panel ────────────────────────────────────────────────

    function readNumber(input) {
        return input.value.trim() === "" ? NaN : Number(input.value);
    }

    function initRange(form) {
        var option = form.elements.namedItem("option");
        var range = document.getElementById("simulation-range");
        var hint = document.getElementById("simulation-range-hint");
        var note = document.getElementById("simulation-literature-note");
        var inputs = ["start", "stop", "step"].map(function (name) {
            return form.elements.namedItem(name);
        });
        var predefined = form.dataset.predefinedMethods.split(" ");
        var maxSteps = Number(form.dataset.maxSteps);

        function updateHint() {
            var values = inputs.map(readNumber);
            var result = CS.format.sweepRange(values[0], values[1], values[2], maxSteps);
            hint.textContent = result.text;
            hint.classList.toggle("is-warning", result.warning);
            hint.hidden = false;
        }

        // Disabled as well as hidden: leftover values in the unused range are then neither
        // sent nor checked by constraint validation (an invalid hidden input would
        // silently block the submit).
        function toggleRange() {
            var isPredefined = predefined.indexOf(option.value) !== -1;
            range.hidden = isPredefined;
            range.disabled = isPredefined;
            note.hidden = !isPredefined;
        }

        option.addEventListener("change", toggleRange);
        inputs.forEach(function (input) {
            input.addEventListener("input", updateHint);
        });
        toggleRange();
        updateHint();
    }

    function initModelHint() {
        var hint = document.getElementById("simulation-model-hint");
        document.addEventListener("cs:model-changed", function () {
            hint.hidden = true;
        });
        document.addEventListener("cs:model-cleared", function () {
            hint.hidden = false;
        });
    }

    // ── Key figures ─────────────────────────────────────────────────────

    /** "Total Runoff Volume [m³]" → {name: "Total Runoff Volume", unit: "m³"}; "[-]" has no unit. */
    function splitLabel(label) {
        var match = /^(.*?)\s*\[([^\]]*)\]\s*$/.exec(label || "");
        if (!match) {
            return { name: label || "", unit: "" };
        }
        return { name: match[1], unit: match[2] === "-" ? "" : match[2] };
    }

    /** About four significant digits, so volumes and small rates both read well. */
    function formatValue(value) {
        var magnitude = Math.abs(value);
        return CS.format.number(value, magnitude >= 100 ? 1 : magnitude >= 1 ? 2 : 4);
    }

    /** "Percent Slope = 4 %": the swept value a figure refers to, as the table prints it. */
    function describeSwept(value, xLabel) {
        var label = splitLabel(xLabel);
        // Non-breaking spaces keep "= 10 %" together when the tile wraps.
        var text = label.name + "\u00a0=\u00a0" + Number(value.toFixed(3));
        return label.unit ? text + "\u00a0" + label.unit : text;
    }

    function percentChange(from, to) {
        if (!from) {
            return "—";
        }
        var change = ((to - from) / Math.abs(from)) * 100;
        return (change > 0 ? "+" : "") + CS.format.number(change, 1);
    }

    function finiteRows(rows, field) {
        return rows.filter(function (row) {
            return Number.isFinite(row[field]);
        });
    }

    /** Tiles for the outputs present in the data, rows ordered by the swept value. */
    function keyFigures(config) {
        var x = config.x;
        var yLabels = config.yLabels || {};
        var rows = finiteRows(config.data, x).sort(function (a, b) {
            return a[x] - b[x];
        });
        var tiles = [];

        var runoffRows = finiteRows(rows, RUNOFF);
        if (runoffRows.length > 1) {
            var lowest = runoffRows[0];
            var highest = runoffRows[runoffRows.length - 1];
            var runoffUnit = splitLabel(yLabels[RUNOFF]).unit;
            tiles.push(
                { label: "Runoff at lowest value", value: formatValue(lowest[RUNOFF]), unit: runoffUnit,
                  note: "At " + describeSwept(lowest[x], config.xLabel) },
                { label: "Runoff at highest value", value: formatValue(highest[RUNOFF]), unit: runoffUnit,
                  note: "At " + describeSwept(highest[x], config.xLabel) },
                { label: "Runoff change", value: percentChange(lowest[RUNOFF], highest[RUNOFF]), unit: "%",
                  note: "From lowest to highest value" }
            );
        }

        var peakRows = finiteRows(rows, PEAK_RATE);
        if (peakRows.length) {
            var peak = peakRows.reduce(function (best, row) {
                return row[PEAK_RATE] > best[PEAK_RATE] ? row : best;
            });
            tiles.push({ label: "Highest peak runoff rate", value: formatValue(peak[PEAK_RATE]),
                         unit: splitLabel(yLabels[PEAK_RATE]).unit, note: "At " + describeSwept(peak[x], config.xLabel) });
        }
        return tiles;
    }

    function element(tag, className, text) {
        var el = document.createElement(tag);
        el.className = className;
        if (text) {
            el.textContent = text;
        }
        return el;
    }

    function renderKeyFigures(list, tiles) {
        list.replaceChildren.apply(list, tiles.map(function (tile) {
            var value = element("dd", "cs-metric__value", tile.value);
            if (tile.unit && tile.value !== "—") {
                value.append(element("span", "cs-metric__unit", tile.unit));
            }
            var group = element("div", "cs-metric");
            group.append(element("dt", "cs-metric__label", tile.label), value,
                         element("dd", "cs-metric__note", tile.note));
            return group;
        }));
        list.hidden = tiles.length === 0;
    }

    // ── Results ─────────────────────────────────────────────────────────

    function readConfig(root) {
        var script = root.querySelector("#chart-config");
        if (!script) {
            return null;
        }
        try {
            var config = JSON.parse(script.textContent);
            return config && Array.isArray(config.data) ? config : null;
        } catch (error) {
            console.error("Invalid simulation chart config:", error);
            return null;
        }
    }

    function chartLabel(config) {
        var outputs = config.y.map(function (field) {
            return splitLabel((config.yLabels || {})[field] || field).name;
        });
        return "Line charts of " + outputs.join(", ") + " against " + config.xLabel +
            ". The results table lists the same values.";
    }

    function reveal(button, isAvailable) {
        if (button) {
            button.hidden = !isAvailable;
        }
    }

    function initResults(root) {
        var table = root.querySelector("#simulation-table");
        if (table) {
            CS.table.sortable(table);
            reveal(root.querySelector("[data-sim-action='copy-table']"), true);
        }
        var config = readConfig(root);
        if (!config) {
            return;
        }
        renderKeyFigures(root.querySelector("#simulation-metrics"), keyFigures(config));
        var pngButton = root.querySelector("[data-sim-action='download-png']");
        var drawn = CS.charts.line(root.querySelector("#simulation-chart"), config.data, config.x, config.y, {
            xLabel: config.xLabel,
            yLabels: config.yLabels,
            ariaLabel: chartLabel(config),
            fileName: pngButton.dataset.filename,
        });
        reveal(pngButton, drawn !== null);
    }

    var ACTIONS = {
        "copy-table": function (canvas) {
            CS.clipboard.copy(CS.table.toCSV(canvas.querySelector("#simulation-table")), "Table copied as CSV.");
        },
        "download-png": function (canvas, button) {
            CS.charts.downloadPng(canvas.querySelector("#simulation-chart"), button.dataset.filename);
        },
    };

    function initActions(canvas) {
        canvas.addEventListener("click", function (event) {
            var button = event.target.closest("[data-sim-action]");
            if (button && ACTIONS[button.dataset.simAction]) {
                ACTIONS[button.dataset.simAction](canvas, button);
            }
        });
    }

    document.addEventListener("DOMContentLoaded", function () {
        var form = document.getElementById("simulation-form");
        var canvas = document.getElementById("simulation-canvas");
        if (!form || !canvas) {
            return;
        }
        initModelHint();
        initRange(form);
        initActions(canvas);
        initResults(canvas);

        CS.asyncForm({
            form: form,
            submitButton: document.getElementById("run-simulation-button"),
            target: canvas,
            loadingEl: document.getElementById("simulation-loading-state"),
            successMessage: "Simulation finished",
            onSuccess: initResults,
        });
    });
})(window.CS);
