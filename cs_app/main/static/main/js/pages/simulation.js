/*
 * Simulation page (main/simulation.html).
 *
 * Panel: a hint in step 1 while no model is loaded; the range fields only for
 * range-based methods, with a live "N runs" hint that warns before the server's
 * step limit is hit; the parameter card and the range suggestion of
 * CS.sweepAssist. The run form is enhanced by CS.asyncForm, with a status panel
 * that says what is running. The results fragment is initialised by initResults()
 * on load and after every swap: the reading line, key figures, the response chart
 * with the model's own value marked, the water balance chart and the sortable
 * table. Result actions are delegated from the canvas, so they keep working
 * across swaps.
 *
 * CS.simulation.reading(config, name) is the pure function behind the reading line.
 */
(function (CS) {
    "use strict";

    var RUNOFF = "runoff";
    var PEAK_RATE = "peak_runoff_rate";
    var INFILTRATION = "infiltration";
    // Relative difference below which two outputs count as equal (SWMM's own noise is larger).
    var SAME_TOLERANCE = 1e-9;
    var MODEL_VALUE_LABEL = "Model value";

    var assist = null;

    // ── Parameters panel ────────────────────────────────────────────────

    function readNumber(input) {
        return input.value.trim() === "" ? NaN : Number(input.value);
    }

    function rangeInputs(form) {
        return ["start", "stop", "step"].map(function (name) {
            return form.elements.namedItem(name);
        });
    }

    function initRange(form) {
        var option = form.elements.namedItem("option");
        var range = document.getElementById("simulation-range");
        var hint = document.getElementById("simulation-range-hint");
        var note = document.getElementById("simulation-literature-note");
        var inputs = rangeInputs(form);
        var predefined = form.dataset.predefinedMethods.split(" ");
        var maxRuns = Number(form.dataset.maxRuns);

        function updateHint() {
            var values = inputs.map(readNumber);
            var result = CS.format.sweepRange(values[0], values[1], values[2], maxRuns);
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

    function initAssist(form) {
        var features = CS.readJson("method-features") || {};
        assist = CS.sweepAssist({
            form: form,
            parameterSelect: form.elements.namedItem("option"),
            featureOf: function (option) {
                return features[option] || "";
            },
            rangeInputs: rangeInputs(form),
            insightEl: document.getElementById("simulation-insight"),
            suggestionEl: document.getElementById("simulation-range-suggestion"),
        });
        return features;
    }

    /** "Running SWMM — 11 runs for S1, Slope 5 → 15 %": what the status panel says. */
    function describeRun(form, features) {
        var option = form.elements.namedItem("option");
        var catchment = form.elements.namedItem("catchment_name").value;
        if (form.dataset.predefinedMethods.split(" ").indexOf(option.value) !== -1) {
            var entry = assist.guide[features[option.value]];
            var name = entry ? entry.name : option.value;
            return "Running SWMM — one run per literature value of " + name + " for " + catchment;
        }
        return assist.describeRun(catchment);
    }

    // ── Number formatting ───────────────────────────────────────────────

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

    var parameter = CS.format.parameter;

    /** "4 %", "2.5 ha", "0.011": value and unit, held together by non-breaking spaces. */
    function withUnit(text, unit) {
        return unit ? text + "\u00a0" + unit : text;
    }

    /** "Percent Slope = 4 %": the swept value a figure refers to, as the table prints it. */
    function describeSwept(value, xLabel) {
        var label = splitLabel(xLabel);
        // Non-breaking spaces keep "= 10 %" together when the tile wraps.
        return withUnit(label.name + "\u00a0=\u00a0" + parameter(value), label.unit);
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

    function sortedRows(config) {
        var x = config.x;
        return finiteRows(config.data, x).sort(function (a, b) {
            return a[x] - b[x];
        });
    }

    /** The model's own value when the sweep covers it, else null. */
    function baselineInRange(config, rows) {
        var baseline = config.baseline;
        if (!Number.isFinite(baseline) || !rows.length) {
            return null;
        }
        var low = rows[0][config.x];
        var high = rows[rows.length - 1][config.x];
        var slack = SAME_TOLERANCE * Math.max(1, Math.abs(baseline));
        return baseline >= low - slack && baseline <= high + slack ? baseline : null;
    }

    // ── Reading line ────────────────────────────────────────────────────

    function same(a, b) {
        return Math.abs(a - b) <= SAME_TOLERANCE * Math.max(1, Math.abs(a), Math.abs(b));
    }

    /** "up", "down", "flat" (every value the same) or "mixed" (up and down). */
    function trend(values) {
        var up = false;
        var down = false;
        for (var i = 1; i < values.length; i += 1) {
            if (!same(values[i], values[i - 1])) {
                if (values[i] > values[i - 1]) {
                    up = true;
                } else {
                    down = true;
                }
            }
        }
        if (up && down) {
            return "mixed";
        }
        return up ? "up" : down ? "down" : "flat";
    }

    function percent(from, to) {
        return CS.format.number(Math.abs(((to - from) / Math.abs(from)) * 100), 1) + "\u00a0%";
    }

    /** "by 38.0 %" or, from zero, "from 0 to 12.3 m³". */
    function byHowMuch(from, to, unit) {
        if (from === 0) {
            return "from 0 to " + withUnit(formatValue(to), unit);
        }
        return "by " + percent(from, to);
    }

    function column(rows, field) {
        return rows.map(function (row) {
            return row[field];
        });
    }

    /**
     * The response of a sweep in one plain sentence, from a #chart-config payload.
     * `name` is the parameter's display name ("Slope"); it defaults to the axis label.
     */
    function reading(config, name) {
        var x = config.x;
        var yLabels = config.yLabels || {};
        var xLabel = splitLabel(config.xLabel);
        var label = name || xLabel.name;
        var volumeUnit = splitLabel(yLabels[RUNOFF]).unit;
        var flowUnit = splitLabel(yLabels[PEAK_RATE]).unit;
        var rows = finiteRows(sortedRows(config), RUNOFF);

        function at(row) {
            return withUnit(parameter(row[x]), xLabel.unit);
        }

        if (!rows.length) {
            return "SWMM returned no runoff volume for these runs.";
        }
        var first = rows[0];
        var last = rows[rows.length - 1];
        var peakRows = finiteRows(rows, PEAK_RATE);
        var hasPeak = peakRows.length === rows.length;

        if (rows.length === 1) {
            return "A single run at " + label + " = " + at(first) + ": the runoff volume is " +
                withUnit(formatValue(first[RUNOFF]), volumeUnit) +
                (hasPeak ? " and the peak rate " + withUnit(formatValue(first[PEAK_RATE]), flowUnit) : "") + ".";
        }

        var span = parameter(first[x]) + " to " + at(last);
        var runoffTrend = trend(column(rows, RUNOFF));
        var peakTrend = hasPeak ? trend(column(rows, PEAK_RATE)) : null;

        if (runoffTrend === "flat" && (peakTrend === "flat" || peakTrend === null)) {
            return "Changing " + label + " from " + span + " does not change the runoff: every run gives " +
                withUnit(formatValue(first[RUNOFF]), volumeUnit) +
                (hasPeak ? " at a peak rate of " + withUnit(formatValue(first[PEAK_RATE]), flowUnit) : "") + ".";
        }

        if (runoffTrend === "mixed") {
            var highest = rows.reduce(function (best, row) {
                return row[RUNOFF] > best[RUNOFF] ? row : best;
            });
            var lowest = rows.reduce(function (best, row) {
                return row[RUNOFF] < best[RUNOFF] ? row : best;
            });
            return "The runoff volume does not move steadily with " + label + ": it is highest at " + at(highest) +
                " (" + withUnit(formatValue(highest[RUNOFF]), volumeUnit) + ") and lowest at " + at(lowest) +
                " (" + withUnit(formatValue(lowest[RUNOFF]), volumeUnit) + ").";
        }

        var sentence = "Raising " + label + " from " + span;
        if (runoffTrend === "flat") {
            sentence += " leaves the runoff volume at " + withUnit(formatValue(first[RUNOFF]), volumeUnit);
        } else {
            sentence += (runoffTrend === "up" ? " increases" : " decreases") + " the runoff volume " +
                byHowMuch(first[RUNOFF], last[RUNOFF], volumeUnit);
            if (first[RUNOFF] !== 0) {
                sentence += " (" + formatValue(first[RUNOFF]) + " → " +
                    withUnit(formatValue(last[RUNOFF]), volumeUnit) + ")";
            }
        }

        if (peakTrend === "flat") {
            sentence += " and leaves the peak rate unchanged";
        } else if (peakTrend === "mixed") {
            sentence += ", while the peak rate changes unevenly";
        } else if (peakTrend !== null) {
            var peakBy = byHowMuch(first[PEAK_RATE], last[PEAK_RATE], flowUnit);
            if (peakTrend === runoffTrend) {
                sentence += " and the peak rate " + peakBy;
            } else {
                sentence += ", while the peak rate " + (peakTrend === "up" ? "rises " : "falls ") + peakBy;
            }
        }

        var infiltrationRows = finiteRows(rows, INFILTRATION);
        if (infiltrationRows.length === rows.length) {
            var infiltrationTrend = trend(column(rows, INFILTRATION));
            if (infiltrationTrend === "up" || infiltrationTrend === "down") {
                sentence += "; infiltration " + (infiltrationTrend === "up" ? "rises " : "falls ") +
                    byHowMuch(first[INFILTRATION], last[INFILTRATION], volumeUnit);
            }
        }
        return sentence + ".";
    }

    // ── Key figures ─────────────────────────────────────────────────────

    /** Tiles for the outputs present in the data, rows ordered by the swept value. */
    function keyFigures(config) {
        var x = config.x;
        var yLabels = config.yLabels || {};
        var rows = sortedRows(config);
        var tiles = [];

        var runoffRows = finiteRows(rows, RUNOFF);
        var runoffUnit = splitLabel(yLabels[RUNOFF]).unit;
        if (runoffRows.length > 1) {
            var lowest = runoffRows[0];
            var highest = runoffRows[runoffRows.length - 1];
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

        var baseline = baselineInRange(config, rows);
        if (baseline !== null) {
            var match = runoffRows.filter(function (row) {
                return same(row[x], baseline);
            })[0];
            tiles.push({
                label: MODEL_VALUE_LABEL,
                value: parameter(baseline),
                unit: splitLabel(config.xLabel).unit,
                note: match
                    ? "Runoff there: " + withUnit(formatValue(match[RUNOFF]), runoffUnit)
                    : "Between two swept values",
            });
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

    function chartLabel(config, baseline) {
        var outputs = config.y.map(function (field) {
            return splitLabel((config.yLabels || {})[field] || field).name;
        });
        var marker = baseline === null ? "" : " A dashed line marks the model's own value, " + parameter(baseline) + ".";
        return "Line charts of " + outputs.join(", ") + " against " + config.xLabel + "." + marker +
            " The results table lists the same values.";
    }

    function reveal(button, isAvailable) {
        if (button) {
            button.hidden = !isAvailable;
        }
    }

    function initReading(root, config) {
        var line = root.querySelector("#simulation-reading");
        if (!line) {
            return;
        }
        var name = line.dataset.parameterName;
        // A name with a comma ("Manning's n, impervious") would read as two clauses.
        line.textContent = reading(config, name && name.indexOf(",") === -1 ? name : "");
        line.hidden = false;
    }

    function balanceMode(root) {
        var checked = root.querySelector("[name='balance-mode']:checked");
        return checked ? checked.value : "share";
    }

    function drawBalance(root, config) {
        var chart = root.querySelector("#simulation-balance-chart");
        var balance = config.balance;
        if (!chart || !balance || !Array.isArray(balance.rows) || !balance.rows.length) {
            return null;
        }
        var mode = balanceMode(root);
        var button = root.querySelector("[data-sim-action='download-balance-png']");
        return CS.charts.balance(chart, balance.rows, {
            mode: mode,
            unit: balance.unit,
            xLabel: config.xLabel,
            ariaLabel: "Stacked bars for each run against " + config.xLabel + ": runoff, infiltration, " +
                "evaporation and the water still on the surface, " +
                (mode === "share" ? "as a share of the rainfall and runon" : "as volumes in " + balance.unit) +
                ". The water balance table lists the volumes.",
            fileName: button ? button.dataset.filename : "water_balance",
        });
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
        var baseline = baselineInRange(config, sortedRows(config));
        initReading(root, config);
        renderKeyFigures(root.querySelector("#simulation-metrics"), keyFigures(config));
        var pngButton = root.querySelector("[data-sim-action='download-png']");
        var drawn = CS.charts.line(root.querySelector("#simulation-chart"), config.data, config.x, config.y, {
            xLabel: config.xLabel,
            yLabels: config.yLabels,
            reference: baseline === null ? null : { x: baseline, label: MODEL_VALUE_LABEL },
            ariaLabel: chartLabel(config, baseline),
            fileName: pngButton.dataset.filename,
        });
        reveal(pngButton, drawn !== null);

        var balanceDrawn = drawBalance(root, config);
        reveal(root.querySelector("[data-sim-balance-mode]"), balanceDrawn !== null);
        reveal(root.querySelector("[data-sim-action='download-balance-png']"), balanceDrawn !== null);
    }

    var ACTIONS = {
        "copy-table": function (canvas) {
            CS.clipboard.copy(CS.table.toCSV(canvas.querySelector("#simulation-table")), "Table copied as CSV.");
        },
        "download-png": function (canvas, button) {
            CS.charts.downloadPng(canvas.querySelector("#simulation-chart"), button.dataset.filename);
        },
        "download-balance-png": function (canvas, button) {
            CS.charts.downloadPng(canvas.querySelector("#simulation-balance-chart"), button.dataset.filename);
        },
    };

    function initActions(canvas) {
        canvas.addEventListener("click", function (event) {
            var button = event.target.closest("[data-sim-action]");
            if (button && ACTIONS[button.dataset.simAction]) {
                ACTIONS[button.dataset.simAction](canvas, button);
            }
        });
        canvas.addEventListener("change", function (event) {
            if (event.target.name === "balance-mode") {
                var config = readConfig(canvas);
                if (config) {
                    drawBalance(canvas, config);
                }
            }
        });
    }

    CS.simulation = { reading: reading };

    document.addEventListener("DOMContentLoaded", function () {
        var form = document.getElementById("simulation-form");
        var canvas = document.getElementById("simulation-canvas");
        if (!form || !canvas) {
            return;
        }
        initModelHint();
        initRange(form);
        var features = initAssist(form);
        initActions(canvas);
        initResults(canvas);

        CS.asyncForm({
            form: form,
            submitButton: document.getElementById("run-simulation-button"),
            target: canvas,
            loadingEl: document.getElementById("simulation-loading-state"),
            successMessage: "Simulation finished",
            onSuccess: initResults,
            runStatus: function () {
                return describeRun(form, features);
            },
        });
    });
})(window.CS);
