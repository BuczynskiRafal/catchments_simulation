/*
 * CS.format.number(value, digits) — locale-stable display of numeric results.
 * CS.format.parameter(value) — a swept value as the server prints it (Python's
 * f"{value:.10g}"): 0.30000000000000004 → "0.3", 5.0 → "5".
 * CS.format.runCount(start, stop, step) — how many runs a sweep range makes, as the
 * package's build_range counts them; NaN for an incomplete or impossible range.
 * CS.format.sweepRange(start, stop, step, maxRuns) — the live run-count hint of a
 * parameter sweep form (simulation and timeseries pages).
 */
(function (CS) {
    "use strict";

    var formatters = {};

    function formatterFor(digits) {
        if (!formatters[digits]) {
            formatters[digits] = new Intl.NumberFormat("en-US", {
                minimumFractionDigits: digits,
                maximumFractionDigits: digits,
            });
        }
        return formatters[digits];
    }

    function parameter(value) {
        return String(Number(Number(value).toPrecision(10)));
    }

    /** start, start + step, … up to stop, never past it (1e-9 absorbs floating-point noise). */
    function runCount(start, stop, step) {
        if (![start, stop, step].every(Number.isFinite) || step <= 0 || start > stop) {
            return NaN;
        }
        return Math.floor((stop - start) / step + 1e-9) + 1;
    }

    CS.format = {
        parameter: parameter,
        runCount: runCount,

        /**
         * Format `value` with exactly `digits` decimals (default 2) and
         * thousands separators. Non-numeric input yields an em dash.
         */
        number: function (value, digits) {
            var number = typeof value === "number" ? value : parseFloat(value);
            if (!Number.isFinite(number)) {
                return "—";
            }
            return formatterFor(digits === undefined ? 2 : digits).format(number);
        },

        /**
         * Describe a sweep range as "N runs · start → stop, step s", or say what is
         * wrong with it. The server remains authoritative; this only warns early.
         *
         * @returns {{text: string, warning: boolean}}
         */
        sweepRange: function (start, stop, step, maxRuns) {
            if (![start, stop, step].every(Number.isFinite)) {
                return { text: "Enter start, stop and step to see the number of runs.", warning: false };
            }
            if (step <= 0) {
                return { text: "Step must be greater than zero.", warning: true };
            }
            if (start > stop) {
                return { text: "Stop must be greater than or equal to start.", warning: true };
            }
            var runs = runCount(start, stop, step);
            var summary = runs + (runs === 1 ? " run" : " runs") + " · " + parameter(start) + " → " +
                parameter(stop) + ", step " + parameter(step);
            // The forms' own limit on the number of runs.
            if (runs > maxRuns) {
                return {
                    text: summary + ". That is over the limit of " + maxRuns +
                        " runs: increase the step or narrow the range.",
                    warning: true,
                };
            }
            return { text: summary, warning: false };
        },
    };
})(window.CS);
