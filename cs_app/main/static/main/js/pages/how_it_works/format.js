/*
 * CS.howItWorks.format — how the interactive model on the "How it works" page
 * prints its numbers: parameter values, rates, volumes, clock times and changes.
 */
(function (CS) {
    "use strict";

    var MINUS = "−";
    var UNCHANGED = "no change";

    /** Shortest text of a value at three significant digits: 0.0112, 5, 113, 1000. */
    function parameter(value) {
        return String(Number(value.toPrecision(3)));
    }

    /** `digits` significant digits, trailing zeros kept, thousands grouped: 0.193, 0.0270, 1.24. */
    function significant(value, digits) {
        if (!Number.isFinite(value)) {
            return "—";
        }
        if (value === 0) {
            return "0";
        }
        var rounded = Number(value.toPrecision(digits));
        var decimals = Math.max(0, digits - 1 - Math.floor(Math.log10(Math.abs(rounded))));
        return CS.format.number(rounded, decimals);
    }

    /** A volume in whole cubic metres: 3,347. */
    function volume(value) {
        return CS.format.number(value, 0);
    }

    /** Clock time from the start of the storm: 8:05. */
    function clock(seconds) {
        var minutes = Math.round(seconds / 60);
        var rest = minutes % 60;
        return Math.floor(minutes / 60) + ":" + (rest < 10 ? "0" : "") + rest;
    }

    /** A duration as the tools print it, leaving out zero parts: 8 h 55 min, 9 h, 45 min. */
    function duration(seconds) {
        var minutes = Math.round(Math.abs(seconds) / 60);
        var parts = [];
        if (minutes >= 60) {
            parts.push(Math.floor(minutes / 60) + " h");
        }
        if (minutes % 60 || !parts.length) {
            parts.push((minutes % 60) + " min");
        }
        return parts.join(" ");
    }

    function sign(value) {
        return value > 0 ? "+" : MINUS;
    }

    /** Relative change in per cent with its sign: +12.4 %, −3.0 %, or "no change"; null without a reference. */
    function change(from, to) {
        if (!from) {
            return null;
        }
        var percent = ((to - from) / Math.abs(from)) * 100;
        if (Math.abs(percent) < 0.05) {
            return UNCHANGED;
        }
        return sign(percent) + CS.format.number(Math.abs(percent), 1) + " %";
    }

    /** A signed difference of two durations: +15 min, −1 h 5 min, or "no change". */
    function durationChange(from, to) {
        var difference = Math.round((to - from) / 60) * 60;
        return difference ? sign(difference) + duration(difference) : UNCHANGED;
    }

    /** A signed difference in percentage points: +5.2 points, or "no change". */
    function pointChange(from, to) {
        var difference = to - from;
        if (Math.abs(difference) < 0.05) {
            return UNCHANGED;
        }
        return sign(difference) + CS.format.number(Math.abs(difference), 1) + " points";
    }

    CS.howItWorks = CS.howItWorks || {};
    CS.howItWorks.format = {
        parameter: parameter,
        significant: significant,
        volume: volume,
        clock: clock,
        duration: duration,
        change: change,
        durationChange: durationChange,
        pointChange: pointChange,
    };
})(window.CS);
