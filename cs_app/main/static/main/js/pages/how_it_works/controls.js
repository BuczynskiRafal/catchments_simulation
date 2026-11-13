/*
 * CS.howItWorks.controls — the parameter sliders of the interactive model and
 * its starting values from the query string.
 *
 * Each slider is a native range input over integer positions 0…max; its
 * .hiw-slider wrapper says how positions map to values (data-scale linear or
 * log, data-min, data-max, data-unit). A log mapping keeps the starting value
 * near mid-track and small values reachable on wide ranges. The model state,
 * not the slider, holds the exact value: a value arriving from the query
 * string or a reset is kept as given, and only a drag snaps it to a position
 * (three significant digits). A starting value outside the range extends it,
 * within the bounds the query string is clamped to.
 */
(function (CS) {
    "use strict";

    var format = CS.howItWorks.format;

    // The range a starting value from the query string is clamped to, per parameter (SI units).
    // Wider than the sliders, which widen to take a value in it, but physically sensible.
    var BOUNDS = {
        area: [0.001, 10000], // ha
        imperviousness: [0, 100], // %
        width: [0.1, 100000], // m
        slope: [0.001, 100], // %
        nImperv: [0.001, 5],
        nPerv: [0.001, 5],
        storageImperv: [0, 100], // mm
        storagePerv: [0, 100], // mm
        zeroImperv: [0, 100], // %
        conductivity: [0.01, 1000], // mm/h
    };

    function Slider(element) {
        this.element = element;
        this.key = element.dataset.hiwParam;
        this.input = element.querySelector("input[type='range']");
        this.output = element.querySelector("output");
        this.tick = element.querySelector(".hiw-slider__tick");
        this.bounds = element.querySelectorAll(".hiw-slider__bound");
        this.log = element.dataset.scale === "log";
        this.min = Number(element.dataset.min);
        this.max = Number(element.dataset.max);
        this.unit = element.dataset.unit || "";
        this.steps = Number(this.input.max);
    }

    /** The value at a (possibly fractional) position, at three significant digits. */
    Slider.prototype.valueAt = function (position) {
        var t = Math.min(Math.max(position / this.steps, 0), 1);
        var value = this.log
            ? this.min * Math.pow(this.max / this.min, t)
            : this.min + t * (this.max - this.min);
        return Number(value.toPrecision(3));
    };

    /** Fractional position of a value on the track (0…1). */
    Slider.prototype.fractionOf = function (value) {
        var t = this.log
            ? Math.log(value / this.min) / Math.log(this.max / this.min)
            : (value - this.min) / (this.max - this.min);
        return Math.min(Math.max(t, 0), 1);
    };

    Slider.prototype.text = function (value) {
        return format.parameter(value) + (this.unit ? " " + this.unit : "");
    };

    /** Widen the range so that `value` lies on it. */
    Slider.prototype.include = function (value) {
        this.min = Math.min(this.min, value);
        this.max = Math.max(this.max, value);
        this.bounds[0].textContent = format.parameter(this.min);
        this.bounds[1].textContent = this.text(this.max);
    };

    /** Show `value` (thumb, readout and the value screen readers announce). */
    Slider.prototype.show = function (value) {
        var text = this.text(value);
        this.input.value = String(Math.round(this.fractionOf(value) * this.steps));
        this.input.setAttribute("aria-valuetext", text);
        this.output.textContent = text;
    };

    /** Place the starting-value tick under the track. */
    Slider.prototype.markStart = function (value) {
        this.element.style.setProperty("--hiw-start", String(this.fractionOf(value)));
    };

    /** `count` values spread evenly over the track (log-spaced on a log slider). */
    Slider.prototype.spread = function (count) {
        var values = [];
        for (var i = 0; i < count; i++) {
            values.push(this.valueAt((i / (count - 1)) * this.steps));
        }
        return values;
    };

    /** Slider objects of every .hiw-slider under root, keyed by parameter. */
    function sliders(root) {
        var result = {};
        root.querySelectorAll(".hiw-slider").forEach(function (element) {
            var slider = new Slider(element);
            slider.include(slider.min);
            result[slider.key] = slider;
        });
        return result;
    }

    /**
     * Starting values: `defaults` overridden by the query-string keys, and the
     * optional `vary` feature name. Keys and values follow CS.runoffModel.DEFAULTS
     * (SI units); a number is clamped to the parameter's BOUNDS, and anything
     * missing or not a number falls back to the default.
     */
    function startingValues(search, defaults) {
        var query = new URLSearchParams(search);
        var values = {};
        Object.keys(BOUNDS).forEach(function (key) {
            var raw = query.get(key);
            var value = raw === null || raw.trim() === "" ? NaN : Number(raw);
            values[key] = Number.isFinite(value)
                ? Math.min(Math.max(value, BOUNDS[key][0]), BOUNDS[key][1])
                : defaults[key];
        });
        return { values: values, vary: query.get("vary") };
    }

    /** Query string of the values that differ from the defaults, plus `vary`. */
    function queryString(values, defaults, vary) {
        var query = new URLSearchParams();
        Object.keys(BOUNDS).forEach(function (key) {
            if (values[key] !== defaults[key]) {
                query.set(key, String(values[key]));
            }
        });
        if (vary) {
            query.set("vary", vary);
        }
        var text = query.toString();
        return text ? "?" + text : "";
    }

    CS.howItWorks.controls = {
        KEYS: Object.keys(BOUNDS),
        sliders: sliders,
        startingValues: startingValues,
        queryString: queryString,
    };
})(window.CS);
