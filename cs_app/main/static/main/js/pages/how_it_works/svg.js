/*
 * CS.howItWorks.svg — helpers for the hand-drawn SVG figures of the interactive
 * model on the "How it works" page: element creation, scales and axis ticks.
 *
 * The figures are sized in CSS pixels (viewBox = measured width × height), so
 * their text keeps its real size at every width; they are rebuilt only when
 * their container is resized and otherwise updated attribute by attribute.
 */
(function (CS) {
    "use strict";

    var NS = "http://www.w3.org/2000/svg";

    /** Set attributes; a null or undefined value removes the attribute. */
    function attrs(element, values) {
        Object.keys(values).forEach(function (name) {
            var value = values[name];
            if (value === null || value === undefined) {
                element.removeAttribute(name);
            } else {
                element.setAttribute(name, value);
            }
        });
        return element;
    }

    /** Create an SVG element with the given attributes (and text), appended to `parent` if given. */
    function el(tag, values, parent, text) {
        var element = attrs(document.createElementNS(NS, tag), values || {});
        if (text !== undefined) {
            element.textContent = text;
        }
        if (parent) {
            parent.appendChild(element);
        }
        return element;
    }

    /** An <svg> filling `container`, its viewBox in CSS pixels. */
    function canvas(container, className) {
        var svg = el("svg", { class: className, "aria-hidden": "true", focusable: "false" });
        container.replaceChildren(svg);
        return svg;
    }

    function size(svg, width, height) {
        attrs(svg, { viewBox: "0 0 " + width + " " + height, width: width, height: height });
    }

    /** Coordinates rounded to 0.1 px keep path data short. */
    function px(value) {
        return Math.round(value * 10) / 10;
    }

    /**
     * Linear map from [d0, d1] to [r0, r1]; `scale.invert` maps back.
     * With `log: true` the domain is mapped logarithmically (both ends > 0).
     */
    function scale(domain, range, log) {
        var transform = log ? Math.log : function (value) {
            return value;
        };
        var d0 = transform(domain[0]);
        var d1 = transform(domain[1]);
        var span = d1 - d0 || 1;
        function map(value) {
            return range[0] + ((transform(value) - d0) / span) * (range[1] - range[0]);
        }
        map.invert = function (position) {
            var value = d0 + ((position - range[0]) / (range[1] - range[0])) * span;
            return log ? Math.exp(value) : value;
        };
        return map;
    }

    /** A 1-2-5 step that splits `span` into about `count` intervals. */
    function niceStep(span, count) {
        var raw = span / Math.max(count, 1);
        var magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
        var steps = [1, 2, 2.5, 5, 10];
        for (var i = 0; i < steps.length; i++) {
            if (raw <= steps[i] * magnitude) {
                return steps[i] * magnitude;
            }
        }
        return 10 * magnitude;
    }

    /** Multiples of a nice step from 0 (or `min`) up to `max`. */
    function ticks(min, max, count) {
        if (!(max > min)) {
            return [min];
        }
        var step = niceStep(max - min, count);
        var values = [];
        for (var value = Math.ceil(min / step - 1e-9) * step; value <= max + step * 1e-9; value += step) {
            values.push(Number(value.toPrecision(12)));
        }
        return values;
    }

    /** 1-2-5 values between min and max (both > 0), for a logarithmic axis. */
    function logTicks(min, max) {
        var values = [];
        for (var exponent = Math.floor(Math.log10(min)); exponent <= Math.ceil(Math.log10(max)); exponent++) {
            [1, 2, 5].forEach(function (factor) {
                var value = Number((factor * Math.pow(10, exponent)).toPrecision(6));
                if (value >= min * (1 - 1e-9) && value <= max * (1 + 1e-9)) {
                    values.push(value);
                }
            });
        }
        return values;
    }

    /** Path data of a polyline through [x, y] points. */
    function line(points) {
        return points.map(function (point, i) {
            return (i ? "L" : "M") + px(point[0]) + " " + px(point[1]);
        }).join("");
    }

    CS.howItWorks = CS.howItWorks || {};
    CS.howItWorks.svg = {
        attrs: attrs,
        el: el,
        canvas: canvas,
        size: size,
        px: px,
        scale: scale,
        ticks: ticks,
        logTicks: logTicks,
        line: line,
    };
})(window.CS);
