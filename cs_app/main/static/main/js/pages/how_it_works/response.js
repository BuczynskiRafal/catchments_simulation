/*
 * CS.howItWorks.response(container) — the response curve of a sweep in the
 * interactive model: one output of every run (runoff volume or peak rate)
 * against the swept value, on a log axis for log sliders. Points take the
 * colours of their hydrographs (the sequential ramp, value order = colour order).
 *
 * set(sweep)        {values, volumes, peaks, shown, log, name, unit, start};
 *                   null shows the empty state
 * setQuantity(key)  "volume" or "peak"
 */
(function (CS) {
    "use strict";

    var svg = CS.howItWorks.svg;
    var format = CS.howItWorks.format;

    var MARGIN = { top: 14, right: 16, bottom: 42, left: 58 };
    var HEIGHT = 224;
    var QUANTITIES = {
        volume: { field: "volumes", title: "Runoff volume [m³]", text: format.volume },
        peak: {
            field: "peaks",
            title: "Peak runoff [m³/s]",
            text: function (value) {
                return format.significant(value, 3);
            },
        },
    };

    function response(container) {
        var root = null;
        var parts = null;
        var g = null;
        var data = null;
        var quantity = "volume";

        function build() {
            var width = Math.round(container.clientWidth);
            if (!width) {
                return false;
            }
            g = {
                width: width,
                left: MARGIN.left,
                right: width - MARGIN.right,
                top: MARGIN.top,
                bottom: HEIGHT - MARGIN.bottom,
            };
            root = svg.canvas(container, "hiw-chart hiw-chart--response");
            svg.size(root, width, HEIGHT);
            parts = {};
            parts.grid = svg.el("path", { class: "hiw-chart__grid" }, root);
            parts.axes = svg.el("path", {
                class: "hiw-chart__axis",
                d: "M" + g.left + " " + g.top + "V" + g.bottom + "H" + g.right,
            }, root);
            parts.marks = svg.el("path", { class: "hiw-chart__axis" }, root);
            parts.start = svg.el("path", { class: "hiw-chart__start" }, root);
            parts.startText = svg.el("text", { class: "hiw-chart__start-text", "text-anchor": "middle", y: g.top + 9 }, root, "current");
            parts.line = svg.el("path", { class: "hiw-chart__response-line" }, root);
            parts.points = svg.el("g", {}, root);
            parts.xTicks = svg.el("g", {}, root);
            parts.yTicks = svg.el("g", {}, root);
            parts.xTitle = svg.el("text", { class: "hiw-chart__title", x: (g.left + g.right) / 2, y: HEIGHT - 6, "text-anchor": "middle" }, root);
            parts.yTitle = svg.el("text", {
                class: "hiw-chart__title", "text-anchor": "middle",
                transform: "translate(14 " + (g.top + g.bottom) / 2 + ") rotate(-90)",
            }, root);
            parts.empty = svg.el("text", {
                class: "hiw-chart__empty", x: (g.left + g.right) / 2, y: (g.top + g.bottom) / 2, "text-anchor": "middle",
            }, root, "Run a sweep to draw the response.");
            return true;
        }

        function ticksText(group, values, attributes) {
            group.replaceChildren();
            values.forEach(function (tick) {
                var values = Object.assign({ class: "hiw-chart__tick" }, attributes(tick));
                svg.el("text", values, group, tick.text);
            });
        }

        function apply() {
            if (!parts) {
                return;
            }
            var show = Boolean(data);
            [parts.start, parts.startText, parts.line, parts.points, parts.xTicks, parts.yTicks, parts.grid, parts.marks].forEach(function (element) {
                element.setAttribute("visibility", show ? "visible" : "hidden");
            });
            parts.empty.setAttribute("visibility", show ? "hidden" : "visible");
            var spec = QUANTITIES[quantity];
            parts.yTitle.textContent = spec.title;
            if (!show) {
                parts.points.replaceChildren();
                parts.line.setAttribute("d", "");
                parts.xTitle.textContent = "Parameter value";
                return;
            }
            parts.xTitle.textContent = data.name + (data.unit ? " [" + data.unit + "]" : "");

            var values = data.values;
            var low = values[0];
            var high = values[values.length - 1];
            var x = svg.scale([low, high], [g.left + 10, g.right - 10], data.log);
            var outputs = data[spec.field];
            var top = Math.max.apply(null, outputs) || 1;
            var y = svg.scale([0, top * 1.15], [g.bottom, g.top]);

            // Ticks: the ends of the sweep and nice values between them.
            var xValues = (data.log ? svg.logTicks(low, high) : svg.ticks(low, high, 4));
            var most = Math.max(2, Math.floor((g.right - g.left) / 56));
            while (xValues.length > most) {
                xValues = xValues.filter(function (value, i) {
                    return i % 2 === 0;
                });
            }
            var xTicks = xValues.map(function (value) {
                return { at: x(value), text: format.parameter(value) };
            });
            var yTicks = svg.ticks(0, top * 1.1, 4).map(function (value) {
                return { at: y(value), text: spec.text(value), value: value };
            });
            ticksText(parts.xTicks, xTicks, function (tick) {
                return { x: svg.px(tick.at), y: g.bottom + 16, "text-anchor": "middle" };
            });
            ticksText(parts.yTicks, yTicks, function (tick) {
                return { x: g.left - 7, y: svg.px(tick.at + 4), "text-anchor": "end" };
            });
            var grid = "";
            var marks = "";
            yTicks.forEach(function (tick) {
                if (tick.value > 0) {
                    grid += "M" + g.left + " " + svg.px(tick.at) + "H" + g.right;
                }
                marks += "M" + (g.left - 4) + " " + svg.px(tick.at) + "h4";
            });
            xTicks.forEach(function (tick) {
                marks += "M" + svg.px(tick.at) + " " + g.bottom + "v4";
            });
            parts.grid.setAttribute("d", grid);
            parts.marks.setAttribute("d", marks);

            var startX = svg.px(x(Math.min(Math.max(data.start, low), high)));
            parts.start.setAttribute("d", "M" + startX + " " + (g.top + 14) + "V" + g.bottom);
            parts.startText.setAttribute("x", startX);

            var points = [];
            parts.points.replaceChildren();
            for (var i = 0; i < data.shown; i++) {
                var point = [x(values[i]), y(outputs[i])];
                points.push(point);
                svg.el("circle", {
                    class: "hiw-chart__point hiw-ramp-" + Math.round((8 * i) / Math.max(values.length - 1, 1)),
                    cx: svg.px(point[0]),
                    cy: svg.px(point[1]),
                    r: 4.5,
                }, parts.points);
            }
            parts.line.setAttribute("d", points.length > 1 ? svg.line(points) : "");
        }

        if (window.ResizeObserver) {
            var lastWidth = 0;
            new ResizeObserver(function () {
                var width = Math.round(container.clientWidth);
                if (width && width !== lastWidth) {
                    lastWidth = width;
                    if (build()) {
                        apply();
                    }
                }
            }).observe(container);
        }

        return {
            set: function (next) {
                data = next;
                if (!parts) {
                    build();
                }
                apply();
            },
            setQuantity: function (key) {
                quantity = key;
                apply();
            },
        };
    }

    CS.howItWorks.response = response;
})(window.CS);
