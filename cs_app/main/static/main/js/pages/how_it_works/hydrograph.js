/*
 * CS.howItWorks.hydrograph(container, options) — the hyetograph + hydrograph of
 * the interactive model, in the visual language of CS.charts.hydrograph: rain
 * as bars hanging from the top axis, runoff as a line below, the peak marked.
 * It is plain SVG rather than Plotly so it stays in lock-step with the
 * animation and takes its colours from the theme's CSS variables.
 *
 * setRuns(result, baseline, storm)  the run to draw; a baseline (the starting
 *                                   values) is drawn as a dashed ghost line
 * setSweep(sweep)                   {runs, shown} draws one line per sweep run
 *                                   on the sequential ramp, or null to clear
 * setCursor(index)                  the time step under the cursor
 *
 * Pointer drags on the chart call options.onSeek(index); the page's time slider
 * gives the same control from the keyboard.
 */
(function (CS) {
    "use strict";

    var svg = CS.howItWorks.svg;
    var format = CS.howItWorks.format;

    // Same proportions as CS.charts.hydrograph: the rain bars fill the top ~40 %,
    // the runoff lines the lower ~60 %.
    var RAIN_AXIS_FACTOR = 2.5;
    var FLOW_AXIS_FACTOR = 1.65;
    var NARROW_WIDTH = 520;
    var MARGIN = { top: 12, right: 50, bottom: 40, left: 54 };
    var HOUR_STEPS = [0.25, 0.5, 1, 2, 3, 4, 6, 12, 24];
    var MONO_ADVANCE = 0.6; // IBM Plex Mono advance width, em
    var LABEL_SIZE = 11;

    function hydrograph(container, options) {
        var root = null;
        var parts = null;
        var g = null; // geometry and scales
        var run = null;
        var baseline = null;
        var storm = null;
        var sweep = null;
        var index = 0;

        // ── Static drawing (rebuilt on resize) ──────────────────────────

        function build() {
            var width = Math.round(container.clientWidth);
            if (!width) {
                return false;
            }
            var height = width < NARROW_WIDTH ? 248 : 300;
            g = {
                width: width,
                height: height,
                left: MARGIN.left,
                right: width - MARGIN.right,
                top: MARGIN.top,
                bottom: height - MARGIN.bottom,
                narrow: width < NARROW_WIDTH,
            };
            root = svg.canvas(container, "hiw-chart");
            svg.size(root, width, height);

            parts = {};
            parts.grid = svg.el("path", { class: "hiw-chart__grid" }, root);
            parts.rain = svg.el("path", { class: "hiw-chart__rain" }, root);
            parts.rainNow = svg.el("rect", { class: "hiw-chart__rain-now" }, root);
            parts.axes = svg.el("path", {
                class: "hiw-chart__axis",
                d: "M" + g.left + " " + g.top + "V" + g.bottom + "H" + g.right + "V" + g.top,
            }, root);
            parts.sweep = svg.el("g", { class: "hiw-chart__sweep" }, root);
            parts.ghost = svg.el("path", { class: "hiw-chart__ghost" }, root);
            parts.runoff = svg.el("path", { class: "hiw-chart__runoff" }, root);

            parts.xTicks = pool(16, { class: "hiw-chart__tick", "text-anchor": "middle", y: g.bottom + 16 });
            parts.yTicks = pool(8, { class: "hiw-chart__tick", "text-anchor": "end", x: g.left - 7 });
            parts.rainTicks = pool(8, { class: "hiw-chart__tick", "text-anchor": "start", x: g.right + 7 });
            parts.tickMarks = svg.el("path", { class: "hiw-chart__axis" }, root);

            var middleY = (g.top + g.bottom) / 2;
            svg.el("text", { class: "hiw-chart__title", x: (g.left + g.right) / 2, y: g.height - 6, "text-anchor": "middle" }, root, "Time [h]");
            svg.el("text", {
                class: "hiw-chart__title", "text-anchor": "middle",
                transform: "translate(14 " + middleY + ") rotate(-90)",
            }, root, "Runoff [m³/s]");
            svg.el("text", {
                class: "hiw-chart__title", "text-anchor": "middle",
                transform: "translate(" + (g.width - 10) + " " + middleY + ") rotate(90)",
            }, root, "Rainfall [mm/h]");

            parts.peak = svg.el("g", { class: "hiw-chart__peak" }, root);
            parts.peakLeader = svg.el("path", { class: "hiw-chart__peak-leader" }, parts.peak);
            parts.peakBox = svg.el("rect", { class: "hiw-chart__peak-box", rx: 3, height: 19 }, parts.peak);
            parts.peakText = svg.el("text", { class: "hiw-chart__peak-text" }, parts.peak);
            parts.peakMark = svg.el("path", { class: "hiw-chart__peak-mark" }, parts.peak);

            parts.cursor = svg.el("path", { class: "hiw-chart__cursor" }, root);
            parts.cursorDot = svg.el("circle", { class: "hiw-chart__cursor-dot", r: 4.5 }, root);
            return true;
        }

        function pool(count, values) {
            var group = svg.el("g", {}, root);
            var items = [];
            for (var i = 0; i < count; i++) {
                items.push(svg.el("text", values, group));
            }
            return items;
        }

        function fillPool(items, values, place) {
            items.forEach(function (item, i) {
                if (i < values.length) {
                    item.textContent = values[i].text;
                    item.setAttribute(place, svg.px(values[i].at));
                } else {
                    item.textContent = "";
                }
            });
        }

        // ── Per run ─────────────────────────────────────────────────────

        function series(result) {
            return result.time.map(function (time, k) {
                return [g.x(time), g.flow(result.runoff[k])];
            });
        }

        function rainBars() {
            var bars = [];
            var duration = run.time[run.time.length - 1];
            storm.rainfall.forEach(function (point, j) {
                var end = j + 1 < storm.rainfall.length ? storm.rainfall[j + 1].time : duration;
                if (point.intensity > 0 && end > point.time) {
                    bars.push({ start: point.time, end: Math.min(end, duration), intensity: point.intensity });
                }
            });
            return bars;
        }

        function timeTicks(duration) {
            var hours = duration / 3600;
            var most = Math.max(3, Math.floor((g.right - g.left) / (g.narrow ? 46 : 64)));
            var step = HOUR_STEPS.filter(function (candidate) {
                return hours / candidate <= most;
            })[0] || hours;
            var values = [];
            for (var hour = 0; hour <= hours + 1e-9; hour += step) {
                values.push({ text: format.parameter(hour), at: g.x(hour * 3600) });
            }
            return values;
        }

        function applyRuns() {
            if (!run || !parts) {
                return;
            }
            var duration = run.time[run.time.length - 1];
            var peakRain = storm.rainfall.reduce(function (most, point) {
                return Math.max(most, point.intensity);
            }, 0) || 1;
            var shownRuns = sweep ? sweep.runs : [run].concat(baseline ? [baseline] : []);
            var peakFlow = shownRuns.reduce(function (most, result) {
                return Math.max(most, result.peak.runoff);
            }, 0);
            var flowMax = peakFlow > 0 ? peakFlow * FLOW_AXIS_FACTOR : 1;

            g.x = svg.scale([0, duration], [g.left, g.right]);
            g.flow = svg.scale([0, flowMax], [g.bottom, g.top]);
            g.rain = svg.scale([0, peakRain * RAIN_AXIS_FACTOR], [g.top, g.bottom]);
            g.timeStep = run.time[1] - run.time[0];
            g.bars = rainBars();

            // Rain: one path of hanging bars, a pixel apart.
            parts.rain.setAttribute("d", g.bars.map(function (bar) {
                var x0 = g.x(bar.start) + 0.5;
                var x1 = Math.max(g.x(bar.end) - 0.5, x0 + 1);
                return "M" + svg.px(x0) + " " + g.top + "H" + svg.px(x1) + "V" + svg.px(g.rain(bar.intensity)) + "H" + svg.px(x0) + "z";
            }).join(""));

            var xTicks = timeTicks(duration);
            var yTicks = svg.ticks(0, flowMax * 0.98, g.narrow ? 4 : 5).map(function (value) {
                return { text: format.parameter(value), at: g.flow(value) + 4, value: value };
            });
            // The rain axis is labelled only over the bars, so it stays clear of the runoff below.
            var rainTicks = svg.ticks(0, peakRain, 3).map(function (value) {
                return { text: format.parameter(value), at: g.rain(value) + 4, value: value };
            });
            fillPool(parts.xTicks, xTicks, "x");
            fillPool(parts.yTicks, yTicks, "y");
            fillPool(parts.rainTicks, rainTicks, "y");

            var grid = "";
            var marks = "";
            yTicks.forEach(function (tick) {
                var y = svg.px(g.flow(tick.value));
                if (tick.value > 0) {
                    grid += "M" + g.left + " " + y + "H" + g.right;
                }
                marks += "M" + (g.left - 4) + " " + y + "h4";
            });
            xTicks.forEach(function (tick) {
                var x = svg.px(tick.at);
                grid += "M" + x + " " + g.top + "V" + g.bottom;
                marks += "M" + x + " " + g.bottom + "v4";
            });
            rainTicks.forEach(function (tick) {
                marks += "M" + g.right + " " + svg.px(g.rain(tick.value)) + "h4";
            });
            parts.grid.setAttribute("d", grid);
            parts.tickMarks.setAttribute("d", marks);

            parts.runoff.setAttribute("d", sweep ? "" : svg.line(series(run)));
            parts.ghost.setAttribute("d", baseline && !sweep ? svg.line(series(baseline)) : "");
            applySweep();
            applyPeak();
            applyCursor();
        }

        function applyPeak() {
            var peak = run.peak;
            if (sweep || !(peak.runoff > 0)) {
                parts.peak.setAttribute("visibility", "hidden");
                return;
            }
            parts.peak.removeAttribute("visibility");
            var x = g.x(peak.time);
            var y = g.flow(peak.runoff);
            var text = "Peak " + format.significant(peak.runoff, 3) + " m³/s";
            var width = text.length * LABEL_SIZE * MONO_ADVANCE + 12;
            var toLeft = x + 18 + width > g.right;
            var boxX = toLeft ? x - 18 - width : x + 18;
            var boxY = Math.max(g.top + 2, y - 34);
            parts.peakMark.setAttribute("d", "M" + svg.px(x) + " " + svg.px(y - 6) + "l6 6-6 6-6-6z");
            parts.peakLeader.setAttribute("d", "M" + svg.px(x) + " " + svg.px(y) + "L" +
                svg.px(toLeft ? boxX + width : boxX) + " " + svg.px(boxY + 9.5));
            svg.attrs(parts.peakBox, { x: svg.px(boxX), y: svg.px(boxY), width: svg.px(width) });
            svg.attrs(parts.peakText, { x: svg.px(boxX + 6), y: svg.px(boxY + 13.5) });
            parts.peakText.textContent = text;
        }

        function applySweep() {
            var paths = parts.sweep.children;
            var runs = sweep ? sweep.runs : [];
            while (paths.length > runs.length) {
                parts.sweep.lastChild.remove();
            }
            runs.forEach(function (result, i) {
                var path = paths[i] || svg.el("path", {}, parts.sweep);
                path.setAttribute("class", "hiw-chart__sweep-line hiw-ramp-" + Math.round((8 * i) / Math.max(runs.length - 1, 1)));
                path.setAttribute("d", i < sweep.shown ? svg.line(series(result)) : "");
            });
        }

        // ── Per frame ───────────────────────────────────────────────────

        function applyCursor() {
            if (!run || !parts) {
                return;
            }
            var k = Math.min(index, run.time.length - 1);
            var time = run.time[k];
            var x = svg.px(g.x(time));
            parts.cursor.setAttribute("d", "M" + x + " " + g.top + "V" + g.bottom);
            if (sweep) {
                parts.cursorDot.setAttribute("visibility", "hidden");
            } else {
                parts.cursorDot.removeAttribute("visibility");
                svg.attrs(parts.cursorDot, { cx: x, cy: svg.px(g.flow(run.runoff[k])) });
            }
            // The rain bar the cursor is in, drawn stronger.
            var start = time - g.timeStep;
            var bar = g.bars.filter(function (candidate) {
                return candidate.start <= start + 1e-6 && start < candidate.end - 1e-6;
            })[0];
            if (bar && k > 0) {
                var x0 = g.x(bar.start) + 0.5;
                svg.attrs(parts.rainNow, {
                    visibility: null,
                    x: svg.px(x0),
                    y: g.top,
                    width: svg.px(Math.max(g.x(bar.end) - 0.5 - x0, 1)),
                    height: svg.px(g.rain(bar.intensity) - g.top),
                });
            } else {
                parts.rainNow.setAttribute("visibility", "hidden");
            }
        }

        function redraw() {
            if (build()) {
                applyRuns();
            }
        }

        // ── Interaction ─────────────────────────────────────────────────

        function lastIndex() {
            return run ? run.time.length - 1 : 0;
        }

        function seek(next) {
            options.onSeek(Math.min(Math.max(Math.round(next), 0), lastIndex()));
        }

        function seekToPointer(event) {
            if (!run || !g) {
                return;
            }
            var box = container.getBoundingClientRect();
            var x = ((event.clientX - box.left) / box.width) * g.width;
            seek(g.x.invert(x) / g.timeStep);
        }

        container.addEventListener("pointerdown", function (event) {
            if (event.button !== 0) {
                return;
            }
            container.setPointerCapture(event.pointerId);
            seekToPointer(event);
        });
        container.addEventListener("pointermove", function (event) {
            if (container.hasPointerCapture(event.pointerId)) {
                seekToPointer(event);
            }
        });
        if (window.ResizeObserver) {
            var lastWidth = 0;
            new ResizeObserver(function () {
                var width = Math.round(container.clientWidth);
                if (width && width !== lastWidth) {
                    lastWidth = width;
                    redraw();
                }
            }).observe(container);
        }

        return {
            setRuns: function (result, start, design) {
                run = result;
                baseline = start;
                storm = design;
                if (!parts) {
                    build();
                }
                applyRuns();
            },
            setSweep: function (next) {
                sweep = next;
                applyRuns();
            },
            setCursor: function (next) {
                index = next;
                applyCursor();
            },
        };
    }

    CS.howItWorks.hydrograph = hydrograph;
})(window.CS);
