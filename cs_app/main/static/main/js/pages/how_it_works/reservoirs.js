/*
 * CS.howItWorks.reservoirs(container) — the three sub-areas of the interactive
 * model drawn as reservoirs side by side: pervious, impervious, and impervious
 * without depression storage.
 *
 * For the current time step each one shows its ponded depth d, its depression
 * storage d_s (a dashed line and the lip the water must rise above; none on the
 * third), the rain falling in, infiltration leaving the pervious one through the
 * soil and the outflow over the lip into a common outlet with the total runoff Q.
 * All three share one square-root depth scale, so 1 mm and 40 mm both read;
 * the values are printed as well. A sub-area with no share of the area is drawn
 * as an empty dashed outline, so the layout never jumps while a slider moves.
 *
 * setRun(result, area) takes a CS.runoffModel.simulate() result (once per run),
 * setFrame(index) the time step to show (every animation frame).
 */
(function (CS) {
    "use strict";

    var svg = CS.howItWorks.svg;
    var format = CS.howItWorks.format;

    var SUBAREAS = [
        { key: "pervious", name: "Pervious", shortName: "Pervious" },
        { key: "impervious", name: "Impervious", shortName: "Impervious" },
        { key: "imperviousZero", name: "Impervious, no storage", shortName: "No storage" },
    ];
    var NARROW_WIDTH = 520;
    var NARROW_TANK = 96;
    // Rain streaks per reservoir; all of them show at RAIN_FULL mm/h. The order spreads
    // the first few across the width.
    var DROPS = 10;
    var DROP_ORDER = [4, 9, 1, 6, 2, 7, 0, 5, 3, 8];
    var RAIN_FULL = 50;
    // Infiltration arrows under the pervious reservoir; all show at INFILTRATION_FULL mm/h.
    var ARROWS = 3;
    var INFILTRATION_FULL = 10;
    var DEPTH_SCALES = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];
    var DEPTH_TICKS = [0, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000];
    var MIN_TICK_SPACING = 13;
    var MIN_LIP = 3;

    function layout(width) {
        var narrow = width < NARROW_WIDTH;
        var axis = 34;
        var gap = narrow ? 22 : 44;
        var tank = (width - axis - 3 * gap) / 3;
        return {
            width: width,
            height: 272,
            narrow: narrow,
            compact: tank < NARROW_TANK,
            axis: axis,
            gap: gap,
            tank: tank,
            nameY: 14,
            shareY: 31,
            storageY: 47,
            rainY: 58,
            top: 86,
            bottom: 198,
            channel: 242,
            outletY: 264,
            left: function (index) {
                return axis + index * (tank + gap);
            },
        };
    }

    function depthScaleMax(result) {
        var deepest = 1e-9;
        SUBAREAS.forEach(function (subarea) {
            var series = result.subareas[subarea.key];
            if (series.share > 0) {
                deepest = Math.max(deepest, series.storage, Math.max.apply(null, series.depth));
            }
        });
        for (var i = 0; i < DEPTH_SCALES.length; i++) {
            if (deepest <= DEPTH_SCALES[i]) {
                return DEPTH_SCALES[i];
            }
        }
        return Math.ceil(deepest / 1000) * 1000;
    }

    /**
     * A label "symbol[subscript] value unit" (symbol in italics); returns the
     * <tspan> that holds the value, the only part that changes afterwards.
     */
    function label(parent, values, symbol, subscript, unit) {
        var text = svg.el("text", values, parent);
        svg.el("tspan", { class: "hiw-res__symbol" }, text, symbol);
        if (subscript) {
            svg.el("tspan", { class: "hiw-res__sub", dy: 3 }, text, subscript);
        }
        var value = svg.el("tspan", subscript ? { dy: -3 } : {}, text);
        if (unit) {
            text.appendChild(document.createTextNode(unit));
        }
        return value;
    }

    function reservoirs(container) {
        var root = null;
        var geometry = null;
        var parts = null;
        var run = null;
        var area = 0;
        var index = 0;
        var playing = false;
        var depthMax = 1;
        var flowMax = 1;

        // ── Static drawing (rebuilt on resize) ──────────────────────────

        function build() {
            var width = Math.round(container.clientWidth);
            if (!width) {
                return false;
            }
            geometry = layout(width);
            root = svg.canvas(container, "hiw-res" + (playing ? " is-playing" : ""));
            svg.size(root, width, geometry.height);
            var g = geometry;

            var defs = svg.el("defs", {}, root);
            var hatch = svg.el("pattern", {
                id: "hiw-res-soil", width: 7, height: 7, patternUnits: "userSpaceOnUse",
                patternTransform: "rotate(45)",
            }, defs);
            svg.el("line", { class: "hiw-res__hatch", x1: 0, y1: 0, x2: 0, y2: 7 }, hatch);

            parts = { tanks: [], segments: [] };
            parts.grid = svg.el("path", { class: "hiw-res__grid" }, root);
            var ticks = svg.el("g", { class: "hiw-res__ticks" }, root);
            parts.ticks = DEPTH_TICKS.map(function () {
                return svg.el("text", { x: g.axis - 8, "text-anchor": "end" }, ticks);
            });
            svg.el("text", { class: "hiw-res__axis-unit", x: g.axis - 8, y: g.top - 10, "text-anchor": "end" }, root, "mm");

            // The outlet channel: a bed line, then the flow that each reservoir adds.
            var dropX = SUBAREAS.map(function (subarea, i) {
                return g.left(i) + g.tank + g.gap / 2;
            });
            var end = g.width - 4;
            svg.el("path", { class: "hiw-res__bed", d: "M" + svg.px(dropX[0]) + " " + g.channel + "H" + end }, root);
            dropX.forEach(function (x, i) {
                var to = i < dropX.length - 1 ? dropX[i + 1] : end - 6;
                parts.segments.push(svg.el("path", {
                    class: "hiw-res__stream",
                    d: "M" + svg.px(x) + " " + g.channel + "H" + svg.px(to),
                }, root));
            });
            svg.el("path", {
                class: "hiw-res__outlet-arrow",
                d: "M" + (end - 8) + " " + (g.channel - 5) + "L" + end + " " + g.channel + "L" + (end - 8) + " " + (g.channel + 5) + "z",
            }, root);
            parts.outlet = label(root, { class: "hiw-res__outlet", x: end, y: g.outletY, "text-anchor": "end" }, "Q", "", " m³/s");

            SUBAREAS.forEach(function (subarea, i) {
                parts.tanks.push(buildTank(subarea, g.left(i), dropX[i]));
            });
            return true;
        }

        function buildTank(subarea, left, dropX) {
            var g = geometry;
            var right = left + g.tank;
            var center = left + g.tank / 2;
            var group = svg.el("g", { class: "hiw-res__tank hiw-res__tank--" + subarea.key }, root);
            var tank = { group: group, left: left, right: right, dropX: dropX, drops: [], arrows: [] };

            svg.el("text", { class: "hiw-res__name", x: center, y: g.nameY, "text-anchor": "middle" }, group,
                   g.compact ? subarea.shortName : subarea.name);
            tank.share = svg.el("text", { class: "hiw-res__share", x: center, y: g.shareY, "text-anchor": "middle" }, group);

            var rain = svg.el("g", { class: "hiw-res__rain" }, group);
            for (var j = 0; j < DROPS; j++) {
                var x = left + ((j + 0.5) * g.tank) / DROPS;
                var y = g.rainY + (j % 2 ? 11 : 0);
                tank.drops.push(svg.el("path", {
                    class: "hiw-res__drop",
                    d: "M" + svg.px(x + 2) + " " + y + "l-3 10",
                }, rain));
            }

            tank.interior = svg.el("rect", {
                class: "hiw-res__interior", x: left, y: g.top, width: svg.px(g.tank), height: g.bottom - g.top,
            }, group);
            tank.water = svg.el("rect", { class: "hiw-res__water", x: left, width: svg.px(g.tank) }, group);
            tank.surface = svg.el("path", { class: "hiw-res__surface" }, group);
            tank.storage = svg.el("path", { class: "hiw-res__storage" }, group);
            tank.walls = svg.el("path", { class: "hiw-res__walls" }, group);
            tank.stream = svg.el("path", { class: "hiw-res__stream hiw-res__stream--drop" }, group);

            tank.storageText = label(group, {
                class: "hiw-res__value hiw-res__value--muted", x: center, y: g.storageY, "text-anchor": "middle",
            }, "d", "s", "");
            // Narrow reservoirs leave the unit to the depth axis.
            tank.depth = label(group, { class: "hiw-res__value", x: left + 6 }, "d", "", g.compact ? "" : " mm");

            if (subarea.key === "pervious") {
                svg.el("rect", {
                    class: "hiw-res__soil", x: left, y: g.bottom + 1, width: svg.px(g.tank), height: 22,
                }, group);
                for (var k = 0; k < ARROWS; k++) {
                    var ax = svg.px(left + ((k + 1) * g.tank) / (ARROWS + 1));
                    tank.arrows.push(svg.el("path", {
                        class: "hiw-res__infiltration",
                        d: "M" + ax + " " + (g.bottom + 4) + "v13m-3.5-4 3.5 4 3.5-4",
                    }, group));
                }
                tank.infiltration = label(group, {
                    class: "hiw-res__value hiw-res__value--soil", x: left, y: g.bottom + 40,
                }, "f", "", " mm/h");
            }
            return tank;
        }

        // ── Per run ─────────────────────────────────────────────────────

        function depthY(depth) {
            var g = geometry;
            return g.bottom - (g.bottom - g.top) * Math.sqrt(Math.min(Math.max(depth, 0) / depthMax, 1));
        }

        function applyRun() {
            if (!run || !parts) {
                return;
            }
            var g = geometry;
            depthMax = depthScaleMax(run);
            flowMax = Math.max(run.peak.runoff, 1e-9);

            // Ticks on the shared depth axis: the top of the scale, then every value far
            // enough below the last one (the square root crowds the large ones).
            var ticks = [];
            DEPTH_TICKS.filter(function (value) {
                return value <= depthMax;
            }).reverse().forEach(function (value) {
                if (!ticks.length || depthY(value) - depthY(ticks[ticks.length - 1]) >= MIN_TICK_SPACING) {
                    ticks.push(value);
                }
            });
            var grid = "";
            parts.ticks.forEach(function (text, t) {
                var value = ticks[t];
                text.textContent = value === undefined ? "" : format.parameter(value);
                if (value !== undefined) {
                    var y = svg.px(depthY(value));
                    text.setAttribute("y", y + 3.5);
                    SUBAREAS.forEach(function (subarea, i) {
                        if (run.subareas[subarea.key].share > 0) {
                            grid += "M" + svg.px(g.left(i)) + " " + y + "h" + svg.px(g.tank);
                        }
                    });
                }
            });
            parts.grid.setAttribute("d", grid);

            SUBAREAS.forEach(function (subarea, i) {
                var series = run.subareas[subarea.key];
                var tank = parts.tanks[i];
                var empty = !(series.share > 0);
                var share = format.parameter(series.share * 100) + " %";
                tank.group.classList.toggle("is-empty", empty);
                tank.share.textContent = g.narrow || empty
                    ? share
                    : share + " · " + format.parameter(series.share * area) + " ha";
                tank.lipY = series.storage > 0 ? Math.min(depthY(series.storage), g.bottom - MIN_LIP) : g.bottom;
                tank.walls.setAttribute("d",
                    "M" + svg.px(tank.left) + " " + g.top + "V" + g.bottom + "H" + svg.px(tank.right) +
                    (series.storage > 0 ? "V" + svg.px(tank.lipY) : ""));
                tank.storage.setAttribute("d", series.storage > 0
                    ? "M" + svg.px(tank.left) + " " + svg.px(tank.lipY) + "H" + svg.px(tank.right)
                    : "");
                tank.storageText.textContent = series.storage > 0
                    ? " " + format.parameter(series.storage) + " mm"
                    : " 0";
            });
        }


        // ── Per frame ───────────────────────────────────────────────────

        function streamWidth(flow) {
            return flow > flowMax * 1e-4 ? 1.25 + 7 * Math.sqrt(Math.min(flow / flowMax, 1)) : 0;
        }

        function applyFrame() {
            if (!run || !parts) {
                return;
            }
            var g = geometry;
            var k = Math.min(index, run.time.length - 1);
            var rain = run.rainfall[k];
            var shown = rain > 0 ? Math.max(1, Math.round((DROPS * Math.min(rain, RAIN_FULL)) / RAIN_FULL)) : 0;
            var carried = 0;

            SUBAREAS.forEach(function (subarea, i) {
                var series = run.subareas[subarea.key];
                var tank = parts.tanks[i];
                var present = series.share > 0;
                var depth = present ? series.depth[k] : 0;
                var flow = present ? series.runoff[k] : 0;
                var y = svg.px(depthY(depth));

                DROP_ORDER.forEach(function (drop, order) {
                    tank.drops[drop].classList.toggle("is-on", present && order < shown);
                });
                svg.attrs(tank.water, { y: y, height: svg.px(g.bottom - y) });
                tank.surface.setAttribute("d", depth > 0 ? "M" + svg.px(tank.left) + " " + y + "H" + svg.px(tank.right) : "");
                tank.depth.textContent = " " + format.significant(depth, 3);
                // The depth rides on the water: above the surface, or just under it near the top.
                tank.depth.parentNode.setAttribute("y", svg.px(y - 6 < g.top + 14 ? y + 15 : y - 6));

                var width = streamWidth(flow);
                // Over the lip and down into the outlet channel.
                var fromY = Math.min(y, tank.lipY);
                tank.stream.setAttribute("d", width ? "M" + svg.px(tank.right - 1) + " " + svg.px(fromY + width / 2) +
                    "C" + svg.px(tank.right + g.gap * 0.4) + " " + svg.px(fromY + width / 2) + " " +
                    svg.px(tank.dropX) + " " + svg.px(fromY + 16) + " " + svg.px(tank.dropX) + " " + g.channel : "");
                tank.stream.setAttribute("stroke-width", width);

                carried += flow;
                parts.segments[i].setAttribute("stroke-width", streamWidth(carried));

                if (tank.infiltration) {
                    var rate = present ? run.infiltration[k] / series.share : 0;
                    var arrows = rate > 1e-3 ? Math.max(1, Math.round((ARROWS * Math.min(rate, INFILTRATION_FULL)) / INFILTRATION_FULL)) : 0;
                    tank.arrows.forEach(function (arrow, a) {
                        arrow.classList.toggle("is-on", a < arrows);
                    });
                    tank.infiltration.textContent = " " + format.significant(rate, 2);
                }
            });
            parts.outlet.textContent = " " + format.significant(run.runoff[k], 3);
        }

        function redraw() {
            if (build()) {
                applyRun();
                applyFrame();
            }
        }

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
            setRun: function (result, areaHectares) {
                run = result;
                area = areaHectares;
                if (!parts) {
                    build();
                }
                applyRun();
                applyFrame();
            },
            setFrame: function (frame) {
                index = frame;
                applyFrame();
            },
            setPlaying: function (isPlaying) {
                playing = isPlaying;
                if (root) {
                    root.classList.toggle("is-playing", isPlaying);
                }
            },
            /** A sentence for screen readers: the state at the current step. */
            describe: function () {
                if (!run) {
                    return "";
                }
                var k = Math.min(index, run.time.length - 1);
                var depths = SUBAREAS.filter(function (subarea) {
                    return run.subareas[subarea.key].share > 0;
                }).map(function (subarea) {
                    return subarea.name.toLowerCase() + " " + format.significant(run.subareas[subarea.key].depth[k], 3) + " mm";
                });
                return "Ponded depth at " + format.clock(run.time[k]) + ": " + depths.join(", ") +
                    ". Total runoff " + format.significant(run.runoff[k], 3) + " m³/s.";
            },
        };
    }

    CS.howItWorks.reservoirs = reservoirs;
})(window.CS);
