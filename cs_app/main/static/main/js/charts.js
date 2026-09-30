"use strict";

/**
 * Theme-aware Plotly charts: `window.CS.charts`.
 *
 * Colours are read from the design tokens (tokens.css) at render time and again
 * on every `cs:themechange` event, when each chart still in the document is
 * redrawn from the data it was given (no refetch). A constant `uirevision`
 * keeps zoom and legend toggles across those redraws. Charts are purged (which
 * frees Plotly's window listeners) when redrawn, when CS.asyncForm is about to
 * replace them (`cs:beforeswap`), and once they have left the document.
 *
 * Every renderer takes a target element or its id, returns the Plotly promise,
 * or `null` when nothing was drawn (missing element, or Plotly not loaded — the
 * latter leaves a `.cs-empty-state` message in the container).
 *
 * The container gets `role="img"` (unless it already has a role) and an
 * `aria-label` from `opts.ariaLabel`, else `opts.title`, else a generic label
 * when it has none. Pages should pair each chart with a data table.
 */
(function () {
    var CS = (window.CS = window.CS || {});

    var FONT_UI = '"Public Sans", system-ui, -apple-system, "Segoe UI", sans-serif';
    var FONT_MONO = '"IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace';
    var UNAVAILABLE_MESSAGE = "Charts could not be loaded. Check your connection and reload the page.";
    var UI_REVISION = "cs-chart";
    var TRANSPARENT = "rgba(0,0,0,0)";
    var MARKER_POINT_LIMIT = 40;
    var NARROW_WIDTH = 576;
    // More swept values than this get a colour bar instead of one legend entry each.
    var LEGEND_VALUE_LIMIT = 8;
    // Rainfall (rate) axis max = 2.5 x visible peak, so the hanging bars fill the top ~40 %;
    // flow axis max = 1.65 x visible peak, so the lines stay in the lower ~60 %.
    var RAIN_AXIS_FACTOR = 2.5;
    var FLOW_AXIS_FACTOR = 1.65;
    // Hydrograph series in a third unit get a panel below the plot: its height and the gap
    // above it (which holds the panel's header), in px, and its axis max = 1.2 x visible peak.
    var PANEL_HEIGHT = 96;
    var PANEL_GAP = 40;
    var PANEL_AXIS_FACTOR = 1.2;
    // Rough height outside the plot area (legend, ticks, range slider; measured ~200 px on desktop,
    // ~265 px on phones), for turning those px into domain fractions.
    var PLOT_MARGINS_ESTIMATE = 200;

    // DESIGN_SPEC §2.1 values, used when a token is not defined by tokens.css.
    var FALLBACK_TOKENS = {
        light: {
            "--cs-ink": "#0F2233",
            "--cs-ink-muted": "#4A5E6D",
            "--cs-line": "#D6E1E8",
            "--cs-surface": "#FFFFFF",
            "--cs-water": "#0B6E8A",
            "--cs-water-strong": "#08566C",
            "--cs-rain": "#5A6FD6",
            "--cs-ochre": "#B7791F",
            "--cs-soil": "#346400",
            "--cs-evap": "#8A5A9E",
        },
        dark: {
            "--cs-ink": "#E3ECF2",
            "--cs-ink-muted": "#9DB1BF",
            "--cs-line": "#23394A",
            "--cs-surface": "#122330",
            "--cs-water": "#3FB8D6",
            "--cs-water-strong": "#6FCDE3",
            "--cs-rain": "#8C9CF0",
            "--cs-ochre": "#E0A546",
            "--cs-soil": "#C1EB95",
            "--cs-evap": "#C29AD6",
        },
    };

    // Categorical order (spec): water, rain, ochre, soil, evaporation.
    var SERIES_ORDER = ["water", "rain", "ochre", "soil", "evap"];

    // Hydrograph flow series: colour key + dash (secondary encoding, so identity is not colour-only).
    var FLOW_STYLES = {
        runoff: { color: "water", dash: "solid" },
        runon: { color: "waterStrong", dash: "dot" },
        infiltration_loss: { color: "soil", dash: "dash" },
        evaporation_loss: { color: "evap", dash: "dashdot" },
    };
    var DEFAULT_FLOW_FIELDS = ["runoff", "runon", "infiltration_loss", "evaporation_loss"];
    var EXTRA_DASHES = ["longdash", "longdashdot", "dot", "dash"];
    // Sweep lines listed in a legend also cycle through dashes: neighbouring ramp steps are close in colour.
    var SWEEP_DASHES = ["solid", "dash", "dot", "dashdot"];
    // Mix amounts of the sequential ramp's ends per theme (see rampEnds).
    var RAMP_MIX = { light: { low: 0.7, high: 0.85 }, dark: { low: 0.56, high: 0.9 } };

    var registry = [];

    // ── Theme ─────────────────────────────────────────────────────────

    function isDarkTheme() {
        var theme = document.documentElement.getAttribute("data-bs-theme");
        if (theme === "dark" || theme === "light") {
            return theme === "dark";
        }
        return Boolean(window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
    }

    function readTheme() {
        var styles = window.getComputedStyle(document.documentElement);
        var mode = isDarkTheme() ? "dark" : "light";
        var fallback = FALLBACK_TOKENS[mode];
        function token(name) {
            return styles.getPropertyValue(name).trim() || fallback[name];
        }
        return {
            mode: mode,
            ink: token("--cs-ink"),
            inkMuted: token("--cs-ink-muted"),
            line: token("--cs-line"),
            surface: token("--cs-surface"),
            water: token("--cs-water"),
            waterStrong: token("--cs-water-strong"),
            rain: token("--cs-rain"),
            ochre: token("--cs-ochre"),
            soil: token("--cs-soil"),
            evap: token("--cs-evap"),
        };
    }

    function parseColor(value) {
        var hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(value);
        if (hex) {
            var digits = hex[1].length === 3 ? hex[1].replace(/./g, "$&$&") : hex[1];
            return [0, 2, 4].map(function (i) {
                return parseInt(digits.substr(i, 2), 16);
            });
        }
        var rgb = /^rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)/i.exec(value);
        return rgb ? [Number(rgb[1]), Number(rgb[2]), Number(rgb[3])] : null;
    }

    function mixColors(from, to, t) {
        var a = parseColor(from);
        var b = parseColor(to);
        if (!a || !b) {
            return to;
        }
        var mixed = a.map(function (channel, i) {
            return Math.round(channel + (b[i] - channel) * t);
        });
        return "rgb(" + mixed.join(", ") + ")";
    }

    /**
     * Ends of the sequential teal ramp: low values are --cs-water mixed towards the
     * surface only as far as keeps 3:1 contrast with it (per theme), high values move
     * from --cs-water-strong most of the way to the ink, so the steps between spread
     * over as much lightness as the theme allows.
     */
    function rampEnds(theme) {
        var mix = RAMP_MIX[theme.mode];
        return [
            mixColors(theme.surface, theme.water, mix.low),
            mixColors(theme.waterStrong, theme.ink, mix.high),
        ];
    }

    function seriesColor(theme, index) {
        return theme[SERIES_ORDER[index % SERIES_ORDER.length]];
    }

    // ── Layout helpers ────────────────────────────────────────────────

    function isNarrow(el) {
        var width = el.clientWidth;
        return width > 0 && width < NARROW_WIDTH;
    }

    /**
     * Shared layout. On narrow containers the in-chart title is dropped (the
     * surrounding card and the aria-label carry it) because a wrapped legend
     * would otherwise collide with it; the legend then pushes the top margin.
     */
    function baseLayout(theme, opts, container, legendOnTop) {
        var narrow = isNarrow(container);
        var showTitle = Boolean(opts.title) && !narrow;
        var titleSpace = showTitle ? 44 : 12;
        var layout = {
            font: { family: FONT_UI, size: 12, color: theme.ink },
            paper_bgcolor: TRANSPARENT,
            plot_bgcolor: TRANSPARENT,
            margin: { l: 16, r: 16, t: titleSpace + (legendOnTop ? 28 : 0), b: 16, pad: 0 },
            hoverlabel: {
                bgcolor: theme.surface,
                bordercolor: theme.line,
                font: { family: FONT_UI, size: 12, color: theme.ink },
            },
            legend: {
                font: { color: theme.ink },
                bgcolor: TRANSPARENT,
                itemclick: "toggle",
                itemdoubleclick: "toggleothers",
            },
            modebar: { bgcolor: TRANSPARENT, color: theme.inkMuted, activecolor: theme.water },
            uirevision: UI_REVISION,
        };
        if (legendOnTop || narrow) {
            Object.assign(layout.legend, { orientation: "h", x: 0, xanchor: "left", y: 1.02, yanchor: "bottom" });
        }
        if (showTitle) {
            layout.title = {
                text: opts.title,
                font: { size: 15, color: theme.ink },
                x: 0,
                xref: "paper",
                xanchor: "left",
                y: 1,
                yref: "container",
                yanchor: "top",
                pad: { t: 12 },
            };
        }
        return layout;
    }

    function axis(theme, title, extra) {
        return Object.assign(
            {
                title: { text: title || "", font: { size: 12, color: theme.inkMuted }, standoff: 8 },
                automargin: true,
                gridcolor: theme.line,
                linecolor: theme.line,
                zerolinecolor: theme.line,
                tickcolor: theme.line,
                ticks: "outside",
                ticklen: 4,
                showline: true,
                tickfont: { family: FONT_MONO, size: 11, color: theme.inkMuted },
                hoverformat: ".4~g",
            },
            extra
        );
    }

    /** One column on phone-width containers, two otherwise (small multiples). */
    function gridFor(el, panelCount) {
        var columns = panelCount > 1 && !isNarrow(el) ? 2 : 1;
        return { rows: Math.ceil(panelCount / columns), columns: columns };
    }

    function axisSuffix(index) {
        return index === 0 ? "" : String(index + 1);
    }

    function pluck(records, field) {
        return records.map(function (row) {
            return row[field];
        });
    }

    function finiteNumbers(values) {
        return values.filter(function (value) {
            return value !== null && value !== "" && isFinite(Number(value));
        }).map(Number);
    }

    /** Largest finite value, or -Infinity when there is none. */
    function finiteMax(values) {
        return finiteNumbers(values).reduce(function (max, value) {
            return value > max ? value : max;
        }, -Infinity);
    }

    /**
     * Keeps the axis of an all non-negative series (volumes, rates) at or above zero,
     * so an all-zero series is drawn on 0…1 instead of Plotly's default −1…1.
     */
    function nonNegativeRange(records, field) {
        var negative = finiteNumbers(pluck(records, field)).some(function (value) {
            return value < 0;
        });
        return negative ? {} : { rangemode: "nonnegative" };
    }

    /** "Runoff [CMS]" → "CMS"; "" when the label has no unit. */
    function unitOf(label) {
        var match = /\[([^\]]+)\]\s*$/.exec(label || "");
        return match ? match[1] : "";
    }

    function sanitizeFileName(fileName) {
        return (fileName || "chart")
            .replace(/[\r\n]+/g, "_")
            .replace(/[<>:"/\\|?*]+/g, "_")
            .replace(/\.png$/i, "") || "chart";
    }

    // ── Rendering core ────────────────────────────────────────────────

    function resolveElement(target) {
        return typeof target === "string" ? document.getElementById(target) : target;
    }

    function showUnavailable(el) {
        el.removeAttribute("role");
        el.removeAttribute("aria-label");
        el.innerHTML = "";
        var message = document.createElement("div");
        message.className = "cs-empty-state";
        message.textContent = UNAVAILABLE_MESSAGE;
        el.appendChild(message);
    }

    function labelChart(el, opts, defaultLabel) {
        if (!el.hasAttribute("role")) {
            el.setAttribute("role", "img");
        }
        var label = opts.ariaLabel || opts.title;
        if (label) {
            el.setAttribute("aria-label", label);
        } else if (!el.hasAttribute("aria-label")) {
            el.setAttribute("aria-label", defaultLabel);
        }
    }

    function plotConfig(fileName) {
        return {
            responsive: true,
            displaylogo: false,
            // The stock camera button would export the transparent background; ours paints the surface.
            modeBarButtonsToRemove: ["toImage", "lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
            modeBarButtonsToAdd: [
                {
                    name: "downloadPng",
                    title: "Download chart as PNG",
                    icon: window.Plotly.Icons.camera,
                    click: function (gd) {
                        downloadPng(gd, fileName);
                    },
                },
            ],
            toImageButtonOptions: { format: "png", filename: sanitizeFileName(fileName), scale: 2 },
        };
    }

    function draw(entry, plotFn) {
        var spec = entry.build(readTheme(), entry.el);
        return plotFn(entry.el, spec.data, spec.layout, plotConfig(entry.fileName));
    }

    /**
     * @param {Element|string} target
     * @param {function(Object, Element): {data: Object[], layout: Object}} build
     *        Pure function of (theme, element); re-run on every theme change.
     */
    function render(target, build, opts, defaultLabel) {
        var el = resolveElement(target);
        if (!el) {
            return null;
        }
        if (typeof window.Plotly === "undefined") {
            showUnavailable(el);
            return null;
        }
        releaseCharts(function (chart) {
            return chart === el || !chart.isConnected;
        });
        window.Plotly.purge(el);
        el.innerHTML = "";
        labelChart(el, opts, defaultLabel);

        var entry = { el: el, build: build, fileName: opts.fileName || opts.title || "chart" };
        registry.push(entry);
        return draw(entry, window.Plotly.newPlot);
    }

    /** Purges and forgets every registered chart for which isReleased(element) is true. */
    function releaseCharts(isReleased) {
        registry = registry.filter(function (item) {
            if (!isReleased(item.el)) {
                return true;
            }
            window.Plotly.purge(item.el);
            return false;
        });
    }

    document.addEventListener("cs:themechange", function () {
        if (typeof window.Plotly === "undefined") {
            return;
        }
        releaseCharts(function (chart) {
            return !chart.isConnected;
        });
        registry.forEach(function (item) {
            draw(item, window.Plotly.react);
        });
    });

    document.addEventListener("cs:beforeswap", function (event) {
        if (typeof window.Plotly === "undefined") {
            return;
        }
        releaseCharts(function (chart) {
            return event.target.contains(chart);
        });
    });

    // ── Public renderers ──────────────────────────────────────────────

    /**
     * Line chart of one y field, or small multiples (two-column grid, one column
     * on narrow containers) when several fields are given.
     *
     * @param {Element|string} el
     * @param {Object[]} records   Row objects (DataFrame `orient="records"`).
     * @param {string} xField
     * @param {string|string[]} yFields
     * @param {Object} [opts]
     * @param {string} [opts.title]
     * @param {string} [opts.xLabel]
     * @param {string} [opts.yLabel]    Axis label of a single-field chart.
     * @param {Object} [opts.yLabels]   Field → axis label.
     * @param {number[]} [opts.xRange]
     * @param {number} [opts.height]    Default 420, or 300 per grid row.
     * @param {string} [opts.ariaLabel]
     * @param {string} [opts.fileName]  PNG export name (defaults to the title).
     * @returns {Promise|null}
     */
    function line(el, records, xField, yFields, opts) {
        opts = opts || {};
        var fields = [].concat(yFields);
        var yLabels = opts.yLabels || {};
        var multiple = fields.length > 1;
        var x = pluck(records, xField);
        var mode = records.length <= MARKER_POINT_LIMIT ? "lines+markers" : "lines";

        return render(el, function (theme, container) {
            var grid = gridFor(container, fields.length);
            var layout = baseLayout(theme, opts, container, false);
            layout.showlegend = false;
            layout.hovermode = "x unified";
            layout.height = opts.height || (multiple ? 300 * grid.rows : 420);
            if (multiple) {
                layout.grid = { rows: grid.rows, columns: grid.columns, pattern: "independent", xgap: 0.3, ygap: 0.3 };
            }

            var data = fields.map(function (field, i) {
                var suffix = axisSuffix(i);
                var yLabel = multiple ? yLabels[field] || field : yLabels[field] || opts.yLabel;
                layout["xaxis" + suffix] = axis(theme, opts.xLabel, opts.xRange ? { range: opts.xRange } : {});
                layout["yaxis" + suffix] = axis(theme, yLabel, nonNegativeRange(records, field));
                return {
                    type: "scatter",
                    mode: mode,
                    name: yLabels[field] || field,
                    x: x,
                    y: pluck(records, field),
                    xaxis: "x" + suffix,
                    yaxis: "y" + suffix,
                    line: { color: theme.water, width: 2 },
                    marker: { color: theme.water, size: 6 },
                };
            });
            return { data: data, layout: layout };
        }, opts, "Line chart");
    }

    function compareParameterValues(a, b) {
        var numberA = parseFloat(a);
        var numberB = parseFloat(b);
        var finiteA = isFinite(numberA);
        var finiteB = isFinite(numberB);
        if (finiteA && finiteB) {
            return numberA - numberB;
        }
        if (finiteA !== finiteB) {
            return finiteA ? -1 : 1;
        }
        return String(a).localeCompare(String(b));
    }

    /**
     * Where each value of an ascending list sits on the colour ramp (0…1):
     * proportional to the value when all are finite, by rank otherwise.
     */
    function rampPositions(numbers) {
        var low = numbers[0];
        var high = numbers[numbers.length - 1];
        var proportional = numbers.every(isFinite) && high > low;
        return numbers.map(function (number, i) {
            if (proportional) {
                return (number - low) / (high - low);
            }
            return numbers.length > 1 ? i / (numbers.length - 1) : 1;
        });
    }

    /** Colour bar that replaces the legend of a long numeric sweep; horizontal above narrow plots. */
    function sweepColorBar(theme, feature, narrow) {
        var colorBar = {
            title: { text: feature, side: "top", font: { size: 12, color: theme.inkMuted } },
            thickness: 12,
            outlinewidth: 0,
            tickfont: { family: FONT_MONO, size: 11, color: theme.inkMuted },
            tickcolor: theme.line,
        };
        if (narrow) {
            Object.assign(colorBar, { orientation: "h", x: 0, xanchor: "left", y: 1.02, yanchor: "bottom", len: 1 });
        }
        return colorBar;
    }

    /**
     * Parameter sweep: one line per parameter value, coloured on a sequential
     * teal ramp in value order, one panel per column. The legend lists the values
     * under the parameter name, and the lines also differ in dash; above
     * LEGEND_VALUE_LIMIT numeric values a colour bar takes their place.
     *
     * @param {Element|string} el
     * @param {Object} sweepData  {parameterValue: [row, ...], ...}
     * @param {string[]} columns  Fields to plot, one panel each.
     * @param {string} feature    Swept parameter name (legend and colour bar title).
     * @param {string} [catchment]
     * @param {Object} [opts]
     * @param {string} [opts.title]   Default "Timeseries sweep: <feature> for <catchment>".
     * @param {string} [opts.xField]  Default "datetime".
     * @param {string} [opts.xLabel]
     * @param {Object} [opts.yLabels] Column → axis label.
     * @param {number} [opts.height]  Default 300 per grid row (min 360).
     * @param {string} [opts.ariaLabel]
     * @param {string} [opts.fileName]
     * @returns {Promise|null}
     */
    function sweep(el, sweepData, columns, feature, catchment, opts) {
        opts = Object.assign({}, opts);
        if (opts.title === undefined) {
            opts.title = "Timeseries sweep: " + feature + (catchment ? " for " + catchment : "");
        }
        var xField = opts.xField || "datetime";
        var yLabels = opts.yLabels || {};
        var values = Object.keys(sweepData).sort(compareParameterValues);
        var numbers = values.map(parseFloat);
        var positions = rampPositions(numbers);
        var colorBarScale = values.length > LEGEND_VALUE_LIMIT && numbers.every(isFinite);

        return render(el, function (theme, container) {
            var grid = gridFor(container, columns.length);
            var ends = rampEnds(theme);
            var layout = baseLayout(theme, opts, container, false);
            layout.hovermode = "closest";
            layout.height = opts.height || Math.max(360, 300 * grid.rows);
            layout.showlegend = !colorBarScale;
            layout.legend.title = { text: feature, font: { color: theme.inkMuted } };
            if (columns.length > 1) {
                layout.grid = { rows: grid.rows, columns: grid.columns, pattern: "independent", xgap: 0.3, ygap: 0.3 };
            }
            columns.forEach(function (column, i) {
                var suffix = axisSuffix(i);
                layout["xaxis" + suffix] = axis(theme, opts.xLabel, {});
                layout["yaxis" + suffix] = axis(theme, yLabels[column] || column, {});
            });

            var data = [];
            values.forEach(function (value, valueIndex) {
                var rows = sweepData[value] || [];
                var x = pluck(rows, xField);
                columns.forEach(function (column, i) {
                    var suffix = axisSuffix(i);
                    data.push({
                        type: "scatter",
                        mode: "lines",
                        name: value,
                        legendgroup: value,
                        showlegend: i === 0,
                        x: x,
                        y: pluck(rows, column),
                        xaxis: "x" + suffix,
                        yaxis: "y" + suffix,
                        hovertemplate: "%{x}<br>" + (yLabels[column] || column) + ": %{y:.4~g}" +
                            "<extra>" + feature + " = " + value + "</extra>",
                        line: {
                            color: mixColors(ends[0], ends[1], positions[valueIndex]),
                            width: 1.75,
                            dash: colorBarScale ? "solid" : SWEEP_DASHES[valueIndex % SWEEP_DASHES.length],
                        },
                    });
                });
            });
            if (colorBarScale) {
                // A point-less trace that only carries the colour bar (Plotly interpolates in RGB, as mixColors does).
                data.push({
                    type: "scatter",
                    mode: "markers",
                    x: [null],
                    y: [null],
                    hoverinfo: "skip",
                    showlegend: false,
                    marker: {
                        color: [numbers[0], numbers[numbers.length - 1]],
                        cmin: numbers[0],
                        cmax: numbers[numbers.length - 1],
                        colorscale: [[0, ends[0]], [1, ends[1]]],
                        showscale: true,
                        colorbar: sweepColorBar(theme, feature, isNarrow(container)),
                    },
                });
            }
            return { data: data, layout: layout };
        }, opts, "Parameter sweep chart");
    }

    /**
     * [low, factor x peak] over the visible traces on one y axis ("y" or "y2"). The
     * factors leave the flow lines the lower part of the plot and the hanging rain
     * bars the upper part.
     */
    function visibleRange(data, yaxis, factor) {
        var values = [];
        data.forEach(function (trace) {
            if (trace.yaxis === yaxis && isShown(trace)) {
                values = values.concat(finiteNumbers(trace.y));
            }
        });
        var peak = finiteMax(values);
        var low = values.reduce(function (min, value) {
            return value < min ? value : min;
        }, 0);
        return [low, peak > 0 ? peak * factor : 1];
    }

    /** The reversed rate axis: zero at the top, so its bars and lines hang down. */
    function rateAxisRange(data) {
        return visibleRange(data, "y2", RAIN_AXIS_FACTOR).reverse();
    }

    function previousVisibility(el, uid, fallback) {
        var previous = (el.data || []).filter(function (trace) {
            return trace.uid === uid;
        })[0];
        return previous && previous.visible !== undefined ? previous.visible : fallback;
    }

    function isShown(trace) {
        return trace.visible === undefined || trace.visible === true;
    }

    /**
     * Hyetograph + hydrograph: rainfall bars hanging from a reversed top axis
     * (y2, right) over flow lines on the bottom axis (y, left), with the runoff
     * peak marked. Every axis holds one unit, taken from the labels: series in
     * the first flow field's unit share its axis, series in the rainfall unit
     * (loss rates such as infiltration) share the rainfall axis, and each other
     * unit (evaporation, reported per day) gets a panel of its own below the
     * plot, on the same time axis, shown while one of its series is visible.
     * Only the first flow field is visible initially; the rest toggle from the
     * legend and the axes rescale to the visible series.
     *
     * @param {Element|string} el
     * @param {Object[]} records
     * @param {Object} [opts]
     * @param {string} [opts.timeField="datetime"]
     * @param {string} [opts.rainField="rainfall"]
     * @param {string[]} [opts.flowFields]  Default: runoff, runon, infiltration_loss,
     *        evaporation_loss — whichever exist in the records.
     * @param {Object} [opts.yLabels]  Field → label with its unit, "Runoff Rate [CMS]"
     *        (also names the series; the unit picks the axis).
     * @param {string} [opts.rainLabel] Rainfall axis/series label.
     * @param {string} [opts.flowLabel] Flow axis label (default: label of the first flow field).
     * @param {string} [opts.xLabel]
     * @param {string} [opts.title]
     * @param {number} [opts.height=440]  Without panels; each shown panel adds its own height.
     * @param {boolean} [opts.rangeSlider=false]
     * @param {string} [opts.ariaLabel]
     * @param {string} [opts.fileName]
     * @returns {Promise|null}
     */
    function hydrograph(el, records, opts) {
        opts = opts || {};
        var timeField = opts.timeField || "datetime";
        var rainField = opts.rainField || "rainfall";
        var yLabels = opts.yLabels || {};
        var sample = records[0] || {};
        var flowFields = (opts.flowFields || DEFAULT_FLOW_FIELDS).filter(function (field) {
            return field in sample;
        });
        var hasRain = rainField in sample;
        var times = pluck(records, timeField);
        var rain = hasRain ? pluck(records, rainField) : [];
        var rainLabel = opts.rainLabel || yLabels[rainField] || "Rainfall";
        var rainUnit = unitOf(rainLabel);
        var mainField = flowFields[0];
        var mainValues = mainField ? pluck(records, mainField) : [];
        var peakValue = finiteMax(mainValues);
        var peakIndex = isFinite(peakValue) ? mainValues.map(Number).indexOf(peakValue) : -1;
        var mainUnit = unitOf(yLabels[mainField]);
        var peakAnnotations = peakIndex !== -1 ? 1 : 0;

        // Series → axis by unit; panels: [{axis: "y3", unit, fields}], one per unit off both main axes.
        var axisByField = {};
        var panels = [];
        flowFields.forEach(function (field) {
            var unit = unitOf(yLabels[field]);
            if (field === mainField || unit === mainUnit) {
                axisByField[field] = "y";
            } else if (hasRain && rainUnit !== "" && unit === rainUnit) {
                axisByField[field] = "y2";
            } else {
                var panel = panels.filter(function (candidate) {
                    return candidate.unit === unit;
                })[0];
                if (!panel) {
                    panel = { axis: "y" + (panels.length + 3), unit: unit, fields: [] };
                    panels.push(panel);
                }
                panel.fields.push(field);
                axisByField[field] = panel.axis;
            }
        });
        var hasRateSeries = flowFields.some(function (field) {
            return axisByField[field] === "y2";
        });

        function layoutKey(axisId) {
            return axisId.replace("y", "yaxis");
        }

        /**
         * Height, domains and x-axis anchor for the panels with a visible series, from
         * the traces alone, so a redraw (theme change) and a legend toggle agree. Hidden
         * panels overlay the main plot, where they take neither room nor drag area.
         */
        function panelLayout(data) {
            var shown = panels.filter(function (panel) {
                return data.some(function (trace) {
                    return trace.yaxis === panel.axis && isShown(trace);
                });
            });
            var height = (opts.height || 440) + shown.length * (PANEL_HEIGHT + PANEL_GAP);
            var px = 1 / (height - PLOT_MARGINS_ESTIMATE);
            var slot = (PANEL_HEIGHT + PANEL_GAP) * px;
            var placement = {
                height: height,
                mainDomain: [shown.length * slot, 1],
                xAnchor: shown.length ? shown[shown.length - 1].axis : "y",
                panels: {},
            };
            panels.forEach(function (panel) {
                var index = shown.indexOf(panel);
                var bottom = (shown.length - 1 - index) * slot;
                // overlaying "free" (not overlaid) lets a shown panel take its own domain.
                placement.panels[panel.axis] = index === -1
                    ? { visible: false, overlaying: "y" }
                    : { visible: true, overlaying: "free", domain: [bottom, bottom + PANEL_HEIGHT * px] };
            });
            return placement;
        }

        function panelHeaderY(placement, panel) {
            var domain = placement.panels[panel.axis].domain;
            return domain ? domain[1] : 0;
        }

        var result = render(el, function (theme, container) {
            var layout = baseLayout(theme, opts, container, true);
            layout.hovermode = "x unified";
            // Unified hover also lists the panels' series, which share the time axis.
            layout.hoversubplots = "axis";
            layout.bargap = 0.1;

            var data = [];
            if (hasRain) {
                data.push({
                    type: "bar",
                    uid: rainField,
                    name: rainLabel,
                    visible: previousVisibility(container, rainField, true),
                    x: times,
                    y: rain,
                    yaxis: "y2",
                    marker: { color: theme.rain, opacity: 0.85, line: { width: 0 } },
                });
            }

            flowFields.forEach(function (field, i) {
                var style = FLOW_STYLES[field] || { color: SERIES_ORDER[(i + 1) % SERIES_ORDER.length], dash: EXTRA_DASHES[i % EXTRA_DASHES.length] };
                data.push({
                    type: "scatter",
                    mode: "lines",
                    uid: field,
                    name: yLabels[field] || field,
                    legendgroup: field,
                    visible: previousVisibility(container, field, i === 0 ? true : "legendonly"),
                    x: times,
                    y: pluck(records, field),
                    yaxis: axisByField[field],
                    line: { color: theme[style.color], width: i === 0 ? 2.25 : 1.75, dash: style.dash },
                });
            });

            var placement = panelLayout(data);
            layout.height = placement.height;
            // An explicit range: autorange would also make room for the peak label and stretch the axis past the data.
            layout.xaxis = axis(theme, opts.xLabel, {
                type: "date",
                anchor: placement.xAnchor,
                range: times.length > 1 ? [times[0], times[times.length - 1]] : undefined,
                hoverformat: "%Y-%m-%d %H:%M",
                rangeslider: opts.rangeSlider
                    ? { visible: true, thickness: 0.08, bgcolor: TRANSPARENT, bordercolor: theme.line, borderwidth: 1 }
                    : { visible: false },
            });

            layout.annotations = [];
            var mainVisible = mainField ? previousVisibility(container, mainField, true) === true : false;
            if (peakAnnotations) {
                var peakLabel = "Peak " + Number(peakValue.toPrecision(4)) + (mainUnit ? " " + mainUnit : "");
                // Label on the side with more room, so the fixed x range never clips it.
                var labelLeft = peakIndex > times.length / 2;
                data.push({
                    type: "scatter",
                    mode: "markers",
                    uid: "peak",
                    name: peakLabel,
                    legendgroup: mainField,
                    showlegend: false,
                    visible: mainVisible,
                    x: [times[peakIndex]],
                    y: [peakValue],
                    yaxis: "y",
                    hoverinfo: "skip",
                    marker: { symbol: "diamond", size: 11, color: theme.ochre, line: { color: theme.surface, width: 2 } },
                });
                layout.annotations.push({
                    x: times[peakIndex],
                    y: peakValue,
                    xref: "x",
                    yref: "y",
                    text: peakLabel,
                    visible: mainVisible,
                    showarrow: true,
                    arrowhead: 0,
                    arrowwidth: 1,
                    arrowcolor: theme.ochre,
                    ax: labelLeft ? -28 : 28,
                    ay: -28,
                    xanchor: labelLeft ? "right" : "left",
                    font: { family: FONT_MONO, size: 11, color: theme.ink },
                    bgcolor: theme.surface,
                    bordercolor: theme.line,
                    borderpad: 3,
                });
            }

            layout.yaxis = axis(theme, opts.flowLabel || yLabels[mainField] || mainField, {
                domain: placement.mainDomain,
                range: visibleRange(data, "y", FLOW_AXIS_FACTOR),
                rangemode: "tozero",
            });
            if (hasRain) {
                var rateTitle = hasRateSeries ? "Rainfall and loss rates [" + rainUnit + "]" : rainLabel;
                layout.yaxis2 = axis(theme, rateTitle, {
                    overlaying: "y",
                    side: "right",
                    range: rateAxisRange(data),
                    showgrid: false,
                    zeroline: false,
                });
            }
            panels.forEach(function (panel) {
                // The header names the series (with the unit, as in the legend); the axis title is the unit.
                layout[layoutKey(panel.axis)] = axis(theme, panel.unit, Object.assign({
                    anchor: "x",
                    range: visibleRange(data, panel.axis, PANEL_AXIS_FACTOR),
                    rangemode: "tozero",
                    nticks: 4,
                }, placement.panels[panel.axis]));
                layout.annotations.push({
                    text: panel.fields.map(function (field) {
                        return yLabels[field] || field;
                    }).join(" · "),
                    visible: placement.panels[panel.axis].visible,
                    xref: "paper",
                    yref: "paper",
                    x: 0,
                    y: panelHeaderY(placement, panel),
                    xanchor: "left",
                    yanchor: "bottom",
                    yshift: 4,
                    showarrow: false,
                    font: { size: 12, color: theme.ink },
                });
            });
            return { data: data, layout: layout };
        }, opts, "Rainfall hyetograph and runoff hydrograph");

        if (result) {
            var container = resolveElement(el);
            result.then(function () {
                // Legend toggles fire plotly_restyle; relayout does not, so this cannot loop.
                container.on("plotly_restyle", function () {
                    var data = container.data;
                    var placement = panelLayout(data);
                    var update = {
                        height: placement.height,
                        "xaxis.anchor": placement.xAnchor,
                        "yaxis.domain": placement.mainDomain,
                        "yaxis.range": visibleRange(data, "y", FLOW_AXIS_FACTOR),
                    };
                    if (hasRain) {
                        update["yaxis2.range"] = rateAxisRange(data);
                    }
                    if (peakAnnotations) {
                        var peak = data.filter(function (trace) {
                            return trace.uid === mainField;
                        })[0];
                        update["annotations[0].visible"] = Boolean(peak) && isShown(peak);
                    }
                    panels.forEach(function (panel, i) {
                        var key = layoutKey(panel.axis);
                        var panelPlacement = placement.panels[panel.axis];
                        Object.keys(panelPlacement).forEach(function (name) {
                            update[key + "." + name] = panelPlacement[name];
                        });
                        update[key + ".range"] = visibleRange(data, panel.axis, PANEL_AXIS_FACTOR);
                        update["annotations[" + (peakAnnotations + i) + "].visible"] = panelPlacement.visible;
                        update["annotations[" + (peakAnnotations + i) + "].y"] = panelHeaderY(placement, panel);
                    });
                    window.Plotly.relayout(container, update);
                });
            });
        }
        return result;
    }

    /**
     * Parity (observed vs predicted) scatter with a dashed 1:1 reference line
     * and equal axis ranges.
     *
     * @param {Element|string} el
     * @param {Object[]} rows
     * @param {Object} [opts]
     * @param {string} [opts.xField="SWMM_Runoff"]  Observed / reference value.
     * @param {string} [opts.yField="ANN_Runoff"]   Predicted value.
     * @param {string} [opts.nameField="Name"]      Shown on hover.
     * @param {string} [opts.xLabel]
     * @param {string} [opts.yLabel]
     * @param {string} [opts.title]
     * @param {number} [opts.height=420]
     * @param {string} [opts.ariaLabel]
     * @param {string} [opts.fileName]
     * @returns {Promise|null}
     */
    function parity(el, rows, opts) {
        opts = opts || {};
        var xField = opts.xField || "SWMM_Runoff";
        var yField = opts.yField || "ANN_Runoff";
        var nameField = opts.nameField || "Name";
        var xLabel = opts.xLabel || xField;
        var yLabel = opts.yLabel || yField;
        var points = rows.filter(function (row) {
            return isFinite(parseFloat(row[xField])) && isFinite(parseFloat(row[yField]));
        });
        var xs = points.map(function (row) { return Number(row[xField]); });
        var ys = points.map(function (row) { return Number(row[yField]); });
        var all = xs.concat(ys);
        var low = all.reduce(function (min, value) { return value < min ? value : min; }, 0);
        var high = all.reduce(function (max, value) { return value > max ? value : max; }, 0);
        var padding = (high - low || 1) * 0.05;
        var range = [low - (low < 0 ? padding : 0), high + padding];

        return render(el, function (theme, container) {
            var layout = baseLayout(theme, opts, container, true);
            layout.hovermode = "closest";
            layout.height = opts.height || 420;
            layout.xaxis = axis(theme, xLabel, { range: range, constrain: "domain" });
            layout.yaxis = axis(theme, yLabel, { range: range, scaleanchor: "x", scaleratio: 1, constrain: "domain" });
            var data = [
                {
                    type: "scatter",
                    mode: "lines",
                    name: "1:1 line",
                    x: range,
                    y: range,
                    hoverinfo: "skip",
                    line: { color: theme.inkMuted, width: 1, dash: "dash" },
                },
                {
                    type: "scatter",
                    mode: "markers",
                    name: opts.pointsLabel || "Subcatchments",
                    x: xs,
                    y: ys,
                    text: points.map(function (row) { return String(row[nameField]); }),
                    hovertemplate: "<b>%{text}</b><br>" + xLabel + ": %{x:.4~g}<br>" + yLabel + ": %{y:.4~g}<extra></extra>",
                    marker: { color: theme.water, size: 10, line: { color: theme.surface, width: 1.5 } },
                },
            ];
            return { data: data, layout: layout };
        }, opts, "Parity chart of observed versus predicted values");
    }

    /**
     * Grouped bars: one bar per series for every category (e.g. SWMM vs ANN per
     * subcatchment).
     *
     * @param {Element|string} el
     * @param {Object[]} rows
     * @param {Object} [opts]
     * @param {string} [opts.categoryField="Name"]
     * @param {{field: string, label?: string, color?: string}[]} [opts.series]
     *        `color` is a palette key: water|rain|ochre|soil|evap. Default:
     *        SWMM_Runoff (water) and ANN_Runoff (ochre).
     * @param {string} [opts.xLabel]
     * @param {string} [opts.yLabel]
     * @param {string} [opts.title]
     * @param {number} [opts.height=400]
     * @param {string} [opts.ariaLabel]
     * @param {string} [opts.fileName]
     * @returns {Promise|null}
     */
    function bars(el, rows, opts) {
        opts = opts || {};
        var categoryField = opts.categoryField || "Name";
        var series = opts.series || [
            { field: "SWMM_Runoff", label: "SWMM", color: "water" },
            { field: "ANN_Runoff", label: "ANN", color: "ochre" },
        ];
        var categories = rows.map(function (row) { return String(row[categoryField]); });

        return render(el, function (theme, container) {
            var layout = baseLayout(theme, opts, container, true);
            layout.hovermode = "x unified";
            layout.height = opts.height || 400;
            layout.barmode = "group";
            layout.bargap = 0.25;
            layout.bargroupgap = 0.08;
            layout.xaxis = axis(theme, opts.xLabel, { type: "category", showgrid: false });
            layout.yaxis = axis(theme, opts.yLabel, { rangemode: "tozero" });
            var data = series.map(function (item, i) {
                var color = theme[item.color] || seriesColor(theme, i);
                return {
                    type: "bar",
                    name: item.label || item.field,
                    x: categories,
                    y: pluck(rows, item.field),
                    marker: { color: color, line: { color: theme.surface, width: 1 } },
                };
            });
            return { data: data, layout: layout };
        }, opts, "Grouped bar chart");
    }

    /**
     * Download a rendered chart as a 2x PNG painted on the current surface colour.
     *
     * @param {Element|string} el
     * @param {string} [fileName]  Sanitized; a trailing ".png" is dropped.
     * @returns {boolean} false when there is no rendered chart or Plotly is missing.
     */
    function downloadPng(el, fileName) {
        var chartEl = resolveElement(el);
        if (!chartEl || !chartEl.data || typeof window.Plotly === "undefined"
            || typeof window.Plotly.downloadImage !== "function") {
            return false;
        }
        var surface = readTheme().surface;
        var figure = {
            data: chartEl.data,
            layout: Object.assign({}, chartEl.layout, { paper_bgcolor: surface, plot_bgcolor: surface }),
        };
        window.Plotly.downloadImage(figure, {
            format: "png",
            filename: sanitizeFileName(fileName),
            width: chartEl.offsetWidth || null,
            height: chartEl.layout.height || chartEl.offsetHeight || null,
            scale: 2,
        }).catch(function (error) {
            console.warn("chart export failed:", error);
        });
        return true;
    }

    CS.charts = {
        line: line,
        sweep: sweep,
        hydrograph: hydrograph,
        parity: parity,
        bars: bars,
        downloadPng: downloadPng,
    };
})();
