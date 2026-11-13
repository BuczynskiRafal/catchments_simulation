/*
 * SWMM vs ANN comparison page (/calculations).
 *
 * The run form is enhanced with CS.asyncForm: the server returns the results
 * fragment (partials/_calculations_results.html), which replaces the contents of
 * #calculations-output. initResults(root) wires up whatever that fragment holds and
 * runs both on page load (results rendered by a no-JS POST) and after every swap.
 * All numbers are computed server-side; the charts read the rows embedded as
 * #calculations-chart-data.
 *
 * "Inside the network" draws the network (#calculations-network) as an SVG and,
 * when the rows carry the network's inputs, traces one subcatchment through it
 * with CS.network.forward: neurons fill by activation, connections show what they
 * contribute, and the inputs are checked against the range of the training data.
 */
(function (CS) {
    "use strict";

    var SVG_NS = "http://www.w3.org/2000/svg";
    // Below this diagram width the labels switch to their short form.
    var COMPACT_WIDTH = 600;
    // Inputs further than this many standard deviations from the training mean are flagged.
    var RANGE_SIGMAS = 2;
    var COLUMN_TITLES = ["Inputs", "Layer 1", "Layer 2", "Output"];
    var COMPACT_COLUMN_TITLES = ["Inputs", "L1", "L2", "Out"];
    var DEFAULT_READOUT = "Point at a neuron, or focus the diagram and use the arrow keys, to read its value.";
    var WEIGHTS_LEGEND = "Line weight shows the strength of each connection; teal adds to the next neuron, ochre subtracts.";

    var inputFormat = new Intl.NumberFormat("en-US", { maximumSignificantDigits: 5 });
    var rangeFormat = new Intl.NumberFormat("en-US", { maximumSignificantDigits: 4 });

    function readRows(root) {
        var payload = root.querySelector("#calculations-chart-data");
        return payload ? JSON.parse(payload.textContent) : null;
    }

    function drawCharts(root, rows) {
        // The volume unit follows the model's flow units (m³ for SI, ft³ for US models).
        var unit = " [" + root.querySelector("[data-calc-unit]").dataset.calcUnit + "]";
        CS.charts.parity(root.querySelector("#calculations-parity-chart"), rows, {
            xLabel: "SWMM runoff" + unit,
            yLabel: "ANN runoff" + unit,
            ariaLabel: "Parity chart of ANN against SWMM runoff per subcatchment, with a 1:1 reference line",
            fileName: "swmm-vs-ann-parity",
        });
        CS.charts.bars(root.querySelector("#calculations-bars-chart"), rows, {
            xLabel: "Subcatchment",
            yLabel: "Runoff" + unit,
            ariaLabel: "Grouped bar chart of SWMM and ANN runoff per subcatchment",
            fileName: "swmm-vs-ann-runoff",
        });
    }

    // ── Formatting ───────────────────────────────────────────────────────

    /** "−0.42" with a true minus sign, as in the rest of the page's copy. */
    function signed(value, digits) {
        var text = CS.format.number(Math.abs(value), digits);
        return value < 0 ? "−" + text : text;
    }

    function withUnit(text, unit) {
        // Manning's n has no unit ("–").
        return unit && unit !== "–" ? text + " " + unit : text;
    }

    function inputText(value, feature) {
        return withUnit(inputFormat.format(value), feature.unit);
    }

    function volumeText(value, unit) {
        return CS.format.number(value, 2) + " " + unit;
    }

    // ── DOM helpers ──────────────────────────────────────────────────────

    function svg(tag, attributes, parent) {
        var element = document.createElementNS(SVG_NS, tag);
        Object.keys(attributes || {}).forEach(function (name) {
            element.setAttribute(name, attributes[name]);
        });
        if (parent) {
            parent.appendChild(element);
        }
        return element;
    }

    function svgText(parent, x, y, className, text, anchor) {
        var element = svg("text", { x: x, y: y, class: className, "text-anchor": anchor || "start" }, parent);
        element.textContent = text;
        return element;
    }

    function html(tag, className, text, parent) {
        var element = document.createElement(tag);
        if (className) {
            element.className = className;
        }
        if (text !== undefined) {
            element.textContent = text;
        }
        if (parent) {
            parent.appendChild(element);
        }
        return element;
    }

    // ── State of the diagram ─────────────────────────────────────────────

    var net = {
        network: null, // main.predictor.network_payload
        rows: [], // the comparison rows
        unit: "", // the page's volume unit
        traceable: false, // whether the rows carry the network's inputs
        traced: null, // index of the traced row in net.rows
        trace: null, // CS.network.forward of the traced row
        current: null, // {column, index} of the neuron read out
        width: 0,
        el: {},
    };

    function columnSize(column) {
        return column === 0 ? net.network.features.length : net.network.layers[column - 1].bias.length;
    }

    function hasInputs(row) {
        return Array.isArray(row.inputs) && row.inputs.every(Number.isFinite);
    }

    // The training range of each input as drawn: two standard deviations around the mean,
    // cut at the values the input can physically take (the server's min and max; no max
    // for a width or a roughness). The flag still follows the z-score.
    function trainingRange(index) {
        var feature = net.network.features[index];
        var mean = net.network.mean[index];
        var spread = RANGE_SIGMAS * net.network.std[index];
        var high = mean + spread;
        if (feature.max !== null && feature.max !== undefined) {
            high = Math.min(feature.max, high);
        }
        return { low: Math.max(feature.min || 0, mean - spread), high: high, mean: mean };
    }

    function outsideInputs() {
        if (!net.trace) {
            return [];
        }
        return net.network.features.filter(function (feature, index) {
            return Math.abs(net.trace.normalized[index]) > RANGE_SIGMAS;
        });
    }

    // ── Geometry ─────────────────────────────────────────────────────────

    function geometry(width) {
        var compact = width < COMPACT_WIDTH;
        var radius = compact ? 7 : 9;
        var left = compact ? 110 : 244; // room for the input labels
        var right = compact ? 50 : 160; // room for the output label
        var top = 26; // room for the column titles
        var rowHeight = compact ? 38 : 42;
        var first = left + radius;
        var last = width - right - radius;
        return {
            compact: compact,
            radius: radius,
            width: width,
            height: top + rowHeight * net.network.features.length,
            columns: [0, 1, 2, 3].map(function (column) {
                return first + ((last - first) * column) / 3;
            }),
            y: function (column, index) {
                var size = columnSize(column);
                // Columns with fewer neurons sit centred against the inputs.
                var offset = (net.network.features.length - size) / 2;
                return top + rowHeight * (index + offset + 0.5);
            },
        };
    }

    // ── Drawing ──────────────────────────────────────────────────────────

    /** The value each connection carries: its weight, or weight × source activation when tracing. */
    function edgeValue(layerIndex, i, j) {
        var weight = net.network.layers[layerIndex].weights[i][j];
        if (!net.trace) {
            return weight;
        }
        var source = layerIndex === 0 ? net.trace.normalized : net.trace.layers[layerIndex - 1];
        return weight * source[i];
    }

    function edgeScales() {
        // Weights share one scale, so the layers compare. Contributions are scaled per
        // layer: activations grow from layer to layer, and one scale would hide the first.
        var maxima = net.network.layers.map(function (layer, layerIndex) {
            var max = 0;
            layer.weights.forEach(function (row, i) {
                row.forEach(function (weight, j) {
                    max = Math.max(max, Math.abs(edgeValue(layerIndex, i, j)));
                });
            });
            return max;
        });
        if (net.trace) {
            return maxima;
        }
        var overall = Math.max.apply(null, maxima);
        return maxima.map(function () {
            return overall;
        });
    }

    function drawEdges(group, g) {
        var scales = edgeScales();
        net.network.layers.forEach(function (layer, layerIndex) {
            layer.weights.forEach(function (row, i) {
                row.forEach(function (weight, j) {
                    var value = edgeValue(layerIndex, i, j);
                    var strength = scales[layerIndex] > 0 ? Math.abs(value) / scales[layerIndex] : 0;
                    var kind = value === 0 ? "idle" : value > 0 ? "positive" : "negative";
                    svg(
                        "line",
                        {
                            class: "calc-net__edge calc-net__edge--" + kind,
                            x1: g.columns[layerIndex],
                            y1: g.y(layerIndex, i),
                            x2: g.columns[layerIndex + 1],
                            y2: g.y(layerIndex + 1, j),
                            "stroke-width": (0.6 + 3.4 * strength).toFixed(2),
                            "stroke-opacity": value === 0 ? 1 : (0.2 + 0.75 * strength).toFixed(2),
                            "data-from": layerIndex + ":" + i,
                            "data-to": layerIndex + 1 + ":" + j,
                        },
                        group
                    );
                });
            });
        });
    }

    function nodeState(column, index) {
        if (!net.trace) {
            return { className: "", fill: null };
        }
        if (column === 0) {
            var outside = Math.abs(net.trace.normalized[index]) > RANGE_SIGMAS;
            return { className: " is-on" + (outside ? " is-outside" : ""), fill: null };
        }
        var activations = net.trace.layers[column - 1];
        var value = activations[index];
        if (value <= 0) {
            return { className: " is-off", fill: null };
        }
        var max = Math.max.apply(null, activations);
        return { className: " is-on", fill: (0.3 + (0.7 * value) / max).toFixed(2) };
    }

    function drawNodes(group, g) {
        [0, 1, 2, 3].forEach(function (column) {
            for (var index = 0; index < columnSize(column); index += 1) {
                var x = g.columns[column];
                var y = g.y(column, index);
                var state = nodeState(column, index);
                var kind = column === 0 ? "input" : column === 3 ? "output" : "hidden";
                var node = svg(
                    "g",
                    {
                        class: "calc-net__node calc-net__node--" + kind + state.className,
                        "data-node": column + ":" + index,
                    },
                    group
                );
                svg("circle", { class: "calc-net__hit", cx: x, cy: y, r: g.radius + 8 }, node);
                var circle = svg("circle", { class: "calc-net__dot", cx: x, cy: y, r: g.radius }, node);
                if (state.fill) {
                    circle.setAttribute("fill-opacity", state.fill);
                }
            }
        });
    }

    function drawLabels(group, g) {
        (g.compact ? COMPACT_COLUMN_TITLES : COLUMN_TITLES).forEach(function (title, column) {
            var anchor = column === 0 ? "end" : column === 3 && !g.compact ? "start" : "middle";
            var x = column === 0 ? g.columns[0] + g.radius : g.columns[column];
            if (column === 3 && !g.compact) {
                x = g.columns[3] - g.radius;
            }
            svgText(group, x, 12, "calc-net__column", title, anchor);
        });

        var labelX = g.columns[0] - g.radius - 8;
        net.network.features.forEach(function (feature, index) {
            var y = g.y(0, index);
            var label = g.compact ? feature.short : feature.label;
            if (!net.trace) {
                svgText(group, labelX, y + 4, "calc-net__label", label, "end");
                return;
            }
            svgText(group, labelX, y - 3, "calc-net__label", label, "end");
            var outside = Math.abs(net.trace.normalized[index]) > RANGE_SIGMAS;
            svgText(
                group,
                labelX,
                y + 11,
                "calc-net__value" + (outside ? " calc-net__value--outside" : ""),
                inputText(net.rows[net.traced].inputs[index], feature),
                "end"
            );
        });

        var outX = g.columns[3] + g.radius + 8;
        var outY = g.y(3, 0);
        var row = net.trace ? net.rows[net.traced] : null;
        if (g.compact) {
            svgText(group, outX, outY + 4, "calc-net__label", "Runoff");
        } else if (row && row.ANN_Runoff !== null) {
            svgText(group, outX, outY - 3, "calc-net__label", "Runoff volume");
            svgText(group, outX, outY + 11, "calc-net__value", volumeText(row.ANN_Runoff, net.unit));
        } else {
            svgText(group, outX, outY + 4, "calc-net__label", "Runoff volume");
        }
    }

    function diagramLabel() {
        if (!net.trace) {
            return "Diagram of the neural network: 8 inputs, two hidden layers of 8 neurons and one output neuron. " +
                "Line weight shows the strength of each connection.";
        }
        var row = net.rows[net.traced];
        var active = net.trace.layers.slice(0, 2).map(function (layer) {
            return layer.filter(function (value) {
                return value > 0;
            }).length;
        });
        var outside = outsideInputs().length;
        return "Neural network traced for subcatchment " + row.Name + ": " + active[0] +
            " of 8 neurons active in the first hidden layer and " + active[1] + " of 8 in the second; " +
            "it predicts " + volumeText(row.ANN_Runoff, net.unit) + " of runoff" +
            (row.SWMM_Runoff !== null ? " against " + volumeText(row.SWMM_Runoff, net.unit) + " from SWMM" : "") +
            ". " + (outside ? outside + " of 8 inputs are outside the training range." : "All inputs are within the training range.");
    }

    function draw() {
        var stage = net.el.stage;
        var width = Math.floor(stage.clientWidth);
        if (!width) {
            return;
        }
        net.width = width;
        var g = geometry(width);
        var root = net.el.svg;
        root.textContent = "";
        root.setAttribute("viewBox", "0 0 " + g.width + " " + g.height);
        root.setAttribute("width", g.width);
        root.setAttribute("height", g.height);
        root.setAttribute("aria-label", diagramLabel());
        drawEdges(svg("g", { class: "calc-net__edges" }, root), g);
        drawLabels(svg("g", { class: "calc-net__labels", "aria-hidden": "true" }, root), g);
        drawNodes(svg("g", { class: "calc-net__nodes" }, root), g);
        highlight();
    }

    // ── Reading a neuron ─────────────────────────────────────────────────

    function neuronName(column, index) {
        if (column === 0) {
            return net.network.features[index].label;
        }
        if (column === 3) {
            return "Output neuron";
        }
        return "Layer " + column + ", neuron " + (index + 1);
    }

    /** The source that pushes this neuron hardest, as "Width (−1.58)". */
    function strongestSource(column, index) {
        var layerIndex = column - 1;
        var best = null;
        for (var i = 0; i < columnSize(column - 1); i += 1) {
            var value = edgeValue(layerIndex, i, index);
            if (!best || Math.abs(value) > Math.abs(best.value)) {
                best = { index: i, value: value };
            }
        }
        var source = column === 1 ? net.network.features[best.index].label : neuronName(column - 1, best.index).toLowerCase();
        return source + " (" + signed(best.value, 2) + ")";
    }

    function describe(column, index) {
        var name = neuronName(column, index);
        var feature = column === 0 ? net.network.features[index] : null;
        if (!net.trace) {
            if (feature) {
                return name + ": standardised with the training mean " + inputText(net.network.mean[index], feature) +
                    " and standard deviation " + inputText(net.network.std[index], feature) + ".";
            }
            var bias = net.network.layers[column - 1].bias[index];
            return name + ": bias " + signed(bias, 2) + "; strongest weight from " + strongestSource(column, index) + ".";
        }
        if (feature) {
            var z = net.trace.normalized[index];
            var side = z < 0 ? "below" : "above";
            return name + ": " + inputText(net.rows[net.traced].inputs[index], feature) + ", " +
                CS.format.number(Math.abs(z), 2) + " standard deviations " + side + " the training mean" +
                (Math.abs(z) > RANGE_SIGMAS ? ", outside the training range." : ".");
        }
        var sum = net.trace.sums[column - 1][index];
        if (column === 3) {
            return name + ": " + CS.format.number(net.trace.output, 4) + " × " + net.network.output.unit +
                " of runoff; biggest contribution from " + strongestSource(column, index) + ".";
        }
        if (sum <= 0) {
            return name + ": off. Its weighted sum, " + signed(sum, 2) + ", is negative, so ReLU passes 0.";
        }
        return name + ": " + CS.format.number(sum, 2) + "; biggest contribution from " + strongestSource(column, index) + ".";
    }

    function highlight() {
        var current = net.current;
        var root = net.el.svg;
        root.classList.toggle("is-focusing", Boolean(current));
        var key = current ? current.column + ":" + current.index : "";
        root.querySelectorAll(".calc-net__edge").forEach(function (edge) {
            edge.classList.toggle("is-linked", Boolean(current) && (edge.dataset.from === key || edge.dataset.to === key));
        });
        root.querySelectorAll(".calc-net__node").forEach(function (node) {
            node.classList.toggle("is-current", node.dataset.node === key);
        });
        net.el.readout.textContent = current ? describe(current.column, current.index) : DEFAULT_READOUT;
    }

    /**
     * Read out a neuron, or none. The visible readout follows pointer and keyboard; only
     * keyboard moves (`announce`) reach the live region, so sweeping the mouse over the
     * diagram does not flood a screen reader, and clearing the readout says nothing.
     */
    function setCurrent(column, index, announce) {
        net.current = column === null ? null : { column: column, index: index };
        highlight();
        if (announce && net.current) {
            net.el.announce.textContent = net.el.readout.textContent;
        }
    }

    function onKeydown(event) {
        var current = net.current || { column: 0, index: 0 };
        var column = current.column;
        var index = current.index;
        switch (event.key) {
            case "ArrowUp":
                index -= 1;
                break;
            case "ArrowDown":
                index += 1;
                break;
            case "ArrowLeft":
                column -= 1;
                break;
            case "ArrowRight":
                column += 1;
                break;
            case "Home":
                index = 0;
                break;
            case "End":
                index = columnSize(column) - 1;
                break;
            case "Escape":
                setCurrent(null);
                return;
            default:
                return;
        }
        event.preventDefault();
        column = Math.min(3, Math.max(0, column));
        if (column !== current.column) {
            // Keep the vertical position when moving between columns of different sizes.
            var y = (current.index + 0.5) / columnSize(current.column);
            index = Math.floor(y * columnSize(column));
        }
        setCurrent(column, Math.min(columnSize(column) - 1, Math.max(0, index)), true);
    }

    function nodeAt(event) {
        var node = event.target.closest && event.target.closest("[data-node]");
        return node ? node.dataset.node.split(":").map(Number) : null;
    }

    // ── Training range ───────────────────────────────────────────────────

    function rangeTrack(index, value) {
        var range = trainingRange(index);
        var span = range.high - range.low;
        var position = function (number) {
            return ((Math.min(range.high, Math.max(range.low, number)) - range.low) / span) * 100 + "%";
        };
        var track = svg("svg", { class: "calc-range__track", "aria-hidden": "true", focusable: "false", height: 14, overflow: "visible" });
        svg("line", { class: "calc-range__bar", x1: "0%", x2: "100%", y1: 7, y2: 7 }, track);
        svg("line", { class: "calc-range__mean", x1: position(range.mean), x2: position(range.mean), y1: 2, y2: 12 }, track);
        if (value === null) {
            return track;
        }
        if (value > range.high || value < range.low) {
            // Clamped at the end of the track, as an arrow pointing past it.
            var holder = svg("svg", { x: value > range.high ? "100%" : "0%", y: 0, overflow: "visible" }, track);
            svg("path", { class: "calc-range__arrow", d: value > range.high ? "M-4 1 6 7-4 13Z" : "M4 1-6 7 4 13Z" }, holder);
        } else {
            svg("circle", { class: "calc-range__marker", cx: position(value), cy: 7, r: 5 }, track);
        }
        return track;
    }

    function drawRanges() {
        var list = net.el.rangeList;
        list.textContent = "";
        var row = net.trace ? net.rows[net.traced] : null;
        net.network.features.forEach(function (feature, index) {
            var value = row ? row.inputs[index] : null;
            var outside = row !== null && Math.abs(net.trace.normalized[index]) > RANGE_SIGMAS;
            var item = html("li", "calc-range__item" + (outside ? " is-outside" : ""), undefined, list);

            var name = html("span", "calc-range__name", feature.label, item);
            if (feature.unit !== "–") {
                html("span", "calc-range__unit", " (" + feature.unit + ")", name);
            }
            if (outside) {
                html("span", "calc-range__flag", "outside the training range", name);
            }
            // Empty before a trace, so the bars line up in the same column either way.
            html("span", "calc-range__value", row ? inputText(value, feature) : "", item);

            var range = trainingRange(index);
            var scale = html("span", "calc-range__scale", undefined, item);
            scale.appendChild(rangeTrack(index, value));
            var ends = html("span", "calc-range__ends", undefined, scale);
            var low = html("span", "", undefined, ends);
            html("span", "visually-hidden", "Training range ", low);
            low.append(rangeFormat.format(range.low));
            var high = html("span", "", undefined, ends);
            html("span", "visually-hidden", " to ", high);
            high.append(withUnit(rangeFormat.format(range.high), feature.unit));
        });

        var summary;
        if (row) {
            var outside = outsideInputs();
            summary = outside.length === 0
                ? "All eight inputs are within the range the network was trained on."
                : outside.length + " of 8 inputs " + (outside.length === 1 ? "is" : "are") +
                  " outside the range the network was trained on (" +
                  outside.map(function (feature) {
                      return feature.label;
                  }).join(", ") + "): treat this prediction as indicative.";
        } else if (net.rows.length) {
            summary = "No subcatchment of this comparison can be placed on these ranges.";
        } else {
            summary = "Run the comparison to check each subcatchment's inputs against these ranges.";
        }
        net.el.rangeSummary.textContent = summary;
        net.el.rangeSummary.classList.toggle("is-outside", Boolean(row) && outsideInputs().length > 0);
    }

    // ── Trace ────────────────────────────────────────────────────────────

    function drawResult() {
        var output = net.el.output;
        var row = net.trace ? net.rows[net.traced] : null;
        output.hidden = !row;
        output.textContent = "";
        if (!row) {
            return;
        }
        // The server's prediction, exactly as in the table; the browser's pass agrees to ~1e-5.
        html("span", "calc-net__result-label", "ANN ", output);
        html("span", "calc-net__result-value", volumeText(row.ANN_Runoff, net.unit), output);
        if (row.SWMM_Runoff !== null) {
            output.append(" · ");
            html("span", "calc-net__result-label", "SWMM ", output);
            html("span", "calc-net__result-value", volumeText(row.SWMM_Runoff, net.unit), output);
        }
        if (row.Difference_pct !== null) {
            output.append(" (" + (row.Difference_pct > 0 ? "+" : "") + signed(row.Difference_pct, 1) + " %)");
        }
    }

    function render() {
        var row = net.traced === null ? null : net.rows[net.traced];
        net.trace = row ? CS.network.forward(net.network, row.inputs) : null;
        net.el.legend.textContent = net.trace
            ? "Tracing " + row.Name + ". Filled neurons are active, dashed ones are off. Line weight shows what each " +
              "connection adds for this subcatchment (its weight times the value it carries); teal adds, ochre subtracts."
            : WEIGHTS_LEGEND;
        // Rows without the network's inputs: the server could not keep them for a large model.
        net.el.notice.hidden = !(net.rows.length && !net.rows.some(hasInputs));
        draw();
        drawRanges();
        drawResult();
    }

    function trace(index) {
        net.traced = index;
        net.el.select.value = index === null ? "" : String(index);
        render();
    }

    /** Point the diagram at the rows of a new comparison (none before the first run). */
    function setRows(rows, unit) {
        if (!net.network) {
            return;
        }
        var previous = net.traced === null ? null : net.rows[net.traced].Name;
        net.rows = rows;
        net.unit = unit;
        net.traceable = rows.length > 0 && rows.every(hasInputs);
        var select = net.el.select;
        select.textContent = "";
        if (net.traceable) {
            rows.forEach(function (row, index) {
                var option = html("option", "", row.Name, select);
                option.value = String(index);
            });
        }
        net.el.traceControl.hidden = !net.traceable;
        var keep = rows.findIndex(function (row) {
            return row.Name === previous;
        });
        trace(net.traceable ? Math.max(0, keep) : null);
    }

    function focusDiagram() {
        var heading = net.el.heading;
        var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        heading.focus({ preventScroll: true });
        heading.scrollIntoView({ block: "start", behavior: reduce ? "auto" : "smooth" });
    }

    function initNetwork() {
        var payload = document.getElementById("calculations-network");
        var network = payload ? JSON.parse(payload.textContent) : null;
        if (!network || !CS.network) {
            return;
        }
        net.network = network;
        net.el = {
            heading: document.getElementById("calc-network-title"),
            stage: document.getElementById("calc-network-stage"),
            svg: document.getElementById("calc-network-diagram"),
            legend: document.getElementById("calc-network-caption"),
            readout: document.getElementById("calc-network-readout"),
            announce: document.getElementById("calc-network-announce"),
            notice: document.getElementById("calc-network-notice"),
            output: document.getElementById("calc-network-output"),
            select: document.getElementById("calc-trace-select"),
            traceControl: document.getElementById("calc-trace-control"),
            rangeList: document.getElementById("calc-range-list"),
            rangeSummary: document.getElementById("calc-range-summary"),
        };
        document.querySelectorAll("[data-calc-network-js]").forEach(function (el) {
            el.hidden = false;
        });

        var stage = net.el.stage;
        stage.addEventListener("keydown", onKeydown);
        stage.addEventListener("focus", function () {
            if (!net.current) {
                setCurrent(0, 0);
            }
        });
        stage.addEventListener("blur", function () {
            setCurrent(null);
        });
        net.el.svg.addEventListener("pointerover", function (event) {
            var node = nodeAt(event);
            if (node) {
                setCurrent(node[0], node[1]);
            }
        });
        net.el.svg.addEventListener("pointerleave", function () {
            if (document.activeElement !== stage) {
                setCurrent(null);
            }
        });
        net.el.select.addEventListener("change", function () {
            trace(Number(net.el.select.value));
        });
        // Redraw at the new width, so the labels keep their size instead of scaling.
        if (window.ResizeObserver) {
            new ResizeObserver(function () {
                if (Math.floor(stage.clientWidth) !== net.width) {
                    draw();
                }
            }).observe(stage);
        }
    }

    // ── Results ──────────────────────────────────────────────────────────

    function initResults(root) {
        var rows = readRows(root);
        var unitHolder = root.querySelector("[data-calc-unit]");
        setRows(rows || [], unitHolder ? unitHolder.dataset.calcUnit : "");
        if (!rows) {
            return;
        }
        // Unhide before drawing: Plotly sizes each chart from its container. The Trace
        // column stays hidden when the page has no network to trace through.
        root.querySelectorAll("[data-calc-requires-js]").forEach(function (el) {
            el.hidden = el.hasAttribute("data-calc-trace-col") && !net.network;
        });
        drawCharts(root, rows);
        CS.table.sortable(root.querySelector("#calculations-table"));
    }

    // Delegated, so the buttons in every swapped-in fragment work.
    document.addEventListener("click", function (event) {
        var traceButton = event.target.closest("[data-calc-trace]");
        if (traceButton && net.traceable) {
            trace(Number(traceButton.dataset.calcTrace));
            focusDiagram();
            return;
        }
        if (!event.target.closest("#copy-calculations-table-button")) {
            return;
        }
        // The Trace column is a control, not data.
        var table = document.getElementById("calculations-table").cloneNode(true);
        table.querySelectorAll("[data-calc-trace-col]").forEach(function (cell) {
            cell.remove();
        });
        CS.clipboard.copy(CS.table.toCSV(table), "Comparison table copied as CSV.");
    });

    document.addEventListener("DOMContentLoaded", function () {
        var output = document.getElementById("calculations-output");
        var form = document.getElementById("calculations-form");
        if (!output || !form) {
            return;
        }
        initNetwork();
        initResults(output);
        CS.asyncForm({
            form: form,
            submitButton: document.getElementById("run-calculations-button"),
            target: output,
            loadingEl: document.getElementById("calculations-loading-state"),
            successMessage: "Comparison finished.",
            onSuccess: initResults,
        });
    });
})(window.CS);
