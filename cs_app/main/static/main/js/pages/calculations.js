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
 * "Inside the network" draws the network (#calculations-network) as an SVG.
 */
(function (CS) {
    "use strict";

    var SVG_NS = "http://www.w3.org/2000/svg";
    // Below this diagram width the labels switch to their short form.
    var COMPACT_WIDTH = 600;
    var COLUMN_TITLES = ["Inputs", "Layer 1", "Layer 2", "Output"];
    var COMPACT_COLUMN_TITLES = ["Inputs", "L1", "L2", "Out"];
    var DEFAULT_READOUT = "Point at a neuron, or focus the diagram and use the arrow keys, to read its value.";

    var inputFormat = new Intl.NumberFormat("en-US", { maximumSignificantDigits: 5 });

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

    // ── State of the diagram ─────────────────────────────────────────────

    var net = {
        network: null, // main.predictor.network_payload
        current: null, // {column, index} of the neuron read out
        width: 0,
        el: {},
    };

    function columnSize(column) {
        return column === 0 ? net.network.features.length : net.network.layers[column - 1].bias.length;
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

    /** The value each connection carries: its weight. */
    function edgeValue(layerIndex, i, j) {
        var weight = net.network.layers[layerIndex].weights[i][j];
        return weight;
    }

    function edgeScales() {
        // Weights share one scale, so the layers compare.
        var maxima = net.network.layers.map(function (layer, layerIndex) {
            var max = 0;
            layer.weights.forEach(function (row, i) {
                row.forEach(function (weight, j) {
                    max = Math.max(max, Math.abs(edgeValue(layerIndex, i, j)));
                });
            });
            return max;
        });
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

    function drawNodes(group, g) {
        [0, 1, 2, 3].forEach(function (column) {
            for (var index = 0; index < columnSize(column); index += 1) {
                var x = g.columns[column];
                var y = g.y(column, index);
                var kind = column === 0 ? "input" : column === 3 ? "output" : "hidden";
                var node = svg(
                    "g",
                    {
                        class: "calc-net__node calc-net__node--" + kind,
                        "data-node": column + ":" + index,
                    },
                    group
                );
                svg("circle", { class: "calc-net__hit", cx: x, cy: y, r: g.radius + 8 }, node);
                svg("circle", { class: "calc-net__dot", cx: x, cy: y, r: g.radius }, node);
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
            svgText(group, labelX, y + 4, "calc-net__label", label, "end");
        });

        var outX = g.columns[3] + g.radius + 8;
        var outY = g.y(3, 0);
        if (g.compact) {
            svgText(group, outX, outY + 4, "calc-net__label", "Runoff");
        } else {
            svgText(group, outX, outY + 4, "calc-net__label", "Runoff volume");
        }
    }

    function diagramLabel() {
        return "Diagram of the neural network: 8 inputs, two hidden layers of 8 neurons and one output neuron. " +
            "Line weight shows the strength of each connection.";
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
        if (feature) {
            return name + ": standardised with the training mean " + inputText(net.network.mean[index], feature) +
                " and standard deviation " + inputText(net.network.std[index], feature) + ".";
        }
        var bias = net.network.layers[column - 1].bias[index];
        return name + ": bias " + signed(bias, 2) + "; strongest weight from " + strongestSource(column, index) + ".";
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

    function initNetwork() {
        var payload = document.getElementById("calculations-network");
        var network = payload ? JSON.parse(payload.textContent) : null;
        if (!network) {
            return;
        }
        net.network = network;
        net.el = {
            stage: document.getElementById("calc-network-stage"),
            svg: document.getElementById("calc-network-diagram"),
            readout: document.getElementById("calc-network-readout"),
            announce: document.getElementById("calc-network-announce"),
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
        // Redraw at the new width, so the labels keep their size instead of scaling.
        if (window.ResizeObserver) {
            new ResizeObserver(function () {
                if (Math.floor(stage.clientWidth) !== net.width) {
                    draw();
                }
            }).observe(stage);
        }
        draw();
    }

    // ── Results ──────────────────────────────────────────────────────────

    function initResults(root) {
        var rows = readRows(root);
        if (!rows) {
            return;
        }
        // Unhide before drawing: Plotly sizes each chart from its container.
        root.querySelectorAll("[data-calc-requires-js]").forEach(function (el) {
            el.hidden = false;
        });
        drawCharts(root, rows);
        CS.table.sortable(root.querySelector("#calculations-table"));
    }

    // Delegated, so the button in every swapped-in fragment works.
    document.addEventListener("click", function (event) {
        if (!event.target.closest("#copy-calculations-table-button")) {
            return;
        }
        var table = document.getElementById("calculations-table");
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
