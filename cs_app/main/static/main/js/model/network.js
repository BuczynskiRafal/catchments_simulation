/*
 * CS.network — the runoff neural network of the calculations page, in the browser.
 *
 * The network reads eight properties of a subcatchment (imperviousness, width,
 * slope, Manning's n and depression storage of both sub-areas, and the share of
 * the impervious area without storage) in SI units, and predicts its runoff
 * volume in 10³ m³:
 *
 *     z  = (x − μ) / σ                       standardise with the training data
 *     h1 = ReLU(z · W1 + b1)                 8 neurons
 *     h2 = ReLU(h1 · W2 + b2)                8 neurons
 *     y  = ReLU(h2 · W3 + b3)                1 neuron: the runoff volume
 *
 * The weights come from the server (main.predictor.network_payload, embedded in
 * the page as #calculations-network): {features, mean, std, layers: [{weights,
 * bias}], output}, with weights[i][j] connecting input i to neuron j. The server
 * runs the same network in float32; this runs it in float64, so the two agree to
 * about 1e-5 relative. It only explains the server's answer: the page shows the
 * server's prediction and uses this pass for what happens inside the network.
 *
 * CS.network.forward(network, inputs) → {normalized, sums, layers, output}   (see forward below)
 * CS.network.zScores(network, inputs) → how many training standard deviations
 *                                       each input lies from the training mean
 */
(function (root, factory) {
    "use strict";
    if (typeof module === "object" && module.exports) {
        module.exports = factory();
    } else {
        root.CS = root.CS || {};
        root.CS.network = factory();
    }
})(typeof self !== "undefined" ? self : this, function () {
    "use strict";

    function relu(value) {
        return value > 0 ? value : 0;
    }

    /** The weighted sums of one dense layer: sum[j] = Σ_i input[i] · weights[i][j] + bias[j]. */
    function weightedSums(layer, input) {
        return layer.bias.map(function (bias, j) {
            var sum = bias;
            for (var i = 0; i < input.length; i += 1) {
                sum += input[i] * layer.weights[i][j];
            }
            return sum;
        });
    }

    function zScores(network, inputs) {
        return inputs.map(function (value, i) {
            return (value - network.mean[i]) / network.std[i];
        });
    }

    /**
     * Run the network on one subcatchment.
     *
     * @param {object} network   the payload of main.predictor.network_payload
     * @param {number[]} inputs  the eight inputs, in the order of network.features (SI units)
     * @returns {{normalized: number[], sums: number[][], layers: number[][], output: number}}
     *   normalized: the standardised inputs; sums: the weighted sums of each dense
     *   layer before ReLU; layers: its activations after ReLU ([8], [8], [1]); output:
     *   the prediction in network.output.unit.
     */
    function forward(network, inputs) {
        if (!inputs || inputs.length !== network.features.length) {
            throw new Error("The network needs " + network.features.length + " inputs.");
        }
        var normalized = zScores(network, inputs);
        var sums = [];
        var layers = [];
        var activation = normalized;
        network.layers.forEach(function (layer) {
            var layerSums = weightedSums(layer, activation);
            activation = layerSums.map(relu);
            sums.push(layerSums);
            layers.push(activation);
        });
        return { normalized: normalized, sums: sums, layers: layers, output: activation[0] };
    }

    return { forward: forward, zScores: zScores };
});
