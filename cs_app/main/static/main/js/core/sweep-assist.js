/*
 * CS.sweepAssist(options) — guidance next to a sweep form's "Parameter to vary"
 * (simulation page; timeseries page in sweep mode).
 *
 * options:
 *   form            the run form; reads data-values-url, data-how-it-works-url and
 *                   data-restored ("true" when the range fields hold the user's values)
 *   parameterSelect the "Parameter to vary" <select>
 *   featureOf       (option value) => SWMM feature name, e.g. "PercSlope"
 *   rangeInputs     [start, stop, step] inputs
 *   insightEl       container of the parameter card (.cs-insight)
 *   suggestionEl    line under the range fields for the suggested range
 *
 * The parameter guide comes from the page's #sweep-guide JSON. The values of the
 * chosen subcatchment are fetched from data-values-url whenever the subcatchment or
 * the model changes; responses for an earlier choice are ignored and failures leave
 * the card without the model's value.
 *
 * Range suggestion: "Suggested for this model: 5 → 15, step 1 · Use". It is applied
 * when the parameter changes (the old range belonged to another parameter) and when the
 * subcatchment or model changes while the range holds no value the user typed since
 * the last fill. A form restored with the user's range (data-restored) keeps it.
 * Filling the fields dispatches `input` on them, so run-count hints follow.
 *
 * One request per subcatchment of a model: the cs:model-changed of a model with a single
 * subcatchment and the `change` its auto-selection fires do not fetch twice.
 *
 * Returns {guide, values()}: the parsed guide and the latest values response (or null).
 */
(function (CS) {
    "use strict";

    var LINK_TEXT = "How it enters the model";
    var PROMPT_TEXT = "Choose a subcatchment to get a range suggested from its value.";
    var CENTRED_TEXT = "Range centred on the model's value.";
    var FULL_RANGE_TEXT = "Range covers every possible value.";
    var DEFAULT_RANGE_TEXT = "Default range for this parameter.";
    var FULL_RANGE_FEATURES = ["PercImperv", "PctZero"];

    var number = CS.format.parameter;

    function element(tag, className, text) {
        var el = document.createElement(tag);
        if (className) {
            el.className = className;
        }
        if (text) {
            el.textContent = text;
        }
        return el;
    }

    CS.sweepAssist = function (options) {
        var form = options.form;
        var select = options.parameterSelect;
        var inputs = options.rangeInputs;
        var guide = CS.readJson("sweep-guide") || {};
        var valuesUrl = form.dataset.valuesUrl;
        var howUrl = form.dataset.howItWorksUrl || "";
        var insightEl = options.insightEl;
        var suggestionEl = options.suggestionEl;

        var values = null;
        var loading = false;
        var requestId = 0;
        // Model generation and subcatchment of the latest request, so each is fetched once.
        var requestedKey = null;
        // The range fields hold values the user chose (typed, or restored from the session).
        var userRange = form.dataset.restored === "true";
        // Apply the suggestion once values for the current parameter arrive.
        var applyWhenKnown = false;
        var filling = false;

        function feature() {
            return options.featureOf(select.value);
        }

        function catchment() {
            var catchmentSelect = document.getElementById("id_catchment_name");
            return catchmentSelect ? catchmentSelect.value : "";
        }

        function suggestion() {
            var ranges = values && values.suggested_ranges;
            return ranges && ranges[feature()] ? ranges[feature()] : null;
        }

        // ── Parameter card ────────────────────────────────────────────

        function renderInsight() {
            var entry = guide[feature()];
            if (!insightEl) {
                return;
            }
            insightEl.hidden = !entry;
            if (!entry) {
                insightEl.replaceChildren();
                return;
            }
            var name = element("p", "cs-insight__name", entry.name + " ");
            name.append(element("span", "cs-insight__symbol", entry.symbol));

            var model = element("p", "cs-insight__model");
            var value = values && values.values ? values.values[entry.feature] : null;
            if (Number.isFinite(value)) {
                // The number in mono, its unit in the text face, as on the metric tiles.
                var unit = values.units[entry.feature];
                model.append("In this model: ", element("span", "cs-mono", number(value)),
                             unit ? "\u00a0" + unit : "");
            } else {
                // Keeps the line's height while the value loads, so nothing below moves.
                model.classList.add("is-pending");
                model.textContent = "In this model: …";
                model.hidden = !catchment();
            }

            var effect = element("p", "cs-insight__effect", entry.effect);
            effect.id = insightEl.id + "-effect";
            var where = element("p", "cs-insight__where cs-mono", entry.where);
            var link = element("a", "cs-insight__link", LINK_TEXT);
            link.href = howUrl + "#" + entry.anchor;

            insightEl.replaceChildren(name, model, effect, where, link);
            select.setAttribute("aria-describedby", effect.id);
        }

        // ── Range suggestion ──────────────────────────────────────────

        function rangeEquals(range) {
            return inputs.every(function (input, i) {
                var current = input.value.trim() === "" ? NaN : Number(input.value);
                return Math.abs(current - range[i]) < 1e-9;
            });
        }

        /** What the applied suggestion is: the whole range, one around the model's value, or a default. */
        function appliedText(range) {
            if (FULL_RANGE_FEATURES.indexOf(feature()) !== -1) {
                return FULL_RANGE_TEXT;
            }
            var modelValue = values.values ? values.values[feature()] : null;
            if (!Number.isFinite(modelValue)) {
                return DEFAULT_RANGE_TEXT;
            }
            if (modelValue >= range[0] && modelValue <= range[1] && modelValue > 0) {
                return CENTRED_TEXT;
            }
            return "Default range: the model's value is " + number(modelValue) + ".";
        }

        function renderSuggestion() {
            if (!suggestionEl) {
                return;
            }
            var range = suggestion();
            if (!range) {
                suggestionEl.replaceChildren(catchment() || loading ? "" : PROMPT_TEXT);
                suggestionEl.classList.remove("is-applied");
                return;
            }
            if (rangeEquals(range)) {
                suggestionEl.replaceChildren(appliedText(range));
                suggestionEl.classList.add("is-applied");
                return;
            }
            var button = element("button", "cs-suggestion__use", "Use");
            button.type = "button";
            button.setAttribute("aria-label", "Use the suggested range");
            button.addEventListener("click", function () {
                applySuggestion();
                inputs[0].focus();
            });
            suggestionEl.replaceChildren(
                "Suggested for this model: " + number(range[0]) + " → " + number(range[1]) +
                    ", step " + number(range[2]) + " · ",
                button
            );
            suggestionEl.classList.remove("is-applied");
        }

        function applySuggestion() {
            var range = suggestion();
            if (!range) {
                return false;
            }
            filling = true;
            inputs.forEach(function (input, i) {
                input.value = number(range[i]);
                input.dispatchEvent(new Event("input", { bubbles: true }));
            });
            filling = false;
            userRange = false;
            applyWhenKnown = false;
            renderSuggestion();
            return true;
        }

        function render() {
            renderInsight();
            renderSuggestion();
        }

        // ── Model values ──────────────────────────────────────────────

        function loadValues() {
            var name = catchment();
            var key = (CS.modelGeneration || 0) + "\n" + name;
            if (key === requestedKey) {
                return;
            }
            requestedKey = key;
            var id = ++requestId;
            values = null;
            loading = Boolean(name && valuesUrl);
            render();
            if (!loading) {
                return;
            }
            fetch(valuesUrl + "?" + new URLSearchParams({ catchment: name }).toString(), {
                headers: { "X-Requested-With": "XMLHttpRequest" },
                credentials: "same-origin",
            })
                .then(function (response) {
                    return response.ok ? response.json() : null;
                })
                .catch(function () {
                    return null;
                })
                .then(function (data) {
                    if (id !== requestId) {
                        return;
                    }
                    loading = false;
                    values = data && data.values ? data : null;
                    if (!values) {
                        // A failed request may be retried by choosing the subcatchment again.
                        requestedKey = null;
                    }
                    if (values && (applyWhenKnown || !userRange)) {
                        applySuggestion();
                    }
                    render();
                });
        }

        select.addEventListener("change", function () {
            // The range on screen belonged to the previous parameter.
            userRange = false;
            if (!applySuggestion()) {
                applyWhenKnown = loading;
            }
            render();
        });
        inputs.forEach(function (input) {
            input.addEventListener("input", function () {
                if (!filling) {
                    userRange = true;
                    applyWhenKnown = false;
                    renderSuggestion();
                }
            });
        });
        document.addEventListener("change", function (event) {
            if (event.target.id === "id_catchment_name") {
                loadValues();
            }
        });
        document.addEventListener("cs:model-changed", loadValues);
        document.addEventListener("cs:model-cleared", loadValues);

        if (suggestionEl) {
            suggestionEl.hidden = false;
        }

        render();
        return {
            guide: guide,
            values: function () {
                return values;
            },
        };
    };
})(window.CS);
