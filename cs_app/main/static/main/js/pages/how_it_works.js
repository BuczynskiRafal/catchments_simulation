/*
 * "How it works" page (main/how_it_works.html).
 *
 * Typesets the equations (CS.math), lets equations and tables that are wider
 * than the screen scroll from the keyboard, and runs the interactive model in
 * #lab: CS.runoffModel solved in the browser, re-run on every slider movement.
 *
 * The interactive model is driven by one state object. Event handlers change
 * the state and ask for a render; render() runs once per animation frame and
 * brings each part of the page up to date only when what it shows has changed
 * (a new run, a new time step, a new sweep line), so dragging a slider or
 * playing the storm only updates attributes of the existing figures.
 */
(function (CS) {
    "use strict";

    var hiw = CS.howItWorks;
    var format = hiw.format;

    var TIME_STEP = 60; // s; the tools' SWMM runs report every 5 min, the model here steps every minute
    var PLAY_DURATION = 12000; // ms for the whole storm
    var SWEEP_RUNS = 9;
    var SWEEP_PAUSE = 280; // ms between sweep runs
    var reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

    // ── Equations and wide content ──────────────────────────────────────

    /** Over-wide equations and tables scroll inside their box; make those reachable by keyboard. */
    function markScrollable() {
        document.querySelectorAll(".hiw-equation__body, [data-hiw-scroll]").forEach(function (element) {
            var scrolls = element.scrollWidth > element.clientWidth + 1;
            if (scrolls && !element.hasAttribute("tabindex")) {
                var number = element.parentElement.querySelector(".hiw-equation__number");
                element.setAttribute("tabindex", "0");
                element.setAttribute("role", "group");
                element.setAttribute("aria-label", number ? "Equation " + number.textContent : "Scrollable table");
            } else if (!scrolls && element.getAttribute("tabindex") === "0") {
                element.removeAttribute("tabindex");
                element.removeAttribute("role");
                element.removeAttribute("aria-label");
            }
        });
    }

    function watchScrollable() {
        var pending = false;
        window.addEventListener("resize", function () {
            if (!pending) {
                pending = true;
                window.requestAnimationFrame(function () {
                    pending = false;
                    markScrollable();
                });
            }
        });
        markScrollable();
    }

    /** Typesetting and the lab change the page's height above a #fragment target; land on it again. */
    function restoreFragment() {
        var target = window.location.hash && document.getElementById(decodeURIComponent(window.location.hash.slice(1)));
        if (target) {
            target.scrollIntoView({ block: "start", behavior: "instant" });
        }
    }

    // ── Interactive model ───────────────────────────────────────────────

    function initLab(root) {
        var model = CS.runoffModel;
        var sliders = hiw.controls.sliders(root);
        var select = document.getElementById("lab-sweep-parameter");
        var elements = {
            storms: document.getElementById("lab-storms"),
            stormDescription: document.getElementById("lab-storm-description"),
            reset: document.getElementById("lab-reset"),
            play: document.getElementById("lab-play"),
            playText: root.querySelector(".hiw-time__play-text"),
            scrubber: document.getElementById("lab-time"),
            timeValue: document.getElementById("lab-time-value"),
            rainValue: document.getElementById("lab-rain-value"),
            reservoirs: document.getElementById("lab-reservoirs"),
            hydrograph: document.getElementById("lab-hydrograph"),
            legend: document.getElementById("lab-legend"),
            metrics: document.getElementById("lab-metrics"),
            balanceTotal: document.getElementById("lab-balance-total"),
            sweepRun: document.getElementById("lab-sweep-run"),
            sweepProgress: document.getElementById("lab-sweep-progress"),
            sweepStatus: document.getElementById("lab-sweep-status"),
            sweepDone: document.getElementById("lab-sweep-done"),
            response: document.getElementById("lab-response"),
            dock: document.getElementById("lab-dock"),
            responseTable: document.querySelector("#lab-response-table tbody"),
        };

        var state = {
            storm: "example",
            start: model.DEFAULTS,
            values: Object.assign({}, model.DEFAULTS),
            result: null,
            baseline: null,
            changed: false,
            index: 0,
            playhead: 0,
            playing: false,
            // Whether the reader put the cursor where it is (a seek or a pause); Play then resumes
            // there. The cursor starts at the peak, and Play from there starts the storm over.
            cued: false,
            sweep: null,
            quantity: "volume",
        };
        var drawn = {};
        var frameRequest = 0;
        var playedAt = null;
        var sweepTimer = 0;

        var reservoirs = hiw.reservoirs(elements.reservoirs);
        var chart = hiw.hydrograph(elements.hydrograph, { onSeek: seek });
        var response = hiw.response(elements.response);

        // ── Model runs ──────────────────────────────────────────────────

        function simulate(values) {
            var storm = model.STORMS[state.storm];
            return model.simulate(Object.assign({}, values, {
                rainfall: storm.rainfall,
                duration: storm.duration,
                timeStep: TIME_STEP,
            }));
        }

        function lastIndex() {
            return state.result.time.length - 1;
        }

        function peakIndex(result) {
            return Math.round(result.peak.time / TIME_STEP);
        }

        /** Re-run the model for the current values; `newStorm` also re-runs the starting values. */
        function recompute(newStorm) {
            state.result = simulate(state.values);
            if (newStorm || !state.baseline) {
                state.baseline = simulate(state.start);
            }
            state.changed = hiw.controls.KEYS.some(function (key) {
                return state.values[key] !== state.start[key];
            });
            state.index = Math.min(state.index, lastIndex());
            state.playhead = state.index;
        }

        function requestRender() {
            if (!frameRequest) {
                frameRequest = window.requestAnimationFrame(render);
            }
        }

        // ── State changes ───────────────────────────────────────────────

        function setValue(key, value) {
            state.values[key] = value;
            clearSweep();
            recompute(false);
            requestRender();
        }

        function setStorm(key) {
            state.storm = key;
            clearSweep();
            stop();
            recompute(true);
            state.index = state.playhead = peakIndex(state.result);
            state.cued = false;
            requestRender();
        }

        function reset() {
            state.values = Object.assign({}, state.start);
            clearSweep();
            recompute(false);
            requestRender();
        }

        function seek(index) {
            state.index = Math.min(Math.max(index, 0), lastIndex());
            state.playhead = state.index;
            state.cued = true;
            requestRender();
        }

        // ── Playback ────────────────────────────────────────────────────

        function play() {
            if (!state.cued || state.index >= lastIndex()) {
                state.index = state.playhead = 0;
            }
            state.playing = true;
            state.cued = true;
            playedAt = null;
            window.requestAnimationFrame(advance);
            requestRender();
        }

        function stop() {
            state.playing = false;
            requestRender();
        }

        function advance(now) {
            if (!state.playing) {
                return;
            }
            if (playedAt !== null) {
                state.playhead += ((now - playedAt) * lastIndex()) / PLAY_DURATION;
            }
            playedAt = now;
            state.index = Math.min(Math.floor(state.playhead), lastIndex());
            if (state.index >= lastIndex()) {
                state.playing = false;
            }
            requestRender();
            if (state.playing) {
                window.requestAnimationFrame(advance);
            }
        }

        // ── Sweep ───────────────────────────────────────────────────────

        function startSweep() {
            var option = select.selectedOptions[0];
            var key = option.getAttribute("data-hiw-param");
            var slider = sliders[key];
            var values = slider.spread(SWEEP_RUNS);
            var runs = values.map(function (value) {
                var changed = Object.assign({}, state.values);
                changed[key] = value;
                return simulate(changed);
            });
            state.sweep = {
                feature: option.value,
                key: key,
                name: option.textContent.trim(),
                unit: slider.unit,
                log: slider.log,
                start: state.values[key],
                values: values,
                runs: runs,
                volumes: runs.map(function (run) {
                    return run.totals.runoff;
                }),
                peaks: runs.map(function (run) {
                    return run.peak.runoff;
                }),
                shown: reducedMotion.matches ? runs.length : 1,
            };
            elements.sweepStatus.textContent = "Running the model for " + SWEEP_RUNS + " values of " +
                state.sweep.name + "…";
            if (state.sweep.shown === runs.length) {
                finishSweep();
            } else {
                sweepTimer = window.setTimeout(nextSweepRun, SWEEP_PAUSE);
            }
            requestRender();
        }

        function nextSweepRun() {
            var sweep = state.sweep;
            sweep.shown += 1;
            if (sweep.shown < sweep.runs.length) {
                sweepTimer = window.setTimeout(nextSweepRun, SWEEP_PAUSE);
            } else {
                finishSweep();
            }
            requestRender();
        }

        function finishSweep() {
            elements.sweepStatus.replaceChildren(elements.sweepDone.content.cloneNode(true));
        }

        function clearSweep() {
            window.clearTimeout(sweepTimer);
            if (state.sweep) {
                state.sweep = null;
                elements.sweepStatus.textContent = "";
            }
        }

        // ── Rendering ───────────────────────────────────────────────────

        function render() {
            frameRequest = 0;
            var result = state.result;
            var runChanged = drawn.result !== result || drawn.changed !== state.changed;
            var sweep = state.sweep;
            var sweepChanged = drawn.sweep !== sweep || (sweep && drawn.shown !== sweep.shown);

            if (runChanged) {
                Object.keys(sliders).forEach(function (key) {
                    sliders[key].show(state.values[key]);
                });
                chart.setRuns(result, state.changed ? state.baseline : null, model.STORMS[state.storm]);
                reservoirs.setRun(result, state.values.area);
                renderMetrics();
                renderBalance();
                renderDock();
                elements.scrubber.max = String(lastIndex());
                legendItem("ghost").hidden = !state.changed || Boolean(sweep);
            }
            if (sweepChanged) {
                chart.setSweep(sweep && { runs: sweep.runs, shown: sweep.shown });
                renderSweep();
            }
            if (runChanged || sweepChanged) {
                elements.hydrograph.setAttribute("aria-label", describeChart());
            }
            if (runChanged || drawn.index !== state.index) {
                renderTime();
            }
            if (drawn.playing !== state.playing) {
                renderPlaying();
            }
            drawn = { result: result, changed: state.changed, sweep: sweep, shown: sweep && sweep.shown, index: state.index, playing: state.playing };
        }

        function legendItem(name) {
            return elements.legend.querySelector("[data-hiw-legend='" + name + "']");
        }

        function renderTime() {
            var k = state.index;
            var result = state.result;
            var time = format.clock(result.time[k]);
            var rain = format.parameter(result.rainfall[k]);
            chart.setCursor(k);
            reservoirs.setFrame(k);
            elements.scrubber.value = String(k);
            elements.scrubber.setAttribute("aria-valuetext", time + ", rainfall " + rain + " mm/h");
            elements.timeValue.textContent = time;
            elements.rainValue.textContent = rain;
            elements.reservoirs.setAttribute("aria-label", reservoirs.describe());
        }

        function renderPlaying() {
            elements.play.classList.toggle("is-playing", state.playing);
            elements.playText.textContent = state.playing ? "Pause" : "Play";
            reservoirs.setPlaying(state.playing);
        }

        function metric(name) {
            var tile = elements.metrics.querySelector("[data-hiw-metric='" + name + "']");
            return { value: tile.querySelector("[data-hiw-value]"), delta: tile.querySelector("[data-hiw-delta]") };
        }

        function summary(result) {
            return {
                peak: result.peak.runoff,
                timeToPeak: result.peak.runoff > 0 ? result.peak.time : null,
                volume: result.totals.runoff,
                coefficient: result.totals.rainfall > 0 ? (result.totals.runoff / result.totals.rainfall) * 100 : 0,
            };
        }

        function renderMetrics() {
            var now = summary(state.result);
            var then = summary(state.baseline);
            var texts = {
                peak: format.significant(now.peak, 3),
                timeToPeak: now.timeToPeak === null ? "—" : format.duration(now.timeToPeak),
                volume: format.volume(now.volume),
                coefficient: CS.format.number(now.coefficient, 1),
            };
            var deltas = {
                peak: format.change(then.peak, now.peak),
                timeToPeak: now.timeToPeak === null || then.timeToPeak === null
                    ? null
                    : format.durationChange(then.timeToPeak, now.timeToPeak),
                volume: format.change(then.volume, now.volume),
                coefficient: format.pointChange(then.coefficient, now.coefficient),
            };
            Object.keys(texts).forEach(function (name) {
                var tile = metric(name);
                tile.value.textContent = texts[name];
                tile.delta.textContent = state.changed && deltas[name] ? deltas[name] + " vs.\u00a0start" : "";
                tile.delta.hidden = !tile.delta.textContent;
            });
        }

        /** Path of a run's hydrograph in the dock's 120 × 36 sparkline box. */
        function sparkline(result, top) {
            var last = result.time.length - 1;
            var points = [];
            for (var i = 0; i <= 120; i++) {
                var k = Math.round((i / 120) * last);
                points.push([i, 34 - (32 * result.runoff[k]) / top]);
            }
            return hiw.svg.line(points);
        }

        function renderDock() {
            var now = summary(state.result);
            var then = summary(state.baseline);
            var top = Math.max(now.peak, state.changed ? then.peak : 0) || 1;
            var texts = {
                peak: format.significant(now.peak, 3),
                peakChange: state.changed ? format.change(then.peak, now.peak) || "" : "",
                volume: format.volume(now.volume),
                volumeChange: state.changed ? format.change(then.volume, now.volume) || "" : "",
            };
            Object.keys(texts).forEach(function (key) {
                elements.dock.querySelector("[data-hiw-dock='" + key + "']").textContent = texts[key];
            });
            elements.dock.querySelector(".hiw-dock__line").setAttribute("d", sparkline(state.result, top));
            elements.dock.querySelector(".hiw-dock__ghost").setAttribute("d", state.changed ? sparkline(state.baseline, top) : "");
        }

        /**
         * Below xl the sliders sit under the figures they move: while any of them is on
         * screen and the hydrograph is not (less than half of it shows under the site
         * header), the dock keeps the run's headline numbers in view.
         */
        function watchDock() {
            if (!window.IntersectionObserver) {
                return;
            }
            var stacked = window.matchMedia("(max-width: 1199.98px)");
            var header = document.querySelector(".cs-header");
            var controlsInView = false;
            var chartInView = true;
            function update() {
                elements.dock.classList.toggle("is-shown", stacked.matches && controlsInView && !chartInView);
            }
            new IntersectionObserver(function (entries) {
                controlsInView = entries[entries.length - 1].isIntersecting;
                update();
            }).observe(document.getElementById("lab-controls"));
            new IntersectionObserver(function (entries) {
                chartInView = entries[entries.length - 1].intersectionRatio >= 0.5;
                update();
            }, {
                rootMargin: "-" + (header ? header.offsetHeight : 0) + "px 0px 0px 0px",
                threshold: [0, 0.5, 1],
            }).observe(root.querySelector(".hiw-hydrograph-card"));
            stacked.addEventListener("change", update);
        }

        function renderBalance() {
            var totals = state.result.totals;
            var parts = {
                runoff: totals.runoff,
                infiltration: totals.infiltration,
                storage: totals.storage,
                evaporation: totals.evaporation,
            };
            // Normalised by the sum of the parts: the solver's continuity error is the tiny remainder.
            var sum = Object.keys(parts).reduce(function (total, key) {
                return total + Math.max(parts[key], 0);
            }, 0) || 1;
            elements.balanceTotal.textContent = format.volume(totals.rainfall) + " m³ of rain on " +
                format.parameter(state.values.area) + " ha";
            root.querySelectorAll(".hiw-balance [data-hiw-part]").forEach(function (element) {
                var key = element.getAttribute("data-hiw-part");
                var share = Math.max(parts[key], 0) / sum;
                // Evaporation is left out when there is none; the other parts always show.
                element.hidden = key === "evaporation" && !(parts[key] > 0);
                if (element.classList.contains("hiw-balance__part")) {
                    element.style.setProperty("--hiw-share", String(share));
                } else {
                    element.querySelector("dd").textContent = format.volume(parts[key]) + " m³ · " +
                        CS.format.number(share * 100, 1) + " %";
                }
            });
        }

        function renderSweep() {
            var sweep = state.sweep;
            var done = sweep && sweep.shown === sweep.runs.length;
            elements.sweepRun.textContent = sweep ? "Clear sweep" : "Run sweep";
            legendItem("sweep").hidden = !sweep;
            legendItem("runoff").hidden = Boolean(sweep);
            legendItem("peak").hidden = Boolean(sweep);
            legendItem("ghost").hidden = !state.changed || Boolean(sweep);
            elements.sweepProgress.textContent = sweep
                ? sweep.shown + " of " + sweep.runs.length + " runs · " + sweep.name + " = " +
                    format.parameter(sweep.values[sweep.shown - 1]) + (sweep.unit ? " " + sweep.unit : "")
                : "";
            response.set(sweep && {
                values: sweep.values,
                volumes: sweep.volumes,
                peaks: sweep.peaks,
                shown: sweep.shown,
                log: sweep.log,
                name: sweep.name,
                unit: sweep.unit,
                start: sweep.start,
            });
            elements.response.setAttribute("aria-label", describeResponse(done));
            elements.responseTable.replaceChildren();
            if (done) {
                sweep.values.forEach(function (value, i) {
                    var row = document.createElement("tr");
                    [format.parameter(value) + (sweep.unit ? " " + sweep.unit : ""), format.volume(sweep.volumes[i]),
                     format.significant(sweep.peaks[i], 3)].forEach(function (text, column) {
                        var cell = document.createElement(column ? "td" : "th");
                        if (!column) {
                            cell.scope = "row";
                        }
                        cell.textContent = text;
                        row.appendChild(cell);
                    });
                    elements.responseTable.appendChild(row);
                });
            }
        }

        function describeChart() {
            var storm = model.STORMS[state.storm].label.toLowerCase();
            var sweep = state.sweep;
            if (sweep) {
                return "Hydrographs of the " + storm + " for " + sweep.runs.length + " values of " + sweep.name +
                    " from " + format.parameter(sweep.values[0]) + " to " +
                    format.parameter(sweep.values[sweep.values.length - 1]) + (sweep.unit ? " " + sweep.unit : "") +
                    ": peak runoff from " + format.significant(Math.min.apply(null, sweep.peaks), 3) + " to " +
                    format.significant(Math.max.apply(null, sweep.peaks), 3) + " m³/s.";
            }
            var now = summary(state.result);
            var text = "Rainfall and runoff of the " + storm + ". Peak runoff " + format.significant(now.peak, 3) +
                " m³/s" + (now.timeToPeak === null ? "" : " after " + format.duration(now.timeToPeak)) +
                ", runoff volume " + format.volume(now.volume) + " m³.";
            if (state.changed) {
                var then = summary(state.baseline);
                text += " With the starting values: peak " + format.significant(then.peak, 3) + " m³/s, volume " +
                    format.volume(then.volume) + " m³.";
            }
            return text;
        }

        function describeResponse(done) {
            var sweep = state.sweep;
            if (!done) {
                return "Response curve: run a sweep to fill it.";
            }
            var volume = state.quantity === "volume";
            var outputs = volume ? sweep.volumes : sweep.peaks;
            var text = volume ? format.volume : function (value) {
                return format.significant(value, 3);
            };
            var unit = volume ? " m³" : " m³/s";
            var last = sweep.values.length - 1;
            return (volume ? "Runoff volume" : "Peak runoff") + " against " + sweep.name + ": " +
                text(outputs[0]) + unit + " at " + format.parameter(sweep.values[0]) + ", " +
                text(outputs[last]) + unit + " at " + format.parameter(sweep.values[last]) +
                (sweep.unit ? " " + sweep.unit : "") + ". The table of results follows.";
        }

        // ── Controls ────────────────────────────────────────────────────

        function buildStorms() {
            Object.keys(model.STORMS).forEach(function (key) {
                var id = "lab-storm-" + key;
                var input = document.createElement("input");
                input.type = "radio";
                input.className = "btn-check";
                input.name = "lab-storm";
                input.id = id;
                input.value = key;
                input.checked = key === state.storm;
                input.setAttribute("aria-describedby", elements.stormDescription.id);
                var label = document.createElement("label");
                label.className = "btn cs-segmented__option";
                label.htmlFor = id;
                label.textContent = model.STORMS[key].label;
                elements.storms.append(input, label);
            });
            elements.stormDescription.textContent = model.STORMS[state.storm].description;
            elements.storms.addEventListener("change", function (event) {
                elements.stormDescription.textContent = model.STORMS[event.target.value].description;
                setStorm(event.target.value);
            });
        }

        function initSliders() {
            Object.keys(sliders).forEach(function (key) {
                var slider = sliders[key];
                slider.include(state.start[key]);
                slider.markStart(state.start[key]);
                slider.input.addEventListener("input", function () {
                    setValue(key, slider.valueAt(Number(slider.input.value)));
                });
            });
        }

        /** The sweep option of a feature name, compared as data. */
        function optionFor(feature) {
            return Array.prototype.find.call(select.options, function (option) {
                return option.value === feature;
            }) || null;
        }

        /** Select a parameter for the sweep and bring its slider into view, focused. */
        function tryParameter(feature) {
            var option = optionFor(feature);
            if (!option) {
                return;
            }
            select.value = feature;
            clearSweep();
            requestRender();
            var slider = sliders[option.getAttribute("data-hiw-param")];
            var header = document.querySelector(".cs-header");
            var room = window.innerHeight - (header ? header.offsetHeight : 0);
            var lab = document.getElementById("lab");
            var reach = slider.element.getBoundingClientRect().bottom - lab.getBoundingClientRect().top;
            // The lab's top when the slider still shows under it, otherwise the slider itself.
            var behavior = reducedMotion.matches ? "instant" : "smooth";
            if (reach < room - 16) {
                lab.scrollIntoView({ block: "start", behavior: behavior });
            } else {
                slider.element.scrollIntoView({ block: "center", behavior: behavior });
            }
            slider.input.focus({ preventScroll: true });
        }

        buildStorms();
        initSliders();

        elements.reset.addEventListener("click", reset);
        elements.play.addEventListener("click", function () {
            if (state.playing) {
                stop();
            } else {
                play();
            }
        });
        elements.scrubber.addEventListener("input", function () {
            seek(Number(elements.scrubber.value));
        });
        elements.sweepRun.addEventListener("click", function () {
            if (state.sweep) {
                clearSweep();
                requestRender();
            } else {
                startSweep();
            }
        });
        select.addEventListener("change", function () {
            clearSweep();
            requestRender();
        });
        root.querySelectorAll("input[name='lab-response']").forEach(function (input) {
            input.addEventListener("change", function () {
                state.quantity = input.value;
                response.setQuantity(input.value);
                elements.response.setAttribute("aria-label", describeResponse(state.sweep && state.sweep.shown === state.sweep.runs.length));
            });
        });
        var tryButtons = document.querySelectorAll("[data-hiw-try]");
        tryButtons.forEach(function (button) {
            button.addEventListener("click", function () {
                tryParameter(button.getAttribute("data-hiw-try"));
            });
        });

        // Shown before the first run, so the figures can measure their width.
        root.hidden = false;
        watchDock();
        recompute(true);
        state.index = state.playhead = peakIndex(state.result);
        render();
        tryButtons.forEach(function (button) {
            button.hidden = false;
        });
    }

    /** In place of a lab that failed to start: the no-JavaScript note, reworded. */
    function labUnavailable(root) {
        root.hidden = true;
        var note = document.createElement("p");
        note.className = "cs-hint cs-hint--note hiw-lab__noscript";
        note.textContent = "The interactive model could not be started. Everything else on this page reads without it.";
        root.after(note);
    }

    /** Run one feature of the page; a failure is logged and leaves the others working. */
    function safely(feature, onError) {
        try {
            feature();
        } catch (error) {
            console.error(error);
            if (onError) {
                onError();
            }
        }
    }

    document.addEventListener("DOMContentLoaded", function () {
        safely(function () {
            CS.math.render(document);
        });
        var lab = document.getElementById("hiw-lab");
        if (lab) {
            safely(function () {
                if (!CS.runoffModel) {
                    throw new Error("CS.runoffModel is not loaded.");
                }
                initLab(lab);
            }, function () {
                labUnavailable(lab);
            });
        }
        safely(watchScrollable);
        safely(restoreFragment);
    });
})(window.CS);
