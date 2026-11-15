/*
 * Workbench progress on the tool pages (simulation, timeseries, calculations).
 *
 * Steps: every .cs-step[data-cs-step="model"|"run"] of the control panel carries
 * data-state "todo" | "active" | "done". The server renders the first state; the upload
 * zone's cs:model-changed / cs:model-cleared keep it current: the model step is done
 * while a model is loaded (active otherwise), the run step is todo until then (muted,
 * still usable) and active after. A visually hidden [data-cs-step-status] in each
 * title says the same to assistive technology.
 *
 * Checklist: the items of the empty results state's [data-cs-checklist]
 * (data-check = "model" | "catchment" | "run") are checked as the user completes them;
 * [data-cs-next] shows the data-next sentence of the first open item.
 *
 * Subcatchment select (#id_catchment_name): [data-cs-catchment-hint] says how many
 * subcatchments there are to choose from. A run form ([data-cs-run-form]) sent without
 * a model or without a subcatchment gets an inline message at the control that needs
 * attention, which takes the focus, instead of only the browser's validation bubble.
 * Sent while the subcatchments of a model just loaded are still being listed (the select
 * is disabled), the run waits for them and starts once the model is announced.
 */
(function () {
    "use strict";

    var STEP_STATUS = { done: " (completed)", active: "", todo: " (load a model first)" };
    var CHECK_STATUS = { done: " (done)", current: " (next)", todo: "" };
    var NO_MODEL_MESSAGE = "Load a model first: drop an .inp file here or try the sample data.";
    var NO_CATCHMENT_MESSAGE = "Choose the subcatchment to run.";
    var WAITING_MESSAGE = "Reading the model's subcatchments; the run starts as soon as they are listed.";
    var ATTENTION_MARK = "data-cs-attention";

    var state = { model: false, count: 0 };
    // A run asked for while the subcatchments were loading: {form, submitter}.
    var queuedRun = null;

    function catchmentSelect() {
        return document.getElementById("id_catchment_name");
    }

    function hasCatchment() {
        var select = catchmentSelect();
        return Boolean(select && select.value);
    }

    function setStepState(step, value) {
        step.dataset.state = value;
        var status = step.querySelector("[data-cs-step-status]");
        if (status) {
            status.textContent = STEP_STATUS[value];
        }
    }

    function updateSteps() {
        document.querySelectorAll(".cs-step[data-cs-step]").forEach(function (step) {
            if (step.dataset.csStep === "model") {
                setStepState(step, state.model ? "done" : "active");
            } else {
                setStepState(step, state.model ? "active" : "todo");
            }
        });
    }

    function updateChecklist() {
        var list = document.querySelector("[data-cs-checklist]");
        if (!list) {
            return;
        }
        var completed = { model: state.model, catchment: state.model && hasCatchment(), run: false };
        var next = null;
        list.querySelectorAll("[data-check]").forEach(function (item) {
            var value = "todo";
            if (completed[item.dataset.check]) {
                value = "done";
            } else if (!next) {
                next = item;
                value = "current";
            }
            item.dataset.state = value;
            var status = item.querySelector("[data-check-status]");
            if (status) {
                status.textContent = CHECK_STATUS[value];
            }
        });
        var nextEl = document.querySelector("[data-cs-next]");
        if (nextEl) {
            nextEl.textContent = next ? next.dataset.next : "";
        }
    }

    function updateCatchmentHint() {
        var hint = document.querySelector("[data-cs-catchment-hint]");
        if (!hint) {
            return;
        }
        var several = state.model && state.count > 1;
        hint.textContent = several ? "Choose one of the " + state.count + " subcatchments in this model." : "";
        hint.hidden = !several;
    }

    function update() {
        updateSteps();
        updateChecklist();
        updateCatchmentHint();
    }

    // ── Inline messages at the control that needs attention ─────────────

    function clearAttention() {
        document.querySelectorAll("[" + ATTENTION_MARK + "]").forEach(function (el) {
            if (el.tagName === "SELECT") {
                var ownId = el.getAttribute(ATTENTION_MARK);
                var describedBy = (el.getAttribute("aria-describedby") || "").split(/\s+/).filter(function (id) {
                    return id && id !== ownId;
                });
                if (describedBy.length) {
                    el.setAttribute("aria-describedby", describedBy.join(" "));
                } else {
                    el.removeAttribute("aria-describedby");
                }
                el.classList.remove("is-invalid");
                el.removeAttribute("aria-invalid");
                el.removeAttribute(ATTENTION_MARK);
            } else {
                el.remove();
            }
        });
    }

    function message(text, id) {
        var el = document.createElement("p");
        el.id = id;
        el.className = "cs-attention";
        el.setAttribute("role", "alert");
        el.setAttribute(ATTENTION_MARK, "");
        el.textContent = text;
        return el;
    }

    function askForModel() {
        var step = document.querySelector('.cs-step[data-cs-step="model"]');
        var zone = step && step.querySelector(".upload-zone-wrapper");
        if (!zone) {
            return false;
        }
        zone.insertAdjacentElement("afterend", message(NO_MODEL_MESSAGE, "cs-model-attention"));
        var trigger = zone.querySelector(".upload-trigger");
        if (trigger) {
            trigger.focus();
        }
        return true;
    }

    function askForCatchment(select) {
        var id = select.id + "-attention";
        var note = message(NO_CATCHMENT_MESSAGE, id);
        note.classList.add("cs-attention--field");
        select.insertAdjacentElement("afterend", note);
        select.classList.add("is-invalid");
        select.setAttribute("aria-invalid", "true");
        select.setAttribute(ATTENTION_MARK, id);
        var describedBy = select.getAttribute("aria-describedby");
        select.setAttribute("aria-describedby", describedBy ? describedBy + " " + id : id);
        select.focus();
        return true;
    }

    /** Run a queued submit once the model is announced (after the other listeners have run). */
    function releaseQueuedRun() {
        var run = queuedRun;
        queuedRun = null;
        if (run) {
            window.setTimeout(function () {
                run.form.requestSubmit(run.submitter);
            }, 0);
        }
    }

    function initRunForm(form) {
        // Registered before the page's CS.asyncForm, which skips a submit prevented here.
        form.addEventListener("submit", function (event) {
            var select = catchmentSelect();
            if (!select || !select.disabled || !form.contains(select)) {
                return;
            }
            event.preventDefault();
            clearAttention();
            var note = message(WAITING_MESSAGE, "cs-run-waiting");
            note.setAttribute("role", "status");
            select.insertAdjacentElement("afterend", note);
            queuedRun = { form: form, submitter: event.submitter || null };
        });
        // `invalid` does not bubble: listen in the capture phase.
        form.addEventListener("invalid", function (event) {
            var select = catchmentSelect();
            if (event.target !== select) {
                return;
            }
            clearAttention();
            if (state.model ? askForCatchment(select) : askForModel()) {
                event.preventDefault();
            }
        }, true);
    }

    document.addEventListener("DOMContentLoaded", function () {
        var modelStep = document.querySelector('.cs-step[data-cs-step="model"]');
        if (!modelStep) {
            return;
        }
        var select = catchmentSelect();
        state.model = modelStep.dataset.state === "done";
        state.count = select
            ? Array.prototype.filter.call(select.options, function (option) {
                return option.value;
            }).length
            : 0;

        document.addEventListener("cs:model-changed", function (event) {
            state.model = true;
            state.count = (event.detail && event.detail.subcatchments || []).length;
            clearAttention();
            update();
            releaseQueuedRun();
        });
        document.addEventListener("cs:model-cleared", function () {
            queuedRun = null;
            state.model = false;
            state.count = 0;
            clearAttention();
            update();
        });
        document.addEventListener("change", function (event) {
            if (event.target === catchmentSelect()) {
                clearAttention();
                updateChecklist();
            }
        });
        document.querySelectorAll("form[data-cs-run-form]").forEach(initRunForm);
        update();
    });
})();
