/*
 * CS.asyncForm(options) — progressive enhancement for run forms (DESIGN_SPEC §4).
 *
 * options:
 *   form          <form> to enhance (without JS it keeps its native POST)
 *   submitButton  the run button; only submits from it (or implicit/requestSubmit
 *                 submits without a submitter) are intercepted, so download buttons
 *                 with formaction keep submitting natively and never show loading
 *   target        element whose innerHTML receives the 200 results fragment; focus then
 *                 moves to its first heading (tabindex=-1 added when needed). A bubbling
 *                 `cs:beforeswap` event is dispatched on it just before the swap (charts.js
 *                 releases the charts about to be replaced)
 *   loadingEl     optional progress element, toggled with the `hidden` attribute;
 *                 a descendant [data-cs-elapsed] shows elapsed seconds
 *   successMessage optional toast text shown after a successful run ("Simulation finished")
 *   onSuccess     optional (target) => void, called after the fragment is swapped in
 *   onError       optional ({status, message, field_errors}) => void, after the inline
 *                 errors and the error toast are shown (do not toast again)
 *
 * Error responses: 4xx/5xx JSON {message, field_errors} render inline Bootstrap
 * field errors plus a role="alert" summary in the form, and a short danger toast;
 * 401 redirects to login; anything else (non-JSON body, network failure) shows
 * TIMEOUT_MESSAGE. Results kept from an earlier run (a [data-cs-results] element in
 * target) get a .cs-stale-notice above them. Every new run dismisses the toasts and
 * the notice left by the previous one.
 *
 * Loading another model or removing it (the upload zone's cs:model-changed, unless
 * `restored`, and cs:model-cleared) marks the results shown as stale too, and a run
 * still in flight is discarded when it returns: both belong to the previous model.
 */
(function (CS) {
    "use strict";

    var TIMEOUT_MESSAGE =
        "The server did not finish the run. It may have timed out — try a smaller range " +
        "or reload the page to check for results.";
    var FIELDS_MESSAGE = "Please correct the highlighted fields.";
    var FAILED_MESSAGE = "The run could not be completed. Please try again.";
    var ERROR_TOAST = "The run did not complete. See the form for details.";
    var STALE_MESSAGE = "Showing results from the previous run: the latest run did not complete.";
    var MODEL_STALE_MESSAGE = "These results are for the previously loaded model. Run again to update them.";
    var DISCARDED_MESSAGE = "A run for the previously loaded model finished; its results were discarded.";
    var ERROR_MARK = "data-cs-async-error";
    var INVALID_MARK = "data-cs-async-invalid";
    var STALE_CLASS = "cs-stale-notice";

    function startElapsed(loadingEl) {
        if (!loadingEl) {
            return function () {};
        }
        var counter = loadingEl.querySelector("[data-cs-elapsed]");
        var startedAt = Date.now();
        function render() {
            if (counter) {
                counter.textContent = Math.floor((Date.now() - startedAt) / 1000) + " s";
            }
        }
        render();
        loadingEl.hidden = false;
        var timer = window.setInterval(render, 1000);
        return function stop() {
            window.clearInterval(timer);
            loadingEl.hidden = true;
        };
    }

    function focusResults(container) {
        var focusTarget = container.querySelector("h1, h2, h3, h4, h5, h6") || container;
        if (!focusTarget.hasAttribute("tabindex")) {
            focusTarget.setAttribute("tabindex", "-1");
        }
        focusTarget.focus();
    }

    function fieldFor(form, name) {
        var field = form.elements.namedItem(name);
        if (field instanceof RadioNodeList) {
            return field[field.length - 1] || null;
        }
        return field;
    }

    function clearFieldError(field) {
        var feedbackId = field.getAttribute(INVALID_MARK);
        if (feedbackId === null) {
            return;
        }
        var feedback = document.getElementById(feedbackId);
        if (feedback) {
            feedback.remove();
        }
        var describedBy = (field.getAttribute("aria-describedby") || "")
            .split(/\s+/)
            .filter(function (id) {
                return id && id !== feedbackId;
            });
        if (describedBy.length) {
            field.setAttribute("aria-describedby", describedBy.join(" "));
        } else {
            field.removeAttribute("aria-describedby");
        }
        field.classList.remove("is-invalid");
        field.removeAttribute("aria-invalid");
        field.removeAttribute(INVALID_MARK);
    }

    function clearErrors(form) {
        form.querySelectorAll("[" + INVALID_MARK + "]").forEach(clearFieldError);
        form.querySelectorAll("[" + ERROR_MARK + "]").forEach(function (el) {
            el.remove();
        });
    }

    function showFieldError(field, messages) {
        var feedbackId = (field.id || field.name) + "-async-error";
        var feedback = document.createElement("div");
        feedback.id = feedbackId;
        feedback.className = "invalid-feedback";
        feedback.setAttribute(ERROR_MARK, "");
        feedback.textContent = messages.join(" ");
        field.insertAdjacentElement("afterend", feedback);

        field.classList.add("is-invalid");
        field.setAttribute("aria-invalid", "true");
        field.setAttribute(INVALID_MARK, feedbackId);
        var describedBy = field.getAttribute("aria-describedby");
        field.setAttribute("aria-describedby", describedBy ? describedBy + " " + feedbackId : feedbackId);
    }

    function showSummary(form, message, details) {
        var alert = document.createElement("div");
        alert.className = "alert alert-danger";
        alert.setAttribute("role", "alert");
        alert.setAttribute(ERROR_MARK, "");

        var text = document.createElement("p");
        text.className = "mb-0";
        text.textContent = message;
        alert.append(text);

        if (details.length) {
            var list = document.createElement("ul");
            list.className = "mb-0 mt-1";
            details.forEach(function (detail) {
                var item = document.createElement("li");
                item.textContent = detail;
                list.append(item);
            });
            alert.append(list);
        }
        form.prepend(alert);
    }

    /** Returns the first field marked invalid, if any. */
    function renderErrors(form, error) {
        var unplaced = [];
        var firstInvalid = null;
        Object.keys(error.field_errors).forEach(function (name) {
            var messages = [].concat(error.field_errors[name]).map(String);
            var field = name === "__all__" ? null : fieldFor(form, name);
            if (field && field.nodeType === Node.ELEMENT_NODE) {
                showFieldError(field, messages);
                firstInvalid = firstInvalid || field;
            } else {
                unplaced = unplaced.concat(messages);
            }
        });
        var summary = error.message || unplaced.shift() || (firstInvalid ? FIELDS_MESSAGE : FAILED_MESSAGE);
        showSummary(form, summary, unplaced);
        return firstInvalid;
    }

    async function readError(response) {
        var isJson = (response.headers.get("Content-Type") || "").indexOf("application/json") !== -1;
        var payload = null;
        if (isJson) {
            try {
                payload = await response.json();
            } catch (parseError) {
                payload = null;
            }
        }
        if (!payload || typeof payload !== "object") {
            return { status: response.status, message: TIMEOUT_MESSAGE, field_errors: {} };
        }
        return {
            status: response.status,
            message: typeof payload.message === "string" ? payload.message : "",
            field_errors:
                payload.field_errors && typeof payload.field_errors === "object" ? payload.field_errors : {},
        };
    }

    async function send(form, submitter) {
        var body = new FormData(form);
        if (submitter && submitter.name) {
            body.append(submitter.name, submitter.value);
        }
        var response;
        try {
            response = await fetch(form.getAttribute("action") || window.location.href, {
                method: "POST",
                body: body,
                headers: { "X-Requested-With": "XMLHttpRequest" },
                credentials: "same-origin",
            });
        } catch (networkError) {
            return { ok: false, error: { status: 0, message: TIMEOUT_MESSAGE, field_errors: {} } };
        }
        if (response.ok) {
            return { ok: true, html: await response.text() };
        }
        return { ok: false, error: await readError(response) };
    }

    function hideToast(toastEl) {
        var toast = window.bootstrap && window.bootstrap.Toast.getInstance(toastEl);
        if (toast) {
            toast.hide();
        }
    }

    function removeStaleNotices(target) {
        target.querySelectorAll("." + STALE_CLASS).forEach(function (notice) {
            notice.remove();
        });
    }

    function showStaleNotice(target, message) {
        if (!target.querySelector("[data-cs-results]")) {
            return;
        }
        removeStaleNotices(target);
        var notice = document.createElement("p");
        notice.className = STALE_CLASS;
        notice.textContent = message;
        target.prepend(notice);
    }

    function swap(target, html) {
        target.dispatchEvent(new CustomEvent("cs:beforeswap", { bubbles: true }));
        target.innerHTML = html;
    }

    CS.asyncForm = function (options) {
        var form = options.form;
        var button = options.submitButton;
        var target = options.target;
        var running = false;
        var runToasts = [];
        // Bumped whenever the model changes, so a run started before can tell it is outdated.
        var modelVersion = 0;

        function toast(message, level) {
            var toastEl = CS.toast(message, level);
            if (toastEl) {
                runToasts.push(toastEl);
            }
        }

        function clearPreviousRun() {
            clearErrors(form);
            runToasts.splice(0).forEach(hideToast);
            removeStaleNotices(target);
        }

        function setRunning(isRunning) {
            running = isRunning;
            button.disabled = isRunning;
            [button, target].forEach(function (el) {
                if (isRunning) {
                    el.setAttribute("aria-busy", "true");
                } else {
                    el.removeAttribute("aria-busy");
                }
            });
        }

        async function run(submitter) {
            clearPreviousRun();
            setRunning(true);
            var runModelVersion = modelVersion;
            var stopElapsed = startElapsed(options.loadingEl);
            var result;
            try {
                result = await send(form, submitter);
            } catch (readFailure) {
                // The body stream broke mid-read: same outcome as a dropped connection.
                result = { ok: false, error: { status: 0, message: TIMEOUT_MESSAGE, field_errors: {} } };
            } finally {
                stopElapsed();
                setRunning(false);
            }

            if (runModelVersion !== modelVersion) {
                toast(DISCARDED_MESSAGE, "info");
                return;
            }
            if (result.ok) {
                swap(target, result.html);
                if (options.onSuccess) {
                    options.onSuccess(target);
                }
                focusResults(target);
                if (options.successMessage) {
                    toast(options.successMessage, "success");
                }
                return;
            }
            if (result.error.status === 401 && CS.redirectToLogin()) {
                return;
            }
            // Disabling the button dropped its focus; return it to where the fix is needed.
            (renderErrors(form, result.error) || button).focus();
            showStaleNotice(target, STALE_MESSAGE);
            toast(ERROR_TOAST, "danger");
            if (options.onError) {
                options.onError(result.error);
            }
        }

        form.addEventListener("submit", function (event) {
            var submitter = event.submitter;
            if (event.defaultPrevented || (submitter && submitter !== button)) {
                return;
            }
            event.preventDefault();
            if (!running) {
                run(submitter);
            }
        });

        function onModelSwitch(event) {
            if (event.detail && event.detail.restored) {
                return;
            }
            modelVersion += 1;
            showStaleNotice(target, MODEL_STALE_MESSAGE);
        }
        document.addEventListener("cs:model-changed", onModelSwitch);
        document.addEventListener("cs:model-cleared", onModelSwitch);

        form.addEventListener("input", function (event) {
            var field = event.target.closest("[" + INVALID_MARK + "]");
            if (field) {
                clearFieldError(field);
            }
        });
    };
})(window.CS);
