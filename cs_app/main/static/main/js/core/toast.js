/*
 * CS.toast(message, level) — transient feedback for async flows.
 * level: "success" | "info" | "warning" | "danger" (default "info").
 * Renders into the base template's .toast-container (aria-live="polite").
 * The message is set via textContent, so user data is never parsed as HTML.
 */
(function (CS) {
    "use strict";

    var LEVELS = ["success", "info", "warning", "danger"];
    var DELAY_MS = { success: 4000, info: 5000, warning: 7000, danger: 8000 };

    CS.toast = function (message, level) {
        var region = document.querySelector(".toast-container");
        if (!region || !window.bootstrap) {
            return null;
        }
        var safeLevel = LEVELS.indexOf(level) !== -1 ? level : "info";

        var toastEl = document.createElement("div");
        toastEl.className = "toast cs-toast cs-toast--" + safeLevel;
        toastEl.setAttribute("aria-atomic", "true");

        var row = document.createElement("div");
        row.className = "d-flex align-items-start";

        var body = document.createElement("div");
        body.className = "toast-body";
        body.textContent = message;

        var close = document.createElement("button");
        close.type = "button";
        close.className = "btn-close me-2 mt-2";
        close.setAttribute("data-bs-dismiss", "toast");
        close.setAttribute("aria-label", "Close");

        row.append(body, close);
        toastEl.append(row);
        region.append(toastEl);

        toastEl.addEventListener("hidden.bs.toast", function () {
            toastEl.remove();
        });
        window.bootstrap.Toast.getOrCreateInstance(toastEl, { delay: DELAY_MS[safeLevel] }).show();
        return toastEl;
    };
})(window.CS);
