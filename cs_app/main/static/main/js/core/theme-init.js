/*
 * Loaded synchronously in <head> so the colour mode is set before first paint.
 * Defines the minimal CS.theme primitives that core/theme.js builds on.
 */
(function () {
    "use strict";

    var STORAGE_KEY = "cs-theme";
    var MODES = ["light", "dark", "auto"];
    var darkQuery = window.matchMedia("(prefers-color-scheme: dark)");

    function storedMode() {
        var value = null;
        try {
            value = window.localStorage.getItem(STORAGE_KEY);
        } catch (error) {
            // Storage can be blocked (privacy mode); fall back to "auto".
        }
        return MODES.indexOf(value) !== -1 ? value : "auto";
    }

    function resolve(mode) {
        if (mode === "auto") {
            return darkQuery.matches ? "dark" : "light";
        }
        return mode;
    }

    function apply(mode) {
        var resolved = resolve(mode);
        document.documentElement.setAttribute("data-bs-theme", resolved);
        return resolved;
    }

    var CS = window.CS = window.CS || {};
    CS.theme = {
        STORAGE_KEY: STORAGE_KEY,
        MODES: MODES,
        darkQuery: darkQuery,
        storedMode: storedMode,
        resolve: resolve,
        apply: apply,
        /** Resolved theme currently on <html>: "light" | "dark". */
        current: function () {
            return document.documentElement.getAttribute("data-bs-theme") === "dark" ? "dark" : "light";
        },
    };

    apply(storedMode());
})();
