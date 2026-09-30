/*
 * Theme toggle (light / dark / auto). Persists the chosen mode in localStorage
 * and dispatches `cs:themechange` on document with {detail: {theme}} whenever
 * the resolved theme changes, including system changes while in "auto".
 * Controls: any button with data-cs-theme-value="light|dark|auto"; its
 * aria-pressed reflects the active mode. [data-cs-theme-label] shows the mode.
 */
(function (CS) {
    "use strict";

    var theme = CS.theme;
    var LABELS = { light: "Light", dark: "Dark", auto: "Auto" };
    var mode = theme.storedMode();

    function dispatchChange(resolved) {
        document.dispatchEvent(new CustomEvent("cs:themechange", { detail: { theme: resolved } }));
    }

    // Switch colours in one step: without this, elements with colour transitions
    // briefly show light-theme text on the dark background (and vice versa).
    var restoreFrame = null;

    function applyWithoutTransitions() {
        var root = document.documentElement;
        window.cancelAnimationFrame(restoreFrame);
        root.classList.add("cs-theme-switching");
        var resolved = theme.apply(mode);
        window.getComputedStyle(root).getPropertyValue("color"); // flush styles before re-enabling
        restoreFrame = window.requestAnimationFrame(function () {
            root.classList.remove("cs-theme-switching");
        });
        return resolved;
    }

    function applyMode() {
        var previous = theme.current();
        if (theme.resolve(mode) === previous) {
            return;
        }
        dispatchChange(applyWithoutTransitions());
    }

    function persist() {
        try {
            if (mode === "auto") {
                window.localStorage.removeItem(theme.STORAGE_KEY);
            } else {
                window.localStorage.setItem(theme.STORAGE_KEY, mode);
            }
        } catch (error) {
            // Not persisted when storage is blocked; the choice still applies to this page.
        }
    }

    function syncControls() {
        document.querySelectorAll("[data-cs-theme-value]").forEach(function (control) {
            control.setAttribute("aria-pressed", String(control.getAttribute("data-cs-theme-value") === mode));
        });
        document.querySelectorAll("[data-cs-theme-label]").forEach(function (label) {
            label.textContent = LABELS[mode];
        });
    }

    /** Selected mode: "light" | "dark" | "auto". */
    theme.mode = function () {
        return mode;
    };

    theme.set = function (nextMode) {
        if (theme.MODES.indexOf(nextMode) === -1) {
            return;
        }
        mode = nextMode;
        persist();
        applyMode();
        syncControls();
    };

    theme.darkQuery.addEventListener("change", function () {
        if (mode === "auto") {
            applyMode();
        }
    });

    document.addEventListener("click", function (event) {
        var control = event.target.closest("[data-cs-theme-value]");
        if (!control) {
            return;
        }
        theme.set(control.getAttribute("data-cs-theme-value"));
        // The dropdown closes and hides the chosen item; keep focus on its toggle, not <body>.
        var menu = control.closest(".dropdown-menu[aria-labelledby]");
        var toggle = menu && document.getElementById(menu.getAttribute("aria-labelledby"));
        if (toggle) {
            toggle.focus();
        }
    });

    syncControls();
})(window.CS);
