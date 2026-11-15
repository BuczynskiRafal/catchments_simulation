/*
 * CS.readJson(id) — the payload of a page's json_script element (<script
 * type="application/json" id="…">): pages hand their scripts data this way, never as
 * inline script. Returns null when the element is missing or its JSON is invalid.
 */
(function (CS) {
    "use strict";

    CS.readJson = function (id) {
        var script = document.getElementById(id);
        if (!script) {
            return null;
        }
        try {
            return JSON.parse(script.textContent);
        } catch (error) {
            console.error("Invalid JSON in #" + id + ":", error);
            return null;
        }
    };
})(window.CS);
