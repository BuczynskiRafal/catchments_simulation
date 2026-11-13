/*
 * "How it works" page (main/how_it_works.html).
 *
 * Typesets the equations (CS.math) and lets equations and tables that are
 * wider than the screen scroll from the keyboard.
 */
(function (CS) {
    "use strict";

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

    /** Typesetting changes the page's height above a #fragment target; land on it again. */
    function restoreFragment() {
        var target = window.location.hash && document.getElementById(decodeURIComponent(window.location.hash.slice(1)));
        if (target) {
            target.scrollIntoView({ block: "start", behavior: "instant" });
        }
    }

    /** Run one feature of the page; a failure is logged and leaves the others working. */
    function safely(feature) {
        try {
            feature();
        } catch (error) {
            console.error(error);
        }
    }

    document.addEventListener("DOMContentLoaded", function () {
        safely(function () {
            CS.math.render(document);
        });
        safely(watchScrollable);
        safely(restoreFragment);
    });
})(window.CS);
