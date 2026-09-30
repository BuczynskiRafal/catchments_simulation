/*
 * Workbench control panel (.cs-panel): on wide screens it sticks below the
 * header (.is-sticky, see app.css) only while it fits the viewport. A taller
 * panel scrolls with the page instead of inside itself, so none of its
 * controls can end up out of reach.
 */
(function () {
    "use strict";

    var WIDE_QUERY = window.matchMedia("(min-width: 992px)");
    // Room kept above and below a sticky panel (1rem each, as in app.css).
    var GUTTER_PX = 32;

    var panels = document.querySelectorAll(".cs-panel");
    if (!panels.length) {
        return;
    }
    var header = document.querySelector(".cs-header");

    function update() {
        var available = window.innerHeight - (header ? header.offsetHeight : 0) - GUTTER_PX;
        panels.forEach(function (panel) {
            panel.classList.toggle("is-sticky", WIDE_QUERY.matches && panel.offsetHeight <= available);
        });
    }

    // The panel grows and shrinks with its content (sweep fields, error summaries, the model chip).
    if (window.ResizeObserver) {
        var observer = new ResizeObserver(update);
        panels.forEach(function (panel) {
            observer.observe(panel);
        });
    }
    window.addEventListener("resize", update);
    update();
})();
