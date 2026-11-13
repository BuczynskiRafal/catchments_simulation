/*
 * CS.math.render(root) — typesets the equations of a page with KaTeX.
 *
 * Every [data-cs-math] element under `root` (default: the document) holds its
 * TeX source as text; data-cs-math="display" sets it as a display equation,
 * any other value inline. Until KaTeX has loaded, and without JavaScript, the
 * element keeps showing the source, which the page styles as a legible
 * fallback. Rendered elements get .is-rendered and are skipped on later calls,
 * so rendering is idempotent and safe to repeat on content added later.
 * A formula KaTeX cannot parse is shown as its source in the error colour
 * (throwOnError: false) instead of breaking the page.
 */
(function (CS) {
    "use strict";

    var SELECTOR = "[data-cs-math]:not(.is-rendered)";

    /**
     * @param {ParentNode} [root=document]
     * @returns {boolean} false when KaTeX is not available (the sources stay as they are).
     */
    function render(root) {
        if (!window.katex) {
            return false;
        }
        (root || document).querySelectorAll(SELECTOR).forEach(function (element) {
            var tex = element.textContent.trim();
            window.katex.render(tex, element, {
                displayMode: element.getAttribute("data-cs-math") === "display",
                throwOnError: false,
                output: "htmlAndMathml",
            });
            element.setAttribute("data-cs-math-source", tex);
            element.classList.add("is-rendered");
        });
        return true;
    }

    CS.math = { render: render };
})(window.CS);
