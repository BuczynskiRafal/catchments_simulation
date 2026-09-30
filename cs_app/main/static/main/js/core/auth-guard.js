/*
 * CS.redirectToLogin() sends the user to body[data-login-url] with ?next= set to
 * the current page. Buttons rendered with data-authenticated="False" use it
 * instead of submitting (delegated, so buttons swapped in later are covered).
 */
(function (CS) {
    "use strict";

    CS.redirectToLogin = function () {
        var loginUrl = document.body.dataset.loginUrl;
        if (!loginUrl) {
            return false;
        }
        var url = new URL(loginUrl, window.location.href);
        url.searchParams.set("next", window.location.pathname + window.location.search);
        window.location.assign(url.href);
        return true;
    };

    document.addEventListener("click", function (event) {
        var button = event.target.closest("button[data-authenticated]");
        if (!button || button.getAttribute("data-authenticated") === "True") {
            return;
        }
        if (CS.redirectToLogin()) {
            event.preventDefault();
        }
    });
})(window.CS);
