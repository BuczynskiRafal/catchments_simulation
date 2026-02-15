"use strict";

document.addEventListener("DOMContentLoaded", function () {
    var body = document.body;
    var loginUrl = body ? body.dataset.loginUrl : "";
    if (!loginUrl) {
        return;
    }

    var actionButtons = document.querySelectorAll("button[data-authenticated]");
    actionButtons.forEach(function (button) {
        button.addEventListener("click", function (event) {
            var isAuthenticated = button.getAttribute("data-authenticated") === "True";
            if (!isAuthenticated) {
                event.preventDefault();
                window.location.href = loginUrl;
            }
        });
    });
});
