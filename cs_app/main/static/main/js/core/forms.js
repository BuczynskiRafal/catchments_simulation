/*
 * Server-rendered forms (log in, create account, contact, profile), marked
 * with [data-cs-form]. Progressive enhancement only; the forms work without it.
 *  - Field errors rendered by crispy are tied to their inputs (aria-invalid,
 *    aria-describedby) and the first invalid field receives focus.
 *  - Password inputs get a show/hide toggle button (aria-pressed).
 */
(function () {
    "use strict";

    var EYE_ICON =
        '<svg class="cs-password-toggle__show" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>' +
        '<svg class="cs-password-toggle__hide" aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3l18 18"/><path d="M10.6 5.1A10.4 10.4 0 0 1 12 5c6.4 0 10 7 10 7a17 17 0 0 1-3.2 4.1M6.6 6.6C3.8 8.4 2 12 2 12s3.6 7 10 7a9.6 9.6 0 0 0 5.4-1.6"/><path d="M9.9 9.9a3 3 0 0 0 4.2 4.2"/></svg>';

    function describeBy(input, id) {
        var ids = (input.getAttribute("aria-describedby") || "").split(/\s+/).filter(Boolean);
        if (ids.indexOf(id) === -1) {
            ids.push(id);
            input.setAttribute("aria-describedby", ids.join(" "));
        }
    }

    function firstField(form) {
        return form.querySelector("input:not([type=hidden]):not([disabled]), select:not([disabled]), textarea:not([disabled])");
    }

    /** Crispy places `.invalid-feedback#error_N_<input id>` and `#<input id>_helptext` next to each field. */
    function linkErrors(form) {
        form.querySelectorAll(".is-invalid").forEach(function (input) {
            input.setAttribute("aria-invalid", "true");
            form.querySelectorAll('.invalid-feedback[id$="_' + input.id + '"]').forEach(function (feedback) {
                describeBy(input, feedback.id);
            });
        });
        form.querySelectorAll("input[id], select[id], textarea[id]").forEach(function (input) {
            if (document.getElementById(input.id + "_helptext")) {
                describeBy(input, input.id + "_helptext");
            }
        });

        // Errors that belong to no field (e.g. wrong credentials) render as one alert.
        var summary = form.querySelector(".alert-danger");
        var target = form.querySelector(".is-invalid") || (summary && firstField(form));
        if (summary && target) {
            summary.id = summary.id || form.id + "-errors";
            describeBy(target, summary.id);
        }
        if (target) {
            target.focus();
        }
    }

    function labelText(input) {
        var label = input.labels && input.labels[0];
        return label ? label.textContent.replace("*", "").trim().toLowerCase() : "password";
    }

    function addPasswordToggle(input) {
        var group = document.createElement("div");
        group.className = "input-group has-validation";
        input.parentNode.insertBefore(group, input);
        group.appendChild(input);

        var button = document.createElement("button");
        button.type = "button";
        button.className = "btn cs-password-toggle";
        button.setAttribute("aria-controls", input.id);
        button.setAttribute("aria-pressed", "false");
        button.setAttribute("aria-label", "Show " + labelText(input));
        button.title = button.getAttribute("aria-label");
        button.innerHTML = EYE_ICON;
        group.appendChild(button);

        // Bootstrap only shows feedback that follows the invalid input inside the group.
        var sibling = group.nextElementSibling;
        while (sibling && sibling.classList.contains("invalid-feedback")) {
            var next = sibling.nextElementSibling;
            group.appendChild(sibling);
            sibling = next;
        }

        button.addEventListener("click", function () {
            var reveal = input.type === "password";
            input.type = reveal ? "text" : "password";
            button.setAttribute("aria-pressed", String(reveal));
        });
    }

    function concealPasswords(form) {
        form.querySelectorAll(".cs-password-toggle[aria-pressed='true']").forEach(function (button) {
            button.click();
        });
    }

    document.querySelectorAll("form[data-cs-form]").forEach(function (form, index) {
        form.id = form.id || "cs-form-" + index;
        form.querySelectorAll("input[type=password][id]").forEach(addPasswordToggle);
        // Submit the password as a password field so browsers offer to save it.
        form.addEventListener("submit", function () {
            concealPasswords(form);
        });
        linkErrors(form);
    });
})();
