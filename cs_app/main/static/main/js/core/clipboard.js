/*
 * CS.clipboard.copy(text, successMessage) — copies text and confirms with a toast.
 * Resolves to true on success, false otherwise (never rejects).
 */
(function (CS) {
    "use strict";

    CS.clipboard = {
        copy: function (text, successMessage) {
            if (!navigator.clipboard || !window.isSecureContext) {
                CS.toast("Copying is not available here. Select the text and copy it manually.", "warning");
                return Promise.resolve(false);
            }
            return navigator.clipboard.writeText(text).then(
                function () {
                    CS.toast(successMessage || "Copied to clipboard.", "success");
                    return true;
                },
                function () {
                    CS.toast("Could not copy. Select the text and copy it manually.", "danger");
                    return false;
                }
            );
        },
    };
})(window.CS);
