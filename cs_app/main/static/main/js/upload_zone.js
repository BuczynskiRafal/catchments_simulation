"use strict";

/*
 * Upload zone (main/_upload_zone.html).
 *
 * Events dispatched on `document`:
 *   "cs:model-changed"  detail: {filename, size, subcatchments: [name, ...], restored}
 *                       fired once the subcatchment list of a newly loaded model
 *                       (upload, sample data or restored session) is known;
 *                       `restored` is true when the session's model is unchanged: shown
 *                       on page load, or the same content loaded again (the server
 *                       answers `unchanged`), so results computed from it stay valid.
 *   "cs:model-cleared"  fired after the user removes the loaded model.
 */
document.addEventListener("DOMContentLoaded", function () {
    if (typeof Dropzone === "undefined") {
        return;
    }

    var dropzoneForm = document.getElementById("my-dropzone");
    // Also stops a second copy of this script when the partial is included twice.
    if (!dropzoneForm || dropzoneForm.dropzone) {
        return;
    }

    var csrfInput = document.querySelector("[name=csrfmiddlewaretoken]");
    if (!csrfInput || !csrfInput.value) {
        return;
    }

    var csrfToken = csrfInput.value;
    var uploadUrl = dropzoneForm.dataset.uploadUrl || dropzoneForm.getAttribute("action");
    var sampleUploadUrl = dropzoneForm.dataset.uploadSampleUrl;
    var clearUrl = dropzoneForm.dataset.uploadClearUrl;
    var statusUrl = dropzoneForm.dataset.uploadStatusUrl;
    var subcatchmentsUrl = dropzoneForm.dataset.subcatchmentsUrl;

    if (!uploadUrl) {
        return;
    }

    var wrapper = dropzoneForm.closest(".upload-zone-wrapper");
    var statusElement = document.getElementById("upload-status");
    var statusText = document.getElementById("upload-status-text");
    var statusMeta = document.getElementById("upload-status-meta");
    var removeButton = wrapper ? wrapper.querySelector(".upload-chip-remove") : null;
    var trigger = dropzoneForm.querySelector(".upload-trigger");

    var currentModel = null;
    // Incremented per subcatchment request so a stale response cannot overwrite a newer model.
    var subcatchmentsRequestId = 0;
    var preferredCatchmentValue = "";
    var catchmentSelect = document.getElementById("id_catchment_name");

    if (catchmentSelect) {
        preferredCatchmentValue = catchmentSelect.value || "";
        catchmentSelect.addEventListener("change", function () {
            preferredCatchmentValue = this.value || "";
        });
    }

    function notify(message, level) {
        if (window.CS && typeof window.CS.toast === "function") {
            window.CS.toast(message, level);
            return true;
        }
        window.alert(message);
        return false;
    }

    function handleUnauthorized() {
        var toastShown = notify("You must be logged in to upload files.", "warning");
        // Leave the toast on screen briefly; an alert has already blocked until dismissed.
        window.setTimeout(window.CS.redirectToLogin, toastShown ? 1500 : 0);
    }

    function formatFileSize(bytes) {
        if (!bytes) {
            return "";
        }
        if (bytes < 1024) {
            return bytes + " B";
        }
        if (bytes < 1024 * 1024) {
            return (bytes / 1024).toFixed(1) + " KB";
        }
        return (bytes / (1024 * 1024)).toFixed(1) + " MB";
    }

    function describeModel(model, subcatchments) {
        var parts = [formatFileSize(model.size)];
        if (subcatchments) {
            var count = subcatchments.length;
            parts.push(count + (count === 1 ? " subcatchment" : " subcatchments"));
        }
        return parts.filter(Boolean).join(" · ");
    }

    function setChipVisible(visible) {
        if (wrapper) {
            wrapper.classList.toggle("has-model", visible);
        }
        if (statusElement) {
            statusElement.style.display = visible ? "" : "none";
        }
    }

    /** names: the model's subcatchments, or null when they could not be read. */
    function populateCatchmentSelect(select, names, previousValue) {
        select.innerHTML = "";
        if (!names || names.length === 0) {
            var emptyOption = document.createElement("option");
            emptyOption.value = "";
            emptyOption.textContent = names
                ? "--- No subcatchments found in this model ---"
                : "--- Could not read subcatchments ---";
            select.appendChild(emptyOption);
            preferredCatchmentValue = "";
            return;
        }
        var placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.textContent = "--- Select subcatchment ---";
        select.appendChild(placeholder);

        var hasSelectedValue = false;
        names.forEach(function (name) {
            var option = document.createElement("option");
            option.value = name;
            option.textContent = name;
            if (name === previousValue) {
                option.selected = true;
                hasSelectedValue = true;
            }
            select.appendChild(option);
        });
        if (!hasSelectedValue) {
            preferredCatchmentValue = "";
        }
    }

    function announceModel(subcatchments) {
        if (!currentModel) {
            return;
        }
        if (statusMeta) {
            statusMeta.textContent = describeModel(currentModel, subcatchments);
        }
        document.dispatchEvent(new CustomEvent("cs:model-changed", {
            detail: {
                filename: currentModel.filename,
                size: currentModel.size,
                subcatchments: (subcatchments || []).slice(),
                restored: currentModel.restored,
            },
        }));
    }

    function fetchSubcatchments() {
        var requestId = ++subcatchmentsRequestId;
        var select = document.getElementById("id_catchment_name");
        var previousValue = "";
        if (select) {
            if (select.value) {
                preferredCatchmentValue = select.value;
            }
            previousValue = preferredCatchmentValue;
            select.disabled = true;
            select.innerHTML = '<option value="">Loading...</option>';
        }

        var request = subcatchmentsUrl
            ? fetch(subcatchmentsUrl, { headers: { "X-Requested-With": "XMLHttpRequest" } })
                .then(function (response) {
                    return response.ok ? response.json() : null;
                })
            : Promise.resolve(null);

        request
            .catch(function (error) {
                console.warn("subcatchments fetch failed:", error);
                return null;
            })
            .then(function (data) {
                if (requestId !== subcatchmentsRequestId) {
                    return;
                }
                // null means "unknown": the chip then shows the file size only.
                var names = data && Array.isArray(data.subcatchments) ? data.subcatchments : null;
                // Re-query in case the page swapped its form while the request was in flight.
                select = document.getElementById("id_catchment_name");
                if (select) {
                    populateCatchmentSelect(select, names, previousValue);
                    select.disabled = false;
                }
                announceModel(names);
            });
    }

    function resetCatchmentDropdown() {
        var select = document.getElementById("id_catchment_name");
        if (!select) {
            return;
        }
        preferredCatchmentValue = "";
        select.disabled = false;
        select.innerHTML = '<option value="">--- Upload a file first ---</option>';
    }

    function setModel(filename, size, restored) {
        currentModel = { filename: filename, size: size || 0, restored: restored };
        if (statusText) {
            statusText.textContent = filename;
        }
        if (statusMeta) {
            statusMeta.textContent = describeModel(currentModel, null);
        }
        setChipVisible(true);
        fetchSubcatchments();
    }

    function parseJsonSafely(response) {
        return response.text().then(function (body) {
            if (!body) {
                return {};
            }
            try {
                return JSON.parse(body);
            } catch (error) {
                return {};
            }
        });
    }

    var previewTemplate =
        '<div class="dz-preview dz-file-preview">' +
        '<div class="dz-details">' +
        '<span class="dz-filename" data-dz-name></span>' +
        '<span class="dz-size" data-dz-size></span>' +
        "</div>" +
        '<div class="dz-progress" aria-hidden="true"><span class="dz-upload" data-dz-uploadprogress></span></div>' +
        '<p class="dz-error-message" role="alert"><span data-dz-errormessage></span></p>' +
        '<button type="button" class="upload-preview-dismiss" data-dz-remove aria-label="Dismiss upload">' +
        '<span aria-hidden="true">×</span></button>' +
        "</div>";

    var dzInstance = new Dropzone(dropzoneForm, {
        url: uploadUrl,
        acceptedFiles: ".inp",
        maxFilesize: 10,
        maxFiles: 1,
        previewTemplate: previewTemplate,
        // Same units as the model chip (formatFileSize).
        filesizeBase: 1024,
        dictFileSizeUnits: { tb: "TB", gb: "GB", mb: "MB", kb: "KB", b: "B" },
        headers: {
            "X-Requested-With": "XMLHttpRequest",
            "X-CSRFToken": csrfToken,
        },
        init: function () {
            this.on("success", function (file, response) {
                setModel(file.name, file.size, Boolean(file.restored || (response && response.unchanged)));
            });
            this.on("removedfile", function () {
                // Dismissing a preview removes the focused button; keep focus in the upload zone.
                if (trigger && document.activeElement === document.body) {
                    trigger.focus();
                }
            });
            this.on("maxfilesexceeded", function (file) {
                this.removeAllFiles();
                this.addFile(file);
            });
        },
        error: function (file, response, xhr) {
            if (xhr && xhr.status === 401) {
                handleUnauthorized();
                return;
            }

            var message = "Upload failed.";
            if (xhr && xhr.status === 413) {
                message = response && response.error ? response.error : "File too large.";
            } else if (typeof response === "string") {
                message = response;
            } else if (response && response.error) {
                message = response.error;
            }

            if (file.previewElement) {
                file.previewElement.classList.add("dz-error");
                var errorElement = file.previewElement.querySelector("[data-dz-errormessage]");
                if (errorElement) {
                    errorElement.textContent = message;
                }
            }
        },
    });

    /** Show a model the server already holds (sample data, or `restored` from the session). */
    function showMockFile(filename, size, restored) {
        dzInstance.removeAllFiles();

        var mockFile = {
            name: filename,
            size: size || 0,
            status: Dropzone.SUCCESS,
            accepted: true,
            restored: restored,
        };

        dzInstance.files.push(mockFile);
        dzInstance.emit("addedfile", mockFile);
        dzInstance.emit("success", mockFile);
        dzInstance.emit("complete", mockFile);
    }

    function clearModel() {
        subcatchmentsRequestId++;
        currentModel = null;
        dzInstance.removeAllFiles(true);
        setChipVisible(false);
        resetCatchmentDropdown();
        if (clearUrl) {
            fetch(clearUrl, {
                method: "POST",
                headers: {
                    "X-CSRFToken": csrfToken,
                    "X-Requested-With": "XMLHttpRequest",
                },
            }).catch(function (error) {
                console.warn("upload clear failed:", error);
            });
        }
        document.dispatchEvent(new CustomEvent("cs:model-cleared"));
        // The remove button just disappeared; keep keyboard users inside the upload zone.
        if (trigger) {
            trigger.focus();
        }
    }

    if (removeButton) {
        removeButton.addEventListener("click", clearModel);
    }

    var sampleButton = document.getElementById("load-sample-data-button");
    if (sampleButton && sampleUploadUrl) {
        sampleButton.hidden = false;
        var defaultText = sampleButton.textContent;
        sampleButton.addEventListener("click", function () {
            var hadFocus = document.activeElement === sampleButton;
            sampleButton.disabled = true;
            sampleButton.textContent = "Loading sample data...";

            fetch(sampleUploadUrl, {
                method: "POST",
                headers: {
                    "X-CSRFToken": csrfToken,
                    "X-Requested-With": "XMLHttpRequest",
                },
            })
                .then(function (response) {
                    if (response.status === 401) {
                        handleUnauthorized();
                        return null;
                    }
                    if (!response.ok) {
                        return parseJsonSafely(response).then(function (payload) {
                            throw new Error(payload.error || "Failed to load sample data.");
                        });
                    }
                    return parseJsonSafely(response);
                })
                .then(function (data) {
                    if (data) {
                        showMockFile(data.filename || "example.inp", data.size, data.unchanged === true);
                    }
                })
                .catch(function (error) {
                    notify(error.message || "Failed to load sample data.", "danger");
                })
                .finally(function () {
                    sampleButton.disabled = false;
                    sampleButton.textContent = defaultText;
                    // Disabling the button dropped its focus; give it back to keyboard users.
                    if (hadFocus && document.activeElement === document.body) {
                        sampleButton.focus();
                    }
                });
        });
    }

    if (statusUrl) {
        fetch(statusUrl, {
            headers: { "X-Requested-With": "XMLHttpRequest" },
        })
            .then(function (response) {
                return response.ok ? response.json() : null;
            })
            .then(function (data) {
                // Ignore the restored state if the user loaded a model while this was in flight.
                if (data && data.has_file && !currentModel) {
                    showMockFile(data.filename, data.size, true);
                }
            })
            .catch(function (error) {
                console.warn("upload status check failed:", error);
            });
    }
});
