"use strict";

document.addEventListener("DOMContentLoaded", function () {
    if (typeof Dropzone === "undefined") {
        return;
    }

    var dropzoneForm = document.getElementById("my-dropzone");
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
    var loginUrl = document.body ? document.body.dataset.loginUrl || "" : "";

    if (!uploadUrl) {
        return;
    }

    var preferredCatchmentValue = "";
    var catchmentSelect = document.getElementById("id_catchment_name");

    if (catchmentSelect) {
        preferredCatchmentValue = catchmentSelect.value || "";
        catchmentSelect.addEventListener("change", function () {
            preferredCatchmentValue = this.value || "";
        });
    }

    function fetchSubcatchments() {
        var select = document.getElementById("id_catchment_name");
        if (!select || !subcatchmentsUrl) {
            return;
        }
        if (select.value) {
            preferredCatchmentValue = select.value;
        }
        var previousValue = preferredCatchmentValue;
        select.disabled = true;
        select.innerHTML = '<option value="">Loading...</option>';
        fetch(subcatchmentsUrl, {
            headers: { "X-Requested-With": "XMLHttpRequest" },
        })
            .then(function (response) {
                return response.ok ? response.json() : null;
            })
            .then(function (data) {
                select.innerHTML = "";
                if (!data || !data.subcatchments || data.subcatchments.length === 0) {
                    var emptyOption = document.createElement("option");
                    emptyOption.value = "";
                    emptyOption.textContent = "--- Upload a file first ---";
                    select.appendChild(emptyOption);
                    preferredCatchmentValue = "";
                    return;
                }
                var placeholder = document.createElement("option");
                placeholder.value = "";
                placeholder.textContent = "--- Select catchment ---";
                select.appendChild(placeholder);

                var hasSelectedValue = false;
                data.subcatchments.forEach(function (name) {
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
            })
            .catch(function (error) {
                console.warn("subcatchments fetch failed:", error);
            })
            .finally(function () {
                select.disabled = false;
            });
    }

    function resetCatchmentDropdown() {
        var select = document.getElementById("id_catchment_name");
        if (!select) {
            return;
        }
        preferredCatchmentValue = "";
        select.innerHTML = '<option value="">--- Upload a file first ---</option>';
    }

    function showUploadStatus(filename) {
        var statusElement = document.getElementById("upload-status");
        var statusText = document.getElementById("upload-status-text");
        if (statusElement && statusText) {
            statusText.textContent = "Loaded: " + filename;
            statusElement.style.display = "";
        }
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

    var replacingUpload = false;
    var dzInstance = new Dropzone(dropzoneForm, {
        url: uploadUrl,
        acceptedFiles: ".inp",
        maxFilesize: 10,
        maxFiles: 1,
        addRemoveLinks: true,
        dictRemoveFile: "\u00d7",
        headers: {
            "X-Requested-With": "XMLHttpRequest",
            "X-CSRFToken": csrfToken,
        },
        init: function () {
            this.on("removedfile", function () {
                var messageElement = document.querySelector("#my-dropzone .dz-message");
                if (messageElement) {
                    messageElement.style.display = "";
                }
                var statusElement = document.getElementById("upload-status");
                if (statusElement) {
                    statusElement.style.display = "none";
                }
                if (!replacingUpload && clearUrl) {
                    fetch(clearUrl, {
                        method: "POST",
                        headers: {
                            "X-CSRFToken": csrfToken,
                            "X-Requested-With": "XMLHttpRequest",
                        },
                    }).catch(function (error) {
                        console.warn("upload clear failed:", error);
                    });
                    resetCatchmentDropdown();
                }
            });
            this.on("addedfile", function () {
                var messageElement = document.querySelector("#my-dropzone .dz-message");
                if (messageElement) {
                    messageElement.style.display = "none";
                }
            });
            this.on("success", function (file) {
                var statusElement = document.getElementById("upload-status");
                var statusText = document.getElementById("upload-status-text");
                if (statusElement && statusText) {
                    statusText.textContent = "Loaded: " + file.name;
                    statusElement.style.display = "";
                }
                fetchSubcatchments();
            });
            this.on("maxfilesexceeded", function (file) {
                replacingUpload = true;
                this.removeAllFiles();
                replacingUpload = false;
                this.addFile(file);
            });
        },
        error: function (file, response, xhr) {
            if (xhr && xhr.status === 401) {
                var redirectUrl = response && response.login_url ? response.login_url : loginUrl;
                alert("You must be logged in to upload files.");
                if (redirectUrl) {
                    window.location.href = redirectUrl + "?next=" + encodeURIComponent(window.location.pathname);
                }
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

    function showMockFile(filename, size) {
        replacingUpload = true;
        dzInstance.removeAllFiles();
        replacingUpload = false;

        var mockFile = {
            name: filename,
            size: size || 0,
            status: Dropzone.SUCCESS,
            accepted: true,
        };

        dzInstance.files.push(mockFile);
        dzInstance.emit("addedfile", mockFile);
        dzInstance.emit("success", mockFile);
        dzInstance.emit("complete", mockFile);
    }

    var sampleButton = document.getElementById("load-sample-data-button");
    if (sampleButton && sampleUploadUrl) {
        var defaultText = sampleButton.textContent;
        sampleButton.addEventListener("click", function () {
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
                        return parseJsonSafely(response).then(function (payload) {
                            var redirectUrl = payload && payload.login_url ? payload.login_url : loginUrl;
                            if (redirectUrl) {
                                window.location.href =
                                    redirectUrl + "?next=" + encodeURIComponent(window.location.pathname);
                            }
                            return null;
                        });
                    }
                    if (!response.ok) {
                        return parseJsonSafely(response).then(function (payload) {
                            throw new Error(
                                payload && payload.error ? payload.error : "Failed to load sample data."
                            );
                        });
                    }
                    return parseJsonSafely(response);
                })
                .then(function (data) {
                    if (!data) {
                        return;
                    }
                    var filename = data.filename || "example.inp";
                    showMockFile(filename, data.size);
                    showUploadStatus(filename);
                })
                .catch(function (error) {
                    alert(error.message || "Failed to load sample data.");
                })
                .finally(function () {
                    sampleButton.disabled = false;
                    sampleButton.textContent = defaultText;
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
                if (data && data.has_file) {
                    showMockFile(data.filename, data.size);
                    showUploadStatus(data.filename);
                    fetchSubcatchments();
                }
            })
            .catch(function (error) {
                console.warn("upload status check failed:", error);
            });
    }
});
