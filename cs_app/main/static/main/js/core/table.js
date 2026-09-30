/*
 * CS.table.sortable(tableEl) — click/Enter/Space on a header sorts the first
 *   tbody by that column (numeric-aware, ascending then descending) and sets
 *   aria-sort on the sorted <th>. Header text is wrapped in a .cs-sort button
 *   unless the header already contains one; <th data-cs-sort="false"> opts out.
 *   A cell's data-sort-value overrides its text as the sort key. Calling it again
 *   on the same table is a no-op.
 * CS.table.toCSV(tableEl) — RFC 4180 CSV of the header and body rows.
 */
(function (CS) {
    "use strict";

    var collator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });

    function cellKey(cell) {
        return cell ? (cell.dataset.sortValue || cell.textContent).trim() : "";
    }

    function toNumber(text) {
        var normalized = text.replace(/[,\s ]/g, "");
        return /^[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?$/i.test(normalized) ? parseFloat(normalized) : NaN;
    }

    // Numbers before text; text compared naturally ("S2" < "S10").
    function compareKeys(a, b) {
        var numA = toNumber(a);
        var numB = toNumber(b);
        var aIsNum = !Number.isNaN(numA);
        var bIsNum = !Number.isNaN(numB);
        if (aIsNum && bIsNum) {
            return numA - numB;
        }
        if (aIsNum !== bIsNum) {
            return aIsNum ? -1 : 1;
        }
        return collator.compare(a, b);
    }

    function ensureSortButton(th) {
        var button = th.querySelector(".cs-sort");
        if (button) {
            return button;
        }
        button = document.createElement("button");
        button.type = "button";
        button.className = "cs-sort";
        button.append.apply(button, Array.prototype.slice.call(th.childNodes));
        th.append(button);
        return button;
    }

    function sortBy(table, headers, th) {
        var body = table.tBodies[0];
        if (!body) {
            return;
        }
        var direction = th.getAttribute("aria-sort") === "ascending" ? "descending" : "ascending";
        var sign = direction === "ascending" ? 1 : -1;
        var column = th.cellIndex;

        Array.prototype.slice
            .call(body.rows)
            .sort(function (rowA, rowB) {
                return sign * compareKeys(cellKey(rowA.cells[column]), cellKey(rowB.cells[column]));
            })
            .forEach(function (row) {
                body.appendChild(row);
            });

        headers.forEach(function (header) {
            header.removeAttribute("aria-sort");
        });
        th.setAttribute("aria-sort", direction);
    }

    function csvField(text) {
        var value = text.trim().replace(/\s+/g, " ");
        // Neutralise spreadsheet formulas in text cells (numbers such as -1.5 stay intact).
        if (/^[=+\-@]/.test(value) && Number.isNaN(toNumber(value))) {
            value = "'" + value;
        }
        return /[",\r\n]/.test(value) ? '"' + value.replace(/"/g, '""') + '"' : value;
    }

    CS.table = {
        sortable: function (tableEl) {
            var headerRow = tableEl.tHead && tableEl.tHead.rows[0];
            if (!headerRow || tableEl.hasAttribute("data-cs-sortable")) {
                return;
            }
            tableEl.setAttribute("data-cs-sortable", "");
            var headers = Array.prototype.filter.call(headerRow.cells, function (th) {
                return th.dataset.csSort !== "false";
            });
            headers.forEach(function (th) {
                ensureSortButton(th).addEventListener("click", function () {
                    sortBy(tableEl, headers, th);
                });
            });
        },

        toCSV: function (tableEl) {
            return Array.prototype.map
                .call(tableEl.querySelectorAll("thead tr, tbody tr"), function (row) {
                    return Array.prototype.map
                        .call(row.cells, function (cell) {
                            return csvField(cell.textContent);
                        })
                        .join(",");
                })
                .join("\r\n");
        },
    };
})(window.CS);
