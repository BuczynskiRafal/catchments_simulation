"use strict";

/**
 * Render a single line chart or subplot grid into the given container.
 *
 * @param {string}   containerId  - DOM id of the target div
 * @param {Object[]} records      - Array of row objects (from DataFrame.to_json orient="records"))
 * @param {string}   xField       - Key used for the x-axis
 * @param {string|string[]} yFields - Key(s) used for the y-axis
 * @param {Object}   [opts]       - Extra options
 * @param {string}   [opts.title]
 * @param {string}   [opts.xLabel]
 * @param {string}   [opts.yLabel]
 * @param {number[]} [opts.xRange]  - e.g. [0, 100]
 */
function renderLineChart(containerId, records, xField, yFields, opts) {
    opts = opts || {};
    var cols = Array.isArray(yFields) ? yFields : [yFields];
    var el = document.getElementById(containerId);
    if (!el) return;

    if (cols.length > 1) {
        var nCols = 2;
        var nRows = Math.ceil(cols.length / nCols);
        var traces = [];
        var layout = {
            height: 350 * nRows,
            title: {text: opts.title || '', x: 0.5, xanchor: 'center'},
            showlegend: false,
            grid: {rows: nRows, columns: nCols, pattern: 'independent'}
        };

        cols.forEach(function(col, i) {
            var r = Math.floor(i / nCols) + 1;
            var c = (i % nCols) + 1;
            var axisIdx = i === 0 ? '' : String(i + 1);
            traces.push({
                x: records.map(function(row) { return row[xField]; }),
                y: records.map(function(row) { return row[col]; }),
                type: 'scatter',
                mode: 'lines',
                name: col,
                xaxis: 'x' + axisIdx,
                yaxis: 'y' + axisIdx
            });
            layout['xaxis' + axisIdx] = {};
            layout['yaxis' + axisIdx] = {title: col};
            if (opts.xRange) {
                layout['xaxis' + axisIdx].range = opts.xRange;
            }
        });

        Plotly.newPlot(el, traces, layout, {responsive: true});
    } else {
        var yCol = cols[0];
        var traceData = [{
            x: records.map(function(row) { return row[xField]; }),
            y: records.map(function(row) { return row[yCol]; }),
            type: 'scatter',
            mode: 'lines',
            name: yCol
        }];
        var singleLayout = {
            title: {text: opts.title || '', x: 0.5, xanchor: 'center'}
        };
        if (opts.xLabel) singleLayout.xaxis = {title: opts.xLabel};
        if (opts.yLabel) singleLayout.yaxis = {title: opts.yLabel};
        if (opts.xRange) {
            singleLayout.xaxis = singleLayout.xaxis || {};
            singleLayout.xaxis.range = opts.xRange;
        }
        Plotly.newPlot(el, traceData, singleLayout, {responsive: true});
    }
}

/**
 * Render a sweep overlay chart with subplots.
 *
 * @param {string}   containerId
 * @param {Object}   sweepData   - {paramValue: [{datetime:…, col1:…, col2:…}, …], …}
 * @param {string[]} columns     - Timeseries column names to plot
 * @param {string}   feature     - Name of the swept parameter
 * @param {string}   catchmentName
 */
function renderSweepChart(containerId, sweepData, columns, feature, catchmentName) {
    var el = document.getElementById(containerId);
    if (!el) return;

    var nCols = 2;
    var nRows = Math.ceil(columns.length / nCols);
    var traces = [];
    var layout = {
        height: 350 * nRows,
        title: {
            text: 'Timeseries sweep: ' + feature + ' for ' + catchmentName,
            x: 0.5,
            xanchor: 'center'
        },
        grid: {rows: nRows, columns: nCols, pattern: 'independent'}
    };

    var paramValues = Object.keys(sweepData);
    paramValues.forEach(function(paramVal) {
        var rows = sweepData[paramVal];
        columns.forEach(function(col, i) {
            var axisIdx = i === 0 ? '' : String(i + 1);
            traces.push({
                x: rows.map(function(r) { return r.datetime; }),
                y: rows.map(function(r) { return r[col]; }),
                type: 'scatter',
                mode: 'lines',
                name: feature + '=' + paramVal,
                legendgroup: paramVal,
                showlegend: i === 0,
                xaxis: 'x' + axisIdx,
                yaxis: 'y' + axisIdx
            });
            layout['xaxis' + axisIdx] = layout['xaxis' + axisIdx] || {};
            layout['yaxis' + axisIdx] = layout['yaxis' + axisIdx] || {title: col};
        });
    });

    Plotly.newPlot(el, traces, layout, {responsive: true});
}
