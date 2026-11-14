/*
 * CS.runoffModel — the subcatchment runoff model of EPA SWMM 5, in the browser.
 *
 * SWMM splits a subcatchment into three sub-areas: pervious, impervious with
 * depression storage and impervious without it. Each one is a nonlinear
 * reservoir whose ponded depth d follows
 *
 *     dd/dt = i − e − f − α · (d − ds)^(5/3),      α = W · √S / (A · n)
 *
 * with rainfall i, evaporation e, infiltration f (pervious sub-area only,
 * Modified Green–Ampt), depression storage ds, width W, slope S, sub-area A
 * and Manning's n. The depth is integrated with an adaptive Cash–Karp
 * Runge–Kutta step, the infiltration with the integrated Green–Ampt equation.
 *
 * The procedure follows subcatch.c, infil.c and odesolve.c of SWMM 5.2, the
 * engine the tools on this site run through pyswmm, including its internal
 * units (feet and seconds). It covers one subcatchment that drains straight
 * to its outlet, without snow, groundwater, LID controls or internal routing
 * between sub-areas. It drives the interactive model on the "How it works"
 * page: it explains the engine and does not replace it.
 *
 * CS.runoffModel.simulate(parameters) → result      (see simulate below)
 * CS.runoffModel.DEFAULTS                           the bundled example subcatchment
 * CS.runoffModel.STORMS                             design storms for the interactive model
 */
(function (root, factory) {
    "use strict";
    if (typeof module === "object" && module.exports) {
        module.exports = factory();
    } else {
        root.CS = root.CS || {};
        root.CS.runoffModel = factory();
    }
})(typeof self !== "undefined" ? self : this, function () {
    "use strict";

    // ── Constants of the SWMM engine (internal units: ft, s) ─────────────

    var MANNING_COEFFICIENT = 1.49;
    var MANNING_EXPONENT = 5 / 3;
    var ODE_TOLERANCE = 1e-4;
    var ZERO = 1e-10;

    var FEET_PER_METRE = 1 / 0.3048;
    var SQUARE_FEET_PER_HECTARE = 1e4 * FEET_PER_METRE * FEET_PER_METRE;
    var FEET_PER_MM = FEET_PER_METRE / 1000;
    var FT_S_PER_MM_H = FEET_PER_MM / 3600;
    var FT_S_PER_MM_DAY = FEET_PER_MM / 86400;
    var CUBIC_METRES_PER_CUBIC_FOOT = Math.pow(0.3048, 3);

    var SUBAREAS = ["imperviousZero", "impervious", "pervious"];
    // Time steps of one run at most: a year in five-minute steps.
    var MAX_TIME_STEPS = 105120;

    /** The subcatchment S1 of the bundled example model (SI units, as in its .inp file). */
    var DEFAULTS = {
        area: 5, // ha
        imperviousness: 25, // % of the area
        width: 100, // m
        slope: 10, // %
        nImperv: 0.01, // Manning's n, impervious
        nPerv: 1, // Manning's n, pervious
        storageImperv: 0.05, // depression storage, mm
        storagePerv: 0.05, // depression storage, mm
        zeroImperv: 25, // % of the impervious area without depression storage
        suction: 3.5, // Green–Ampt capillary suction head, mm
        conductivity: 0.5, // saturated hydraulic conductivity, mm/h
        deficit: 0.25, // initial moisture deficit, fraction
        evaporation: 0, // mm/day
        timeStep: 300, // s, SWMM's WET_STEP
    };

    /**
     * The range each parameter is kept in. Far wider than any real subcatchment: the
     * bounds only keep the arithmetic finite whatever a caller passes in.
     */
    var LIMITS = {
        area: [1e-6, 1e6],
        imperviousness: [0, 100],
        width: [1e-3, 1e7],
        slope: [1e-6, 1e4],
        nImperv: [1e-6, 1e3],
        nPerv: [1e-6, 1e3],
        storageImperv: [0, 1e4],
        storagePerv: [0, 1e4],
        zeroImperv: [0, 100],
        suction: [0, 1e4],
        conductivity: [0, 1e4],
        deficit: [0, 1],
        evaporation: [0, 1e4],
        timeStep: [1, 86400],
    };

    function hourly(intensities, startHour) {
        return intensities.map(function (intensity, index) {
            return { time: (startHour + index) * 3600, intensity: intensity };
        });
    }

    /**
     * Design storms: step functions of rainfall intensity (mm/h) over time (s).
     * An intensity holds until the next breakpoint; the last one ends the rain.
     */
    var STORMS = {
        example: {
            label: "Example storm",
            description: "The 12-hour storm of the bundled example model, peaking at 20 mm/h.",
            duration: 12 * 3600,
            rainfall: hourly([1, 2, 4, 4, 12, 13, 11, 20, 15, 10, 5, 3], 1),
        },
        cloudburst: {
            label: "Cloudburst",
            description: "A short, intense burst: 45 minutes of heavy rain on a dry catchment.",
            duration: 4 * 3600,
            rainfall: [
                { time: 900, intensity: 12 },
                { time: 1800, intensity: 48 },
                { time: 2700, intensity: 30 },
                { time: 3600, intensity: 8 },
                { time: 4500, intensity: 0 },
            ],
        },
        drizzle: {
            label: "Steady rain",
            description: "Eight hours of gentle, constant rain at 3 mm/h.",
            duration: 12 * 3600,
            rainfall: [
                { time: 3600, intensity: 3 },
                { time: 9 * 3600, intensity: 0 },
            ],
        },
        doublePeak: {
            label: "Two showers",
            description: "Two showers two hours apart: the second falls on wet ground.",
            duration: 8 * 3600,
            rainfall: [
                { time: 1800, intensity: 18 },
                { time: 3600, intensity: 6 },
                { time: 5400, intensity: 0 },
                { time: 12600, intensity: 18 },
                { time: 14400, intensity: 6 },
                { time: 16200, intensity: 0 },
            ],
        },
    };

    // ── Depth integration (odesolve.c: Cash–Karp Runge–Kutta, adaptive step) ──

    var SAFETY = 0.9;
    var GROW_EXPONENT = -0.2;
    var SHRINK_EXPONENT = -0.25;
    var ERROR_LIMIT = 1.89e-4; // (5 / SAFETY)^(1 / GROW_EXPONENT)
    var TINY = 1e-30;
    // Cash–Karp steps (accepted or rejected) allowed for one call. An ordinary runoff step
    // needs a handful; a reservoir that needs more drains so fast that it is at
    // equilibrium with its inflow long before the step ends.
    var STEP_BUDGET = 50;

    function depthDerivative(depth, inflow, alpha, storage) {
        var excess = depth - storage;
        return excess > 0 ? inflow - alpha * Math.pow(excess, MANNING_EXPONENT) : inflow;
    }

    /** One Cash–Karp step of size h from depth y; returns [new depth, error estimate]. */
    function cashKarpStep(y, dydx, h, inflow, alpha, storage) {
        var k2 = depthDerivative(y + h * 0.2 * dydx, inflow, alpha, storage);
        var k3 = depthDerivative(y + h * (0.075 * dydx + 0.225 * k2), inflow, alpha, storage);
        var k4 = depthDerivative(y + h * (0.3 * dydx - 0.9 * k2 + 1.2 * k3), inflow, alpha, storage);
        var k5 = depthDerivative(
            y + h * ((-11 / 54) * dydx + 2.5 * k2 - (70 / 27) * k3 + (35 / 27) * k4),
            inflow, alpha, storage
        );
        var k6 = depthDerivative(
            y + h * ((1631 / 55296) * dydx + (175 / 512) * k2 + (575 / 13824) * k3 +
                (44275 / 110592) * k4 + (253 / 4096) * k5),
            inflow, alpha, storage
        );
        var next = y + h * ((37 / 378) * dydx + (250 / 621) * k3 + (125 / 594) * k4 + (512 / 1771) * k6);
        var error = h * ((37 / 378 - 2825 / 27648) * dydx + (250 / 621 - 18575 / 48384) * k3 +
            (125 / 594 - 13525 / 55296) * k4 - (277 / 14336) * k5 + (512 / 1771 - 0.25) * k6);
        return [next, error];
    }

    /** The depth at which the outflow of a reservoir balances its inflow. */
    function equilibriumDepth(inflow, alpha, storage) {
        return inflow > 0 ? storage + Math.pow(inflow / alpha, 1 / MANNING_EXPONENT) : storage;
    }

    /**
     * Integrate dd/dt = inflow − α·(d − ds)^(5/3) over `duration` seconds. A reservoir
     * too stiff for the step budget (a tiny area behind a very wide outlet) is taken to
     * its equilibrium depth instead, which it reaches within a fraction of the step.
     */
    function integrateDepth(depth, inflow, alpha, storage, duration) {
        var elapsed = 0;
        var h = duration;
        var budget = STEP_BUDGET;
        while (budget > 0) {
            var dydx = depthDerivative(depth, inflow, alpha, storage);
            var scale = Math.abs(depth) + Math.abs(dydx * h) + TINY;
            if (elapsed + h > duration) {
                h = duration - elapsed;
            }
            var result;
            var errorRatio;
            for (;;) {
                budget--;
                result = cashKarpStep(depth, dydx, h, inflow, alpha, storage);
                errorRatio = Math.abs(result[1] / scale) / ODE_TOLERANCE;
                if (errorRatio <= 1) {
                    break;
                }
                // Also taken when the error is not a number: the step cannot be trusted.
                if (!(budget > 0) || !Number.isFinite(errorRatio)) {
                    return equilibriumDepth(inflow, alpha, storage);
                }
                h = Math.max(SAFETY * h * Math.pow(errorRatio, SHRINK_EXPONENT), 0.1 * h);
            }
            elapsed += h;
            depth = result[0];
            if (elapsed >= duration) {
                return depth;
            }
            h = errorRatio > ERROR_LIMIT ? SAFETY * h * Math.pow(errorRatio, GROW_EXPONENT) : 5 * h;
        }
        return equilibriumDepth(inflow, alpha, storage);
    }

    // ── Infiltration (infil.c: Modified Green–Ampt) ──────────────────────

    function createSoil(parameters) {
        var conductivity = parameters.conductivity * FT_S_PER_MM_H;
        return {
            suction: parameters.suction * FEET_PER_MM,
            conductivity: conductivity,
            maxDeficit: parameters.deficit,
            // Depth of the upper soil zone (ft), Mein's equation on Ks in in/h.
            upperZoneDepth: (4 * Math.sqrt(conductivity * 12 * 3600)) / 12,
            deficit: parameters.deficit,
            infiltrated: 0, // F, cumulative infiltration of the current event (ft)
            upperZoneWater: 0, // Fu (ft)
            saturated: false,
            timeToNewEvent: 0, // T (s)
        };
    }

    /** New cumulative infiltration after `duration` seconds of saturated infiltration. */
    function greenAmptVolume(infiltrated, head, conductivity, duration) {
        var minimum = infiltrated + conductivity * duration;
        if (head === 0) {
            return minimum;
        }
        if (duration < 10 && infiltrated > 0.01 * head) {
            return Math.max(infiltrated + conductivity * (1 + head / infiltrated) * duration, minimum);
        }
        // Newton–Raphson on F2 − F1 − c·ln((F2 + c) / (F1 + c)) = Ks·t.
        var constant = head * Math.log(infiltrated + head) - conductivity * duration;
        var volume = infiltrated;
        for (var iteration = 0; iteration < 20; iteration++) {
            var change = (volume - infiltrated - head * Math.log(volume + head) + constant) /
                (1 - head / (volume + head));
            if (Math.abs(change) < 1e-5) {
                return Math.max(volume, minimum);
            }
            volume -= change;
        }
        return minimum;
    }

    function addInfiltration(soil, volume, upperZoneCapacity) {
        soil.infiltrated += volume;
        soil.upperZoneWater = Math.min(soil.upperZoneWater + volume, upperZoneCapacity);
    }

    function saturatedInfiltration(soil, step, rainfall, depth, upperZoneCapacity) {
        var available = rainfall + depth / step;
        if (available < ZERO) {
            return 0;
        }
        soil.timeToNewEvent = 5400 / soil.upperZoneDepth;
        var head = (soil.suction + depth) * soil.deficit;
        var volume = greenAmptVolume(soil.infiltrated, head, soil.conductivity, step) - soil.infiltrated;
        if (volume > available * step) {
            volume = available * step;
            soil.saturated = false;
        }
        addInfiltration(soil, volume, upperZoneCapacity);
        return volume / step;
    }

    function unsaturatedInfiltration(soil, step, rainfall, depth, upperZoneCapacity) {
        var available = rainfall + depth / step;
        if (available < ZERO) {
            available = 0;
        }

        // Dry step: the upper zone recovers its moisture deficit.
        if (available === 0) {
            if (soil.upperZoneWater <= 0) {
                return 0;
            }
            var recovered = (soil.upperZoneDepth / 90000) * upperZoneCapacity * step;
            soil.infiltrated -= recovered;
            soil.upperZoneWater -= recovered;
            if (soil.upperZoneWater <= 0) {
                soil.upperZoneWater = 0;
                soil.infiltrated = 0;
                soil.deficit = soil.maxDeficit;
            } else if (soil.timeToNewEvent <= 0) {
                soil.deficit = (upperZoneCapacity - soil.upperZoneWater) / soil.upperZoneDepth;
                soil.infiltrated = 0;
            }
            return 0;
        }

        // Supply below the conductivity: everything infiltrates.
        if (available <= soil.conductivity) {
            addInfiltration(soil, available * step, upperZoneCapacity);
            return available;
        }

        soil.timeToNewEvent = 5400 / soil.upperZoneDepth;
        var toSaturate = (soil.conductivity * (soil.suction + depth) * soil.deficit) /
            (available - soil.conductivity);
        if (soil.infiltrated > toSaturate) {
            soil.saturated = true;
            return saturatedInfiltration(soil, step, rainfall, depth, upperZoneCapacity);
        }
        if (soil.infiltrated + available * step < toSaturate) {
            addInfiltration(soil, available * step, upperZoneCapacity);
            return available;
        }

        // The surface layer saturates part-way through the step.
        var saturatedTime = Math.max(step - (toSaturate - soil.infiltrated) / available, 0);
        var head = (soil.suction + depth) * soil.deficit;
        var total = Math.min(
            greenAmptVolume(toSaturate, head, soil.conductivity, saturatedTime),
            toSaturate + available * saturatedTime
        );
        var volume = total - soil.infiltrated;
        soil.infiltrated = total;
        soil.upperZoneWater = Math.min(soil.upperZoneWater + volume, upperZoneCapacity);
        soil.saturated = true;
        return volume / step;
    }

    /** Infiltration rate (ft/s) over one step; updates the soil moisture state. */
    function infiltrationRate(soil, step, rainfall, depth) {
        var upperZoneCapacity = soil.maxDeficit * soil.upperZoneDepth;
        soil.timeToNewEvent -= step;
        return soil.saturated
            ? saturatedInfiltration(soil, step, rainfall, depth, upperZoneCapacity)
            : unsaturatedInfiltration(soil, step, rainfall, depth, upperZoneCapacity);
    }

    // ── Sub-areas (subcatch.c) ───────────────────────────────────────────

    function createSubareas(parameters) {
        var area = parameters.area * SQUARE_FEET_PER_HECTARE;
        var width = parameters.width * FEET_PER_METRE;
        var slope = parameters.slope / 100;
        var impervious = parameters.imperviousness / 100;
        var zero = parameters.zeroImperv / 100;
        var shares = {
            imperviousZero: impervious * zero,
            impervious: impervious * (1 - zero),
            pervious: 1 - impervious,
        };
        var storage = {
            imperviousZero: 0,
            impervious: parameters.storageImperv * FEET_PER_MM,
            pervious: parameters.storagePerv * FEET_PER_MM,
        };
        return SUBAREAS.map(function (name) {
            var isPervious = name === "pervious";
            // Both impervious sub-areas share the flow width of the whole impervious area.
            var groupArea = area * (isPervious ? shares.pervious : impervious);
            var roughness = isPervious ? parameters.nPerv : parameters.nImperv;
            return {
                name: name,
                share: shares[name],
                area: area * shares[name],
                storage: storage[name],
                alpha: groupArea > 0
                    ? (MANNING_COEFFICIENT * width * Math.sqrt(slope)) / (groupArea * roughness)
                    : 0,
                depth: 0,
            };
        });
    }

    /** Advance the ponded depth of a sub-area by one step with net inflow `inflow` (ft/s). */
    function updatePondedDepth(subarea, inflow, step) {
        if (subarea.depth + inflow * step <= subarea.storage) {
            subarea.depth += inflow * step;
        } else {
            var remaining = step;
            var toFill = subarea.storage - subarea.depth;
            if (toFill > 0 && inflow > 0) {
                remaining -= toFill / inflow;
                subarea.depth = subarea.storage;
            }
            if (subarea.alpha > 0 && remaining > 0) {
                subarea.depth = integrateDepth(subarea.depth, inflow, subarea.alpha, subarea.storage, remaining);
            } else {
                subarea.depth += inflow * Math.max(remaining, 0);
            }
        }
        if (subarea.depth < 0) {
            subarea.depth = 0;
        }
    }

    /**
     * One runoff step of a sub-area. Returns its rates over the step in ft/s:
     * {runoff, infiltration, evaporation}, runoff being the rate at the end of the step.
     */
    function stepSubarea(subarea, soil, rainfall, potentialEvaporation, step) {
        var moisture = subarea.depth / step;
        var evaporation = Math.min(moisture, potentialEvaporation);
        var infiltration = soil ? infiltrationRate(soil, step, rainfall, subarea.depth) : 0;
        moisture += rainfall;

        if (evaporation + infiltration >= moisture) {
            subarea.depth = 0;
        } else {
            updatePondedDepth(subarea, rainfall - evaporation - infiltration, step);
        }

        var excess = subarea.depth - subarea.storage;
        return {
            runoff: excess > ZERO ? subarea.alpha * Math.pow(excess, MANNING_EXPONENT) : 0,
            infiltration: infiltration,
            evaporation: evaporation,
        };
    }

    // ── Simulation ───────────────────────────────────────────────────────

    function intensityAt(rainfall, time) {
        var intensity = 0;
        for (var index = 0; index < rainfall.length && rainfall[index].time <= time; index++) {
            intensity = rainfall[index].intensity;
        }
        return intensity;
    }

    /**
     * Run the model.
     *
     * parameters: DEFAULTS overridden by the given keys, plus
     *   rainfall  [{time (s), intensity (mm/h)}, …]; a step function
     *   duration  length of the simulation (s), run as a whole number of time steps
     *             (at most MAX_TIME_STEPS)
     * Parameters outside LIMITS are moved to the nearest bound.
     *
     * Returns series sampled at the end of every time step (index 0 is the start):
     *   time          s from the start
     *   rainfall      mm/h, intensity during the step that ends at `time`
     *   runoff        m³/s, total runoff rate
     *   infiltration  mm/h, averaged over the whole subcatchment
     *   evaporation   mm/h, averaged over the whole subcatchment
     *   subareas      {pervious|impervious|imperviousZero: {share, storage (mm),
     *                 depth (mm, series), runoff (m³/s, series)}}
     *   totals        {rainfall, runoff, infiltration, evaporation, storage} in m³;
     *                 storage is the water still ponded on the surface at the end.
     *                 Like SWMM, runoff volume is the end-of-step rate times the step,
     *                 so the parts add up to the rainfall only within `continuity`
     *   continuity    SWMM's continuity error of the run, % of the rainfall
     *   peak          {runoff (m³/s), time (s)}
     */
    function simulate(parameters) {
        var settings = {};
        Object.keys(DEFAULTS).forEach(function (key) {
            var value = parameters && parameters[key];
            var limits = LIMITS[key];
            settings[key] = Number.isFinite(value)
                ? Math.min(Math.max(value, limits[0]), limits[1])
                : DEFAULTS[key];
        });
        var storm = ((parameters && parameters.rainfall) || STORMS.example.rainfall)
            .slice()
            .sort(function (a, b) {
                return a.time - b.time;
            });
        var duration = parameters && parameters.duration;
        if (!Number.isFinite(duration) || duration <= 0) {
            duration = STORMS.example.duration;
        }
        var step = settings.timeStep;
        var stepCount = Math.min(Math.max(1, Math.round(duration / step)), MAX_TIME_STEPS);

        var subareas = createSubareas(settings);
        var soil = createSoil(settings);
        var totalArea = subareas.reduce(function (sum, subarea) {
            return sum + subarea.area;
        }, 0);
        var potentialEvaporation = settings.evaporation * FT_S_PER_MM_DAY;

        var result = {
            time: [0],
            rainfall: [0],
            runoff: [0],
            infiltration: [0],
            evaporation: [0],
            subareas: {},
        };
        subareas.forEach(function (subarea) {
            result.subareas[subarea.name] = {
                share: subarea.share,
                storage: subarea.storage / FEET_PER_MM,
                depth: [0],
                runoff: [0],
            };
        });
        var volumes = { rainfall: 0, runoff: 0, infiltration: 0, evaporation: 0 };
        var peak = { runoff: 0, time: 0 };

        for (var index = 0; index < stepCount; index++) {
            var start = index * step;
            var intensity = Math.max(intensityAt(storm, start), 0);
            var rainfall = intensity * FT_S_PER_MM_H;
            var runoff = 0;
            var infiltration = 0;
            var evaporation = 0;

            subareas.forEach(function (subarea) {
                var series = result.subareas[subarea.name];
                var rates = { runoff: 0, infiltration: 0, evaporation: 0 };
                if (subarea.area > 0) {
                    rates = stepSubarea(
                        subarea, subarea.name === "pervious" ? soil : null, rainfall, potentialEvaporation, step
                    );
                }
                var flow = rates.runoff * subarea.area;
                runoff += flow;
                infiltration += rates.infiltration * subarea.area;
                evaporation += rates.evaporation * subarea.area;
                series.depth.push(subarea.depth / FEET_PER_MM);
                series.runoff.push(flow * CUBIC_METRES_PER_CUBIC_FOOT);
            });

            volumes.rainfall += rainfall * totalArea * step;
            volumes.runoff += runoff * step;
            volumes.infiltration += infiltration * step;
            volumes.evaporation += evaporation * step;

            var end = start + step;
            var runoffRate = runoff * CUBIC_METRES_PER_CUBIC_FOOT;
            if (runoffRate > peak.runoff) {
                peak = { runoff: runoffRate, time: end };
            }
            result.time.push(end);
            result.rainfall.push(intensity);
            result.runoff.push(runoffRate);
            result.infiltration.push(infiltration / totalArea / FT_S_PER_MM_H);
            result.evaporation.push(evaporation / totalArea / FT_S_PER_MM_H);
        }

        result.totals = {};
        Object.keys(volumes).forEach(function (key) {
            result.totals[key] = volumes[key] * CUBIC_METRES_PER_CUBIC_FOOT;
        });
        result.totals.storage = subareas.reduce(function (sum, subarea) {
            return sum + subarea.depth * subarea.area;
        }, 0) * CUBIC_METRES_PER_CUBIC_FOOT;
        var totals = result.totals;
        result.continuity = totals.rainfall > 0
            ? ((totals.rainfall - totals.runoff - totals.infiltration - totals.evaporation - totals.storage) /
                totals.rainfall) * 100
            : 0;
        result.peak = peak;
        return result;
    }

    return { simulate: simulate, DEFAULTS: DEFAULTS, STORMS: STORMS };
});
