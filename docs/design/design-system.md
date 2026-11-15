# Design system — Catchment Simulation web app

The visual language, layout, interaction contracts and front-end conventions of
the Django application in `cs_app/`. Read it before changing templates, CSS or
JavaScript. Do not invent new tokens, fonts or colours; extend this file first
if something is missing.

## 1. Subject and audience

- **Product:** web front-end for the `catchment_simulation` package — runs SWMM
  subcatchment parameter sweeps, timeseries (hydrograph) analysis and a SWMM-vs-ANN
  runoff comparison.
- **Audience:** hydrologists, civil/environmental engineers, students. Technical,
  data-literate, desktop-first but must work on phones.
- **Job of the app:** load a SWMM model → choose a subcatchment and a parameter →
  run → read the answer (chart, numbers, export).

## 2. Visual direction — "survey sheet"

Cool, precise, engineering-drawing calm. Water colours, not a generic SaaS blue.
Boldness is spent in exactly one place: the **hyetograph + hydrograph** chart
(rainfall as inverted bars hanging from the top axis, runoff lines below) — the
most characteristic artefact of hydrology. Everything around it stays quiet.

### 2.1 Colour tokens (CSS custom properties on `:root`)

| Token | Light | Dark | Use |
|---|---|---|---|
| `--cs-ink` | `#0F2233` | `#E3ECF2` | primary text |
| `--cs-ink-muted` | `#4A5E6D` | `#9DB1BF` | secondary text, labels |
| `--cs-paper` | `#F4F7F9` | `#0B1620` | page background |
| `--cs-surface` | `#FFFFFF` | `#122330` | cards, panels |
| `--cs-surface-sunken` | `#EBF1F5` | `#0E1C27` | code blocks, inputs bg, table stripes |
| `--cs-line` | `#D6E1E8` | `#23394A` | borders, dividers, chart grid |
| `--cs-water` | `#0B6E8A` | `#3FB8D6` | primary accent: buttons, links, runoff line |
| `--cs-water-strong` | `#08566C` | `#6FCDE3` | hover/active of accent |
| `--cs-rain` | `#5A6FD6` | `#8C9CF0` | rainfall series / hyetograph bars |
| `--cs-ochre` | `#B7791F` | `#E0A546` | peak markers, ANN series, large accents only (fails AA as small text) |
| `--cs-ochre-text` | `#8A5A12` | `#E9B96A` | ochre-tinted small text (AA) |
| `--cs-danger` | `#B42318` | `#F97066` | errors |
| `--cs-success` | `#1F7A4D` | `#4CC38A` | success states |

Bootstrap 5.3 variables (`--bs-primary`, `--bs-body-bg`, `--bs-body-color`,
`--bs-border-color`, `--bs-link-color`, …) are **mapped** to these tokens in
`tokens.css` so stock Bootstrap components inherit the palette. Dark mode uses
Bootstrap 5.3 colour modes: `<html data-bs-theme="light|dark">`.

Chart series order (categorical): water, rain, ochre, `#346400` / dark `#C1EB95`
(infiltration, "soil"; darker / lighter than ochre so the pair stays apart under
protan/deutan vision), `#8A5A9E` / dark `#C29AD6` (evaporation). Water left on the surface
(`--cs-pond`, `#8796A3` / dark `#5F7383`, 3:1 on the surface) is a quiet neutral that is
also hatched in charts, so it never depends on colour alone. Sweeps use a
sequential teal ramp from `--cs-water` (70 % over the surface) to
`--cs-water-strong` pushed most of the way towards the ink (85 %, dark 90 %; see
`RAMP_MIX` in `charts.js`), so every step keeps 3:1 contrast
with the surface in both themes (value order = colour order), never a rainbow.

### 2.2 Typography (Google Fonts only)

- **Display:** `Archivo` (variable, use `font-stretch: 112%–125%`, weight 600–700)
  — headings H1–H3, hero, card titles. Used with restraint.
- **Body/UI:** `Public Sans` 400/500/600.
- **Data/code:** `IBM Plex Mono` 400/500 — numbers in tables and metric tiles,
  code snippets, file names, parameter values. Use `font-variant-numeric: tabular-nums`.

Scale (rem): 0.8125 caption · 0.9375 small · 1 body · 1.25 h4 · 1.5 h3 · 2 h2 · 2.75 h1/hero.
Line-height 1.55 body, 1.15 display. Eyebrow labels: Public Sans 600, 0.75rem,
`letter-spacing: .08em`, uppercase, `--cs-ink-muted`.

### 2.3 Shape, spacing, depth

- Radius: `--cs-radius-sm: 6px`, `--cs-radius: 10px`, `--cs-radius-lg: 16px`.
- Spacing on a 4px grid; section gaps 48–64px desktop, 32px mobile.
- Shadows minimal: `--cs-shadow: 0 1px 2px rgb(15 34 51 / .06), 0 4px 12px rgb(15 34 51 / .05)`.
  Dark mode relies on borders, not shadows.
- Focus ring: 3px `color-mix(in srgb, var(--cs-water) 45%, transparent)` outline, offset 2px. Never removed.

### 2.4 Motion

- One orchestrated moment: results panel reveal after a run (fade + 8px rise, 220ms,
  staggered metric tiles 40ms). Chart draws with Plotly transition only.
- Micro: buttons/links 120ms colour transitions.
- `@media (prefers-reduced-motion: reduce)` disables all transforms/transitions.
- The interactive model on "How it works" (§3.4) is the one place with continuous
  motion, and only on request: nothing plays until the reader presses Play (the storm
  then runs once, in about 12 s, and stops at the end), rain streaks fall only while it
  plays, and a sweep reveals its runs one by one (~0.3 s apart). With reduced motion
  the rain stands still while playing and a sweep is drawn at once. Figures follow
  sliders and the time cursor by attribute updates, never by transitions.

## 3. Layout

### 3.1 App shell (`base.html`)

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ [logo] Catchment Simulation    Home  Simulation  Timeseries  Calculations   ☾  [Account ▾] │
│                                How it works  About                                       │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│  page content                                                                            │
├──────────────────────────────────────────────────────────────────────────────────────────┤
│ ©  ·  [logo]  ·  How it works · About · Contact · GitHub · PyPI                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```
- Sticky top navbar, Bootstrap `navbar-expand-xl`: below 1200px the six links and the
  tools collapse behind a toggler.
- On xl+ the bar is a three-column grid (brand | links | tools) with equal side columns,
  so the links sit on the page's centre line; between 1200 and 1399px the links sit
  closer together. Nothing in the bar wraps.
- Active link: `aria-current="page"` + accent underline.
- Theme toggle (light/dark/auto) persisted in `localStorage` (`cs-theme`), applied
  before paint by a tiny **external synchronous** script
  `<script src="{% static 'main/js/core/theme-init.js' %}"></script>` in `<head>`
  (NOT inline — a unit test allows only `Dropzone.autoDiscover=false;` as inline JS).
  On every change the theme module dispatches
  `document.dispatchEvent(new CustomEvent("cs:themechange", {detail: {theme: "light"|"dark"}}))`
  (resolved theme, never "auto"); charts listen to it and re-layout.
- Skip link "Skip to content" → `#main-content`.
- Django messages stay **inline dismissible alerts** at the top of the content
  (`.alert.alert-danger|success|warning|info` — tests rely on `alert-danger` + text),
  restyled with tokens. Errors must persist, so they are not toasts.
- Transient feedback of async flows uses Bootstrap **toasts** via
  `CS.toast(message, level)` (`level`: success|info|warning|danger), stacked bottom-right
  in a `.toast-container` with `aria-live="polite"`.

### 3.2 Workbench pages (Simulation, Timeseries, ANN comparison)

```
┌─────────────── page header: title + one-line purpose ─────────────────┐
├──────────── control panel (sticky, 360px) ─┬──── results canvas ───────┤
│ ① Model   upload zone / current file chip   │  empty state → invitation │
│ ② Parameters  (fields, live hint:           │  after run:               │
│    "11 runs · 0 → 100 step 10")             │   metric tiles            │
│ [ Run simulation ]  progress + elapsed time │   chart card (toolbar)    │
│                                             │   data table (sortable)   │
│                                             │   export actions          │
└─────────────────────────────────────────────┴───────────────────────────┘
```
On < lg the panel stacks above results. On ≥ lg it is sticky only while it fits
the viewport (set by `core/workbench.js`); a taller panel scrolls with the page,
never inside itself. The numbered ①② markers are legitimate: the workflow is a
real sequence.

### 3.3 Home (documentation) page

Hero = thesis sentence + live hyetograph/hydrograph of the bundled example model
(signature). Below: three tool cards (Simulation, Timeseries, ANN comparison),
then the package documentation with a sticky table of contents (scrollspy) on ≥ lg,
code blocks with a copy button.

### 3.4 How it works (`/how-it-works`)

Public explainer of the model behind every tool: page header, then the home page's
sticky table of contents (`.home-toc` + scrollspy) on ≥ lg beside the sections
`#pipeline`, `#model`, `#lab`, `#parameters` (entries `#parameter-<Feature>`, link
targets from the tool pages, with a visible `:target` state), `#infiltration`,
`#numerics`, `#sweep`, `#network`, `#references`.
- Equations are KaTeX (§5), written as TeX in the template and typeset by
  `CS.math.render()`; until then, and without JS, the TeX source shows in mono.
  Display equations are numbered `.hiw-equation` rows (1)…(18) that scroll sideways
  inside their own box on narrow screens; symbols in running text are inline math.
- Diagrams are inline SVG coloured through the tokens, with a text alternative.
- `#lab`, the interactive model, is where this page spends its boldness: one bordered
  instrument with the controls (storm, sliders, reset) beside the stage on ≥ xl and
  below it otherwise. The stage holds the three sub-areas as reservoirs (shared
  square-root depth scale, values printed), a time control, the hyetograph/hydrograph
  as custom SVG in the signature-chart style (ghost line for the starting values, peak
  in ochre, sweep runs on the sequential ramp), key numbers with changes "vs. start",
  the "Where the rain went" balance bar and a sweep demonstration with its response
  curve. It runs `CS.runoffModel` in the browser on every input event; starting values
  come from the query string (`?area=…&vary=<Feature>`). Without JS it shows a note.

## 4. Interaction contract

- **Progressive enhancement.** Every form still works without JS (POST → redirect →
  GET). With JS, run forms submit via `fetch` and swap in a server-rendered results
  fragment — no full page reload. Server remains the single source of markup.
- Async requests send header `X-Requested-With: XMLHttpRequest`; CSRF from the form's
  `csrfmiddlewaretoken`. Server contract for run endpoints
  (`simulation_view`, `timeseries_view`, `calculations` POST):
  - **Success:** `200`, `Content-Type: text/html`, body = the results fragment
    rendered from the SAME partial template the full page includes
    (`main/partials/_<page>_results.html`). Session/cache/token state is updated
    exactly as in the non-AJAX path.
  - Each tool's latest result stays in the cache (file-based, survives restarts,
    12 h) under a token in the session, so its page renders it again when the user
    comes back. Loading a model with identical content keeps the results.
  - **Validation / input error:** `4xx` (`400` form invalid or bad input, `413` too
    large, `401` not authenticated) with JSON
    `{"message": str, "field_errors": {field_name: [str, ...]}}`
    (`field_errors` may be `{}`; `__all__` for non-field errors).
  - **Server error:** `500` JSON `{"message": str, "field_errors": {}}`.
  - In the AJAX path the server must NOT call `messages.*` (they would leak into the
    next full page load).
  - Client helper `CS.asyncForm` must treat any non-2xx response whose body is not
    JSON (e.g. a proxy 502/504 HTML page after a gunicorn timeout) and network
    failures as: "The server did not finish the run. It may have timed out — try a
    smaller range or reload the page to check for results." and re-enable the form.
  - Without JS (no header) everything behaves exactly as before (POST → redirect → GET).
- While running: button disabled + `aria-busy`, inline progress with elapsed seconds,
  results region `aria-busy="true"`. After success: results swapped, focus moved to
  the results heading, toast "Simulation finished". On error: inline alert in the
  control panel explaining what to fix + toast; button re-enabled.
- Download/export buttons never trigger the run spinner.
- Client-side hints never replace server validation.

### 4.1 Guided workbench (Simulation, Timeseries, ANN comparison)

- **Step states.** Each panel step is `.cs-step[data-cs-step="model"|"run"]` with
  `data-state="todo"|"active"|"done"`, server-rendered from the session's model and kept
  current by `core/steps.js` from `cs:model-changed` / `cs:model-cleared`: the model step
  is done while a model is loaded (its number becomes a check), the run step is todo
  (muted header, controls usable) until then. A visually hidden
  `[data-cs-step-status]` in the title says "(completed)" / "(load a model first)".
- **Checklist.** The empty results state of the sweep pages lists "Load a model ·
  Choose a subcatchment · Run …" (`partials/_workbench_checklist.html`); items carry
  `data-check` and `data-state="done"|"current"|"todo"`, and `[data-cs-next]` shows the
  `data-next` sentence of the first open item.
- **Ready to run.** A model with one subcatchment selects it (and fires `change`);
  with several, `[data-cs-catchment-hint]` says how many. A run form
  (`form[data-cs-run-form]`) sent without a model or subcatchment shows an inline
  `.cs-attention` message at the control that needs the user and focuses it. The
  upload responses carry `is_sample`: "Try sample data" is hidden while the sample is
  loaded and reads "Use sample data instead" while another model is.
- **Model identity.** `CS.modelGeneration` (upload zone) goes up when the server takes a
  different model or drops it, before `cs:model-changed` (which waits for the
  subcatchment list) and carried in its `detail.generation`. `CS.asyncForm` discards a
  run only when the generation changed while it was in flight, and marks results stale
  only when they come from an earlier generation. A run sent while the subcatchment list
  is still loading waits for it (`core/steps.js`) and then starts.
- **Model values.** `GET /subcatchments/values/?catchment=<id>` (`main:subcatchment_values`)
  returns `{catchment, values, units, suggested_ranges, lab_url}` for the session's
  model (`main/insights.py`); errors are `{"message"}` with 400 (no model, no
  catchment), 404 (unknown subcatchment) or 401. Nothing per subcatchment is kept in the
  session; parsed values are cached in-process, keyed on the file as it is on disk
  (path, modification time, size), in a bounded LRU.
- **Parameter card and range suggestion** (`core/sweep-assist.js`, `CS.sweepAssist`):
  `.cs-insight` under "Parameter to vary" shows the guide entry from the page's
  `#sweep-guide` JSON (`main.guide.parameter_guide_payload()`), "In this model: …" and
  a link to `<how-it-works>#parameter-<Feature>`. `.cs-suggestion` offers the
  suggested range with a "Use" button; it is applied on parameter change, and on
  subcatchment or model change while the user has not typed in the range since the last
  fill. Forms restored from the session carry `data-restored="true"` and keep their range.
  The forms carry `data-values-url` and `data-how-it-works-url`; the simulation page also
  embeds `#method-features` (option → SWMM feature).
- **Run status.** `CS.asyncForm` option `runStatus` (or the form's
  `data-cs-run-status`, with `data-cs-run-stages`) shows `.cs-run-status` at the top of
  the results while a run is in flight: what runs, the pipeline stages, the elapsed time
  and an indeterminate bar (no invented progress; static under reduced motion). It is
  `aria-hidden`; the loading element stays the one polite announcement.
- **Results that explain themselves** (simulation): `chart_config` gains optional
  `baseline` (the model's value of the swept parameter), `balance`
  (`{unit, rows: [{x, rainfall, runon, runoff, infiltration, evaporation, storage}]}`)
  and `labUrl`; results cached without them still render. The page writes a reading
  line (`CS.simulation.reading`), marks the model value on every panel
  (`CS.charts.line` option `reference`) and in a "Model value" tile, draws "Where the
  rain goes" (`CS.charts.balance`: stacked runoff / infiltration / evaporation / still on
  the surface, share or volume) with a data table, and links to the parameter's guide
  entry and to the interactive model.

## 5. Code conventions

- No build step, no framework. Plain ES2017+ scripts, `"use strict"`, one global
  namespace `window.CS` (`CS.theme`, `CS.toast`, `CS.charts`, `CS.asyncForm`, …).
- Files: `static/main/css/tokens.css`, `static/main/css/app.css` (components),
  page CSS only when needed; `static/main/js/core/*.js`, `static/main/js/pages/*.js`.
- Vendor libs from CDN with SRI (`integrity` + `crossorigin`): Bootstrap 5.3.x,
  Plotly (per page only), Dropzone (upload pages only), Prism (home only),
  KaTeX 0.19.1 (`katex.min.js` + `katex.min.css`, "How it works" only).
- Templates: English UI copy everywhere, sentence case, active verbs
  ("Download results (.xlsx)"). The primary run buttons ("Run Simulation",
  "Run Timeseries Analysis", "Run Calculations") and "Download Results" keep their
  texts: many tests key on them.
- No inline `style=""` attributes, with one exemption: `#upload-status` keeps its
  inline `display:none` toggling (tests read `style.display`).
- Use `{% static 'img/logo.png' %}` style paths (no leading slash) and
  `{% url %}` instead of hard-coded hrefs.
- Accessibility: WCAG 2.1 AA contrast in both themes, visible focus, labelled
  controls, charts have `role="img"` + `aria-label` and a data table alternative.
- Existing e2e page objects are the contract for ids; when markup changes, update
  the page object in the same change and keep tests green.
