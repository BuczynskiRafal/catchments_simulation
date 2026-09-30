# UI Modernization — Design Spec

Single source of truth for every agent working on the UI modernization.
Read it fully before touching templates, CSS or JS. Do not invent new tokens,
fonts or colours; extend this file first if something is missing.

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
protan/deutan vision), `#8A5A9E` / dark `#C29AD6` (evaporation). Sweeps use a
sequential teal ramp from `--cs-water` (70 % over the surface) to
`--cs-water-strong` pushed 55 % towards the ink, so every step keeps 3:1 contrast
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

## 3. Layout

### 3.1 App shell (`base.html`)

```
┌──────────────────────────────────────────────────────────────┐
│ [logo] Catchment Simulation   Simulation Timeseries ANN  Docs │  ☾  [Account ▾]
├──────────────────────────────────────────────────────────────┤
│  page content                                                 │
├──────────────────────────────────────────────────────────────┤
│ footer: © · About · Contact · GitHub · PyPI                    │
└──────────────────────────────────────────────────────────────┘
```
- Sticky top navbar, Bootstrap `navbar-expand-lg` with a real collapse toggler on mobile.
- On lg+ the bar is a three-column grid (brand | links | tools) with equal side columns,
  so the links sit on the page's centre line; between 992 and 1199px the wordmark
  stacks on two lines. Nothing in the bar wraps.
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
- Download/export buttons never trigger the run spinner (fixes audit S2).
- Client-side hints never replace server validation.

## 5. Code conventions

- No build step, no framework. Plain ES2017+ scripts, `"use strict"`, one global
  namespace `window.CS` (`CS.theme`, `CS.toast`, `CS.charts`, `CS.asyncForm`, …).
- Files: `static/main/css/tokens.css`, `static/main/css/app.css` (components),
  page CSS only when needed; `static/main/js/core/*.js`, `static/main/js/pages/*.js`.
- Vendor libs from CDN with SRI (`integrity` + `crossorigin`): Bootstrap 5.3.x,
  Plotly (per page only), Dropzone (upload pages only), Prism (home only).
- Templates: English UI copy everywhere (replace remaining Polish: "Kontakt",
  "Logowanie", "Wyślij", "Adres email", "Tytuł", "Treść", "Prześlij", alt "Opis obrazu")
  and update the tests/page objects asserting them in the same change. Sentence case,
  active verbs ("Download results (.xlsx)"). Keep the existing texts of the primary
  run buttons ("Run Simulation", "Run Timeseries Analysis", "Run Calculations") and
  "Download Results" since many tests key on them.
- No inline `style=""` attributes, with one exemption: `#upload-status` keeps its
  inline `display:none` toggling (tests read `style.display`).
- Use `{% static 'img/logo.png' %}` style paths (no leading slash) and
  `{% url %}` instead of hard-coded hrefs.
- Accessibility: WCAG 2.1 AA contrast in both themes, visible focus, labelled
  controls, charts have `role="img"` + `aria-label` and a data table alternative.
- Existing e2e page objects are the contract for ids; when markup changes, update
  the page object in the same change and keep tests green.
