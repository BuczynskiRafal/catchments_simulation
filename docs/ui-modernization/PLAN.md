# Plan modernizacji interfejsu — catchments_simulation

Branch: `ui-modernization` · specyfikacja wizualna i kontrakty: [`DESIGN_SPEC.md`](DESIGN_SPEC.md)

## Decyzja architektoniczna

Zostajemy przy Django renderującym HTML po stronie serwera + **progressive enhancement**
(vanilla JS bez kroku budowania, zgodnie z kierunkiem „static-first” z #76).
Formularze uruchamiające obliczenia wysyłane są przez `fetch`, a serwer zwraca
**fragment HTML wyników z tego samego szablonu częściowego**, którego używa pełna
strona — jedno źródło markupu, brak przeładowania strony, a bez JS wszystko działa
jak dotąd (POST → redirect → GET). Rewrite na SPA (React/Vite) odrzucony: duży koszt,
złamałby 250 testów i architekturę bez proporcjonalnego zysku dla użytkownika.

Legenda: **[nowe]** nowa funkcjonalność · **[poprawa]** naprawa błędu/niespójności ·
**[wzmocnienie]** ulepszenie istniejącego elementu.

## Zakres zmian

### Fundament (shell aplikacji)
- [wzmocnienie] Bootstrap 5.1.3 → 5.3.x (SRI liczone z plików), natywne color modes.
- [poprawa] crispy `bootstrap4` pack przy CSS Bootstrap 5 → `crispy-bootstrap5` (+ `uv.lock`).
- [nowe] System design tokenów (`tokens.css`) + komponenty (`app.css`), typografia Archivo / Public Sans / IBM Plex Mono.
- [nowe] Tryb ciemny (light/dark/auto) bez migotania, zapamiętywany, wykresy przełączają motyw na żywo.
- [wzmocnienie] Nawigacja: sticky navbar z hamburgerem na mobile (dziś brak), aktywny link `aria-current`, menu konta (profil/wyloguj), skip-link.
- [nowe] Toasty dla informacji zwrotnej z operacji asynchronicznych (`CS.toast`).
- [poprawa] Zepsuta klasa `me-md-autolink-dark`, `href="/"` → `{% url %}`, `{% static '/img/…' %}` z wiodącym `/` (psuje się przy manifest storage).
- [wzmocnienie] Vendor JS ładowany tylko tam, gdzie potrzebny (Plotly/Dropzone/Prism nie na każdej stronie).

### Backend pod interaktywność
- [nowe] Kontrakt AJAX dla `simulation_view`, `timeseries_view`, `calculations` (fragment HTML / JSON `{message, field_errors}`), bez wycieku `messages` do sesji.
- [nowe] Szablony częściowe wyników `main/partials/_*_results.html` współdzielone przez pełną stronę i odpowiedź AJAX.
- [poprawa] `contact`: `reverse("contact")` bez namespace → NoReverseMatch po poprawnym wysłaniu; wynik `send_message` ignorowany (audyt W6 — część UI: komunikat o niepowodzeniu).
- [poprawa] `userprofile`: zepsuty `form_action` i `reverse("userprofile")`.
- [wzmocnienie] `calculations` POST wymaga zalogowania po stronie serwera (GET pozostaje publiczny — test e2e tego oczekuje).

### Komponenty współdzielone
- [wzmocnienie] `charts.js`: wykresy świadome motywu (kolory z tokenów), spójna paleta, sekwencyjna rampa dla sweepów zamiast tęczy, uporządkowany modebar, `role="img"` + etykiety.
- [nowe] Wykres **hietogram + hydrogram** (opad jako odwrócone słupki z górnej osi, odpływ poniżej) — sygnatura wizualna aplikacji.
- [wzmocnienie] Strefa uploadu: styl z tokenów (dark mode), „chip” z aktualnym plikiem i liczbą zlewni, zdarzenie `cs:model-changed` dla stron.

### Symulacja (sweep parametru)
- [nowe] Układ „workbench”: panel sterowania (① Model → ② Parametry → Uruchom) + płótno wyników.
- [nowe] Uruchamianie bez przeładowania, postęp z licznikiem czasu, fokus na wynikach, toast.
- [nowe] Podpowiedź na żywo: „11 przebiegów · 0 → 100 co 10”, ostrzeżenie przy limicie 100.
- [nowe] Tabela wyników z sortowaniem, kopiowaniem, formatowaniem liczb; kafelki podsumowania (min/max/zmiana odpływu).
- [poprawa] Przycisk „Download Results” blokował przycisk Run do odświeżenia strony (audyt S2).
- [poprawa] Tekst spinnera „neural network” na stronie symulacji SWMM.

### Timeseries (hydrogram)
- [nowe] Hietogram + hydrogram dla trybu single; sweep z rampą kolorów i legendą wartości.
- [wzmocnienie] Kafelki metryk (czas do szczytu, objętość odpływu, szczyt), eksport PNG/CSV/XLSX w jednym pasku akcji.
- [nowe] Uruchamianie bez przeładowania jak w symulacji.

### Porównanie SWMM vs ANN (Calculations)
- [nowe] Wykres parzystości (SWMM vs ANN z linią 1:1) i słupkowy per zlewnia.
- [nowe] Metryki błędu (MAE, RMSE, średni błąd %) i różnica per wiersz.
- [wzmocnienie] Opis modelu w zwijanych sekcjach, czytelny układ dwukolumnowy, poprawny `alt` obrazka.

### Strony treściowe
- [nowe] Strona główna: hero z hydrogramem przykładowego modelu, karty narzędzi, dokumentacja ze sticky spisem treści (scrollspy), przycisk „Copy” przy kodzie.
- [wzmocnienie] Logowanie/rejestracja/kontakt/profil: karty formularzy, przełącznik widoczności hasła, spójne komunikaty.
- [poprawa] Mieszanka PL/EN → spójny angielski UI; tytuł „Kontakt” na stronie profilu.

### Jakość
- [wzmocnienie] Dostępność: axe bez naruszeń krytycznych w obu motywach, fokus, kontrast AA.
- [wzmocnienie] Testy: aktualizacja page objects, nowe testy kontraktu AJAX, e2e ścieżek async, dark mode, mobile.

### Poza zakresem (świadomie)
Audyt K1/K2/K3/W1/W3/W4 (współbieżność, kolejka zadań, storage, logika pakietu) — to zmiany
backendowe/infrastrukturalne; UI jedynie obsługuje ich skutki (np. timeout → czytelny komunikat).

## Wykonanie i własność plików

Agenci pracują w jednym drzewie roboczym, więc każdy ma rozłączny zestaw plików.

| Faza | Agent | Pliki (wyłączna własność) |
|---|---|---|
| 1 | Shell | `base.html`, `css/tokens.css`, `css/app.css`, `css/base.css`, `js/core/*`, `js/base_ui.js`, `settings.py` (crispy), `pyproject.toml`, `uv.lock`, `main/tests/test_views.py`, `e2e/pages/base_page.py`, `e2e/pages/nav_component.py`, `e2e/test_navigation.py`, `e2e/test_responsive.py` |
| 1 | Backend | `main/views.py`, `main/forms.py`, `main/templates/main/partials/*`, (minimalnie) sekcje wyników w `simulation.html`/`timeseries.html`/`calculations.html`, nowy `main/tests/test_async_views.py` |
| 1 | Komponenty | `js/charts.js`, `js/upload_zone.js`, `css/upload_zone.css`, `_upload_zone.html`, `e2e/pages/upload_component.py`, `e2e/test_upload.py` |
| 2 | Symulacja | `simulation.html`, `partials/_simulation_results.html`, `js/pages/simulation.js`, `css/simulation.css`, page object + e2e symulacji |
| 2 | Timeseries | `timeseries.html`, `partials/_timeseries_results.html`, `js/pages/timeseries.js`, `css/timeseries.css`, page object + e2e |
| 2 | ANN | `calculations.html`, `partials/_calculations_results.html`, `js/pages/calculations.js`, `css/calculations.css`, page object + e2e |
| 2 | Treści | `main_view.html`, `about.html`, `contact.html`, `userprofile.html`, `login.html`, `register.html`, `js/pages/main_view.js`, dane hero, `forms.py` (etykiety), page objects + e2e tych stron |
| 3 | Walidacja | Gemini (agy, read-only) per komponent, pełne testy (także `slow`), zrzuty ekranu light/dark/mobile, przegląd Codex (gpt-6-sol), poprawki |

Kryterium wyjścia każdego agenta: `pytest cs_app/main/tests` + jego pliki e2e zielone,
potem walidacja Gemini i poprawki znalezionych problemów.

## Wynik (2026-09-29)

Zrealizowano cały zakres powyżej w trzech fazach (fundament → strony → integracja + przegląd
krzyżowy Codex gpt-6-sol / Gemini / QA wizualne → poprawki). Stan końcowy: 317 testów unit,
229 e2e (łącznie z wolnymi, realny SWMM), 145 testów pakietu, ruff czysty; test uploadu
prawdziwego pliku (wcześniej `xfail`) przechodzi.

Dodatkowo naprawione po drodze (wykryte przez agentów i zweryfikowane):
- [poprawa] Upload przez Dropzone zwracał 500 w realnym stosie middleware (audyt K3).
- [poprawa] Porównanie SWMM vs ANN zestawiało megalitry z m³ (audyt S7): `TotalRunoffMG`
  to 10⁶ L (SI) / 10⁶ gal (US) — przeliczane teraz do m³ / ft³ (przykład: 3.36 → 3360 m³ vs ANN 2466 m³).
- [poprawa] `evaporation_loss` jest raportowane przez SWMM w mm/dzień (in/dzień), nie mm/h —
  poprawione etykiety, osobny panel na wykresie, dokumentacja pakietu.
- [poprawa] Dane przykładowe na stronie głównej pochodziły z innego modelu niż dołączony
  `example.inp` — zregenerowane skryptem `cs_app/data/build_home_data.py` + test zgodności.
- [poprawa] Formularz profilu ujawniał listę wszystkich użytkowników i pozwalał podać cudze id.
- [poprawa] Brak modelu w sesji → 500 (nieistniejący fallback `example.inp`) → teraz 400 z komunikatem.
- [poprawa] Wyniki poprzedniego modelu przeżywały wgranie nowego (audyt W2).
- [wzmocnienie] Calculations: Post/Redirect/Get także bez JS.

### Poprawki po odbiorze (2026-09-30)
- [poprawa] Wyniki analiz znikały po przejściu do innej zakładki i powrocie: cache wyników był
  w pamięci procesu (`LocMemCache`), czyszczony przy każdym auto-reloadzie `runserver` i osobny
  dla każdego workera gunicorna. Teraz `FileBasedCache` (`DJANGO_CACHE_DIR`, domyślnie
  `cs_app/.cache/django`), TTL 30 min → 12 h; testy nadal na `LocMemCache`.
- [wzmocnienie] Ponowne wczytanie modelu o identycznej treści (np. „Try sample data” na innej
  zakładce) nie kasuje już wyników ani ustawień formularzy (porównanie SHA-256).
- [poprawa] Nawigacja: linki wyśrodkowane względem strony (siatka brand | linki | narzędzia),
  przyciski „Log in” / „Create account” nie łamią się przy 992–1199 px.

### Decyzje do potwierdzenia przez właściciela
1. Przeliczenie jednostek w porównaniu SWMM vs ANN zakłada, że sieć przewiduje m³ (zgodne rzędem
   wielkości na przykładzie). Warto potwierdzić jednostkę celu treningowego ANN.
2. Teksty przycisków auth zmienione na „Log in” / „Create account” (testy zaktualizowane).
3. Rejestracja od razu loguje nowego użytkownika (wcześniej tylko przekierowanie).
4. Sweep > 8 wartości pokazuje pasek kolorów zamiast legendy (także na desktopie).
5. Kolory serii „evaporation” vs „runoff” wciąż słabo rozróżnialne dla daltonistów
   (łagodzone wzorami linii) — do decyzji przy ewentualnej zmianie palety.
6. Wyniki trzymane są 12 h w cache plikowym na hoście. Przy wdrożeniu na kilku hostach
   potrzebny wspólny backend (Redis/baza) — wystarczy zmiana `CACHES`.

### Poza zakresem, nadal otwarte (audyt)
K1/K2 (współbieżność, synchroniczne symulacje; W1 — cache per proces — rozwiązane dla wyników przez `FileBasedCache`), W3/W4 (logika pakietu),
wyścig sesji przy zmianie modelu w trakcie trwającego przebiegu.
