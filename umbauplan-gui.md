# Umbauplan: Warchhold-Trading-GUI (Frontend)

## Gegenstand und Vorgehen

**Paket:** `warchhold-gui-frontend-20260928T172137Z.zip` mit 23 Dateien.

| Datei | Umfang |
|---|---|
| `templates/app.html` | 3 280 Zeilen |
| `static/app.js` | 7 415 Zeilen |
| 17 weitere `static/*.js` | – |
| `app.css` | 4 411 Zeilen |
| `app_mobile.css` | 493 Zeilen |

**Vorgehen:**
1. Statisch gelesen und maschinell gezählt (Skripte über alle Dateien).
2. Zusätzlich die Vorlage **lokal ohne Backend** gerendert:
   - Platzhalter `__CSS_VER__`, `__JS_VER__` und `{{ max_spread }}` durch Testwerte ersetzt.
   - socket.io und Schriften lokal gespiegelt.
   - Headless-Chromium mit 1440 × 900 und 390 × 844 Pixeln.
3. Grenzen des Renderings:
   - Alle API-Aufrufe enden dabei mit 404. Es gab keine Anmeldung, keine Verbindung zum Trading-Server und keine Order.
   - Gerendert wurde deshalb nur der Zustand „abgemeldet, ohne Daten“. Alles, was erst mit echten Daten oder Operator-Rechten sichtbar wird, ist statisch abgeleitet.

**Kennzeichnung der Befunde:**
- **[R]** im lokalen Rendering bestätigt
- **[S]** statisch aus dem Code abgeleitet

**Leitlinie für den Umbau:**
- Jeder Schritt ist ein einzelner Commit.
- Jeder Schritt lässt sich einzeln ausliefern und einzeln mit `git revert` zurücknehmen.
- Bei jeder Auslieferung wird `__JS_VER__`/`__CSS_VER__` erhöht, damit kein Browser alte Dateien mischt.
- Kein Schritt ändert gleichzeitig Verhalten und Struktur.

---

## Kurzfazit

1. **Aufbau:** Die GUI ist funktional reich und im Geldpfad an vielen Stellen sorgfältig:
   - Bestätigungsdialog und Pending-Sperre
   - 25-s-Timeout mit Abbruch
   - `client_request_id` bei der Order-Eröffnung
   - zweistufiger Kill-Switch

   Strukturell ist sie ein globaler Namensraum:
   - `app.js` legt 81 Variablen und 337 Funktionen global an.
   - 13 IIFE-Module exportieren zusammen 188 `window.*`-Namen.
   - Gesteuert wird über 276 Inline-Handler im HTML und 81 weitere in JS-erzeugtem HTML.
2. **Kritischster Befund [R]:** `app_manual.js` überschreibt `window.manualClose` mit einer Funktion, die einen Button erwartet. Die CLOSE-Knöpfe der Orderliste in `app.js` übergeben aber eine Deal-ID als Text. Der Klick wirft `btn.closest is not a function`. Es erscheint kein Dialog, und es wird nichts gesendet.
3. **Zweitkritischster Befund [R]:** Ohne das CDN-Skript socket.io bricht `app.js` in Zeile 9 ab, und die gesamte Bedienlogik fehlt.
4. **Mobil [R]/[S]:** Der Kontotyp DEMO/ECHT ist ausgeblendet. In der Orderliste ist außerdem die Spalte mit dem Echtgeld-Kennzeichen „ECHT“ ausgeblendet.
5. **Reihenfolge des Plans:**
   - Zuerst ein Sicherheitsnetz: Rauchtest und Vertragstests mit Attrappen-Backend.
   - Dann vier kleine Sofortkorrekturen mit hohem Nutzen.
   - Danach Fundament ohne Verhaltensänderung: API-Helfer, Designvariablen, toter Code.
   - Dann Bereich für Bereich Inline-Styles, Event-Delegation und Sprache, zuerst in den Research-Bereichen.
   - Den Geldpfad zuletzt und nur mit Tests.
   - Die Zerlegung von `app.js` am Ende als reines Verschieben.

---

## 1. Bestandsaufnahme

### 1.1 Dateien, Ladereihenfolge, Muster

`app.html:2227-2241` lädt die Skripte klassisch in fester Reihenfolge. Klassische Skripte teilen sich einen globalen Gültigkeitsbereich. Die Spalte „nutzt app.js“ nennt die in `app.js` definierten globalen Namen, die im Modul vorkommen; `ctx`/`canvas` sind vermutlich lokale Namensgleichheiten.

| Nr. | Datei | Zeilen | Geladen | Muster | `window.*`-Exporte | nutzt `app.js`-Globale |
|---|---|---:|---|---|---:|---|
| – | `socket.io.min.js` (cdnjs) | – | ja, Z. 6 | extern | – | – |
| – | `lightweight-charts.min.js` | 7 | ja, Z. 7 | Bibliothek | – | – |
| – | Inline `<script>` | 1 | ja, Z. 14 | liest `localStorage` für den Umschalter `ui-v2` | – | – |
| – | Inline `<script>` | 1 | ja, Z. 2227 | `window._APP_CFG={maxSpread:{{ max_spread }}}` | 1 | – |
| 1 | `app.js` | 7 415 | ja | globale Funktionen, 1 IIFE | 15 | – |
| 2 | `app_backtest.js` | 1 446 | ja | IIFE | 19 | `_esc`, `openBacktest`, `closeBacktest` |
| 3 | `app_library.js` | 810 | ja | IIFE | 19 | – (überschreibt `openLibrary`/`closeLibrary`) |
| 4 | `app_orch.js` | 1 232 | ja | IIFE | 26 | `_esc` |
| 5 | `app_live_runner.js` | 428 | ja | IIFE | 3 | `_esc` |
| 6 | `app_param_optimizer.js` | 1 468 | ja | IIFE | 21 | – (überschreibt `open`/`closeParamOptimizer`) |
| 7 | `app_calendar.js` | 506 | ja | IIFE | 5 | – |
| 8 | `app_autobt.js` | 819 | ja | IIFE | 6 | – |
| 9 | `app_signal_mining.js` | 938 | ja | IIFE | 12 | – |
| 10 | `app_regime.js` | 530 | ja | 2 IIFE | 8 | `_esc` |
| 11 | `app_regime_lab.js` | 1 449 | ja | IIFE | 22 | `_esc` |
| 12 | `app_affinity.js` | 420 | ja | IIFE | 4 | `_esc` |
| 13 | `app_manual.js` | 1 642 | ja | IIFE | 30 | `_esc`, `socket`, `checkAuth`, `showToast`, **`manualClose` (überschreibt)** |
| 14 | `app_sysmon.js` | 298 | ja | IIFE | 4 | – |
| – | Inline `<script>` (Mobil) | ca. 297 | ja, ab Z. 2921 | überschreibt `updatePrice`, `updateChart`, `drawChart`, `updateAuthUI` durch Umhüllen | 4 + `mobTab` u. a. | – |
| – | `app_core.js`, `app_accounts.js`, `app_board.js`, `app_controls.js` | 3 794 | **nein** | verwaiste Teilkopien von `app.js` (71–97 % ihrer Zeilen stehen wortgleich in `app.js`) | – | – |
| – | `backtest_engine_export.zip` | 55 Dateien | nicht referenziert | Python-Quellcode unter `/static` | – | – |

Die Abschnitte in `app.js` (Kommentarköpfe) bilden bereits die natürlichen Modulgrenzen:

| Zeile | Abschnitt |
|---:|---|
| 35 | Auth |
| 294 | Socket-Events |
| 357 | Log |
| 835 | Preis |
| 1382 | Decision Board |
| 1701 | Stats |
| 1723 | Performance-Filter |
| 1980 | Strategie-Übersicht |
| 2115 | Gesamtbewertung |
| 2223 | Daily Briefing |
| 2594 | Restart-Overlay |
| 2739 | Backtest-Modal |
| 4052 | Signal-Family-Helfer |
| 4455 | Discovery-Tabs |
| 5271 | Trainingsmodus |
| 5336 | News-Pause |
| 5344 | Emergency Stop |
| 5386 | Daily-Loss-Limit |
| 5414 | Wirtschaftskalender |
| 5510 | Orders |
| 5688 | Order-Historie |
| 5864 | Systemstatus-Leiste |
| 5916 | Systemstatus-Ampel |
| 6009 | Tools-Panel |
| 6033 | Bot-Steuerung |
| 6062 | Collector |
| 6093 | Toast |
| 6120 | Konto-Management |
| 6302 | Signal-Board-Tabs |
| 6394 | Konten |

### 1.2 Globale Variablen und Überschreibungen

- **`app.js`:** 81 `let`/`const`/`var` und 337 `function` auf oberster Ebene. Alle landen im globalen Bereich.
- **Kollisionen und Doppelungen:** Doppelte `let`/`const` über Dateien hinweg gibt es nicht; das hätte einen Syntaxfehler zur Folge. Doppelte Funktionsnamen über Dateien gibt es ebenfalls nicht.
- **Überschreibungen von `app.js`-Funktionen durch spätere Skripte:**

| Name | Überschrieben von | Art |
|---|---|---|
| `openLibrary`, `closeLibrary` | `app_library.js:27ff.` | gewollt („override top-level stubs“) |
| `openParamOptimizer`, `closeParamOptimizer` | `app_param_optimizer.js` | gewollt |
| `updatePrice`, `updateChart`, `drawChart`, `updateAuthUI` | Inline-Skript `app.html:3175-3206` | Umhüllen und Original aufrufen; funktioniert, ist aber versteckte Kopplung |
| **`manualClose`** | `app_manual.js:596` | **nicht kompatibel**, siehe Befund F1 |

### 1.3 Inline-Handler

| Ort | Anzahl |
|---|---:|
| `app.html`: `on*`-Attribute gesamt | **276** |
| davon `onclick` | 238 |
| davon `onchange` | 31 |
| davon `oninput` | 3 |
| davon `onkeydown` | 2 |
| davon `onmouseover`/`onmouseout` | je 1 |
| aufgerufene verschiedene Funktionen | 183 |
| davon im geladenen Code nicht definiert | 0 (alle aufgelöst) |
| `onclick` auf `div`/`span`/`td`/`tr`/`label`/`a` (nicht tastaturbedienbar) | 28 |
| `onclick` in JS-erzeugtem HTML (geladene Dateien) | 81 |
| Buttons ohne `type` | 189 von 200 (kein `<form>` im Dokument, daher derzeit folgenlos) |
| reine Symbol-Buttons (✕, ×, ↻ …) | 52; davon 6 mit `aria-label`, 15 nur mit `title` |

### 1.4 Inline-Styles

| Ort | Anzahl | häufigste Eigenschaften |
|---|---:|---|
| `app.html`: `style="…"` | **265** | `display` 106 (davon `display:none` 88), `margin-top` 47, `color` 45, `font-size` 43, `width` 28, `margin-left` 27, `background` 20 |
| JS: `style="…"` in erzeugtem HTML | 366 | Spitzenreiter `app.js` 130, `app_param_optimizer.js` 76, `app_regime_lab.js` 35 |
| JS: direkte Zuweisungen `.style.x =` | 296 | überwiegend `.style.display` zum Ein-/Ausblenden |

### 1.5 CSS

**`app.css`:**
- Ein `:root` mit 26 Variablen: Farben `--bg0…5`, `--t1…3`, `--acc`, `--buy`, `--sell`, `--warn` und drei Schriftfamilien.
- Daneben stehen 671 feste Hex-Farben und 64 `!important`.
- 746 Schriftgrößen in px, davon **391 unter 10 px** und **145 bei höchstens 8 px**; das Minimum sind 6 px.
- `body` hat 11 px Grundgröße. 72-mal `white-space:nowrap`, 85-mal `overflow:hidden`, 20-mal `text-overflow:ellipsis`.
- Zwei Designvarianten: Die Klasse `ui-v2` (101 Regeln) wird per `localStorage` umgeschaltet (`app.html:14`, `app.js:7378-7408`).

**`app_mobile.css`:**
- Wird nur per `media="(max-width:768px)"` geladen.
- Enthält **229 `!important`** und keine eigenen Variablen.
- Hat keine Cache-Kennung `?v=` (`app.html:10`), im Gegensatz zu `app.css`.

### 1.6 Netzwerk und schreibende Aufrufe

- **Zählung:** 178 `fetch`-Aufrufe in den geladenen Dateien. Davon folgen 137 dem Muster `r => r.json()`; nur 20 Stellen prüfen `r.ok`.
- **Weitere Zählungen:**

  | Muster | Anzahl |
  |---|---:|
  | `setInterval` | 32 |
  | `alert` | 65 |
  | `confirm` | 28 |
  | `prompt` | 5 |

- **CSRF:** Kein schreibender Aufruf sendet ein CSRF-Token. Ob der Server das anders absichert, ist hier nicht prüfbar (Frage F-O4).

Schreibende Endpunkte, nach Risikoklasse:

| Klasse | Endpunkte (Datei) |
|---|---|
| **A – Geld und Handelsbetrieb** | `/manual/order/open`, `/manual/order/close`, `/manual/order/reverse`, `/manual/order/update`, `/manual/order/validate`, `/manual/kill-switch/set`, `/manual/kill-switch/reset` (`app_manual.js`)<br>`/close_position`, `/emergency_stops`, `/daily_loss_limit`, `/training_mode`, `/news_training`, `/accounts/active`, `/accounts/save`, `/accounts/delete`, `/accounts/instrument/sync`, `/manual/account/active`, `/manual/same-account-override`, `/app/restart` (`app.js`)<br>Socket-Ereignisse `start_bot`, `stop_bot`, `start_collector`, `stop_collector` (`app.js:6034-6066`)<br>`/api/orchestrator/slots` (`app_orch.js`), `/api/param-optimizer/adopt` (`app_param_optimizer.js`) |
| **B – Forschung und Rechenlast** | `/api/nbtest/*`, `/api/backtest/strategies/*`, `/api/library/strategies/*`, `/api/calendar/*`, `/api/signal-mining/*`, `/api/regime-lab/*`, `/api/param-optimizer/start`, `/api/param-optimizer/reject`, `/api/param-optimizer/runs/*`, `/strategy-discovery/*`, `/api/system/monitor/cancel/*` |
| **C – Anzeige und Konto des Nutzers** | `/api/login`, `/api/logout`, `/api/change_password`, `/api/instruments/active_view`, `/manual/chart/layers/save` |

### 1.7 Sprache

In den 200 Buttons stehen Deutsch und Englisch sowie Groß- und Normalschreibung gemischt:

| Bedeutung | Vorkommende Beschriftungen |
|---|---|
| Neu laden | „RELOAD“ (2×), „Reload“ (2×), „Aktualisieren“ (2×), „↻“ (6×) |
| Abbrechen | „■ ABBRECHEN“, „■ Abbrechen“, „Abbrechen“ (6×) |
| Starten | „▶ STARTEN“, „▶ Starten“, „Mining starten“ |
| Englisch im Auftragsbereich | „CLOSE“, „REV“, „LIVE/ALLE/SHADOW“, „HISTORY“, „ORDERS“, „PERF“, „STRATS“ |

Umlaute sind uneinheitlich. In sichtbaren Texten stehen unter anderem:

| Umschrift | Vorkommen |
|---|---:|
| „fuer“ | 20 |
| „verfuegbar“ | 9 |
| „schliessen“ | 7 |
| „Ungueltig“ | 2 |

Daneben stehen allein in `app.html`, `app.js` und `app_manual.js` rund 90 Texte mit echten Umlauten.

---

## 2. Befunde mit Sofortwirkung (vor jedem Umbau zu kennen)

| Nr | Fundstelle | Befund | Art |
|---|---|---|---|
| **F1** | `app.js:5616`, `:5621`, `:5674` gegen `app_manual.js:596-613` | **CLOSE in der Orderliste ist wirkungslos.**<br>• `app.js` rendert `onclick="manualClose(this.dataset.dealId)"`, übergibt also eine Zeichenkette.<br>• `app_manual.js` lädt später und ersetzt `window.manualClose` durch `function(btn){ const row = btn.closest(...) … }`.<br>• Im lokalen Aufruf mit genau diesem Muster: `TypeError: btn.closest is not a function`. Es erscheint kein Bestätigungsdialog, und es wird keine Anfrage gesendet.<br>• Die beiden Varianten rufen **verschiedene Endpunkte** mit verschiedener Bedeutung: `/close_position` („Close-Signal gilt global“) bzw. `/manual/order/close` mit Deal-ID und Größe. | [R] |
| **F2** | `app.html:6`, `app.js:9` | **Harte Abhängigkeit vom CDN.**<br>• Wird `socket.io.min.js` von cdnjs nicht geladen (Störung, Firmen-Proxy, Werbeblocker), wirft `app.js` in Zeile 9 `io is not defined`. Der Rest von `app.js` läuft nicht.<br>• Auch `app_manual.js` meldet `socket is not defined`.<br>• Im Test mit blockiertem CDN reproduziert. | [R] |
| **F3** | `app_mobile.css:20-30` (`#topbar #account-info` ausgeblendet); `app_mobile.css:361-366` (Spalte 3 der Orderliste ausgeblendet); Inhalt der Spalte: `app.js:5568-5574` | **Mobil fehlt jede Kontoanzeige.**<br>• Der Kopf mit dem Badge `DEMO`/`ECHT` ist ausgeblendet.<br>• In der Orderliste ist genau die Spalte „Modus“ weg. Sie trägt den Chip `ECHT` („Echtes Geld“) je Zeile.<br>• Die Konto-Leiste unten (`app.html:2874-2887`) zeigt nur Einstellungen, Passwort und Log.<br>• Mit Operator-Rechten stehen CLOSE-Knöpfe in derselben Liste. | [R] Kopf im Screenshot ohne Badge; [S] Spalte |
| **F4** | `app.js:5568` | Ein fehlendes `account_env` wird als `DEMO` gewertet (`t.account_env \|\| 'DEMO'`). Ohne Angabe fällt die Anzeige also auf die harmlosere Kennzeichnung zurück. | [S] |
| **F5** | `app_manual.js:701-713` | **Kill-Switch-Status bei Ladefehler unbekannt, aber nicht so angezeigt.**<br>• Scheitert `GET /manual/kill-switch`, wird nur `console.warn` geschrieben.<br>• Banner und Knopf bleiben im letzten Zustand; beim ersten Laden ist das „nicht aktiv“. | [S] |
| **F6** | Muster `r => r.json()` ohne `r.ok` (137×); Beispiel `app.js:5414-5425` | **Fehler erscheinen als Leerzustand.**<br>• Der Wirtschaftskalender zeigt nach einem 404 „Keine Hoch-Prio-Ereignisse heute“ und stempelt die Uhrzeit, als wären die Daten frisch.<br>• Im Rendering ohne Backend beobachtet. | [R] |
| **F7** | `app_manual.js:774-808` | **Hotkeys hängen an Text und CSS-Klasse.**<br>• „R“ sucht den Knopf, dessen Text genau `REV` lautet.<br>• „C“ sucht `.mw-pos-btn.danger`.<br>• Eine Übersetzung („UMKEHREN“) oder eine neue Klasse schaltet die Hotkeys **ohne Fehlermeldung** ab. Das ist relevant für die Sprachvereinheitlichung. | [S] |
| **F8** | `app_manual.js:596-613`, `:615-643` | **Close und Reverse ohne Absicherung gegen Doppelsenden.**<br>• Sie senden ohne Pending-Sperre, ohne Timeout und ohne `.catch`.<br>• Ein Netzfehler erzeugt keine Meldung; die Anfrage kann doppelt gesendet werden.<br>• Die Eröffnung (`:520-585`) hat alle drei Absicherungen. | [S] |

---

## 3. Sichtbare Fehler, die sofort auffallen

| Nr | Stelle | Beobachtung | Art |
|---|---|---|---|
| V1 | `app.html:294` | In der Statusleiste steht wörtlich **`MARKT: –`**. Die JS-Escape-Folge steht im HTML und wird dort nicht ausgewertet. | [R] |
| V2 | Kopfleiste mobil | Nur „EDGE“, Preisplatzhalter, BID/ASK/SPREAD und die SYSTEM-Pille. **Kein Kontotyp** (F3). Das Logo ist auf „EDGE“ gekürzt. | [R] |
| V3 | `app.html:1001` (`&#128462;` 🗎 „BEISPIEL“); `ⓘ` (U+24D8, 3×, u. a. „ⓘ Details“) | **Fehlende Icons.**<br>• U+1F5CE steht nicht in Noto Color Emoji, DejaVu oder Liberation. Im Testsystem zeichnet ihn nur die Notschrift Unifont. Auf Systemen ohne solche Notschrift erscheint ein leeres Kästchen.<br>• U+24D8 fehlt ebenfalls in den drei genannten Schriften.<br>• Insgesamt dienen 82 verschiedene Emoji und Dingbats als Icons. Ihre Darstellung hängt von der Systemschrift ab: teils farbig (📊 🔒), teils einfarbig (⚙ ▶ ✕). Welche Zeichen auf dem Gerät des Betreibers fehlen, lässt sich nur dort feststellen. | [S] Schriftabdeckung per `fc-list` |
| V4 | CSS gesamt | **Sehr kleine Schrift.**<br>• 391 von 746 Größenangaben liegen unter 10 px, das Minimum bei 6 px.<br>• Gemessen im Startzustand: 72 sichtbare Textelemente unter 10 px am Desktop, 35 mobil. Die Beschriftungen der unteren Mobilnavigation haben 8 px.<br>• Mit `nowrap` (72×) und `overflow:hidden` (85×) werden längere Werte abgeschnitten. Im leeren Startzustand wurde noch keine Abschneidung gemessen; sie entsteht erst mit echten Daten und muss dann im Browser geprüft werden. | [R]/[S] |
| V5 | Performance-Kachel, `app.js:1918-1925` | **Fehlender Wert erscheint als „Win Rate 0.0%“ in Rot.**<br>• `parseFloat(d.win_rate\|\|0)` macht aus einem fehlenden Wert 0 und färbt ihn als schlecht (`db-nok`).<br>• Im Rendering ohne Daten sichtbar, neben „0 Trades“.<br>• Dasselbe Muster gilt für die PnL-Felder daneben.<br>• Außerdem steht dort ein Dezimalpunkt statt Komma. | [R] |
| V6 | Wirtschaftskalender | Bei gescheitertem Laden steht „Keine Hoch-Prio-Ereignisse heute“ (F6). | [R] |
| V7 | Orderliste mobil | Die Kopfzeile ist schmaler als der Tabellenrahmen, weil Spalte 3 per `nth-child` ausgeblendet ist und die Zeile „Keine Trades“ weiter über alle Spalten reicht. | [R] |
| V8 | Mobil gesamt | **Der Chart ist mobil immer ausgeblendet** (`app_mobile.css`, `#main > .panel:nth-child(2)`).<br>• `html, body { overflow:hidden }` doppelt (`app.css:15`, `app_mobile.css:8-11`); nur Innenbereiche scrollen.<br>• Englische Navigation: „PERF“, „STRATS“, „ORDERS“, „HISTORY“. | [R] |
| V9 | Knöpfe | 52 reine Symbol-Knöpfe, davon 46 ohne `aria-label`. „Schließen“ erscheint als ✕ (17×) und als × (10×). | [S] |

---

## 4. Zielbild

1. **Ein globaler Name statt 600.** Alles hängt an `window.EL`:
   - `EL.api` für Netzwerk
   - `EL.ui` für Dialog, Toast und Hinweis
   - `EL.actions` für die Aktionsregistrierung
   - `EL.fmt` für Zahlen, Einheiten und Datum
   - `EL.text` für die deutschen Texte
   - dazu die Fachmodule `EL.orders`, `EL.manual`, `EL.accounts`, `EL.backtest` …

   Zunächst bleiben die Module klassische Skripte in fester Reihenfolge; das braucht kein Build-Werkzeug. Später ist der Wechsel auf `<script type="module">` möglich, aber nicht nötig.
2. **Ein Netzwerkweg:**
   - `EL.api.get/post` prüft `r.ok`, begrenzt die Wartezeit und liefert `{ ok, status, data, error }`.
   - Aufrufe der Klasse A laufen zusätzlich über `EL.api.order`. Dieser Weg vergibt immer eine `client_request_id`, verhindert Doppelsenden und meldet jeden Ausgang sichtbar.
3. **Event-Delegation statt `onclick`:**
   - Markup: `<button type="button" data-action="manual.close" data-deal-id="…">`.
   - Ein Listener pro Dokument löst über `EL.actions.register(name, fn)` auf.
   - Klickbare `div`/`span` werden zu `<button>`.
   - Hotkeys lösen dieselben Aktionen aus, nicht Buttons über ihren Text.
4. **Designvariablen statt Inline-Styles:**
   - `:root` wird um Abstände (`--space-1…6`), Schriftgrößen (`--fs-xs` = 11 px als Untergrenze, `--fs-s`, `--fs-m` …), semantische Farben (`--c-danger`, `--c-live-money` …), Radien und Ebenen (z-index) erweitert.
   - Ein-/Ausblenden läuft über das Attribut `hidden` bzw. `.is-hidden` statt `style.display`.
   - Wiederkehrende Inline-Muster werden zu wenigen Hilfsklassen.
   - Die Variante `ui-v2` wird nach einer Entscheidung die einzige.
5. **Einheitlich Deutsch:**
   - Echte Umlaute, eine Schreibweise je Begriff, Normalschreibung statt durchgehender Versalien.
   - Fachbegriffe nach einem kurzen Glossar festlegen, z. B.:

     | Englisch / heute | Einheitlich |
     |---|---|
     | CLOSE | „Schließen“ |
     | REV | „Umkehren“ |
     | LIVE / SHADOW | „Echtbetrieb“ / „Schattenbetrieb“ (oder „Beobachtung“) |
     | SL/TP | bleibt, mit Erklärung |

   - Alle Texte stehen in `EL.text`, damit Hotkeys und Tests nie an Beschriftungen hängen.
6. **Konto immer sichtbar:** Kontotyp und Echtgeld-Kennzeichen stehen auf jeder Breite in der Kopfzeile. Ein unbekannter Kontotyp heißt „UNBEKANNT“, nie „DEMO“.
7. **Keine fremden Laufzeitquellen:** socket.io und die Schriften liegen unter `/static`. Symbole kommen aus einem kleinen SVG-Sprite statt aus Emoji.

Zielstruktur der Dateien:

```
static/
  vendor/socket.io.min.js, lightweight-charts.min.js
  css/tokens.css      :root-Variablen, einzige Quelle für Farben, Größen, Abstände
      base.css        Grundelemente, Hilfsklassen (.is-hidden, .u-mt-2 …)
      app.css         Komponenten (schrumpft schrittweise)
      mobile.css      nur Layout-Umschaltung, ohne !important-Kaskaden
  js/core/            el.js (Namensraum), api.js, actions.js, ui.js, fmt.js, text.de.js
  js/features/        orders.js, manual.js, killswitch.js, accounts.js, bot.js,
                      econ.js, board.js, backtest.js, library.js, orch.js, …
  icons.svg
templates/app.html    ohne onclick, ohne style="…", Skripte in fester Reihenfolge
tests/gui/            Playwright: Rauchtest, Vertragstests Geldpfad, Screenshots 390/1440
```

---

## 5. Reihenfolge

Grundregeln:

- **Zuerst** das Sicherheitsnetz (Phase 0). Ohne diese Tests geht kein Schritt der Phasen 3–5 live.
- **Früh** kommt, was sichtbar hilft und wenig riskiert (Phase 1 ohne Geldpfad, Phase 2).
- **Nie ohne Tests:** alles, was Klasse-A-Endpunkte, die Sichtbarkeit von Bedienelementen der Klasse A, den Bestätigungsdialog, die Hotkeys, das Operator-Gating (`_authState.is_operator`) oder die Kontoanzeige berührt. Diese Schritte sind unten mit **Geldpfad** markiert.
  - Sie brauchen grüne Vertragstests und eine manuelle Abnahme auf einem DEMO-Konto.
  - Sie werden einzeln ausgeliefert, nie zusammen mit anderen Schritten.

Übersicht:

| Schritt | Kurzname | Geldpfad | Risiko | sichtbarer Nutzen |
|---|---|---|---|---|
| 0.1 | Rauchtest mit Attrappen-Backend | – | keins (nur Tests) | – |
| 0.2 | Vertragstests Geldpfad | – | keins (nur Tests) | – |
| 1.1 | CLOSE-Konflikt beheben | **ja** | mittel | hoch |
| 1.2 | socket.io lokal + Schutz | – | niedrig | hoch |
| 1.3 | `–` korrigieren | – | sehr niedrig | mittel |
| 1.4 | Kontotyp und ECHT mobil sichtbar | **ja** (Anzeige) | niedrig | hoch |
| 1.5 | Fehler statt Leerzustand (Kalender) | – | niedrig | mittel |
| 1.6 | Kill-Switch „Status unbekannt“ | **ja** (Anzeige) | niedrig | hoch |
| 1.7 | Fehlende Icons, 0 Trades → „—“ | – | sehr niedrig | mittel |
| 2.1 | Verwaiste Dateien entfernen | – | niedrig | – |
| 2.2 | `EL`-Namensraum und `EL.api` (unbenutzt einführen) | – | sehr niedrig | – |
| 2.3 | Designvariablen und Hilfsklassen einführen | – | sehr niedrig | – |
| 2.4 | Cache-Kennung für `app_mobile.css` | – | sehr niedrig | – |
| 3.1 | Lesende Aufrufe auf `EL.api` | – | niedrig | mittel |
| 3.2 | Inline-Styles → Klassen, je Bereich | – | niedrig–mittel | mittel |
| 3.3 | Event-Delegation, je Bereich (Research zuerst) | – | mittel | – |
| 3.4 | Sprache vereinheitlichen, je Bereich (außer Geldpfad) | – | niedrig | hoch |
| 3.5 | Schrift-Untergrenze 11 px | – | mittel (Layout) | hoch |
| 4.1 | Hotkeys von Text und Klassen lösen | **ja** | mittel | – |
| 4.2 | Close/Reverse/Update über `EL.api.order` | **ja** | hoch | mittel |
| 4.3 | Delegation und Sprache im Geldpfad | **ja** | hoch | hoch |
| 4.4 | `account_env` fehlt → „UNBEKANNT“ | **ja** (Anzeige) | niedrig | hoch |
| 5.1 | `app.js` nach Abschnitten zerlegen (nur verschieben) | berührt ihn | mittel | – |
| 5.2 | Designvariante `ui-v2` als einzige | – | mittel | mittel |

---

## 6. Plan im Detail

Jeder Schritt nennt die betroffenen Dateien, das Risiko, den Prüfweg und einen kleinen Beispiel-Diff. Die Diffs sind nicht ausgeführt; sie zeigen die Richtung im Stil des vorhandenen Codes.

### Phase 0 – Sicherheitsnetz

#### 0.1 Rauchtest mit Attrappen-Backend

- **Dateien:** neu `tests/gui/smoke.spec.js`, `tests/gui/stub-server.js`. Die Vorlage wird wie im Review mit Testwerten für die Platzhalter ausgeliefert.
- **Risiko:** keins, weil der Produktivcode unverändert bleibt.
- **Prüfweg:** Der Test läuft grün gegen den heutigen Stand. Er prüft:
  - keine `pageerror`
  - jeder `onclick`-Name ist nach dem Laden eine Funktion
  - Screenshots bei 390 und 1440 px zum Vergleich
  - kein Aufruf nach außen
- **Rücknahme:** Datei löschen.

```diff
+// tests/gui/smoke.spec.js
+const { test, expect } = require('@playwright/test');
+for (const vp of [{ width: 1440, height: 900 }, { width: 390, height: 844 }]) {
+  test(`lädt ohne Skriptfehler @${vp.width}`, async ({ page }) => {
+    const errors = [];
+    page.on('pageerror', e => errors.push(e.message));
+    await page.route(/^https?:\/\/(?!127\.0\.0\.1)/, r => r.abort());   // nichts nach außen
+    await page.setViewportSize(vp);
+    await page.goto('http://127.0.0.1:8765/');
+    const unresolved = await page.evaluate(() =>
+      [...document.querySelectorAll('[onclick]')]
+        .flatMap(el => (el.getAttribute('onclick').match(/[A-Za-z_$][\w$]*(?=\s*\()/g) || []))
+        .filter(n => !['confirm', 'event'].includes(n) && typeof window[n] !== 'function'));
+    expect(errors).toEqual([]);
+    expect([...new Set(unresolved)]).toEqual([]);
+    await expect(page).toHaveScreenshot(`start-${vp.width}.png`, { maxDiffPixelRatio: 0.01 });
+  });
+}
```

Hinweis: Die Namensprüfung meldet heute zwei Fehlalarme (`rgba`, `replace` stehen in Attributwerten). Die Filterliste ist entsprechend zu ergänzen.

#### 0.2 Vertragstests für den Geldpfad

- **Dateien:** neu `tests/gui/money-path.spec.js`.
- **Risiko:** keins.
- **Prüfweg:** Alle Klasse-A-Aufrufe werden per `page.route` abgefangen, nie an einen Server gegeben. Geprüft wird je Aktion:
  1. Ohne Bestätigung geht keine Anfrage raus.
  2. Nach Bestätigung genau eine Anfrage, mit richtiger URL und richtigem Inhalt, einschließlich `client_request_id`.
  3. Ein Doppelklick erzeugt genau eine Anfrage.
  4. Eine Fehlerantwort erzeugt eine sichtbare Meldung.
  5. Die Hotkeys B/S/C/R wirken nur im Manual-Modus.
- **Erwartung heute:** Der Test „CLOSE in Orderliste“ ist **rot**. Das belegt F1 und ist gewollt. Punkt 3 ist für Close/Reverse rot (F8).

```diff
+// tests/gui/money-path.spec.js (Ausschnitt)
+test('CLOSE in der Orderliste fragt nach und sendet genau einmal', async ({ page }) => {
+  const posts = [];
+  await page.route('**/close_position', r => { posts.push(r.request().postDataJSON()); r.fulfill({ json: { ok: true } }); });
+  await page.route('**/manual/order/close', r => { posts.push(r.request().postDataJSON()); r.fulfill({ json: { ok: true } }); });
+  await loginAsOperatorStub(page);                 // Attrappe für /auth_status
+  await renderOrdersWithOpenTrade(page, 'D-1');    // Attrappe für die Orderliste
+  page.once('dialog', d => d.accept());
+  await page.click('#orders-table .btn-close-order');
+  await page.click('#orders-table .btn-close-order', { noWaitAfter: true }).catch(() => {});
+  expect(posts).toHaveLength(1);
+});
```

### Phase 1 – Sofortkorrekturen

#### 1.1 CLOSE-Konflikt beheben (Geldpfad)

- **Dateien:** `static/app.js` (3 Zeilen).
- **Risiko:** mittel, weil es den Geldpfad betrifft.
- **Kern der Korrektur:** Die Korrektur stellt das **ursprüngliche** Verhalten der Orderliste wieder her: `/close_position` mit `confirm`. Dazu erhält die `app.js`-Funktion einen eigenen Namen, und `app_manual.js` bleibt unverändert.
- **Offene Entscheidung:** Welcher Endpunkt fachlich richtig ist, entscheidet der Betreiber (Frage F-O1). Die Korrektur legt das nicht fest, sondern beseitigt nur den Namenskonflikt.
- **Prüfweg:**
  - Test 0.2 „CLOSE in der Orderliste“ wird grün.
  - Test „CLOSE im Manual-Workspace“ bleibt grün.
  - Manuelle Abnahme auf DEMO.
- **Rücknahme:** `git revert`.

```diff
--- a/static/app.js
+++ b/static/app.js
@@ -5616 +5616 @@
-          ? `<td class="action-cell"><button class="btn-close-order" data-deal-id="${_esc(t.deal_id||'')}" onclick="manualClose(this.dataset.dealId)">CLOSE</button></td>`
+          ? `<td class="action-cell"><button class="btn-close-order" data-deal-id="${_esc(t.deal_id||'')}" onclick="ordersClosePosition(this.dataset.dealId)">CLOSE</button></td>`
@@ -5621 +5621 @@
-          ? `<td class="action-cell"><button class="btn-close-order" data-deal-id="${_esc(t.deal_id||'')}" onclick="manualClose(this.dataset.dealId)">CLOSE</button></td>`
+          ? `<td class="action-cell"><button class="btn-close-order" data-deal-id="${_esc(t.deal_id||'')}" onclick="ordersClosePosition(this.dataset.dealId)">CLOSE</button></td>`
@@ -5674 +5674 @@
-function manualClose(dealId){
+// Eigener Name: app_manual.js belegt window.manualClose mit einer Button-Signatur.
+function ordersClosePosition(dealId){
```

#### 1.2 socket.io lokal ausliefern und absichern

- **Dateien:** neu `static/vendor/socket.io.min.js` in genau Version 4.7.2; `templates/app.html:6`; `static/app.js:9`.
- **Risiko:** niedrig. Es ist dieselbe Datei, nur von eigener Adresse.
- **Prüfweg:**
  - Rauchtest mit blockiertem Außennetz ist grün; heute ist er rot.
  - Die Socket-Verbindung im Betrieb baut sich auf (Log „connected“).
- **Rücknahme:** `git revert`.

```diff
--- a/templates/app.html
+++ b/templates/app.html
-<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
+<script src="/static/vendor/socket.io.min.js?v=__JS_VER__"></script>
--- a/static/app.js
+++ b/static/app.js
-const socket = io({
+if (typeof io !== 'function') {
+  document.body.insertAdjacentHTML('afterbegin',
+    '<div class="fatal-banner" role="alert">Echtzeitverbindung nicht verfügbar – Anzeige veraltet. Seite neu laden.</div>');
+}
+const socket = (typeof io === 'function' ? io : () => ({ on(){}, emit(){}, connected: false }))({
```

(Die Attrappe verhindert den Totalausfall der übrigen Oberfläche. Das Banner macht den Zustand sichtbar.)

#### 1.3 `–` in der Statusleiste

- **Dateien:** `templates/app.html:294`.
- **Risiko:** sehr niedrig.
- **Prüfweg:** Screenshot-Vergleich; dort steht dann „MARKT: –“.

```diff
-<span id="sb-mkt-v">MARKT: –</span>
+<span id="sb-mkt-v">MARKT: –</span>
```

#### 1.4 Kontotyp und Echtgeld-Kennzeichen mobil sichtbar (Geldpfad, Anzeige)

- **Dateien:** `static/app_mobile.css:20-30`, `:361-366`.
- **Risiko:** niedrig, da nur Anzeige. Die Kopfzeile wird aber enger, deshalb stattdessen die Uhr auslassen.
- **Prüfweg:**
  - Screenshot 390 px zeigt `DEMO`/`ECHT` in der Kopfzeile.
  - Die Orderliste mit einer Attrappen-Zeile `account_env: "LIVE"` zeigt `ECHT`.
  - Kein horizontales Scrollen bei 390 px.

```diff
--- a/static/app_mobile.css
+++ b/static/app_mobile.css
 #topbar .vsep,
-#topbar #account-info,
 #topbar #app-rating-launch,
@@
+/* Kontotyp ist Pflichtanzeige: nur Badge, ohne Name/Epic/Knopf */
+#topbar #account-info { display: flex !important; gap: 4px !important; }
+#topbar #account-info > :not(#acc-type-badge) { display: none !important; }
@@
-/* Hide Mode column to save space */
-#orders-table .col-mode,
-#orders-table th:nth-child(3),
-#orders-table td:nth-child(3) {
-  display: none !important;
-}
+/* Modus-Spalte bleibt: sie trägt das ECHT-Kennzeichen. Nur der Strategie-Chip entfällt. */
+#orders-table .ord-strat-chip { display: none !important; }
```

#### 1.5 Fehlerzustand statt Leerzustand (Wirtschaftskalender als Muster)

- **Dateien:** `static/app.js:5414-5425`.
- **Risiko:** niedrig.
- **Prüfweg:** Attrappe antwortet mit 500. Dann steht „Kalender nicht verfügbar (HTTP 500)“, nicht „Keine Hoch-Prio-Ereignisse heute“, und die Uhrzeit wird nicht gesetzt.

```diff
-  fetch('/economic_calendar').then(r=>r.json()).then(d=>{
+  fetch('/economic_calendar').then(r => {
+    if(!r.ok) throw new Error('HTTP ' + r.status);
+    return r.json();
+  }).then(d=>{
     if(d && d.error){ el.textContent = 'Fehler: ' + d.error; return; }
-    renderEconCal(Array.isArray(d) ? d : []);
+    if(!Array.isArray(d)){ el.textContent = 'Kalender nicht lesbar (unerwartetes Format)'; return; }
+    renderEconCal(d);
     document.getElementById('econ-ts').textContent = new Date().toLocaleTimeString('de-DE',{hour12:false});
   }).catch(err=>{
-    el.textContent = 'Nicht verfuegbar (' + err + ')';
+    el.textContent = 'Kalender nicht verfügbar (' + err.message + ')';
   });
```

#### 1.6 Kill-Switch: „Status unbekannt“ anzeigen (Geldpfad, Anzeige)

- **Dateien:** `templates/app.html:1855` (neues Element direkt unter dem bestehenden Banner), `static/app_manual.js:701-713`.
- **Risiko:** niedrig. Das Senden ändert sich nicht, nur die Anzeige und die Sperre der Order-Knöpfe.
- **Warum ein eigenes Element:** Das bestehende Banner `mt-kill-banner` enthält `mt-kill-by` und den Reset-Knopf. Ein eigenes Element vermeidet, diese zu überschreiben.
- **Prüfweg:** Vertragstest: Die Attrappe für `GET /manual/kill-switch` antwortet mit 503. Dann erscheint „Kill-Switch-Status unbekannt“, und die Knöpfe BUY/SELL sind gesperrt. Nach einer Antwort 200 mit `active:false` verschwindet der Hinweis, und die Knöpfe sind wieder frei.

```diff
--- a/templates/app.html
   <div id="mt-kill-banner" class="mt-kill-banner" style="display:none">
     …
   </div>
+  <div id="mt-kill-unknown" class="mt-kill-banner" role="alert" hidden>
+    Kill-Switch-Status unbekannt – Orders gesperrt, bis der Status wieder lesbar ist.
+  </div>
--- a/static/app_manual.js
   function manualLoadKillSwitch(){
-    fetch('/manual/kill-switch').then(r=>r.json()).then(d=>{
+    fetch('/manual/kill-switch').then(r=>{ if(!r.ok) throw new Error('HTTP '+r.status); return r.json(); }).then(d=>{
       _state.killSwitchActive = !!(d && d.active);
+      _state.killSwitchUnknown = false;
+      const unk = $('mt-kill-unknown'); if(unk) unk.hidden = true;
       const btn = $('mt-kill-btn');
@@
-      _setOrderButtonsEnabled(!_state.killSwitchActive && !_state.pending);
-    }).catch(e=>console.warn('[manual] kill-switch load failed:', e));
+      _setOrderButtonsEnabled(!_state.killSwitchActive && !_state.pending);
+    }).catch(e=>{
+      console.warn('[manual] kill-switch load failed:', e);
+      _state.killSwitchUnknown = true;
+      const unk = $('mt-kill-unknown'); if(unk) unk.hidden = false;
+      _setOrderButtonsEnabled(false);
+    });
   }
```

(`_setOrderButtonsEnabled(true)` steht außerdem in `app_manual.js:540` und `:578`, nach jeder Order. Das gibt die Knöpfe heute schon wieder frei, **auch wenn der Kill-Switch aktiv ist**, bis zur nächsten Abfrage nach bis zu 15 s. Ob der Server in diesem Fall ablehnt, ist hier nicht prüfbar. Beide Stellen sollten `!_state.killSwitchActive && !_state.killSwitchUnknown` berücksichtigen; der Vertragstest deckt es ab.)

#### 1.7 Fehlende Icons und Nullwerte

- **Dateien:** `templates/app.html:1001` und die Stellen mit `ⓘ`; `static/app.js:1918-1925` (Win Rate).
- **Risiko:** sehr niedrig.
- **Prüfweg:** Screenshot-Vergleich. Zusätzlich die Schriftabdeckung mit `fc-list ":charset=…"` für jedes verbleibende Symbol prüfen; es darf keines nur in Notschriften vorkommen.

```diff
--- a/templates/app.html
+++ b/templates/app.html
-onclick="ifaceCopy('minimal_example')" title="Minimalstrategie in Zwischenablage">&#128462; BEISPIEL</button>
+onclick="ifaceCopy('minimal_example')" title="Minimalstrategie in Zwischenablage">&#128203; BEISPIEL</button>
```

```diff
--- a/static/app.js
+++ b/static/app.js
@@ function loadPerfSummary(){
-      const wr = parseFloat(d.win_rate||0);
+      const wr = d.win_rate == null ? null : parseFloat(d.win_rate);
       const wrEl = document.getElementById('pc-wr');
-      if(wrEl){ wrEl.textContent=wr.toFixed(1)+'%';
-        wrEl.className='perf-row-val '+(wr>=50?'db-ok':wr>=45?'db-neu':'db-nok'); }
+      if(wrEl){
+        wrEl.textContent = (wr == null || !isFinite(wr)) ? '—'
+          : wr.toLocaleString('de-DE', {minimumFractionDigits: 1, maximumFractionDigits: 1}) + ' %';
+        wrEl.className = 'perf-row-val ' + (wr == null ? 'db-neu' : wr>=50 ? 'db-ok' : wr>=45 ? 'db-neu' : 'db-nok'); }
```

(Dieselbe Behandlung braucht `daily_pnl`, `daily_pts` und `total_pnl` in den Zeilen darunter.)

### Phase 2 – Fundament ohne Verhaltensänderung

#### 2.1 Verwaiste Dateien entfernen

- **Dateien:** `static/app_core.js`, `app_accounts.js`, `app_board.js`, `app_controls.js` (zusammen 3 794 Zeilen, nicht eingebunden). `app_accounts.js` liest sogar `window._APP_CFG.strategyId`, das die Vorlage nicht mehr setzt.
- **Vorbehalt:** `static/backtest_engine_export.zip` erst nach Klärung entfernen (Frage F-O3).
- **Risiko:** niedrig.
- **Prüfweg:**
  - `grep -r "app_core\|app_accounts\|app_board\|app_controls" templates/ app/` bleibt ohne Treffer. Auch der Python-Code, der nicht im Paket ist, muss geprüft werden.
  - Der Rauchtest ist grün.

```diff
-static/app_core.js
-static/app_accounts.js
-static/app_board.js
-static/app_controls.js
```

#### 2.2 Namensraum und API-Helfer einführen (noch unbenutzt)

- **Dateien:** neu `static/js/core/el.js` und `api.js`; `app.html` lädt beide **vor** `app.js`.
- **Risiko:** sehr niedrig, da noch nichts sie aufruft.
- **Prüfweg:** Rauchtest grün; `window.EL.api` ist vorhanden.

```diff
+// static/js/core/api.js
+(function (EL) {
+  'use strict';
+  async function request(method, url, body, { timeoutMs = 15000 } = {}) {
+    const ctrl = new AbortController();
+    const t = setTimeout(() => ctrl.abort(), timeoutMs);
+    try {
+      const r = await fetch(url, { method, signal: ctrl.signal,
+        headers: body ? { 'Content-Type': 'application/json' } : undefined,
+        body: body ? JSON.stringify(body) : undefined });
+      const data = await r.json().catch(() => null);
+      if (!r.ok || (data && data.ok === false))
+        return { ok: false, status: r.status, data, error: (data && data.error) || `HTTP ${r.status}` };
+      return { ok: true, status: r.status, data, error: null };
+    } catch (e) {
+      return { ok: false, status: 0, data: null, error: e.name === 'AbortError' ? 'Zeitüberschreitung' : String(e.message || e) };
+    } finally { clearTimeout(t); }
+  }
+  EL.api = { get: (u, o) => request('GET', u, null, o), post: (u, b, o) => request('POST', u, b, o) };
+})(window.EL = window.EL || {});
```

#### 2.3 Designvariablen und Hilfsklassen einführen

- **Dateien:** neu `static/css/tokens.css` (vor `app.css` geladen) und Hilfsklassen am Ende von `app.css`.
- **Risiko:** sehr niedrig. Es kommen nur neue Namen hinzu; die bestehenden `--bg…`/`--t…` bleiben als Aliase.
- **Prüfweg:** Screenshot-Vergleich ohne Unterschied.

```diff
+/* static/css/tokens.css */
+:root {
+  --space-1: 2px; --space-2: 4px; --space-3: 8px; --space-4: 12px; --space-5: 16px; --space-6: 24px;
+  --fs-xs: 11px; --fs-s: 12px; --fs-m: 14px; --fs-l: 18px;
+  --c-danger: var(--sell); --c-ok: var(--buy); --c-warn: var(--warn);
+  --c-live-money: #ff3355;          /* Echtgeld-Kennzeichen, nur dafür verwenden */
+  --radius-s: 4px; --radius-m: 8px;
+  --z-dropdown: 100; --z-modal: 1000; --z-toast: 1100;
+}
+.is-hidden { display: none !important; }
+.u-mt-2 { margin-top: var(--space-2); } .u-mt-3 { margin-top: var(--space-3); }
+.u-muted { color: var(--t3); }
```

#### 2.4 Cache-Kennung für die Mobil-CSS

- **Dateien:** `templates/app.html:10`.
- **Risiko:** sehr niedrig.
- **Prüfweg:** Nach der Auslieferung lädt ein Mobilgerät die neue Datei ohne Leeren des Caches.

```diff
-<link rel="stylesheet" href="/static/app_mobile.css" media="(max-width:768px)">
+<link rel="stylesheet" href="/static/app_mobile.css?v=__CSS_VER__" media="(max-width:768px)">
```

### Phase 3 – Bereich für Bereich (ohne Geldpfad)

Reihenfolge der Bereiche, vom geringsten zum höchsten Risiko:

1. System-Monitor
2. Kalender-Backtest
3. Signal-Mining
4. Regime/Regime-Lab/Affinity
5. Bibliothek
6. Parameter-Optimierer (außer „Übernehmen“)
7. Backtest
8. Daily Briefing
9. Wirtschaftskalender
10. Systemstatus-Leiste
11. Performance
12. Log

Jeder Bereich ist ein eigener Commit je Unterschritt 3.1–3.4.

#### 3.1 Lesende Aufrufe auf `EL.api` umstellen

- **Dateien:** je Bereich eine Datei, z. B. `static/app_regime.js:483-492`. `app_sysmon.js:73-90` prüft `r.ok` und den Content-Type bereits und kann als Vorlage dienen.
- **Risiko:** niedrig.
- **Prüfweg:**
  - Attrappe mit 200, 404, 500 und Zeitüberschreitung.
  - Jeder Fall zeigt einen eigenen Text.
  - Der Rauchtest bleibt grün.

```diff
--- a/static/app_regime.js
+++ b/static/app_regime.js
-    fetch('/api/regime/timeline?since_minutes=1440').then(r=>r.json()).then(d=>{
-      _renderTimeline(d && d.items);
-    }).catch(()=>{});
+    EL.api.get('/api/regime/timeline?since_minutes=1440').then(res=>{
+      if(!res.ok){
+        const p = document.getElementById('re-pane-timeline');
+        if(p) p.innerHTML = '<div class="re-tl-empty">Zeitleiste nicht verfügbar: '+_esc(res.error)+'</div>';
+        return;
+      }
+      _renderTimeline(res.data && res.data.items);
+    });
```

#### 3.2 Inline-Styles in Klassen überführen

- **Dateien:** `templates/app.html` (Bereich), die zugehörige JS-Datei, `app.css`.
- **Risiko:** niedrig bis mittel. Wo JS `el.style.display = …` setzt, muss **im selben Commit** auf `classList`/`hidden` umgestellt werden. Sonst kämpfen Klasse und Inline-Wert gegeneinander.
- **Prüfweg:** Screenshot-Vergleich des Bereichs (geöffnet und geschlossen) bei 390 und 1440 px, ohne Unterschied.

Beispiel: das Hinweisschild „RELOAD“ im Live-Runner (`app.html:663`, `app_live_runner.js:340-341`). Es ist heute ein klickbares `div` mit fest kodierten Farben:

```diff
--- a/templates/app.html
-          <div id="lr-reload-banner" class="pbadge" style="display:none;background:rgba(255,100,0,.18);color:#ff6400;cursor:pointer" onclick="location.reload()" title="Schema-Version veraltet – Seite neu laden">RELOAD</div>
+          <button type="button" id="lr-reload-banner" class="pbadge pbadge-warn is-hidden" onclick="location.reload()" title="Schema-Version veraltet – Seite neu laden">Neu laden</button>
--- a/static/app.css
+.pbadge-warn { background: color-mix(in srgb, var(--c-warn) 18%, transparent); color: var(--c-warn); cursor: pointer; }
--- a/static/app_live_runner.js
-    if(banner) banner.style.display=(data.schema_version!==EXPECTED_SCHEMA_VERSION)?'':'none';
+    if(banner) banner.classList.toggle('is-hidden', data.schema_version===EXPECTED_SCHEMA_VERSION);
```

(Der Farbton wechselt dabei von `#ff6400` auf `--warn` `#ffaa00`. Das ist bewusst und im Screenshot-Vergleich freizugeben. Sonst eine eigene Variable für dieses Orange anlegen.)

#### 3.3 Event-Delegation statt `onclick`

- **Dateien:** neu `static/js/core/actions.js`; je Bereich `app.html` und die Moduldatei.
- **Risiko:** mittel.
  - Ein Element darf nie gleichzeitig `onclick` und `data-action` tragen, sonst wird doppelt ausgelöst.
  - Klickbare `div`/`span` (28×) werden zu `<button type="button">`.
- **Prüfweg:**
  - Rauchtest mit erweiterter Prüfung: Jede `data-action` ist registriert, und kein Element trägt beides.
  - Tastaturbedienung (Tab, Enter/Leertaste) im Bereich manuell prüfen.

```diff
+// static/js/core/actions.js
+(function (EL) {
+  const registry = new Map();
+  EL.actions = { register(name, fn) { registry.set(name, fn); } };
+  document.addEventListener('click', ev => {
+    const el = ev.target.closest('[data-action]');
+    if (!el) return;
+    const fn = registry.get(el.dataset.action);
+    if (!fn) { console.error('Unbekannte Aktion', el.dataset.action); return; }
+    fn(el, ev);
+  });
+})(window.EL = window.EL || {});
```

```diff
--- a/templates/app.html   (Z. 1453)
-              <button class="sd-btn sd-btn-primary" id="sd-start-btn" onclick="_sdStartRun()">
+              <button type="button" class="sd-btn sd-btn-primary" id="sd-start-btn" data-action="mining.start">
                 Mining starten
--- a/static/app_signal_mining.js   (Z. 109)
-  window._sdStartRun = function () {
+  function startRun () {
     …
-  };
+  }
+  EL.actions.register('mining.start', startRun);
+  window._sdStartRun = startRun;   // Übergang, bis kein Aufrufer mehr existiert (grep), dann entfernen
```

#### 3.4 Sprache vereinheitlichen (ohne Geldpfad)

- **Dateien:** je Bereich `app.html` und JS; neu `static/js/core/text.de.js` für wiederkehrende Beschriftungen.
- **Risiko:** niedrig. **Nicht** im Bereich Manual/Orders, solange Schritt 4.1 fehlt (F7).
- **Prüfweg:** Screenshot-Vergleich (bewusste Änderung, neu freigeben). `grep` nach „fuer|verfuegbar|schliessen|Ungueltig“ im Bereich liefert keine Treffer mehr.

```diff
--- a/templates/app.html   (Z. 212, 325, 735: dreimal „Neu laden“ in drei Schreibweisen)
-        <button id="btn-app-rating-reload" onclick="loadAppRating(true)">Reload</button>
+        <button type="button" id="btn-app-rating-reload" onclick="loadAppRating(true)" aria-label="Bewertung neu laden">Neu laden</button>
-        <button class="btn-app-rating-reload" onclick="loadSystemMatrix(true)">Reload</button>
+        <button type="button" class="btn-app-rating-reload" onclick="loadSystemMatrix(true)" aria-label="System-Matrix neu laden">Neu laden</button>
-        <button id="btn-systeminfo-reload" class="btn-app-rating-reload" onclick="loadSystemInfo()" aria-label="Systeminfo neu laden">RELOAD</button>
+        <button type="button" id="btn-systeminfo-reload" class="btn-app-rating-reload" onclick="loadSystemInfo()" aria-label="Systeminfo neu laden">Neu laden</button>
--- a/static/app.js   (Z. 2470, 2473)
-    root.innerHTML = '<div class="sm-compact-empty">Keine Matrix-Daten verfuegbar.</div>';
+    root.innerHTML = '<div class="sm-compact-empty">Keine Matrix-Daten verfügbar.</div>';
-    if(metaEl) metaEl.textContent = 'System-Matrix derzeit nicht verfuegbar';
+    if(metaEl) metaEl.textContent = 'System-Matrix derzeit nicht verfügbar';
```

#### 3.5 Untergrenze für Schriftgrößen

- **Dateien:** `app.css`, `app_mobile.css`.
- **Risiko:** mittel. Größere Schrift verdrängt Inhalt; Abschneidungen werden sichtbar.
- **Vorgehen:** Bereichsweise umstellen, beginnend mit der unteren Mobilnavigation (8 px) und den Tabellen der Orderliste (mobil 9 px).
- **Prüfweg:**
  - Screenshot 390 und 1440 px mit Attrappen-Daten, die lange Werte enthalten (z. B. lange Strategienamen, sechsstellige Preise).
  - Das Skript aus dem Review meldet keine Elemente mit `scrollWidth > clientWidth` bei `overflow:hidden`.

```diff
--- a/static/app_mobile.css
 #mob-nav-konto .mob-nav-lbl {
-  font-size: 9px !important;
+  font-size: var(--fs-xs) !important;
```

### Phase 4 – Geldpfad (nur mit grünen Vertragstests, einzeln ausliefern)

#### 4.1 Hotkeys von Text und Klassen lösen (Geldpfad)

- **Dateien:** `static/app_manual.js:391-403`, `:774-808`.
- **Risiko:** mittel.
- **Prüfweg:**
  - Vertragstest: Hotkeys B, S, C und R lösen jeweils genau den Bestätigungsdialog aus, nie direkt eine Anfrage.
  - In Eingabefeldern und außerhalb des Manual-Modus sind sie wirkungslos.
  - Der Test bleibt grün, wenn die Beschriftung „REV“ testweise zu „Umkehren“ geändert wird.

```diff
-              +     (isManual ? '<button class="mw-pos-btn" onclick="manualReverse(this)">REV</button>' : '')
-              +     '<button class="mw-pos-btn danger" onclick="manualClose(this)">CLOSE</button>'
+              +     (isManual ? '<button type="button" class="mw-pos-btn" data-role="reverse" onclick="manualReverse(this)">REV</button>' : '')
+              +     '<button type="button" class="mw-pos-btn danger" data-role="close" onclick="manualClose(this)">CLOSE</button>'
@@
-        const closeBtn = rows[0].querySelector('.mw-pos-btn.danger');
+        const closeBtn = rows[0].querySelector('[data-role="close"]');
@@
-        const allBtns = rows[0].querySelectorAll('.mw-pos-btn');
-        for(const b of allBtns){ if(b.textContent.trim() === 'REV'){ b.click(); break; } }
+        const revBtn = rows[0].querySelector('[data-role="reverse"]');
+        if(revBtn) revBtn.click();
```

#### 4.2 Close, Reverse und SL/TP-Update absichern (Geldpfad)

- **Dateien:** `static/app_manual.js:596-660`; neu `EL.api.order` in `api.js`.
- **Risiko:** hoch, weil es das Senden verändert. Deshalb eigener Commit, eigene Auslieferung und DEMO-Abnahme.
- **Prüfweg:**
  - Vertragstests: Doppelklick sendet genau einmal.
  - Netzfehler, 500 und Zeitüberschreitung zeigen je eine Meldung und geben die Knöpfe wieder frei.
  - Die `client_request_id` ist je Versuch eindeutig.
  - Positionen werden nach jedem Ausgang neu geladen.

```diff
   window.manualClose = function(btn){
+    if(_state.pending){ _toast('Andere Order läuft noch', 'error'); return; }
     const row = btn.closest('[data-deal-id]');
@@
       function(){
-        fetch('/manual/order/close', {
-          method:'POST', headers:{'Content-Type':'application/json'},
-          body: JSON.stringify({deal_id: dealId, size: size, is_partial: false,
-                                client_request_id: _uuid()}),
-        }).then(r=>r.json()).then(d=>{
-          if(d.ok){ _toast('Geschlossen', 'success'); manualLoadPositions(); }
-          else { _toast('Close fehlgeschlagen: '+(d.error||''), 'error'); }
-        });
+        _state.pending = true; btn.disabled = true;
+        EL.api.post('/manual/order/close',
+          {deal_id: dealId, size: size, is_partial: false, client_request_id: _uuid()},
+          {timeoutMs: 25000}
+        ).then(res=>{
+          if(res.ok) _toast('Position geschlossen', 'success');
+          else _toast('Schließen fehlgeschlagen: ' + res.error + ' – Positionsliste prüfen.', 'error');
+        }).finally(()=>{ _state.pending = false; btn.disabled = false; manualLoadPositions(); });
       });
```

#### 4.3 Delegation und Sprache im Geldpfad (Geldpfad)

- **Dateien:** `app_manual.js`; `app.js` (Orders, Konten, Bot-Steuerung, Emergency Stop, Daily-Loss, Trainingsmodus); die zugehörigen Stellen in `app.html`.
- **Voraussetzung:** 4.1 und 4.2.
- **Risiko:** hoch. Deshalb **ein Bedienelement je Commit**, jeweils nach dem Muster aus 3.3 und 3.4.
- **Prüfweg:**
  - Vertragstest je Element.
  - Operator-Gating: Ohne `is_operator` erscheint kein Element der Klasse A. Das wird mit einer Attrappe für `/auth_status` geprüft.
  - DEMO-Abnahme.

```diff
--- a/templates/app.html
-      <button class="mt-kill-btn" id="mt-kill-btn" onclick="manualKillSwitchOpen()">KILL-SWITCH</button>
+      <button type="button" class="mt-kill-btn" id="mt-kill-btn" data-action="killswitch.open">Not-Aus (Kill-Switch)</button>
--- a/static/app_manual.js
-  window.manualKillSwitchOpen = function(){
+  function killSwitchOpen(){
     …
-  };
+  }
+  EL.actions.register('killswitch.open', killSwitchOpen);
+  window.manualKillSwitchOpen = killSwitchOpen;   // Übergang für Hotkeys/Altaufrufer
```

#### 4.4 Unbekanntes Konto nie als DEMO anzeigen (Geldpfad, Anzeige)

- **Dateien:** `static/app.js:5568-5574`.
- **Risiko:** niedrig.
- **Prüfweg:** Eine Attrappen-Zeile ohne `account_env` zeigt „Konto ?“ mit Warnfarbe.

```diff
-    const envRow = String(t.account_env || 'DEMO').toUpperCase();
-    const envHtml = envRow === 'LIVE'
-      ? '<span class="mode-chip echtgeld" title="Echtes Geld">ECHT</span>'
-      : '';
+    const envRow = String(t.account_env || '').toUpperCase();
+    const envHtml = envRow === 'LIVE'
+      ? '<span class="mode-chip echtgeld" title="Echtes Geld">ECHT</span>'
+      : envRow === 'DEMO' ? ''
+      : '<span class="mode-chip unknown" title="Kontotyp nicht übermittelt">Konto ?</span>';
```

### Phase 5 – Struktur

#### 5.1 `app.js` nach Abschnitten zerlegen (reines Verschieben)

- **Dateien:** `static/app.js`, neu `static/js/features/*.js`, `templates/app.html`.
- **Vorgehen:**
  - Je Commit **ein** Abschnitt aus der Liste in 1.1, zeilengleich verschoben.
  - Die neue Datei wird an derselben Stelle der Ladereihenfolge eingebunden (direkt nach dem Rest von `app.js`, bzw. davor, wenn spätere Teile sie beim Laden brauchen).
  - Funktionen bleiben zunächst global; da klassische Skripte einen gemeinsamen globalen Bereich teilen, ändert sich kein Verhalten.
  - Erst danach je Datei auf eine IIFE mit ausdrücklichen `EL.*`-Exporten umstellen.
- **Reihenfolge:**
  1. Toast, Log und Formatierung: werden von allen genutzt, also zuerst in `core`.
  2. Die Anzeigeabschnitte.
  3. Zuletzt Orders, Konten, Bot, Emergency und Daily-Loss (Geldpfad-Regeln).
- **Risiko:** mittel. Top-Level-Code, der beim Laden läuft, z. B. `let _histFilter = localStorage…` in `app.js:5689`, muss in derselben Reihenfolge ausgeführt werden.
- **Prüfweg:**
  - `git diff --stat` zeigt nur Verschiebungen.
  - `git log --follow`-fähig verschieben (Datei zuerst kopieren, dann aus `app.js` löschen).
  - Rauchtest und Vertragstests grün.

```diff
--- a/templates/app.html
     <script src="/static/app.js?v=__JS_VER__"></script>
+    <script src="/static/js/features/econ.js?v=__JS_VER__"></script>
--- a/static/app.js
-// ── Wirtschaftskalender ──────────────────────────────────────────────────
-function loadEconCal(){ … }
-function _translate(title){ … }
-function renderEconCal(events){ … }
+++ b/static/js/features/econ.js
+// ── Wirtschaftskalender (aus app.js:5414-5509 verschoben, unverändert) ──
+function loadEconCal(){ … }
+function _translate(title){ … }
+function renderEconCal(events){ … }
```

#### 5.2 Designvariante festlegen

- **Dateien:**
  - `templates/app.html:14`
  - `app.js:7378-7408`
  - `app.css`: 101 Regeln mit `.ui-v2`
- **Voraussetzung:** Entscheidung des Betreibers, welche Variante bleibt; heute ist V2 die Vorgabe.
- **Risiko:** mittel. Nutzer mit `uiV2 = '0'` sehen danach V2.
- **Prüfweg:** Screenshot-Vergleich gegen den V2-Stand. Mit der entfernten Umschaltung darf sich für V2-Nutzer nichts ändern.

```diff
--- a/templates/app.html
-<script>try{if((localStorage.getItem('uiV2')||'1')==='1')document.body.classList.add('ui-v2');}catch(e){document.body.classList.add('ui-v2');}</script>
+<script>document.body.classList.add('ui-v2');</script>
```

(Im Folge-Commit werden die `.ui-v2`-Präfixe in `app.css` aufgelöst und die Klassik-Regeln gelöscht.)

---

## 7. Prüfwege im Überblick

| Prüfweg | Werkzeug | Pflicht bei |
|---|---|---|
| Rauchtest (keine Skriptfehler, alle Handler aufgelöst, keine Aufrufe nach außen) | Playwright gegen Attrappen-Server | jedem Schritt |
| Screenshot-Vergleich 390 × 844 und 1440 × 900 | Playwright `toHaveScreenshot` | Styles, Sprache, Layout |
| Vertragstests Geldpfad (Bestätigung vor Anfrage, genau eine Anfrage, Payload, Fehlerfälle, Hotkeys, Operator-Gating) | Playwright mit `page.route`, nie echter Server | allen Schritten mit **Geldpfad** |
| Tastatur und Fokus im Bereich | manuell, 5 Minuten | 3.3, 4.3 |
| DEMO-Abnahme (eine Order öffnen, SL ändern, schließen, Kill-Switch setzen und zurücksetzen) | manuell, DEMO-Konto | allen Schritten mit **Geldpfad**, vor Freigabe |
| Rücknahme | `git revert <commit>` + `__JS_VER__`/`__CSS_VER__` erhöhen | jederzeit |

---

## 8. Offene Fragen an den Betreiber

1. **F-O1:** Soll CLOSE in der Orderliste (AUTO-Workspace) das globale Close-Signal `/close_position` senden oder eine bestimmte Position über `/manual/order/close` schließen? Schritt 1.1 stellt nur das frühere Verhalten wieder her.
2. **F-O2:** Welche Gerätegruppe nutzt die Mobilansicht mit Operator-Rechten? Davon hängt ab, ob CLOSE mobil überhaupt angeboten werden soll.
3. **F-O3:** Ist `/static/backtest_engine_export.zip` absichtlich öffentlich abrufbar? Die GUI verweist nicht darauf.
4. **F-O4:** Wie schützt der Flask-Server die schreibenden Endpunkte gegen Cross-Site-Anfragen (SameSite-Cookie, Origin-Prüfung, Token)? Im Frontend wird kein Token gesendet.
5. **F-O5:** Wird `{{ max_spread }}` (`app.html:2227`) ebenfalls per `.replace()` gesetzt? Bleibt der Platzhalter stehen, ist das Inline-Skript ein Syntaxfehler, und `app.js:857` liest `undefined`.
