# Code-Review: Warchhold Research Hub (Next.js-Frontend)

**Gegenstand:** `warchhold-hub-frontend-20260928T172136Z.zip`. Laut `EXPORT_MANIFEST.json` umfasst das Paket 106 Dateien mit rund 14 400 Zeilen TS/TSX (Next.js 15, React 19, App Router).
**Art der Prüfung:** Rein statisch. Nichts wurde gebaut, installiert oder ausgeführt, und es lagen keine Laufzeitdaten vor.

- **Wie Aussagen zum Laufzeitverhalten zustande kommen:** Sie sind aus dem Code und dem dokumentierten Verhalten von Next.js abgeleitet und als Ableitung gekennzeichnet.
- **Was nicht geprüft wurde:** Welche Werte in den JSON-Dateien stehen, lässt sich ohne die Daten nicht prüfen. Dazu steht hier nichts.
- **Zitierweise:** Zeilenangaben beziehen sich auf die Dateien im Paket.

Schwere:

| Stufe | Bedeutung |
|---|---|
| **Hoch** | Führt zu falscher oder irreführender Anzeige, falschem HTTP-Status oder Funktionsattrappen auf öffentlichen Seiten. |
| **Mittel** | Robustheit, Wartbarkeit oder Performance merklich betroffen. |
| **Niedrig** | Aufräumen, Konsistenz. |

---

## Kurzfazit

1. **Seiten und Leser:** Die Seiten sind ordentlich getrennt. Sie lesen fast überall parallel (`Promise.all`), und die meisten sind Server-Komponenten mit ISR. Die Client-Komponenten sind bis auf den 3D-Graphen klein.
2. **Hauptschwäche Datenschicht:** Rund 30 Leser in 11 Dateien öffnen Dateien unter fest verdrahteten Serverpfaden. Jeder parst die Dateien selbst und erfindet eigene Ersatzwerte.
   - „Quelle fehlt“, „Datei defekt“ und „wirklich leer“ ergeben denselben Rückgabewert (`[]`, `{}`, `null`, `0`).
3. **Folgen dieser Schwäche:**
   - Seiten behaupten „Noch keine Läufe“, wo die Datei vielleicht nur fehlt.
   - Eine einzige defekte JSONL-Zeile lässt die ganze Vorwärtsbilanz verschwinden.
   - Fehlende Kennzahlen erscheinen als `0`.
4. **Einheiten und Urteile:**
   - Ein Feld namens `pnl_eur` wird an zwei Stellen als „Pkt“ beschriftet.
   - Ein Regex zählt „NICHT BESTAETIGT“ als bestätigt.
   - Die Urteilsübersetzung existiert trotz des erklärten Grundsatzes „eine Übersetzung“ noch dreimal.
5. **Ladeverhalten:** `app/algostrategien/loading.tsx` legt eine Suspense-Grenze um den ganzen Hub. Dadurch ist der HTTP-Status gesendet, bevor `notFound()` greifen kann. Unbekannte Dossier-URLs liefern deshalb vermutlich 200 statt 404 (weiche 404).
6. **Attrappen:** Auf der Strategieseite stehen Kommentar- und Bewertungs-Bedienelemente ohne jede Funktion. Dazu kommen Bewertungszahlen, die als Literale im Code stehen.
7. **Wirksamster Hebel:** eine kleine gemeinsame Leseschicht (`lib/server/`). Sie liefert Pfade aus der Konfiguration, liest `readJson` bzw. `readJsonl` zeilenweise tolerant und gibt einen Ergebnistyp mit Zustand zurück. Darauf setzt eine Hinweis-Komponente für die Seiten auf. Das behebt etwa die Hälfte der Befunde strukturell.

## Was gut ist (beibehalten, als Muster nutzen)

- **`lib/program-runtime.ts:973-1016` (`readProgramLedger`):** Parst zeilenweise, sammelt Fehler je Zeile mit Zeilennummer und bricht nicht ab. Das ist das richtige Muster für alle JSONL-Leser.
- **`lib/program-runtime.ts:1114-1168` (`inspectSource`/`readPortalSourceHealth`):** Unterscheidet bereits `missing`, `invalid`, `stale` und `ok`. Diese Unterscheidung fehlt überall sonst.
- **`app/algostrategien/research/export/[file]/route.ts:10, 47-53`:** Allowlist-Regex plus Prüfung von `path.resolve`/`startsWith` gegen Path Traversal. Die Route ist `no-store` und prüft Admin-Rechte serverseitig.
- **`lib/system-brain-docs.ts:22-40`:** Die Login-Prüfung kennt drei Zustände und bleibt bei Fehlern geschlossen (fail-closed).
- **`components/strategies/strategy-explorer.tsx:304-317`:**
  - Die Tabelle ist mit `aria-sort`, `scope="col"` und einer fokussierbaren Scroll-Region umgesetzt.
  - Die Filter-Chips haben aussagekräftige `aria-label`.
- **`components/layout/mobile-nav.tsx`:** `aria-expanded`, Escape-Taste und Fokusführung sind sauber.
- **`lib/strategy-status.ts`:** Status und Ergebnis sind als zwei getrennte Achsen modelliert, und Farbe steht nie allein.
- **`lib/verdict-labels.ts`:** Als „die EINE Übersetzung“ angelegt, das ist richtig. Sie wird nur noch nicht überall genutzt (siehe M5).

---

## Befunde

### Hoch

| Nr | Datei:Zeile | Befund | Vorschlag |
|---|---|---|---|
| H1 | `lib/build-dossier.ts:224` → `:450`; `lib/research-files.ts:266` → `app/algostrategien/research/bestenliste/page.tsx:54` und `:75` | **Einheit falsch beschriftet.**<br>• Die Vorwärtsmonate werden aus `r.pnl_eur` gelesen, im Dossier-Leser mit Rückfall auf `r.pnl`.<br>• Angezeigt werden sie als „… Pkt“ (Zeitleiste im Dossier; Spalte „Forward“ im Kandidatenvergleich).<br>• Der Feldname sagt EUR, die Anzeige sagt Punkte. Eines von beiden ist falsch; welches, lässt sich ohne Daten nicht entscheiden.<br>• Der Rückfall `pnl_eur ?? pnl` kann je Zeile sogar unterschiedliche Einheiten mischen. | Die Einheit im Leser festlegen und mit dem Wert zurückgeben (`{ value, unit }`). Keine Rückfallkette über Felder verschiedener Einheit. Beschriftung aus `unit` erzeugen (Diff D4). |
| H2 | `lib/research-files.ts:239-248`; `lib/build-dossier.ts:204-210` | **Eine defekte JSONL-Zeile verwirft die gesamte Vorwärtsbilanz.**<br>• `raw.split("\n").filter(Boolean).map((l) => JSON.parse(l))` steht in *einem* `try`.<br>• Ein Parse-Fehler in irgendeiner Zeile führt zu `return {}` bzw. `return []`.<br>• Die Seiten zeigen dann „—“, als gäbe es keine Vorwärtsdaten. | Zeilenweise parsen wie in `program-runtime.ts:987-999`, defekte Zeilen zählen und als Hinweis ausgeben. Gemeinsamer Helfer `readJsonl` (Diff D2). |
| H3 | `lib/research-files.ts:50-56, 82-88, 123-129, 239-248`; `lib/build-dossier.ts:185-192`; `lib/evidence-overview.ts:32-37`; `lib/job-calendar.ts:84-95` → z. B. `app/algostrategien/research/kandidaten/page.tsx:33-35` | **„Quelle fehlt“, „defekt“ und „leer“ sind nicht unterscheidbar.**<br>• Jeder Leser fängt alle Fehler ab und gibt einen Leerwert zurück.<br>• Die Kandidatenseite schreibt bei leerem Array „Noch keine Auto-Build-Läufe.“ Das wäre bei fehlendem Mount oder fehlenden Rechten eine falsche Tatsachenbehauptung.<br>• `research/page.tsx:37` formuliert vorsichtiger („Keine lesbaren Versuchsberichte“). Die Seiten sind also selbst uneinheitlich. | Einheitlicher Ergebnistyp `SourceResult<T> = { state: "ok" \| "empty" \| "missing" \| "invalid", data, issues }` (Diff D2). Eine Komponente `<SourceNotice>` formuliert den Hinweis einheitlich. |
| H4 | `app/algostrategien/loading.tsx:6-8` in Verbindung mit `research/build/[slug]/page.tsx:39`, `[slug]/page.tsx:39`, `research/gehirn/dokumente/[slug]/page.tsx:26` | **Weiche 404 und doppelte Landmarken durch den Lade-Platzhalter auf Hub-Ebene.**<br>• `loading.tsx` umschließt jede Seite des Hubs mit Suspense.<br>• Laut Next.js-Dokumentation zu `loading.js` („Status Codes“) sind beim Streaming die Header mit Status 200 bereits gesendet. Ein späteres `notFound()` kann den Status nicht mehr ändern (Next fügt nur `noindex` ein).<br>• Unbekannte Dossier-Slugs bei `dynamicParams = true` (`build/[slug]/page.tsx:9`) liefern deshalb 200. Das passt zur im externen Review beobachteten weichen 404.<br>• Der Platzhalter bringt ein eigenes `<main>` und `<h1>Inhalte werden geladen</h1>` in das gestreamte HTML. Ohne JavaScript und für Crawler bleibt dieser Text vor dem eigentlichen Inhalt stehen. | `loading.tsx` auf Hub-Ebene entfernen. Ladezustände nur dort, wo wirklich dynamisch gerendert wird (Kalender, Dokumente), und als Teil der Seite (Diff D5). Den CLS-Grund aus dem Dateikommentar dort gezielt lösen (reservierte Mindesthöhe im Inhaltsbereich statt Vollbild-`main`). |
| H5 | `components/comments/comment-section.tsx:36-39, 59-63`; `components/ratings/rating-panel.tsx:10, 22-24, 33`; `components/ui/button.tsx:22`; `lib/data.ts:55-65, 49-50`; Aufruf in `app/algostrategien/[slug]/page.tsx:61, 71, 135, 138` | **Funktionsattrappen und Zahlen ohne Herkunft auf einer öffentlichen Strategieseite.**<br>• Das Formular „Kommentar erstellen“ hat keinen Handler. `Button` rendert `<button>` ohne `type`, also einen Submit, und das Formular lädt die Seite neu.<br>• Die sechs Abstimmungsknöpfe haben keinen `onClick`.<br>• Das Rating startet mit den Vorgaben `[4, 4, 3, 4]`, und „Bewertung speichern“ tut nichts.<br>• `lib/supabase.ts` wird nirgends importiert.<br>• `average_rating: 4.1`, `rating_count: 7`, `community_confidence_score: 64`, `trust_score: 42` usw. stehen als Literale in `lib/data.ts`. Im Code gibt es keinen Pfad, der sie erzeugt oder speichert. Ob sie stimmen, ist hier nicht prüfbar.<br>• `tradingview_url`/`github_url` zeigen auf die Startseiten der Dienste. | Kommentar- und Rating-Bereich entfernen, bis ein Backend existiert. Bis dahin ist höchstens ein nicht interaktiver Hinweis „geplant“ vertretbar. Literale Scores aus `data.ts` entfernen oder mit Quelle versehen. `Button` bekommt `type="button"` als Vorgabe (Diff D8). Platzhalter-URLs löschen. |
| H6 | `lib/research-files.ts:196-200` | **„NICHT BESTAETIGT“ zählt als bestätigt.**<br>• `const confirmed = /BESTAETIGT/.test(doc.content)` trifft jede Teilzeichenkette im ganzen Markdown, auch Verneinungen und Zitate.<br>• Daraus wird im News-Feed „Optimierung bestaetigt (OOS)“. | Das Urteil maschinenlesbar aus einer JSON-Begleitdatei oder einer Frontmatter-Zeile lesen. Übergangsweise nur eine definierte Kopfzeile prüfen, z. B. `/^Urteil:\s*BESTAETIGT\b/m`, und die Verneinung ausschließen. |

### Mittel

| Nr | Datei:Zeile | Befund | Vorschlag |
|---|---|---|---|
| M1 | 31 Vorkommen von `/opt/trading-app` in 11 Dateien, u. a.:<br>• `lib/research-files.ts:11, 240, 282`<br>• `lib/build-dossier.ts:18, 510`<br>• `lib/program-runtime.ts:5-18, 1049-1084`<br>• `components/ui/data-stand.tsx:16`<br>• `research/export/[file]/route.ts:8`<br>Flask-URL `http://127.0.0.1:13133` dreimal: `lib/live-system.ts:5`, `lib/system-brain-docs.ts:29`, `export/[file]/route.ts:23` | **Fest verdrahtete Serverpfade.**<br>• Allein in `research-files.ts` steht die Wurzel dreimal, einmal als lokale Konstante in einer Funktion.<br>• Lokal, im Test oder auf einem zweiten Server lässt sich nichts ausführen.<br>• Ein Umzug erfordert Änderungen an 31 Stellen. | Ein Modul `lib/server/config.ts` mit `DATA_ROOT`, `HUB_ROOT` und `TRADING_API` aus Umgebungsvariablen, mit den heutigen Werten als Vorgabe. Alle Pfade werden daraus abgeleitet (Diff D1). |
| M2 | `lib/evidence-overview.ts:34`; `lib/research-files.ts:125, 769-777`; `lib/job-calendar.ts:87`; `app/algostrategien/research/gehirn/karte/page.tsx:20-27` | **`JSON.parse(...) as T` ohne Prüfung.**<br>• Fehlt ein verschachteltes Feld, fällt die Seite beim Rendern mit einem TypeError in `error.tsx`, statt nur den betroffenen Abschnitt auszublenden.<br>• Beispiel `research/evidenz/page.tsx:39` und `:49`: `d.hypothesen.je_evidenzklasse` und `d.erzeugt.slice(...)`.<br>• `job-calendar.ts` prüft wenigstens `schema` und `jobs`.<br>• `karte/page.tsx` gibt `any` ungeprüft als `Graph` an den Client. | Kleine Typwächter nach dem vorhandenen Muster `isProgramEvent` (`program-runtime.ts:957`), je Datei ein Format- und Versionsfeld prüfen. Tiefe Felder optional typisieren und auf der Seite abschnittsweise behandeln. Eine Schema-Bibliothek wäre möglich, ist aber nicht nötig. |
| M3 | `lib/research-files.ts:105-106`; `lib/build-dossier.ts:286-296, 332-335` → `research/build/[slug]/page.tsx:43`, `bestenliste/page.tsx:40` | **Fehlende Zahlen werden zu 0.**<br>• `Number(r.pf ?? 0)` macht aus „nicht vorhanden“ den Wert PF 0,00. Das verfälscht die Auswahl des „besten Trials“ (`reduce` über `pf`) und die Sortierung.<br>• Im Vorwärtsvertrag werden fehlende Schwellen zu `0` und dann als Regel angezeigt (z. B. „mind. 0 Trades“). | Helfer `num(v): number \| null` und Anzeige „—“. Auswahl des besten Trials nur über Trials mit `pf != null`. Fehlt eine Schwelle, ist der Vertrag `invalid` (Diff D3). |
| M4 | `lib/research-files.ts:94` und `bestenliste/page.tsx:56`, `research-files.ts:189`; `lib/build-dossier.ts:185-192` vs. `:196` | **Dossier-Links werden rekonstruiert statt übernommen.**<br>• Der Leser übernimmt `date: raw.date` ungeprüft. Die Links bauen `${d.date}_${d.strategy_id}` zusammen.<br>• Fehlt `date` oder weicht die `strategy_id` vom Dateinamen ab, entsteht `…/build/undefined_…` oder ein Tippfehler-Link.<br>• `listBuildSlugs` listet jede `*.json`, `readRaw` akzeptiert nur `^\d{4}-\d{2}-\d{2}_[a-z0-9_]+$`. `generateStaticParams` und `sitemap.ts` erzeugen so eventuell URLs, die 404 liefern. | Der Leser gibt den Dateistamm als `slug` mit, und Links nutzen nur diesen. `listBuildSlugs` filtert mit demselben Regex wie `readRaw` (eine Konstante). |
| M5 | `lib/research-files.ts:710-718`; `bestenliste/page.tsx:76-80`; `lib/component-registry.ts:70`; `components/research/shared.tsx:13` (`verdictMeta`) | **Die Urteilsklassifikation existiert neben `verdict-labels.ts` noch mehrfach, mit unterschiedlichen Schreibweisen.**<br>• `readResearchFunnel` erwartet `"kein Edge"`/`"schwaches Signal"` (Leerzeichen, groß, case-sensitiv).<br>• Die Bestenliste erwartet `"kein_edge"`/`"schwaches_signal"`.<br>• `verdict-labels` normalisiert auf Kleinschreibung und akzeptiert beides.<br>• Mindestens einer der Zähler trifft also nicht alle Werte. Die Trichterzahlen auf „Über mich“ (`ueber-mich/page.tsx:135-139`) können deshalb zu niedrig sein. | Eine Funktion `verdictClass(raw): "kein_edge" \| "schwach" \| "kandidat" \| …` in `verdict-labels.ts` und überall nur diese verwenden. Unbekannte Werte zählen als eigene Klasse „unbekannt“ und werden angezeigt. |
| M6 | Autobuild-Ergebnisse: `research-files.ts:15, 85-92, 702-720`, `build-dossier.ts:20, 185-200`, `component-registry.ts:6, 145`<br>`forward_runs.jsonl`: `research-files.ts:239-271`, `build-dossier.ts:204-226` | **Dieselben Quellen werden von drei Modulen mit drei Parsern gelesen.**<br>• Die Deduplizierung der Vorwärtsläufe unterscheidet sich: `candidate_id\|month` bzw. nur `month` nach der Zuordnung. Zwei Seiten können so für dieselbe Strategie unterschiedliche Summen zeigen.<br>• `forward_candidates.json` wird je Dossier-Aufruf zweimal gelesen (`build-dossier.ts:214, 231`). | Je Datenquelle genau ein Modul unter `lib/server/sources/` mit Parser, Validierung und Deduplizierung. Die übrigen Module importieren nur noch dieses. |
| M7 | `research/build/[slug]/page.tsx:18` und `:38` | **`readBuildDossier` läuft pro Anfrage zweimal** (Metadaten und Seite). Jeder Lauf liest Ergebnis, Vorwärtsläufe, Kandidatenliste (zweimal), Review, Reproduktion und das Optimizer-Verzeichnis. | Mit `cache` aus React umhüllen (Diff D6). Dasselbe gilt für andere Leser, die in `generateMetadata` und der Seite aufgerufen werden. |
| M8 | `research/export/[file]/route.ts:21-44`; `lib/system-brain-docs.ts:27-40, 43-45`; `components/layout/auth-widget.tsx:18-22` | **Die Login-Prüfung ist dreimal implementiert, und die Kopien verhalten sich unterschiedlich.**<br>• Die Export-Route wertet eine Nicht-JSON-Antwort (z. B. 502 von nginx) über `.catch(() => null)` als „nicht eingeloggt“ (401). Genau das verbietet der Kommentar in `system-brain-docs.ts:24-26`.<br>• `hasTradingAppLogin` ist tot. | Ein Modul `lib/server/auth.ts` mit `readTradingAppSession(cookie) → { state: "ok" \| "unauthorized" \| "unreachable", isAdmin }`. Route und Dokumentseite nutzen es (Diff D7). |
| M9 | `lib/job-calendar.ts:98-102, 108-114` | **Zeitzonen gemischt.**<br>• `jobsOnDay` vergleicht `next_run.slice(0, 10)` (Datum im Offset der Zeichenkette) mit `day.toISOString().slice(0, 10)` (UTC).<br>• `idleWindows` nimmt `getHours()` in der Zeitzone des Node-Prozesses.<br>• Trägt `next_run` einen Offset wie `+02:00`, landen Läufe zwischen 00:00 und 02:00 Berliner Zeit am falschen Tag. Die freien Fenster hängen von der Server-TZ ab. | Tagesschlüssel und Stunden explizit mit `Intl.DateTimeFormat("de-DE", { timeZone: "Europe/Berlin" })` bilden (Diff D9). |
| M10 | `lib/strategy-status.ts:85-90`; `lib/explorer-rows.ts:69` | **Status „Läuft · positiv“ ohne PF.**<br>• Bei `pf == null` und n ≥ 20 liefert `deriveStatus` den Wert `"live"`, beschriftet als „positiv“. Das Ergebnis wird aber getrennt über `pnl` bestimmt (`deriveResult`), beide Achsen können sich also widersprechen.<br>• `explorer-rows.ts:69` wiederholt die Schwelle 20 als Literal statt `MIN_N`. | Bei `pf == null` einen neutralen Zustand ohne Wertung verwenden (Diff D10). `MIN_N` importieren. |
| M11 | `lib/hub-navigation.ts:51-52, 55-56, 62-64`; `components/layout/hub-navigation.tsx:12-37` | **Navigation über Positionsindizes.**<br>• `hubSections[1]`, `[4]` und `[2]` plus Non-Null-`!` brechen still, sobald ein Abschnitt umsortiert wird.<br>• Jeder unbekannte Pfad unter `/algostrategien/*` wird als „Strategiedossier“ ausgewiesen, auch auf 404-Seiten.<br>• Der gesamte Seitenrahmen mit Sidebar und Brotkrumen ist eine Client-Komponente, nur um den aktiven Eintrag aus `usePathname()` zu bestimmen. | Abschnitte über `id` suchen (Diff D11). Ohne Treffer keine Detail-Brotkrume. Rahmen und Sidebar als Server-Komponente; nur ein kleines `<ActiveLink>` als Client-Komponente. |
| M12 | `components/research/brain-graph.tsx:37-40`; `app/algostrategien/research/gehirn/karte/page.tsx:20-27, 78` | **3D-Graph zu früh und zu breit geladen.**<br>• `ForceGraph3D` wird mit `ssr: false` dynamisch geladen, `UnrealBloomPass` (three/examples) und `three-spritetext` aber statisch am Modulkopf. Sie landen im Seiten-Chunk und werden beim SSR der Client-Komponente ausgewertet.<br>• Der komplette Graph-JSON geht als Prop in die RSC-Nutzlast der HTML-Seite, obwohl dieselbe Datei unter `public/brain-graph.json` ohnehin öffentlich ausgeliefert wird.<br>• 30 `any` in der Datei. | Die ganze `BrainGraph` über einen kleinen Client-Wrapper mit `dynamic(() => import(...), { ssr: false })` laden. Den Graphen clientseitig von `/brain-graph.json` holen oder serverseitig auf die gebrauchten Felder kürzen. |
| M13 | `components/ui/source-state.tsx:2-8`; `components/ui/data-stand.tsx` (Alterslogik mit `Date.now()`) | **Die Altersprüfung friert unter ISR ein.** Das Alter wird beim Erzeugen der Seite berechnet. Bei `revalidate = 300`/`1800`, und erst recht wenn eine Neugenerierung scheitert, zeigt die ausgelieferte Seite „aktuell“, obwohl die Daten inzwischen älter sind. | Serverseitig nur den Zeitstempel ausgeben, z. B. `data-stand`. Das Alter und die Warnung berechnet eine winzige Client-Komponente beim Betrachten. |
| M14 | `app/algostrategien/error.tsx:2, 6` | **Der Fehler-Rahmen ignoriert `error`.** Es werden weder `digest` noch Protokoll ausgegeben, sodass eine Nutzermeldung nicht mit dem Serverlog verknüpft werden kann. Weil viele Leser bei Formfehlern werfen (M2), ersetzt ein Formfehler die ganze Seite. | `error.digest` klein anzeigen („Fehlerkennung: …“) und `console.error(error)` in `useEffect`. Grundsätzlich aber Fehler in Lesern abfangen (H3), damit Seiten abschnittsweise degradieren. |
| M15 | `app/sitemap.ts:9, 27, 13-33` | **Sitemap anfällig und ohne Aussagekraft.**<br>• `new Date(`${slug.slice(0, 10)}T12:00:00Z`)` ergibt für jeden Dateinamen ohne Datumspräfix ein `Invalid Date`. `listBuildSlugs` filtert nicht (M4). Beim Serialisieren wirft `toISOString()` vermutlich einen RangeError, und die ganze Sitemap scheitert.<br>• Alle übrigen Einträge tragen `lastModified: new Date()`, also bei jedem Abruf „jetzt“.<br>• Anker-URLs (`alpha-faktoren#…`) gehören nicht in eine Sitemap. | Slugs mit dem Regex filtern. `lastModified` nur setzen, wo eine echte Änderungszeit bekannt ist (Dateizeit der Quelle). Anker-Einträge entfernen. |
| M16 | Tote Module und Exporte | **Toter Code** (außerhalb der eigenen Datei nicht referenziert):<br>• `lib/supabase.ts` samt zwei Abhängigkeiten in `package.json`<br>• `components/strategies/strategy-browser.tsx`, `strategy-table.tsx` (nur vom Browser genutzt), `live-vault.tsx`<br>• `lib/imprint.ts` (ganzes Modul; es gibt keine Impressum-Route im Paket)<br>• `lib/constants.ts:49, 59` (`roadmapPhases`, `methodologyPoints`)<br>• `lib/research-files.ts:280` (`readModelConfig`)<br>• `lib/system-brain-docs.ts:43` (`hasTradingAppLogin`)<br>• `app/algostrategien/research/layout.tsx` (reiner Durchreicher)<br>• Verwaister JSDoc-Block `research-files.ts:403-410`: Er beschreibt `readResearchFunnel`, steht aber über dem Mathe-Abschnitt. | Löschen. Den JSDoc-Block an `readResearchFunnel` (`:696`) verschieben. Klären, wo das Impressum tatsächlich ausgeliefert wird. |

### Niedrig

| Nr | Datei:Zeile | Befund | Vorschlag |
|---|---|---|---|
| N1 | `research/build/[slug]/page.tsx:48-51` und `components/layout/hub-navigation.tsx:28-33` | Auf Dossierseiten stehen zwei Brotkrumen-Leisten; die eigene ist ein `<p>` ohne `nav`. | Die seiteneigene Leiste entfernen und die Hub-Brotkrume nutzen (Label „Versuchsdossier“ existiert dort schon). |
| N2 | `research/build/[slug]/page.tsx:32-34, 311` | Emojis als Zeitleisten-Symbole ohne `aria-hidden`. Screenreader lesen „Hammer“, „Waage“ usw. vor. | `aria-hidden="true"` an den `<span>`. Die Art des Eintrags steht ohnehin im Titel. Besser Lucide-Icons wie im Rest des Hubs. |
| N3 | `error.tsx:6`, `not-found.tsx:2`, `build/[slug]/page.tsx:49`, `bestenliste/page.tsx:93`, `gehirn/dokumente/[slug]/page.tsx:47` | Interne Links als `<a href>` statt `Link`, was jedes Mal einen vollständigen Neuladevorgang auslöst. | Durch `next/link` ersetzen. |
| N4 | `components/ui/source-state.tsx:6` | `role="status"` auf statischem Text macht jede Quellzeile zur Live-Region. | Rolle entfernen; Live-Regionen nur für nachträglich geänderten Text. |
| N5 | `research/kalender/page.tsx:20ff.` (`stateMeta`); `lib/verdict-labels.ts:57-61` | Zwei Farbsysteme: Rohpaletten `emerald`/`red`/`yellow` neben den semantischen Tokens `success`/`danger`/`warning` (`strategy-status.ts:42-79`). | Nur die semantischen Tokens verwenden, damit Kontrast und Dunkelmodus an einer Stelle gepflegt werden. |
| N6 | `research/betrieb/page.tsx:374` | Absolute Serverpfade (`source.source`) werden öffentlich ausgegeben (die Seite steht in der Sitemap). | Nur eine Kennung und das Label anzeigen; Pfade im internen Log. |
| N7 | `lib/live-system.ts:10-22` | Ersatzspeicher `_lastGood` auf Modulebene: Bei einem Ausfall der Flask-API werden alte Daten geliefert, ohne dass der Aufrufer es erfährt. | `{ data, stale: true }` zurückgeben und auf der Seite kennzeichnen. |
| N8 | `lib/research-files.ts:184-188` | `d.results[0] ?? { pf: 0, n: 0 }` macht `best` immer wahr. `best.pf?.toFixed?.(2)` ist eine Umgehung für uneinheitliche Typen. | Mit `num()` aus M3 typisieren; ohne Trials „keine Trials“ schreiben. |
| N9 | `lib/research-files.ts:567-580` | Ungültige Zeitstempel ergeben `NaN` in `overlap`. `NaN < 0.25` ist falsch, die Zeile zählt still nicht. Die Logik ist außerdem aus Python dupliziert, entgegen dem Kommentar „Regel an genau EINER Stelle“ (`:540-541`). | Die Zahl `settledNegatives` von der Python-Seite schreiben lassen und hier nur lesen; sonst `NaN` explizit als ungültig zählen. |
| N10 | `lib/research-files.ts:644-660` | `readTradingEmpirics` prüft zuerst, ob `trades.db` existiert, liest dann aber einen JSON-Auszug. Fehlt die Datenbank im Frontend-Container, verschwinden die Daten, obwohl der Auszug da ist. | Nur den Auszug prüfen. |
| N11 | `package.json` | `lint` ruft `next lint` auf, das Paket enthält aber keine ESLint-Konfiguration. Es gibt kein Test-Skript. | `eslint.config.mjs` ergänzen. Leser-Tests mit kleinen, synthetischen Fixture-Dateien für die Fälle fehlt, defekt, leer und Teilfelder, ohne echte Werte. |
| N12 | `lib/research-data.ts` (481 Z.) vs. `lib/research-files.ts` | Fast gleichnamige Module mit völlig verschiedener Rolle: das eine kuratierter Inhalt (statisch), das andere Laufzeitleser. Kuratierte Kandidaten (`researchCandidates`) werden in `bestenliste/page.tsx:60-82` mit Laufzeitdaten gemischt. | Kuratierte Inhalte nach `content/` verschieben (siehe Zielstruktur). |

---

## Die 10 wirksamsten Änderungen

Geordnet nach Nutzen pro Aufwand.

| # | Änderung | Behebt | Aufwand |
|---|---|---|---|
| 1 | **Gemeinsame Leseschicht** `lib/server/source.ts`:<br>• `readJson`/`readJsonl`/`listDir` mit `SourceResult<T>` (ok / empty / missing / invalid + Zeilenfehler)<br>• dazu `<SourceNotice>` für einheitliche Hinweise | H2, H3, M2 (Grundlage), M14, N8 | mittel |
| 2 | **`loading.tsx` auf Hub-Ebene entfernen**, Ladezustände nur in den wirklich dynamischen Segmenten | H4 (echte 404, kein Platzhalter-`main`/`h1` im HTML) | klein |
| 3 | **Attrappen entfernen**: Kommentar- und Rating-Bereich, Literal-Scores, Platzhalter-URLs; `Button` mit `type="button"` | H5 | klein |
| 4 | **Einheit am Wert führen** (`{ value, unit }`) und keine Rückfallkette `pnl_eur ?? pnl` | H1 | klein |
| 5 | **Konfigurationsmodul** für `DATA_ROOT` und `TRADING_API`, alle 31 Pfade und 3 URLs daraus ableiten | M1, erlaubt lokale Tests | klein bis mittel |
| 6 | **Eine Urteilsklassifikation** (`verdictClass`) in `verdict-labels.ts`; Regex-„BESTAETIGT“ ersetzen | H6, M5 | klein |
| 7 | **Ein Modul je Datenquelle** (`sources/autobuild-results.ts`, `sources/forward-runs.ts` …) inkl. Deduplizierung und `slug` aus dem Dateinamen | M4, M6, M15 | mittel |
| 8 | **`num()` statt `Number(x ?? 0)`** und Auswahl des besten Trials nur über gültige Werte | M3, N8 | klein |
| 9 | **`cache()` für Leser**, die Metadaten und Seite teilen; **eine Auth-Funktion** mit drei Zuständen | M7, M8 | klein |
| 10 | **Navigation über IDs**, Rahmen als Server-Komponente; Kalender in Europe/Berlin | M11, M9 | klein |

---

## Zielstruktur (Vorschlag)

```
lib/
  server/                      ← nur serverseitig; jede Datei beginnt mit import "server-only"
    config.ts                  DATA_ROOT, HUB_ROOT, TRADING_API aus env (heutige Werte als Vorgabe)
    source.ts                  readJson / readJsonl / listDir → SourceResult<T>
    auth.ts                    readTradingAppSession(cookie) → { state, isAdmin }
    sources/                   genau EIN Leser je Datei/Verzeichnis, mit Typwächter
      autobuild-results.ts       (heute verteilt auf research-files, build-dossier, component-registry)
      forward-runs.ts            (heute research-files + build-dossier)
      forward-candidates.ts
      evidence-overview.ts
      job-calendar.ts
      program-ledger.ts          (aus program-runtime.ts; dort bereits gutes Muster)
      …                          (program-runtime.ts mit 1228 Z. in Quellen-Module zerlegen)
    views/                     seitenspezifische Zusammenstellungen (z. B. build-dossier.ts),
                               die nur sources/* verwenden und selbst kein fs importieren
  domain/                      reine Funktionen, ohne fs/fetch, gut testbar
    verdict.ts                   verdictLabel + verdictClass (eine Übersetzung)
    strategy-status.ts
    units.ts                     Wert mit Einheit, Formatierung Pkt/EUR
    time.ts                      Tagesschlüssel Europe/Berlin
    format.ts, taxonomy.ts
content/                       kuratierte, versionierte Texte (heute research-data.ts, data.ts, glossar.ts)
components/
  data/
    source-notice.tsx            ein Hinweis für fehlt / defekt / leer / veraltet
    data-stand.tsx               Zeitstempel serverseitig, Alter clientseitig
  layout/
    hub-frame.tsx                Server: Sidebar, Brotkrume, <main id="hub-content">
    active-link.tsx              Client: nur aria-current aus usePathname()
app/algostrategien/
  layout.tsx                   rendert <HubFrame> mit genau einem <main>
  (live)/research/kalender/…     nur hier eigene loading.tsx, falls nötig
  research/gehirn/karte/…        Graph-Wrapper mit dynamic(..., { ssr: false })
```

Leitregeln:

1. **Pfade, Dateien und Seiten:**
   - Kein Modul außerhalb von `lib/server/` kennt Pfade.
   - Keine Seite liest Dateien direkt; heute tun das `karte/page.tsx` und `ueber-mich/page.tsx`.
2. **Rückgabewerte der Leser:**
   - Jeder Leser gibt `SourceResult<T>` zurück und wirft nie.
   - Seiten entscheiden abschnittsweise, was sie zeigen.
3. **Fehlende Werte:** Ein fehlender Wert ist `null` und wird als „—“ angezeigt, nie als `0`. Eine Einheit reist mit dem Wert.
4. **Übersetzungen:** Urteile, Status und Einheiten haben je genau eine Übersetzung in `lib/domain/`.
5. **Layout:** Genau ein `<main>` je Seite, im Layout, nicht in jeder Seite, im Platzhalter und im 404 wiederholt.

---

## Konkrete Code-Vorschläge (kleine Diffs)

Die Diffs sind nicht kompiliert (statisches Review). Sie zeigen die Richtung und passen zum vorhandenen Stil.

### D1 – Konfiguration statt fest verdrahteter Pfade

```diff
+// lib/server/config.ts
+import "server-only";
+import path from "path";
+
+export const DATA_ROOT = process.env.HUB_DATA_ROOT ?? "/opt/trading-app/app";
+export const HUB_ROOT = path.join(DATA_ROOT, "warchhold-algo-research-hub");
+export const TRADING_API = process.env.HUB_TRADING_API ?? "http://127.0.0.1:13133";
+export const dataPath = (...parts: string[]) => path.join(DATA_ROOT, ...parts);
```

```diff
--- a/lib/research-files.ts
+++ b/lib/research-files.ts
-const APP_ROOT = "/opt/trading-app/app";
-const HUB_ROOT = path.join(APP_ROOT, "warchhold-algo-research-hub");
+import { DATA_ROOT as APP_ROOT, HUB_ROOT, dataPath } from "@/lib/server/config";
@@ export async function readForwardLedger()
-  const APP = "/opt/trading-app/app";
@@
-    const raw = await fs.readFile(path.join(APP, "research_agent/status/forward_runs.jsonl"), "utf-8");
+    const raw = await fs.readFile(dataPath("research_agent/status/forward_runs.jsonl"), "utf-8");
```

(`server-only` ist ein kleines Paket; ohne Installation kann die erste Zeile vorerst entfallen.)

### D2 – Ein Leser mit Zustand statt stillem Leerwert

```diff
+// lib/server/source.ts
+import { promises as fs } from "fs";
+
+export type SourceState = "ok" | "empty" | "missing" | "invalid";
+export type SourceResult<T> = { state: SourceState; data: T; issues: string[] };
+
+const isMissing = (e: unknown) => (e as NodeJS.ErrnoException)?.code === "ENOENT";
+
+export async function readJson<T>(file: string, guard: (v: unknown) => v is T): Promise<SourceResult<T | null>> {
+  try {
+    const value: unknown = JSON.parse(await fs.readFile(file, "utf8"));
+    return guard(value)
+      ? { state: "ok", data: value, issues: [] }
+      : { state: "invalid", data: null, issues: ["Format oder Pflichtfelder passen nicht"] };
+  } catch (e) {
+    return { state: isMissing(e) ? "missing" : "invalid", data: null,
+             issues: [e instanceof Error ? e.message : "unbekannter Lesefehler"] };
+  }
+}
+
+export async function readJsonl<T>(file: string, guard: (v: unknown) => v is T): Promise<SourceResult<T[]>> {
+  let raw: string;
+  try {
+    raw = await fs.readFile(file, "utf8");
+  } catch (e) {
+    return { state: isMissing(e) ? "missing" : "invalid", data: [], issues: [String(e)] };
+  }
+  const rows: T[] = [];
+  const issues: string[] = [];
+  raw.split("\n").forEach((line, i) => {
+    if (!line.trim()) return;
+    try {
+      const v: unknown = JSON.parse(line);
+      if (guard(v)) rows.push(v); else issues.push(`Zeile ${i + 1}: Pflichtfelder fehlen`);
+    } catch {
+      issues.push(`Zeile ${i + 1}: ungültiges JSON`);
+    }
+  });
+  return { state: rows.length ? "ok" : issues.length ? "invalid" : "empty", data: rows, issues };
+}
```

Auf der Seite (Beispiel Kandidaten):

```diff
--- a/app/algostrategien/research/kandidaten/page.tsx
+++ b/app/algostrategien/research/kandidaten/page.tsx
-        {drafts.length === 0 ? (
-          <p …>Noch keine Auto-Build-Läufe.</p>
+        {drafts.state !== "ok" ? (
+          <SourceNotice result={drafts} quelle="Auto-Build-Ergebnisse" />
```

`SourceNotice` formuliert je Zustand:

| Zustand | Hinweis |
|---|---|
| `missing` | „Quelle derzeit nicht erreichbar“ |
| `invalid` | „Quelle nicht lesbar (n Zeilen verworfen)“ |
| `empty` | „Noch keine Einträge“ |

### D3 – Fehlende Zahl ist `null`, nicht `0`

```diff
+// lib/domain/num.ts
+export const num = (v: unknown): number | null => {
+  if (v === null || v === undefined || v === "") return null;
+  const n = typeof v === "number" ? v : Number(v);
+  return Number.isFinite(n) ? n : null;
+};
```

```diff
--- a/lib/build-dossier.ts
+++ b/lib/build-dossier.ts
-    n: Number(r.n ?? 0),
-    wr: Number(r.wr ?? 0),
-    pf: Number(r.pf ?? 0),
-    pnl: Number(r.pnl ?? 0),
+    n: num(r.n),
+    wr: num(r.wr),
+    pf: num(r.pf),
+    pnl: num(r.pnl),
```

```diff
--- a/app/algostrategien/research/build/[slug]/page.tsx
+++ b/app/algostrategien/research/build/[slug]/page.tsx
-  const best = d.trials.reduce((a, b) => (b.pf > a.pf ? b : a), d.trials[0]);
+  const withPf = d.trials.filter((t) => t.pf != null);
+  const best = withPf.reduce<typeof withPf[number] | undefined>(
+    (a, b) => (!a || b.pf! > a.pf! ? b : a), undefined);
```

### D4 – Einheit am Wert, keine Rückfallkette

```diff
--- a/lib/build-dossier.ts
+++ b/lib/build-dossier.ts
-    .map((r) => ({ month: String(r.month), pnl: Number(r.pnl_eur ?? r.pnl ?? 0), n: Number(r.n ?? 0) }))
+    .map((r) => ({
+      month: String(r.month),
+      // Einheit folgt dem Feld, aus dem gelesen wurde — nie mischen.
+      pnl: r.pnl_eur != null ? { value: num(r.pnl_eur), unit: "EUR" as const }
+                             : { value: num(r.pnl), unit: "Pkt" as const },
+      n: num(r.n),
+    }))
@@
-      detail: `${fm.pnl >= 0 ? "+" : ""}${fm.pnl.toFixed(1)} Pkt · ${fm.n} Trades (ungesehene Daten)`
+      detail: `${fmtSigned(fm.pnl.value, 1)} ${fm.pnl.unit} · ${fm.n ?? "—"} Trades (ungesehene Daten)`
```

Falls `pnl_eur` in Wahrheit Punkte enthält, ist das Feld in der Quelle umzubenennen, nicht die Anzeige.

### D5 – Kein Lade-Platzhalter um den ganzen Hub

```diff
--- a/app/algostrategien/loading.tsx
+++ /dev/null
-export default function Loading() {
-  return <main className="container-shell min-h-[100svh] py-10"><h1>Inhalte werden geladen</h1>…</main>;
-}
```

```diff
--- a/app/algostrategien/research/layout.tsx
+++ /dev/null
-export default function ResearchLayout({ children }: { children: ReactNode }) {
-  return children;
-}
```

Wo ein Ladezustand gebraucht wird, z. B. im Kalender (`force-dynamic`), gehört er als `<Suspense>` um den datenabhängigen Abschnitt. Dann stehen Überschrift, Status 200/404 und `<main>` bereits vor dem Streamen fest.

### D6 – Doppeltes Lesen pro Anfrage vermeiden

```diff
--- a/lib/build-dossier.ts
+++ b/lib/build-dossier.ts
+import { cache } from "react";
@@
-export async function readBuildDossier(slug: string): Promise<BuildDossier | null> {
+export const readBuildDossier = cache(async (slug: string): Promise<BuildDossier | null> => {
   …
-}
+});
```

### D7 – Eine Login-Prüfung mit drei Zuständen

```diff
+// lib/server/auth.ts
+import { TRADING_API } from "./config";
+export type Session = { state: "ok" | "unauthorized" | "unreachable"; isAdmin: boolean };
+
+export async function readTradingAppSession(cookie: string): Promise<Session> {
+  try {
+    const r = await fetch(`${TRADING_API}/auth_status`, {
+      headers: { cookie }, cache: "no-store", signal: AbortSignal.timeout(4000),
+    });
+    const d = (await r.json().catch(() => null)) as { logged_in?: boolean; is_admin?: boolean } | null;
+    if (d === null) return { state: "unreachable", isAdmin: false };
+    return { state: d.logged_in ? "ok" : "unauthorized", isAdmin: d.is_admin === true };
+  } catch {
+    return { state: "unreachable", isAdmin: false };
+  }
+}
```

```diff
--- a/app/algostrategien/research/export/[file]/route.ts
+++ b/app/algostrategien/research/export/[file]/route.ts
-  try {
-    const cookie = req.headers.get("cookie") ?? "";
-    const auth = await fetch("http://127.0.0.1:13133/auth_status", { … });
-    const d = (await auth.json().catch(() => null)) as … | null;
-    if (!d?.logged_in) { return new Response("Login erforderlich …", { status: 401, … }); }
-    if (d.is_admin !== true) { return new Response("Export nur fuer Admins.", { status: 403, … }); }
-  } catch {
-    return new Response("Login-Pruefung nicht erreichbar …", { status: 503, … });
-  }
+  const session = await readTradingAppSession(req.headers.get("cookie") ?? "");
+  if (session.state === "unreachable") return text("Login-Prüfung nicht erreichbar — Export derzeit nicht möglich.", 503);
+  if (session.state !== "ok") return text("Login erforderlich …", 401);
+  if (!session.isAdmin) return text("Export nur für Admins.", 403);
```

### D8 – Knopf ist standardmäßig kein Submit

```diff
--- a/components/ui/button.tsx
+++ b/components/ui/button.tsx
-type Props = {
+type Props = React.ButtonHTMLAttributes<HTMLButtonElement> & {
   children: ReactNode;
   href?: string;
   className?: string;
   variant?: "primary" | "secondary" | "ghost";
 };
 
-export function Button({ children, href, className, variant = "primary" }: Props) {
+export function Button({ children, href, className, variant = "primary", type = "button", ...rest }: Props) {
@@
-  return <button className={styles}>{children}</button>;
+  return <button type={type} className={styles} {...rest}>{children}</button>;
```

### D9 – Kalendertag in Europe/Berlin

```diff
--- a/lib/job-calendar.ts
+++ b/lib/job-calendar.ts
+const BERLIN_DAY = new Intl.DateTimeFormat("sv-SE", { timeZone: "Europe/Berlin" }); // liefert YYYY-MM-DD
+const berlinDay = (d: Date) => BERLIN_DAY.format(d);
+
 export function jobsOnDay(jobs: CalendarJob[], day: Date): CalendarJob[] {
-  const key = day.toISOString().slice(0, 10);
+  const key = berlinDay(day);
   return jobs
-    .filter((job) => (job.next_run ?? "").slice(0, 10) === key)
+    .filter((job) => job.next_run != null && berlinDay(new Date(job.next_run)) === key)
```

(Analog in `idleWindows` die Stunden mit `timeZone: "Europe/Berlin"` statt `getHours()` bestimmen.)

### D10 – Kein „positiv“ ohne PF

```diff
--- a/lib/strategy-status.ts
+++ b/lib/strategy-status.ts
 export function deriveStatus(liveSlots: number, n: number, pf: number | null): StatusKey {
   const enough = n >= MIN_N;
   if (liveSlots > 0) {
-    if (!enough) return "observation";
+    if (!enough || pf == null) return "observation";
     if (pf != null && pf < 1) return "live_negative";
     return "live";
   }
```

(Die Beschreibung von `observation` müsste dann „zu wenig Trades oder keine PF-Angabe“ lauten.)

### D11 – Navigation über IDs statt Positionen

```diff
--- a/lib/hub-navigation.ts
+++ b/lib/hub-navigation.ts
+const sectionById = (id: string) => hubSections.find((s) => s.id === id);
+
 export function resolveHubLocation(pathname: string) {
   …
   if (path.startsWith(`${research}/build/`)) {
-    const section = hubSections[1];
-    return { section, item: section.items.find((item) => item.href.endsWith("/kandidaten"))!, detail: true, detailLabel: "Versuchsdossier" };
+    const section = sectionById("research");
+    const item = section?.items.find((i) => i.href.endsWith("/kandidaten"));
+    if (section && item) return { section, item, detail: true, detailLabel: "Versuchsdossier" };
   }
@@
-  if (path.startsWith(`${root}/`) && !path.startsWith(`${research}/`)) {
-    return { section: hubSections[2], item: hubSections[2].items[0], detail: true, detailLabel: "Strategiedossier" };
-  }
+  // Unbekannte Pfade bekommen keine erfundene Detail-Brotkrume.
   return { section: hubSections[0], item: hubSections[0], detail: false };
```

(IDs laut `lib/hub-navigation.ts:8-34`: `hubSections[1]` = `"research"`, `[2]` = `"strategies"`, `[4]` = `"knowledge"`; der Dokument-Zweig nutzt entsprechend `sectionById("knowledge")`.)

### D12 – Urteil nicht per Teilzeichenkette

```diff
--- a/lib/research-files.ts
+++ b/lib/research-files.ts
-    const confirmed = /BESTAETIGT/.test(doc.content);
+    // Nur die definierte Urteilszeile zählt; "NICHT BESTAETIGT" darf nicht treffen.
+    const verdictLine = doc.content.match(/^\s*(?:\*\*)?Urteil(?:\*\*)?:\s*(.+)$/m)?.[1] ?? "";
+    const confirmed = /^BESTAETIGT\b/.test(verdictLine.trim());
```

(Das Zeilenformat „Urteil: …“ ist ein Vorschlag. Das tatsächliche Format der Optimizer-Berichte liegt nicht im Paket und ist mit der erzeugenden Seite abzustimmen.)

---

## Offene Punkte, die nur mit Serverzugriff zu klären sind

1. **Einheit von `pnl_eur` in `forward_runs.jsonl`:** Enthält das Feld EUR oder Punkte? (H1)
2. **Schreibweise der `verdict`-Werte in `autobuild/results/*.json`:** Kommt `kein_edge`, `kein Edge` oder beides vor? (M5)
3. **Format von `next_run` im Jobkalender:** Mit Offset oder UTC? Und in welcher TZ läuft der Node-Prozess? (M9)
4. **Weiche 404 nachprüfen:** Liefert `curl -I https://warchhold.com/algostrategien/research/build/2099-01-01_gibt_es_nicht` heute 200? Wenn ja, bestätigt das H4. Nach D5 sollte es 404 sein.
5. **Impressum:** Wo wird es ausgeliefert, wenn `lib/imprint.ts` im Hub nirgends verwendet wird? (M16)
