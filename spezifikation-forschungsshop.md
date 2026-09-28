# Spezifikation: Forschungsshop für Warchhold Research

**Version:** 0.1 (Entwurf zur Freigabe)

**Grundlage:**
- **Portal:** Quelltext `warchhold-hub-frontend-20260928T172136Z.zip`, vor allem:
  - `lib/build-dossier.ts`
  - `app/algostrategien/research/build/[slug]/page.tsx`
  - `app/algostrategien/research/bestenliste/page.tsx`
  - `lib/live-system.ts`, `lib/strategy-status.ts`
- **Methodikpaket:** `warchhold-methodik-20260928T172136Z.zip`, vor allem:
  - `CLAUDE_research_agents.md`
  - `forschungsmethodik-werkzeuge-2026-09-03.md`
  - `audit-umsetzung-research-hub-2026-09-26.md`
- **Frühere Reviews:** `review-hub-code.md` in diesem Repository.

**Was diese Spezifikation nicht enthält:**
- **Keine Laufzeitdaten.** Sie liegen nicht vor, und es wurden keine erfunden. Alle Zahlen in diesem Dokument sind entweder Konfigurationswerte, die der Betreiber festlegt (mit „Vorschlag“ markiert), oder Schwellen, die die Plattform heute schon verwendet (mit Fundstelle).
- **Digistore24-Schnittstelle nur teilweise belegt:**
  - Belegt ist das Signaturverfahren aus der offiziellen Beispieldatei [`sha_sign.php`](https://www.digistore24.com/download/ipn/examples/ipn/sha_sign.php): SHA-512, Parameter nach Namen sortiert, `sha_sign` ausgenommen, leere Werte übersprungen, Passphrase eingebunden, Ergebnis in Großbuchstaben.
  - Ebenfalls belegt: Der Ereignisname kommt im Parameter `event`, und es gibt das Ereignis `on_payment`.
  - Alle weiteren Feld- und Ereignisnamen sind mit **[DS24 prüfen]** markiert. Sie werden vor Phase 3 gegen die aktuelle [Entwicklerdokumentation](https://dev.digistore24.com/hc/en-us/sections/32544282072977-IPN) und den Testmodus festgelegt. Die Doku war für diese Recherche nicht maschinell abrufbar (HTTP 403).

**Feste Entscheidungen des Betreibers** (nicht Gegenstand dieser Spezifikation, nur umgesetzt):
- Digistore24 als Merchant of Record, angebunden per IPN mit Signaturprüfung, ohne eigene Zahlungsabwicklung.
- „Forschungspakete“ ohne Ertragsversprechen.
- Startpreise 4,99 € (Strategie) und 7,99 € (Familie), als Konfiguration.
- Keine Rohdaten.
- Angebot durch die KI nach einer bestätigten Aufnahmeregel.
- Der Betreiber setzt Preise, Hinweistext, Aufnahmeregel und den Hauptschalter.
- Gescheiterte Funde bleiben sichtbar.

---

## 0. Begriffe

| Begriff | Bedeutung | Quelle im Bestand |
|---|---|---|
| **Fund** | Ein Auto-Build mit Ergebnisdatei | `research_agent/autobuild/results/<datum>_<strategy_id>.json`; Slug = Dateistamm (`build-dossier.ts:185-200`) |
| **Träger** | Eigenständige Strategie | Build ohne `carrier` |
| **Baustein** | Ergänzung eines Trägers (Filter, Stop, Gate) | `spec.candidate_label` beginnt mit „Baustein“; Träger aus `spec.carrier`, für Altbestand aus `LEGACY_CARRIER` (`build-dossier.ts:37-55`, `:324-327`) |
| **Variante** | Weiterer Build mit demselben Träger | `variants` (`build-dossier.ts:344-362`) |
| **Familie** | Ein Träger mit allen Bausteinen und Varianten, die auf ihn zeigen | abgeleitet |
| **Forschungspaket** | Käufliche, versionierte Zusammenstellung zu einem Fund (Strategie-Paket) oder einer Familie (Familien-Paket) | neu |
| **Evidenzklasse** | **in-sample:** Backtest auf der Bauhistorie<br>**vorwärts:** simuliert auf Daten nach dem Einfrieren<br>**live:** DEMO-Betrieb mit Spielgeld | `evidence_overview.json` (`lib/evidence-overview.ts:15`); DEMO: `lib/live-system.ts:31-38` |
| **Einstufung** | explorativ (n < 20), vorläufig (n < 50), belastbarer (n ≥ 50) | `shared_intelligence/public_evidence.py` laut `audit-umsetzung-research-hub-2026-09-26.md:25-26` |
| **Shop-Status** | in Beobachtung / bestätigt / gescheitert (Abschnitt 3.4) | neu, aus Evidenzfeldern abgeleitet |

---

## 1. Schaufenster (kostenlos)

### 1.1 Grundsätze

1. **Vollständigkeit statt Auswahl:**
   - Das Schaufenster zeigt **jeden** Fund aus dem Ergebnisverzeichnis, auch gescheiterte und nicht angebotene.
   - Kopf der Übersicht: „N Funde untersucht · davon G gescheitert · B in Beobachtung · K bestätigt · A im Angebot“, jeweils mit Link auf die gefilterte Liste.
   - Die Zahlen werden aus dem Katalog gezählt, nicht eingetippt.
2. **Keine Rangliste:**
   - Standardsortierung nach Baudatum, neueste zuerst.
   - Andere Sortierungen nur nach Name oder Familie.
   - Keine Sortierung nach PF, PnL oder Score im Shop. Die Bestenliste bleibt die Forschungsansicht und wird unverändert verlinkt.
3. **Mehrfachtest offenlegen:** Jede Paketseite nennt, aus wie vielen registrierten Versuchen der Fund stammt (`manifest.n_registered_trials`, `build-dossier.ts:151`). Die plattformweite Signalrate verlinkt auf die Methodik (Kennzahl 4, `forschungsmethodik-werkzeuge-2026-09-03.md:100-116`).
4. **Nur Punkte, nie Euro:**
   - Jede Ergebnisgröße erscheint in Indexpunkten („Pkt“).
   - Felder, deren Einheit Euro ist oder nicht eindeutig feststeht, werden nicht angezeigt. Das gilt heute für `pnl_eur` in `forward_runs.jsonl` und `live.pnl_eur` in der öffentlichen API (siehe 1.5 und Voraussetzung V1).
5. **Jede Zahl mit Pflichtangaben:** Stichprobe n, Einstufung und Evidenzklasse (Abschnitt 4.3).
6. **Kein Druck:**
   - Keine Countdowns, keine „nur noch heute“-Hinweise, keine Streichpreise, keine Beliebtheitsangaben, keine vorausgewählten Optionen.
   - Keine Superlative (Abschnitt 4.2).

### 1.2 Übersichtsseite `/algostrategien/forschungspakete`

Aufbau von oben nach unten:

1. **Kopf:**
   - Titel „Forschungspakete“.
   - Ein Satz aus dem Hinweistext des Betreibers (Konfiguration).
   - Die Zählzeile aus 1.1.
2. **Einordnung** (drei feste Sätze aus der Konfiguration):
   - Was ein Paket enthält.
   - Was es nicht enthält: keine Kursdaten, keine Empfehlung, kein Ertragsversprechen.
   - Dass gescheiterte Funde kostenlos einsehbar sind.
3. **Filter:**
   - Status (alle / in Beobachtung / bestätigt / gescheitert)
   - Familie
   - Art (Träger / Baustein)
   - im Angebot (ja / nein)

   Der Zustand steht in der URL, damit gefilterte Ansichten verlinkbar sind.
4. **Familienblöcke:**
   - Ein Block je Träger, darin die Karte des Trägers und darunter die Karten seiner Bausteine und Varianten.
   - Funde ohne Familie erscheinen als eigener Block.
5. **Karte je Fund:**
   - Name (Klartextname aus `candidate_label`)
   - Status-Plakette mit Text und Symbol (nicht nur Farbe, Muster aus `lib/strategy-status.ts`)
   - ein Satz Idee
   - Evidenzleiste: in-sample n · vorwärts n · live (DEMO) n, jeweils mit Einstufung
   - „im Angebot“ oder „nicht im Angebot – Grund“
   - Link „Ansehen“

   Kein Preis auf der Karte, damit die Übersicht eine Forschungsübersicht bleibt.
6. **Fuß:**
   - Hinweis- und Haftungstext (Konfiguration).
   - Link auf Methodik und Lesehilfe.
   - Datenstand des Katalogs (Erzeugungszeit, Muster `components/ui/data-stand.tsx`).

Bei ausgeschaltetem Shop bleibt die Seite vollständig erhalten. Nur „im Angebot“ wird zu „Verkauf pausiert“.

### 1.3 Paketseite `/algostrategien/forschungspakete/[paketId]`

`paketId` = Build-Slug (`<datum>_<strategy_id>`) für Strategie-Pakete bzw. `familie_<strategy_id des Trägers>` für Familien-Pakete. Unbekannte IDs liefern eine echte 404 (siehe Befund H4 in `review-hub-code.md`: kein `loading.tsx` über dieser Route).

| Bereich | Inhalt | Quelle |
|---|---|---|
| **Kopf** | Name, Status-Plakette, Art (Träger/Baustein), bei Bausteinen „Baustein für: <Träger>“ mit Link, Baudatum, Familie | `candidate_label`, `isBaustein`, `carrier`, `date`, `family` |
| **Idee** | Die vor dem Test registrierte Hypothese, unverändert zitiert und als Zitat gekennzeichnet („Vor dem Test festgehalten am …“) | Hypothesenregister-Eintrag (`mechanism`, `prediction`, `refutation`, `forschungsmethodik-werkzeuge-2026-09-03.md:35-42`), ersatzweise `spec.rationale` |
| **Mechanik in Klartext** | Zwei bis vier Sätze: welches Marktverhalten ausgenutzt werden soll, in welchem Zeitfenster, in welche Richtung. **Keine** Parameterwerte, keine Einstiegsschwellen (die sind bezahlter Inhalt, 2.1). | Generiert nach Schablone aus Hypothese, `session`, Richtung (Abschnitt 4) |
| **Status und Begründung** | Shop-Status plus ein Satz, **welches Feld** ihn begründet, z. B. „Vorwärtstest: Abbruchkriterium nach Monat 2 erreicht“ | Abschnitt 3.4 |
| **DEMO-Verlauf** (nur wenn vorhanden) | Linie der kumulierten Punkte je Trade, darunter n, Einstufung, 90-%-Bereich des PF (sofern geliefert), Zeitraum, Hinweis „Spielgeld-Konto, keine Echtgeld-Ergebnisse“. Bei n < 20 kein PF, nur „explorativ“ (Regel `pf_als_leistung`, `live-system.ts:37`). | **Voraussetzung V1:** eine Punkte-Zeitreihe aus der öffentlichen Evidenzquelle; heute liefert `live-system.ts:35` nur `pnl_eur` |
| **Vorwärtstest** (falls Vertrag vorhanden) | Eingefrorene Konfiguration (Datum), Monate, Abbruch- und Endkriterien im Wortlaut, Entscheidungsstand. Bei Bausteinen gepaart: Träger ohne und mit Baustein, je Monat. | `ForwardContract` (`build-dossier.ts:96-127`) |
| **Backtest-Übersicht (in-sample)** | Tabelle je Trial: Bezeichnung, n, Trefferquote, PF, Punkte, max. Drawdown in Punkten. Robustheit der **vorregistrierten** Konfiguration (nicht des besten Trials): Bootstrap-Bereich des PF, Block-Bereich, PF bei +1 Pkt Slippage, Top-3-Anteil, Monatsabhängigkeit, Placebo-Perzentil, Auflösbarkeit (MDE). **Keine Parameterwerte** in dieser Tabelle (siehe 1.5). | `DossierTrial` (`build-dossier.ts:57-84`) |
| **Bekannte Schwächen** | Mechanisch erzeugte Liste (Abschnitt 4.4), z. B. „Ergebnis hängt an einem Monat“, „Stichprobe unter der Auflösungsgrenze“, „nach +1 Pkt Slippage PF unter 1“, „Obduktion: Überanpassung“. Leer ist nur erlaubt, wenn keine Regel greift. Dann steht dort: „Keine der geprüften Schwächen-Regeln hat angeschlagen. Das schließt andere Schwächen nicht aus.“ | Abschnitt 4.4 |
| **Prüfspur** | Unabhängige Review (Entscheidung, Kopfzeile), Reproduktion (exakt/abweichend), Nikkei-Transfer, Zahl der registrierten Versuche | `review`, Reproduktion, `nikkei_check`, `manifest` |
| **Kaufbereich** (nur wenn im Angebot und Shop an) | Zwei gleich große Karten nebeneinander (mobil untereinander, Familie zuerst). Die Familienkarte zählt sachlich auf, was darin ist. Die Strategiekarte trägt den Satz „Bei Bausteinen ist der Träger immer enthalten.“ Je Karte ein Knopf „Weiter zu Digistore24“. Darunter: Hinweistext, Link auf Widerrufsbelehrung und Lizenzbedingungen (Texte aus der Konfiguration), „Was ist im Paket?“ (Inhaltsverzeichnis aus 2). | Konfiguration, Katalog |
| **Nicht im Angebot** | Statt Kaufbereich: „Nicht im Angebot. Grund: <Regel-ID und Klartext>“. Bei gescheiterten Funden zusätzlich der vollständige Befund (Obduktion, Vorwärtsentscheidung). | Aufnahmeprotokoll (3.3) |
| **Fuß** | Hinweis-/Haftungstext, Katalogstand, Link auf das Forschungsdossier `/algostrategien/research/build/<slug>` | – |

**Familien-Paket als naheliegende Wahl, ohne Druck:** Die Darstellung stützt sich nur auf sachliche Gründe.
- Ein Baustein ist ohne seinen Träger nicht nutzbar.
- Die Familie enthält die gepaarten Vergleiche aller Bausteine.
- Die Karte nennt die Anzahl der enthaltenen Funde und deren Status. Gescheiterte Bausteine sind darin **mitgezählt und so benannt**.

Die Familie wird zuerst gezeigt (mobil oben, am Desktop links), ist aber weder vorausgewählt noch farblich hervorgehoben.

### 1.4 Verhältnis zur bestehenden Dossierseite (Entscheidung nötig)

Die öffentliche Dossierseite zeigt heute bereits:
- **Parameterwerte je Trial:** `build/[slug]/page.tsx:190-192`, `Object.entries(t.params)`.
- **Optimizer-Raster:** `:141-149`.
- **Quelle der Idee:** `:135-138`.

Damit ist ein Teil des späteren Regelwerks frei sichtbar. Die Spezifikation schlägt vor, das **Forschungsdossier unverändert öffentlich** zu lassen. Gründe:
- die Nachvollziehbarkeit der Forschung
- die Pflicht, Gescheitertes sichtbar zu halten
- die Suchmaschinen-Präsenz

Den Paketwert bilden Regelwerk, Pseudocode, Referenzcode, Evidenzblatt und Aktualisierungen.

Die Alternative wäre, die Parameterspalten des Dossiers für angebotene Funde auszublenden. Das ändert die Forschungsdarstellung und ist eine **Betreiberentscheidung E1** (Abschnitt 11).

### 1.5 Was das Schaufenster nie zeigt

Ausgeschlossen sind:
- Parameterwerte angebotener Funde im Shop-Bereich (siehe E1 für das Dossier)
- Pseudocode, Code
- Euro-Beträge jeder Art
- Echtgeld-Ergebnisse
- Kontonamen, Kontokennungen
- Kurse, Kerzen, Ticks
- Einstiegs- und Ausstiegspreise einzelner Trades
- Zeitstempel einzelner Trades
- interne Pfade (`manifest.strategy_source_path`, `build-dossier.ts:153`)
- Engine-Fingerabdrücke

---

## 2. Paketinhalt (bezahlt)

### 2.1 Strategie-Paket

Ein ZIP-Archiv `warchhold-forschungspaket_<paketId>_v<version>_<lizenz>.zip`:

| Datei | Inhalt | Entstehung |
|---|---|---|
| `LIESMICH.pdf` | Inhaltsverzeichnis, Lizenzkennung, Hinweis- und Haftungstext, Lizenzbedingungen, Versionsstand | Schablone + Konfiguration |
| `regelwerk.pdf` | Regelwerk in Klartext: Instrument (nur Name), Zeitraster, Handelszeiten und Sperrzeiten, Einstieg Long/Short, Ausstieg (Ziel, Stop, Zeit, Sitzungsende), Filter, Positionsgröße (nur als Einheit), **vollständige Parametertabelle** mit Wert, Einheit, Bedeutung, Herkunft (vorregistriert / optimiert / fest). Das Regelwerk der vorregistrierten bzw. im Vorwärtstest eingefrorenen Konfiguration kommt zuerst; weitere Trials folgen im Anhang. | KI-Entwurf, mechanisch gegen Code geprüft (2.4) |
| `pseudocode.txt` | Sprachneutraler Pseudocode derselben Logik, gleiche Parameternamen wie im Regelwerk | KI-Entwurf, mechanisch geprüft (2.4) |
| `referenz/<strategy_id>.py` | Original-Plugin-Code als Referenz, byte-gleich mit der gebauten Fassung, mit vorangestelltem Kommentarkopf (Lizenzkennung, SHA-256 des Originals, „Referenz für das Warchhold-Backtest-Interface; ohne dieses Interface nicht lauffähig“) | Kopie aus `research_strategies/`, Hash-geprüft (3.2 Regel A4) |
| `evidenzblatt.pdf` | Alle Zahlen aus 1.3 mit Pflichtangaben, plus Robustheit je Trial, Placebo, Obduktion, Vorwärtsvertrag im Wortlaut, Prüfspur mit Hashes (Ergebnisdatei, Code, Trial-Ledger-ID), Mehrfachtest-Angabe, Schwächenliste, Datenstand | Schablone, nur Evidenzfelder (Abschnitt 4) |
| `manifest.json` | Paket-ID, Version, Lizenzkennung, Erzeugungszeit, Aufnahmeregel-Version und deren SHA-256, Liste aller Dateien mit SHA-256 | Generator |

**Bei Bausteinen enthält das Strategie-Paket immer:**
- den Träger vollständig (Regelwerk, Pseudocode, Referenzcode, Kurzfassung der Evidenz),
- einen Abschnitt **„Gepaarter Vergleich: Träger ohne / mit Baustein“** im Evidenzblatt:
  - in-sample aus den A/B-Trials des Bausteins
  - vorwärts aus dem gepaarten Vertrag (`ForwardContract.baselineLabel/overlayLabel`, Entscheidungsmetriken `baseline_pnl`, `overlay_pnl`, `overlay_n`, `overlay_profit_factor`, `positive_overlay_months`)

  Beides nur in Punkten; ist die Einheit nicht belegt, fehlt der Wert mit Hinweis (V1).

### 2.2 Familien-Paket

Ein ZIP-Archiv mit:
- dem Träger als Strategie-Paket-Inhalt
- je Baustein und Variante einem Unterordner mit Regelwerk, Pseudocode, Referenzcode und Evidenzblatt
- einer Familienübersicht `familie.pdf`: Tabelle aller Mitglieder mit Status, Evidenzklassen und n, gepaarte Vergleiche nebeneinander

**Gescheiterte Mitglieder sind enthalten und als gescheitert gekennzeichnet**, mit Befund.

Umfang zeitlich: Die Familie ist der Stand zum Kaufzeitpunkt. Ob später gebaute Bausteine als Aktualisierung nachgeliefert werden und wie lange, ist Konfiguration `aktualisierung.familie_neue_mitglieder_monate`. Der Vorschlag ist 12; Betreiberentscheidung **E2**.

### 2.3 Ausgeschlossen aus jedem Paket

Wie 1.5, außer Parametern und Code. Zusätzlich ausgeschlossen:
- Trade-Listen mit Zeitstempeln oder Preisen
- Monats-PnL aus Echtgeld
- `data_tick_count`, `engine_fingerprint`, `strategy_source_path`
- Server- und Hostnamen, Epic- bzw. Kontokennungen
- Agenten- und Modellprotokolle

Der Bereinigungsfilter (8.4) erzwingt das.

### 2.4 Prüfung der KI-Texte gegen den Code

Regelwerk und Pseudocode sind die einzigen frei formulierten Teile. Deshalb gilt:

1. **Parameter-Abgleich:**
   - Der Generator liest die Parameter des Plugins mechanisch aus: Klassenattribute, Konstruktor-Vorgaben und `params` der Trials, per AST-Analyse, ohne Ausführung.
   - Die Parametertabelle im Regelwerk muss **genau** diese Namen und Werte enthalten, nicht mehr und nicht weniger.
   - Zahlen im Fließtext sind nur als Verweis auf Tabellenzeilen erlaubt.
2. **Zeiten-Abgleich:** Handelszeiten und Sperrzeiten im Text müssen `spec.session` bzw. den Zeitkonstanten im Code entsprechen.
3. **Zweitprüfung:**
   - Ein zweiter, unabhängiger Agentenlauf erhält nur Code und Text, ohne Evidenz.
   - Er beantwortet eine feste Frageliste: „Beschreibt der Text jeden Einstiegspfad des Codes? jeden Ausstiegspfad? jeden Filter?“
   - Jede Antwort „nein“ oder „unklar“ verhindert die Aufnahme (Regel A9).
   - Das Muster entspricht der bestehenden unabhängigen Review (`AutobuildReview`, `build-dossier.ts:129-141`).
4. **Keine Wertung im Regelwerk:** Das Regelwerk ist deskriptiv. Der Filter für verbotene Formulierungen (4.2) läuft darüber.

### 2.5 Spätere Stufe: TradingView-Fassung (Pine Script)

Nur mit einer Gleichheitsprüfung gegen das Original.

1. **Vergleich:** Pine-Fassung und Original laufen auf **derselben Kerzenreihe** eines Referenzzeitraums.
2. **Herkunft der Kerzen:** Die Kerzen müssen aus einer Quelle stammen, deren Nutzung dafür erlaubt ist. Broker-Daten scheiden ohne Klärung aus; siehe Rechts-Checkliste.
3. **Verglichen werden die Signal- und Tradelisten:** Zeitpunkt, Richtung, Ausstiegsgrund.
4. **Aufnahmekriterium, Vorschlag:**
   - 100 % Übereinstimmung der Einstiegszeitpunkte und Richtungen.
   - Abweichungen nur bei Ausstiegen durch Intrabar-Reihenfolge, einzeln aufgelistet.
5. **Fehlschlag:** Scheitert die Prüfung, wird keine Pine-Fassung ausgeliefert.
6. **Paketinhalt:** Die Pine-Datei trägt dieselbe Lizenzkennung. Das Evidenzblatt enthält das Prüfprotokoll.

Diese Stufe ist nicht Teil der Phasen 1–4.

---

## 3. Aufnahmeregel

### 3.1 Ablage und Bestätigung

**Dateien** (unter `shop/config/`, versioniert im Repository der Plattform):
- `aufnahmeregel.v1.json`: die Regel (Schema unten)
- `aufnahmeregel.v1.bestaetigung.json`:

  ```json
  { "regel_datei": "aufnahmeregel.v1.json", "regel_sha256": "<64 hex>",
    "bestaetigt_von": "betreiber", "bestaetigt_am": "<ISO-Datum>", "hinweis": "<freier Text>" }
  ```

**Durchsetzung:**
- Der Generator lädt die Regel nur, wenn der SHA-256 der Regeldatei mit der Bestätigung übereinstimmt. Sonst bricht er ab, und es wird **nichts** neu aufgenommen; Bestehendes bleibt unverändert.
- Eine Änderung der Regel erzeugt `v2` und verlangt eine neue Bestätigung.
- Jedes Paket merkt sich, unter welcher Regelversion es aufgenommen wurde.
- Beim Wechsel der Regelversion werden **alle** Pakete neu bewertet. Fällt ein Paket durch, greift 3.5.

**Rolle der KI:**
- Die Plattform führt den Generator zeitgesteuert aus (Vorschlag: täglich nach dem Methodik-Lauf, `forschungsmethodik-werkzeuge-2026-09-03.md:142-147`).
- Die Aufnahmeentscheidung ist **deterministisch** aus Regel und Evidenzfeldern. Frei formuliert sind nur Regelwerk und Pseudocode, geprüft nach 2.4.
- Es gibt keine manuelle Einzelauswahl.

### 3.2 Kriterien (Vorschlag)

Alle Kriterien sind mechanisch prüfbar. Schwellen, die die Plattform bereits verwendet, sind mit Fundstelle markiert; die übrigen sind **Vorschläge** zur Bestätigung.

| ID | Kriterium | Prüfung | Herkunft der Schwelle |
|---|---|---|---|
| A1 | Fund vollständig lesbar | Ergebnisdatei parsebar; Slug passt zu `^\d{4}-\d{2}-\d{2}_[a-z0-9_]+$`; Pflichtfelder vorhanden (`spec.strategy_id`, `results[]`, `verdict`, `manifest`) | Slug-Regel `build-dossier.ts:196` |
| A2 | Unabhängige Review liegt vor und verwirft nicht | `autobuild_reviews/<slug>.v1.json` gültig nach `build-dossier.ts:372-379`; `decision ∈ {advance, hold}`; `result_sha256` = SHA-256 der Ergebnisdatei | bestehende Prüfung |
| A3 | Exakte Reproduktion | `reproductions/<slug>.v1.json` mit `status = "exact_match"`; `source_sha256` = Code-Hash | Feldwert aus `build-dossier.ts:409` |
| A4 | Code unverändert seit Bau | SHA-256 von `research_strategies/<strategy_id>.py` = `manifest.strategy_source_sha256` = `review.strategy_source_sha256` | – |
| A5 | Mindeststichprobe in-sample | n der **vorregistrierten bzw. eingefrorenen** Konfiguration ≥ 50 („belastbarer“). Nicht das beste Trial: Auswahl nach PF wäre selbst Überanpassung. | Einstufungsgrenze 50 (`audit-umsetzung…:25`) |
| A6 | Kostenfest | Robustheit dieser Konfiguration: `pf_slip_10` (PF bei +1,0 Pkt Slippage) > 1,0 **und** `pf_noslip`-Lücke dokumentiert | +1,0 Pkt ist die bestehende Sensitivitätsprüfung (`CLAUDE_research_agents.md:484`); Schwelle 1,0 = Vorschlag |
| A7 | Nicht von einem Monat getragen | `robustness.single_month_dependent = false` | Feld `build-dossier.ts:75` |
| A8 | Besser als Zufall | `placebo.candidate_pct_rank ≥ 0,95` | Vorschlag |
| A9 | Regelwerk-Prüfung bestanden | 2.4 Punkte 1–3 fehlerfrei | – |
| A10 | Keine Überanpassung laut Obduktion | Obduktionsursache ∉ {`ueberanpassung`, `kosten`, `vorzeichenwechsel`}. `stichprobe` bzw. `verlust_ohne_klare_ursache` schließen nur aus, wenn ein Vorwärtstest vorliegt. | Ursachenliste `forschungsmethodik-werkzeuge-2026-09-03.md:76-86` |
| A11 | Hypothese nicht widerlegt, nicht nachträglich | Register: kein Urteil `refuted`; `post_hoc = false` | Registerregeln `forschungsmethodik-werkzeuge-2026-09-03.md:44-54` |
| A12 | Vorwärtstest nicht gescheitert | Kein Vorwärtsentscheid mit Wirkung „gescheitert“ (Abbildung 3.4) | Vertrag `build-dossier.ts:96-127` |
| A13 | Robustheits-Score mindestens „Befund“ | `robustness_score.score ≥ 45` | „ein hoher PF mit Score unter 45 ist eine Anekdote“ (`bestenliste/page.tsx:94-96`) |
| A14 | Auflösbar | `robustness.power_ok = true` | Feld `build-dossier.ts:74` |
| A15 | Paket bereinigt | Bereinigungsfilter (8.4) ohne Treffer | – |
| A16 | Bausteine nur mit Träger | Träger ermittelbar (A1 für den Träger erfüllt); Träger selbst muss **nicht** aufgenommen sein, wird aber vollständig beigelegt | – |

**Familien-Paket:** Aufnahme, wenn der Träger A1–A15 erfüllt. Mitglieder werden unabhängig von ihrem Status beigelegt.

Die Regeldatei (Auszug) mit Regel-ID, Feldpfad, Vergleich und Schwelle je Kriterium:

```json
{
  "format": "warchhold-shop-aufnahmeregel",
  "version": 1,
  "gilt_fuer": ["strategie", "familie"],
  "konfiguration_waehlen": "eingefroren_sonst_vorregistriert",
  "kriterien": [
    { "id": "A5",  "feld": "konfiguration.n",                          "op": ">=", "wert": 50 },
    { "id": "A6",  "feld": "konfiguration.robustness.pf_slip_10",       "op": ">",  "wert": 1.0 },
    { "id": "A7",  "feld": "konfiguration.robustness.single_month_dependent", "op": "==", "wert": false },
    { "id": "A8",  "feld": "konfiguration.placebo.candidate_pct_rank",  "op": ">=", "wert": 0.95 },
    { "id": "A13", "feld": "robustness_score.score",                    "op": ">=", "wert": 45 },
    { "id": "A10", "feld": "obduktion.ursache", "op": "nicht_in", "wert": ["ueberanpassung", "kosten", "vorzeichenwechsel"] }
  ],
  "fehlendes_feld": "nicht_aufnehmen"
}
```

**Grundsatz:** Fehlt ein Feld, gilt das Kriterium als **nicht erfüllt**. Ein fehlender Wert wird nie zu 0 oder `true`; das ist der Fehler M3 aus `review-hub-code.md`.

**Offener Punkt O1:** Woran die „vorregistrierte Konfiguration“ in der Ergebnisdatei erkennbar ist (Feld, Trial-Label oder Hypothesen-Eintrag), ist im Portal-Code nicht sichtbar. Das Portal wählt heute das Trial mit dem höchsten PF (`build/[slug]/page.tsx:43`, `bestenliste/page.tsx:40`). Solange O1 offen ist, gilt:
- mit Vorwärtsvertrag: die eingefrorene Konfiguration
- ohne Vorwärtsvertrag: keine Aufnahme

### 3.3 Aufnahmeprotokoll

Je Lauf entsteht `shop/aufnahme/<datum>.jsonl` mit einer Zeile je Fund. Sie enthält:
- `paketId`
- Regelversion und deren SHA-256
- Ergebnis je Kriterium: `erfuellt`, `nicht_erfuellt` oder `feld_fehlt`, mit gelesenem Wert
- Gesamtergebnis
- Hashes der gelesenen Dateien

Das Schaufenster zeigt bei „nicht im Angebot“ das erste nicht erfüllte Kriterium mit Klartext aus einer festen Tabelle, z. B. „A5: Stichprobe unter 50 Trades“.

### 3.4 Shop-Status

Die Abbildung geht von Evidenzfeldern auf drei Zustände. Die erste zutreffende Zeile gilt:

| Shop-Status | Bedingung (Vorschlag) |
|---|---|
| **gescheitert** | Vorwärtsentscheid mit Wirkung Abbruch oder Endprüfung nicht bestanden; **oder** Review `decision = reject`; **oder** Obduktion `ueberanpassung`/`kosten`/`vorzeichenwechsel`; **oder** Hypothese `refuted`; **oder** Urteil „kein Edge“ laut `verdictLabel` (`lib/verdict-labels.ts`, Tonalität `negative`) |
| **bestätigt** | Vorwärtsentscheid Endprüfung **bestanden** nach Vertrag (alle Endkriterien `finalMin…` erfüllt) |
| **in Beobachtung** | alles andere, auch „nicht entscheidbar“ |

**Offener Punkt O2:** Die tatsächlichen Werte von `decision.status` und `decision.effect` in `forward_pair_decisions/*.json` sind im Portal-Code nicht aufgezählt; nur der Rückfallwert `evidence_invalid` ist sichtbar (`build-dossier.ts:269`).

Deshalb gilt:
- Die Abbildung wird als **Tabelle in der Aufnahmeregel** geführt (`status_abbildung: { "<Rohwert>": "gescheitert" | "bestaetigt" | "beobachtung" }`).
- Ein unbekannter Rohwert führt zu Status „unklar“, Paket nicht im Angebot und einer Meldung an den Betreiber.

„Bestätigt“ heißt im Schaufenster ausdrücklich: „Vorwärtskriterien erfüllt. Kein Nachweis künftiger Ergebnisse.“

### 3.5 Automatische Herausnahme

| Auslöser | Wirkung auf das Angebot | Wirkung auf Käufer |
|---|---|---|
| Status wird „gescheitert“ | sofort aus dem Angebot; Seite bleibt mit Befund | automatische Befund-Mail (6.5) |
| Code-Hash ≠ Bau-Hash (A4 verletzt) | sofort aus dem Angebot, „geändert seit Bau“ | keine; Paketinhalt bleibt, eine neue Fassung ist ein neuer Fund |
| Reproduktion nicht mehr exakt | aus dem Angebot | Hinweis-Mail mit Evidenzblatt-Aktualisierung |
| Neue Regelversion nicht erfüllt | aus dem Angebot, „Aufnahmeregel v<N>“ | keine |
| Quelle unlesbar (fehlt/defekt) | **pausiert**, nicht zurückgezogen; Kaufknopf aus, Hinweis „Evidenz derzeit nicht prüfbar“ | keine |
| Hauptschalter aus | alles „Verkauf pausiert“ | keine; Zustellung läuft weiter (6.6) |

Zurückgezogene Pakete können nur durch einen neuen Aufnahmelauf mit allen Kriterien wieder ins Angebot kommen, nie manuell.

---

## 4. Texte

### 4.1 Erzeugung aus Evidenzfeldern

Alle Texte im Schaufenster, im Evidenzblatt, in `LIESMICH.pdf` und in den Mails entstehen aus **Satzschablonen** mit benannten Platzhaltern. Die Schablonen liegen in `shop/texte/de.v1.json` und sind versioniert.

- Jeder Platzhalter ist an genau ein Evidenzfeld gebunden, mit Formatierer (Punkte, Prozent, Datum).
- Fehlt das Feld, entfällt der **ganze Satz**. Es wird nie „0“ oder „—“ in einen Behauptungssatz eingesetzt.
- Zitierte Texte (Hypothese, Mechanismus, Review-Kopfzeile) stehen immer in Anführungszeichen mit Quelle und Datum und werden nicht umformuliert.
- Beispielschablone:

  ```
  "vorwaerts_satz": "Auf {vorwaerts.monate} Monaten nach dem Einfrieren (Vorwärtstest, simuliert): {vorwaerts.punkte} Pkt bei n = {vorwaerts.n} ({vorwaerts.einstufung})."
  ```

### 4.2 Verbotene Formulierungen

Ein Filter läuft über **jede** erzeugte Textdatei und jede Schaufensterseite. Die Liste steht in `shop/texte/verboten.v1.json`; Groß- und Kleinschreibung spielen keine Rolle. Die Stammformen sind als Regex hinterlegt.

| Gruppe | Beispiele (Stämme) |
|---|---|
| Superlative, Rang | beste, top, führend, stärkste, einzigartig, unschlagbar, Nr. 1, Gewinner, Champion, Hall of Fame |
| Gewinnversprechen | garantiert, sicher(er) Gewinn, passives Einkommen, Rendite von, verdienen Sie, profitabel (als Zusage), bewährt, funktioniert, schlägt den Markt |
| Druck | nur heute, nur noch, limitiert, jetzt zugreifen, verpassen, bald teurer, exklusiv |
| Anlageberatung | empfehlen (Kauf/Verkauf), Signal (als Handlungsaufforderung), sollten Sie kaufen/verkaufen, Kursziel |
| Einheiten | €, EUR, Euro, Gewinn in Euro, Konto, Echtgeld (außer in festen Hinweissätzen) |

Treffer bedeuten: Die Erzeugung schlägt fehl, und das Paket wird nicht aufgenommen (Regel A15). Feste Hinweissätze des Betreibers stehen auf einer Ausnahmeliste mit SHA-256.

### 4.3 Pflichtangaben je Zahl

Jede Ergebniszahl erscheint als Einheit aus:
1. Wert mit Einheit (Pkt, %, Anzahl)
2. **n**
3. **Einstufung** (explorativ / vorläufig / belastbarer, Regeln aus `public_evidence.py`)
4. **Evidenzklasse** (in-sample / vorwärts / live-DEMO)
5. **Zeitraum**
6. beim PF: 90-%-Bereich, sofern vorhanden

Die Umsetzung ist eine Komponente `<Kennzahl wert n einstufung klasse zeitraum bereich?>`. Ohne n oder Klasse rendert sie nicht und schreibt einen Fehler in das Erzeugungsprotokoll. Bei n < 20 wird der PF nicht als Leistung gezeigt (`pf_als_leistung`).

### 4.4 Schwächenliste (mechanisch)

| Regel | Bedingung | Satz |
|---|---|---|
| W1 | `single_month_dependent` | „Ergebnis hängt am Monat {best_month}; ohne ihn {pnl_ex_best_month} Pkt.“ |
| W2 | `power_ok = false` | „Stichprobe zu klein, um einen Vorteil von {mde_pts} Pkt je Trade von null zu trennen.“ |
| W3 | `pf_slip_10 ≤ 1` | „Bei +1 Pkt zusätzlicher Slippage PF {pf_slip_10}.“ |
| W4 | `top3_share ≥ 0,5` (Vorschlag) | „{top3_share} % des Ergebnisses stammen aus den drei größten Trades.“ |
| W5 | Obduktion ≠ `lebt` | „Obduktion: {ursache_klartext}.“ |
| W6 | nur in-sample | „Bisher nur auf der Bauhistorie gemessen; nicht auf neuen Daten.“ |
| W7 | Nikkei-Transfer PF < 1 | „Auf den Nikkei übertragen: PF {pf} bei n = {n}.“ |
| W8 | Regime-Split: ein Regime mit PF < 1 und n ≥ 20 | „Im Regime {name} PF {pf} (n = {n}).“ |

---

## 5. Kaufablauf Ende zu Ende

```
Paketseite ──Knopf──▶ Digistore24-Bestellformular (Produkt „Strategie-Paket“ oder „Familien-Paket“,
                      Paket-ID als Durchreich-Parameter [DS24 prüfen])
     │
     ▼ Zahlung bei Digistore24 (inkl. PayPal)
Digistore24 ──IPN (POST)──▶ shop-dienst /shop/ipn
     1 Signatur prüfen ──falsch──▶ 403, Protokoll, Alarm ab 3/h
     2 Ereignis dauerhaft speichern (Rohdaten minimiert + Hash)  ──▶ Antwort an DS24 [DS24 prüfen: „OK“]
     3 Idempotenz: (order_id, event, transaktions-id) schon verarbeitet? ──ja──▶ fertig
     4 Fachprüfung: Produkt-ID bekannt · Währung EUR · Betrag passt zur Preiskonfiguration zum Bestellzeitpunkt
                    · Paket-ID existiert, Typ passt zum Produkt, war zum Bestellzeitpunkt im Angebot
          └─ nicht bestanden ──▶ Bestellung „prüfen“, keine Zustellung, Betreiber-Meldung (Rückerstattung über DS24)
     5 Bestellbuch: Bestellung „bezahlt“
     6 Lizenz erzeugen (zufällige Kennung), Paket personalisieren, ZIP ablegen
     7 Download-Token erzeugen (nur Hash speichern)
     8 Mail über Transaktionsdienst: Regelwerk-PDF als Anhang + Download-Link (befristet)
     9 Zustellprotokoll; Zustellfehler ──▶ Wiederholung, danach Betreiber-Meldung
Käufer ──▶ Danke-Seite (generisch, ohne personenbezogene Daten) mit Hinweis Spam-Ordner + Nachlieferung
Käufer ──▶ /shop/nachlieferung (E-Mail + Bestellnummer) ──▶ neuer Link an die **gespeicherte** Adresse
```

Einzelheiten:

1. **Produkte bei Digistore24:** Zwei Produkte, „Strategie-Paket“ und „Familien-Paket“. Die Paket-ID reist als Durchreich-Parameter (Name und Signaturabdeckung **[DS24 prüfen]**). So muss die KI keine Produkte bei Digistore24 anlegen.
   - Die Käuferin kann den Parameter in der URL verändern. Deshalb prüft Schritt 4 immer, ob das Paket existierte, im Angebot war und zum Produkt passt.
   - Alternative, falls Durchreich-Parameter nicht signiert sind: je Paket ein eigenes DS24-Produkt, angelegt über die DS24-API **[DS24 prüfen]** oder vom Betreiber. Entscheidung **E3**.
2. **Preise:** `shop/config/shop.v1.json` führt `preise.strategie` und `preise.familie` mit `betrag_cent`, `waehrung`, `ds24_produkt_id` und `gueltig_ab`.
   - Ein Prüfjob vergleicht täglich die Konfiguration mit den bei DS24 hinterlegten Preisen (per API **[DS24 prüfen]**) und meldet Abweichungen.
   - Welches IPN-Feld den Bruttobetrag trägt und ob er je Land variiert, wird im Testmodus festgestellt **[DS24 prüfen]**. Bis dahin prüft Schritt 4 Produkt-ID und Währung und protokolliert den Betrag nur.
3. **Antwort an DS24:** Die Antwort kommt erst, wenn das Ereignis dauerhaft gespeichert ist (SQLite-Transaktion abgeschlossen). Schritte 3–9 laufen danach in einem Arbeitsprozess, damit langsame Mail oder PDF-Erzeugung keine Zeitüberschreitung bei DS24 auslöst. Ob und wie DS24 bei fehlender Bestätigung wiederholt: **[DS24 prüfen]**.
4. **Personalisierung:**
   - Die Lizenzkennung steht in `manifest.json`, im Fuß jeder PDF-Seite, im Kommentarkopf des Referenzcodes und im Dateinamen.
   - Kein Kopierschutz, nur Zuordenbarkeit.
   - Die Kennung enthält keine personenbezogenen Daten.
5. **Mail:**
   - Anhang: `regelwerk.pdf` (klein, deshalb Anhang).
   - Link: das vollständige Paket.
   - Betreff und Text aus Schablonen (4.1).
   - Absenderdomain mit SPF, DKIM und DMARC.
6. **Link:**
   - Gültigkeit und Höchstzahl der Abrufe sind Konfiguration; Vorschlag 14 Tage, 5 Abrufe.
   - Nach Ablauf gibt es über die Nachlieferung einen neuen Link, solange die Lizenz nicht gesperrt ist.
7. **Nachlieferung:**
   - Das Formular antwortet **immer gleich** („Wenn eine passende Bestellung existiert, geht in den nächsten Minuten eine Mail an die bei der Bestellung verwendete Adresse.“). So lässt sich nicht ausprobieren, welche Adressen Kunden sind.
   - Der Link geht nur an die gespeicherte Adresse.
   - Begrenzung: 3 Anfragen je Bestellung und Tag (Vorschlag).
8. **Danke-Seite:** Eine eigene, generische Seite `/algostrategien/forschungspakete/danke`.
   - Falls DS24 signierte Parameter an die Danke-Seite übergibt **[DS24 prüfen]**, kann sie nach Signaturprüfung zusätzlich einen Sofort-Download anzeigen. Das hilft gegen Spam-Filter.
   - Sonst zeigt sie nur Hinweise.

---

## 6. Sonderfälle

| Nr | Fall | Erkennung | Verhalten |
|---|---|---|---|
| 6.1 | **Rückerstattung** | IPN-Ereignis Rückerstattung **[DS24 prüfen: Name]** | Bestellung „erstattet“, Lizenz gesperrt, alle Tokens ungültig, keine Aktualisierungs-Mails mehr. Keine Mail an den Käufer (die sendet DS24). Datensatz bleibt für die Aufbewahrung (7.3). |
| 6.2 | **Rückbuchung** (Chargeback) | IPN-Ereignis Rückbuchung **[DS24 prüfen]** | wie 6.1, Status „rückgebucht“, Betreiber-Meldung |
| 6.3 | **Doppelte Meldung** | gleicher Idempotenzschlüssel | nichts tun, gleiche Antwort wie beim ersten Mal |
| 6.4 | **Doppelte Zahlung** | zweite **eigene** Bestellung (andere Bestellnummer) mit gleicher Käufer-Kennung (7.2) und gleichem Paket innerhalb von 7 Tagen (Vorschlag) | normal zustellen (es ist eine gültige Bestellung), Bestellung markieren, Betreiber-Meldung mit Vorschlag Rückerstattung über DS24. Keine automatische Rückerstattung, da es keine eigene Zahlungsabwicklung gibt. |
| 6.5 | **Paket gescheitert** | Statuswechsel nach 3.4 | Aus dem Angebot (3.5). Alle Käufer mit ungesperrter Lizenz erhalten automatisch eine Mail „Befund zu Ihrem Forschungspaket“ mit aktualisiertem Evidenzblatt als Anhang und Kurzbefund aus Schablone. Keine Wertung, kein Angebot. |
| 6.6 | **Paket aktualisiert** | neue Paketversion: neue Vorwärtsmonate, neue DEMO-Trades, Textkorrektur, neues Familienmitglied innerhalb der Frist E2 | Käufer erhalten eine Mail „Aktualisierung verfügbar“ mit Änderungsliste (mechanisch aus dem Vergleich der Manifeste) und neuem Link. Höchstens eine Mail je Paket und Woche (Vorschlag); Zusammenfassung weiterer Änderungen. Codeänderungen sind **keine** Aktualisierung (A4). |
| 6.7 | **Zustellung fehlgeschlagen** | Rückmeldung des Mail-Dienstes (Bounce, Beschwerde) per Webhook; oder kein Download nach 72 h (Vorschlag) | Bounce: bis zu 3 Wiederholungen mit Abstand, danach Betreiber-Meldung. Kein Abruf nach 72 h: eine Erinnerung mit Hinweis auf den Spam-Ordner. Beschwerde (Spam-Markierung): keine weiteren Mails außer auf Anfrage über die Nachlieferung. |
| 6.8 | **Shop ausgeschaltet** | Hauptschalter `shop_aktiv = false` | Kaufknöpfe verschwinden, Schaufenster bleibt. IPNs für **vorher begonnene** Bestellungen werden weiter verarbeitet und zugestellt. Downloads, Nachlieferung, Befund- und Aktualisierungs-Mails laufen weiter. Eingehende Bestellungen nach dem Abschalten (z. B. aus gespeicherten Links) werden zugestellt und gemeldet. |
| 6.9 | **Paket-ID ungültig oder nicht im Angebot** | Schritt 4 | keine Zustellung, Status „prüfen“, Betreiber-Meldung; Rückerstattung über DS24 |
| 6.10 | **Paketerzeugung scheitert** | Generatorfehler oder Bereinigungsfilter-Treffer beim Personalisieren | keine Zustellung eines unvollständigen Pakets; drei Wiederholungen, dann Betreiber-Meldung und automatische Mail an den Käufer „Zustellung verzögert“ |

---

## 7. Datenmodell, Datensparsamkeit, Aufbewahrung

### 7.1 Speicher

Speicherorte:
- **SQLite-Datei** `shop/data/shop.sqlite`: nur der Shop-Dienst liest und schreibt; WAL-Modus; tägliche verschlüsselte Sicherung.
- **Paketdateien:** `shop/pakete/` (Vorlagen je Version) und `shop/auslieferung/` (personalisierte ZIPs). Beide liegen außerhalb jedes Web-Wurzelverzeichnisses.

Tabellen:

| Tabelle | Felder (Auswahl) | Hinweis |
|---|---|---|
| `paket` | `paket_id`, `typ` (strategie/familie), `traeger_id`, `version`, `status_shop`, `im_angebot`, `grund`, `regel_version`, `regel_sha256`, `vorlage_sha256`, `erzeugt_am`, `zurueckgezogen_am` | eine Zeile je Paketversion |
| `paket_mitglied` | `paket_id`, `version`, `fund_slug`, `rolle` (traeger/baustein/variante), `status_shop` | Familien |
| `ipn_ereignis` | `id`, `empfangen_am`, `event`, `order_id`, `transaktions_id`, `idempotenz_schluessel` (eindeutig), `signatur_ok`, `nutzdaten_sha256`, `nutzdaten_minimiert` (JSON ohne Namen und Adresse), `verarbeitet_am`, `ergebnis` | unveränderlich, nur anhängen |
| `bestellung` | `order_id` (DS24), `produkt`, `paket_id`, `paket_version`, `betrag_cent`, `waehrung`, `status` (bezahlt/prüfen/erstattet/rückgebucht), `kaeufer_id`, `angelegt_am`, `geaendert_am` | – |
| `kaeufer` | `kaeufer_id` (zufällig), `email_verschluesselt`, `email_suchschluessel` (HMAC-SHA-256 mit Geheimnis), `angelegt_am`, `loeschen_ab` | keine Namen, keine Anschrift |
| `lizenz` | `lizenz_id` (128 Bit zufällig, Base32), `order_id`, `paket_id`, `version`, `gesperrt`, `gesperrt_grund` | – |
| `download_token` | `token_hash` (SHA-256), `lizenz_id`, `gueltig_bis`, `max_abrufe`, `abrufe`, `angelegt_am` | Token im Klartext nie gespeichert |
| `zustellung` | `id`, `lizenz_id`, `art` (kauf/nachlieferung/aktualisierung/befund/erinnerung), `dienst_nachricht_id`, `status` (angenommen/zugestellt/bounce/beschwerde), `versuche`, `letzter_fehler`, `zeit` | – |
| `abruf` | `token_hash`, `zeit`, `ergebnis` | keine IP-Adresse gespeichert (Vorschlag); Missbrauchserkennung über Zähler |
| `betreiber_meldung` | `id`, `art`, `bezug`, `text`, `erledigt` | – |
| `konfig_protokoll` | `zeit`, `datei`, `sha256_alt`, `sha256_neu` | Hauptschalter, Preise, Texte |

### 7.2 Käufer-E-Mail datensparsam

- **Welche Daten:** Gespeichert wird nur die E-Mail-Adresse. Sie ist für Zustellung, Nachlieferung, Befund- und Aktualisierungs-Mails nötig und damit Vertragserfüllung. Namen, Anschrift, Land und Zahlungsart werden verworfen, bevor irgendetwas gespeichert wird.
- **Verschlüsselung:**
  - Die Adresse liegt verschlüsselt vor, mit einem Schlüssel aus dem Geheimnisspeicher (8.1).
  - Gesucht wird über einen HMAC-Suchschlüssel, sodass für die Doppelzahlungs-Erkennung und die Nachlieferung kein Klartextindex nötig ist.
- **Protokolle:** Logs enthalten nur `kaeufer_id` und `order_id`, nie die Adresse.
- **Mail-Dienst:** Er ist Auftragsverarbeiter. Vertrag zur Auftragsverarbeitung, Serverstandort EU (Auswahl 9.3).

### 7.3 Aufbewahrung (Vorschläge, rechtlich zu bestätigen)

| Datum | Frist | Danach |
|---|---|---|
| E-Mail-Adresse | bis Ende der Aktualisierungsfrist des Pakets (E2) + 3 Monate, mindestens bis zum Ende der Widerrufs- und Rückbuchungsfrist | Feld `email_verschluesselt` gelöscht; `kaeufer_id` bleibt |
| Bestellung, Lizenz, IPN-Ereignis (minimiert) | handels- und steuerrechtliche Frist (Rechts-Checkliste, Frage S-6) | gelöscht |
| Download-Tokens | 30 Tage nach Ablauf | gelöscht |
| Zustellprotokoll | 12 Monate | gelöscht |
| personalisierte ZIPs | bis Ablauf des letzten Tokens + 7 Tage; bei Bedarf deterministisch neu erzeugbar | gelöscht |
| Anwendungsprotokolle | 30 Tage | gelöscht |

Ein täglicher Löschlauf setzt diese Fristen um und protokolliert nur die Anzahl der gelöschten Einträge.

---

## 8. Sicherheit

### 8.1 Geheimnisse

- **Welche:** IPN-Passphrase, Mail-API-Schlüssel, Schlüssel für die E-Mail-Verschlüsselung, HMAC-Schlüssel, Signaturschlüssel für die Danke-Seite (falls genutzt).
- **Wo sie liegen:** Nur in einer Umgebungsdatei bzw. als systemd-Credentials, lesbar nur für den Dienstnutzer `shop`. Nie im Repository, nie im Next.js-Build, nie mit Präfix `NEXT_PUBLIC_`.
- **Wechsel:** Die IPN-Passphrase ist wechselbar. Während eines Wechsels werden zwei Passphrasen parallel geprüft (Konfiguration mit Ablaufdatum).
- **Kontrolle:** Ein Prüfskript im Repository sucht vor jedem Commit nach Schlüsselmustern und nach bekannten Werten aus der Umgebungsdatei, dort nur als Hash.

### 8.2 Webhook-Endpunkt gehärtet

- **Anfragen:**
  - Nur `POST`, nur `application/x-www-form-urlencoded` **[DS24 prüfen]**.
  - Körper höchstens 64 KB.
  - Zeitlimit 10 s.
  - Begrenzung auf 60 Anfragen/Minute.
- **Signatur:**
  - Geprüft nach der offiziellen Beispielfunktion `digistore_signature()` aus `sha_sign.php` (Verfahren siehe Kopf).
  - Vergleich in konstanter Zeit.
  - Die Implementierung wird mit Testnachrichten aus dem DS24-Testmodus als festen Testvektoren abgesichert.
- **Weitere Absicherung:**
  - Keine Weiterverarbeitung vor erfolgreicher Signaturprüfung.
  - Fehlermeldungen ohne Details nach außen.
  - Wiederholungen werden über den Idempotenzschlüssel abgefangen.
  - Die Signatur ist der Schutz. Eine IP-Allowlist nur zusätzlich, falls DS24 feste Absender-IPs dokumentiert **[DS24 prüfen]**.
- **Netz:** Der Dienst lauscht nur auf `127.0.0.1`. nginx leitet ausschließlich `/shop/ipn`, `/shop/download/` und `/shop/nachlieferung` weiter.

### 8.3 Download-Links

- Token: 32 Byte aus einem kryptografischen Zufallsgenerator, Base64url. Gespeichert wird nur der SHA-256.
- Der Link enthält weder Bestellnummer noch Lizenz noch E-Mail.
- Prüfung bei jedem Abruf: Token bekannt, nicht abgelaufen, Abrufe < Maximum, Lizenz nicht gesperrt.
- Auslieferung über nginx `X-Accel-Redirect` auf einen `internal`-Ort. So liest der Dienst die Datei nicht selbst, und der Speicherpfad ist nie in einer Antwort sichtbar.
- Antwort-Header: `Cache-Control: no-store`, `Referrer-Policy: no-referrer`, `Content-Disposition: attachment`.

### 8.4 Pakete ohne interne Daten (Bereinigungsfilter)

Vor dem Aufnehmen **und** vor jedem Ausliefern prüft der Filter jede Datei im Paket (Text und PDF-Textebene). Treffer führen zum Abbruch (A15, 6.10). Geprüft werden:

- absolute Pfade (`/opt/`, `/home/`, `/root/`, `/var/`, `C:\`)
- Hostnamen und IPs (`127.0.0.1`, `localhost`, interne Domains)
- Portangaben
- Felder bzw. Wörter `pnl_eur`, `account`, `konto`, `epic`, `api_key`, `password`, `token`, `secret`
- Euro-Zeichen und „EUR“ außerhalb der Ausnahmeliste
- Muster für Kontonummern und Broker-Kontokennungen (Liste vom Betreiber)
- Tick- und Kerzenstrukturen: Zeitstempel-Preis-Paare, Spalten `open/high/low/close/bid/ask`, mehr als 20 aufeinanderfolgende Preiszahlen
- Manifestfelder `strategy_source_path`, `engine_fingerprint`, `data_tick_count`

Der Referenzcode wird zusätzlich auf Importe außer dem Backtest-Interface und Kommentare mit Pfaden geprüft. Treffer werden nicht automatisch entfernt, weil der Code byte-gleich bleiben muss. Das Paket wird dann nicht aufgenommen.

### 8.5 Trennung vom Handelssystem

- **Konto:** Der Shop-Dienst läuft unter eigenem Nutzer.
- **Leserechte:** Er liest nur die Evidenzdateien (lesend) und `research_strategies/*.py` (lesend).
- **Kein Zugriff:** Er hat keinen Zugriff auf Kontodaten, `trades.db` oder die Flask-Trading-App. Er teilt keine Sitzung und kein Cookie mit der Trading-GUI.
- **Hub:** Das Portal (Next.js) liest nur den öffentlichen Katalog `shop/katalog/katalog.v1.json` und hat keine Schreibrechte auf `shop/`.

---

## 9. Architektur

### 9.1 Entscheidung: separater kleiner Dienst für Webhook, Zustellung und Download

| Kriterium | Next.js-Route im Hub | separater Dienst `shop-dienst` |
|---|---|---|
| Schreibende Last, Geheimnisse | Der Hub ist heute rein lesend und öffentlich. Eine Schreibroute und Geheimnisse vergrößern die Angriffsfläche des ganzen Portals. | isoliert, eigener Nutzer, eigene Rechte (8.5) |
| Verfügbarkeit | Hub-Deploys über `HUB_BUILD_DIR` und Neustarts unterbrechen den Webhook | unabhängig deploybar |
| Hintergrundarbeit (PDF, ZIP, Mail, Wiederholung) | in Next.js unhandlich; eine Route ist anfragegebunden | Arbeitsprozess und Zeitsteuerung natürlich |
| Wiederverwendung | müsste Parser der Evidenzdateien in TypeScript neu bauen; heute schon dreifach vorhanden (`review-hub-code.md` M6) | der Paketgenerator nutzt die Python-Parser der Plattform (`results_digest.py`, `public_evidence.py`, Obduktion, Hypothesenregister) |
| Aufwand | geringer Startaufwand | ein weiterer Prozess (systemd), SQLite |

**Empfehlung:** separater Dienst in Python. Die Evidenzlogik (Einstufung, Obduktion, Register) liegt bereits in Python, und der Generator muss sie **identisch** anwenden. Es ist ausdrücklich **nicht** Teil der Flask-Trading-App (8.5).

Das Schaufenster bleibt im Next.js-Hub: Es ist lesend, nutzt ISR und liest den vom Generator geschriebenen Katalog.

### 9.2 Bausteine

```
┌──────────────────────────── Server ─────────────────────────────┐
│ Plattform (bestehend)          shop-generator (Zeitplan, Python)│
│  results/, reviews/, …  ──lesen──▶ Aufnahmeregel ─▶ Pakete       │
│  research_strategies/*.py        Texte ─▶ PDF ─▶ Bereinigung     │
│                                  schreibt: shop/katalog/, pakete/│
│                                                                  │
│ Hub (Next.js, bestehend) ──liest──▶ shop/katalog/katalog.v1.json │
│  /algostrategien/forschungspakete[/…]                            │
│                                                                  │
│ shop-dienst (Python, 127.0.0.1)  ◀── nginx /shop/{ipn,download,  │
│  IPN · Bestellbuch (SQLite) · Lizenz · Personalisierung          │
│  Arbeitsprozess: Mail, Wiederholung, Befund-/Aktualisierungsmails│
│       │                                     ▲                    │
│       ▼                                     │ Bounce-Webhook     │
│  Transaktions-Mail-Dienst (EU) ─────────────┘                    │
└──────────────────────────────────────────────────────────────────┘
Digistore24 ──IPN──▶ nginx ──▶ shop-dienst
```

- **Katalog:** Der Generator schreibt ihn atomar (temporäre Datei, dann umbenennen) mit Formatfeld und SHA-256.
- **Lesen im Hub:** Der Hub liest den Katalog über einen Leser mit Zustandsrückgabe (ok / fehlt / defekt, siehe `review-hub-code.md` D2).
- **Fehlt oder ist der Katalog defekt:** Das Schaufenster zeigt „Katalog derzeit nicht lesbar“ und **keine** Kaufknöpfe.
- **PDF:** Die Erzeugung läuft aus HTML-Schablonen mit einem Werkzeug, das auf dem Server ohnehin verfügbar ist, z. B. Headless-Chromium. Die Ausgabe ist deterministisch: feste Schriften, keine Zeitstempel im Inhalt außer dem Datenstand.

### 9.3 Transaktions-Mail-Dienst

Anforderungen:
- EU-Verarbeitung und Vertrag zur Auftragsverarbeitung
- API mit Anhängen bis mindestens 5 MB
- Webhooks für Zustellung, Bounce und Beschwerde
- eigene Absenderdomain mit DKIM

Die Auswahl trifft der Betreiber (**E4**); die Spezifikation legt sich auf keinen Anbieter fest. Die Anbindung erfolgt über eine schmale Schnittstelle `senden(an, betreff, text, html, anhaenge) → nachricht_id`, damit ein Wechsel möglich bleibt.

### 9.4 Konfiguration (Betreiber)

`shop/config/shop.v1.json`:

```json
{
  "shop_aktiv": false,
  "preise": {
    "strategie": { "betrag_cent": 499, "waehrung": "EUR", "ds24_produkt_id": "<vom Betreiber>", "gueltig_ab": "<ISO>" },
    "familie":   { "betrag_cent": 799, "waehrung": "EUR", "ds24_produkt_id": "<vom Betreiber>", "gueltig_ab": "<ISO>" }
  },
  "hinweistext_datei": "hinweis.v1.md",
  "hinweistext_sha256": "<64 hex>",
  "lizenzbedingungen_datei": "lizenz.v1.md",
  "download": { "gueltig_tage": 14, "max_abrufe": 5 },
  "aktualisierung": { "paket_monate": 12, "familie_neue_mitglieder_monate": 12 },
  "aufnahmeregel": "aufnahmeregel.v1.json"
}
```

- **Startwerte:** Die Preise 4,99 € und 7,99 € sind Betreibervorgaben. Download- und Aktualisierungswerte sind Vorschläge.
- **Änderungen:** Jede Änderung schreibt `konfig_protokoll`.
- **Hauptschalter:** Er wird über ein kleines Kommandozeilenwerkzeug gesetzt (`shopctl an|aus`), das Datei und Protokoll gemeinsam ändert. Der Hub liest den Schalter über den Katalog.

---

## 10. Umsetzungsphasen mit Prüfweg

### Voraussetzungen

- **V1 (Einheiten):** Die Einheit von `pnl_eur` in `forward_runs.jsonl` ist geklärt, und für DEMO und Vorwärtstest steht ein Punkte-Feld bereit (heute nur `pnl_eur`, `lib/live-system.ts:35`, `build-dossier.ts:224`; vgl. `review-hub-code.md` H1). Ohne V1 zeigen Schaufenster und Evidenzblatt für diese Klassen **keine** Ergebniswerte, nur n.
- **V2 (robustes Lesen):** JSONL wird zeilenweise robust gelesen (`review-hub-code.md` H2), und es gibt eine einheitliche Urteilsklassifikation (M5). Der Shop darf nicht auf den heutigen Einzellesern aufbauen.
- **V3 (Konfiguration):** O1 und O2 sind beantwortet, d. h. es ist festgelegt, wie die vorregistrierte Konfiguration erkannt wird und welche Werte der Vorwärtsentscheid annimmt.
- **V4 (Recht):** Die Rechts-Checkliste ist beantwortet (`rechts-checkliste-shop.md`), mindestens die Fragen, die mit „vor Phase 4“ markiert sind.

### Phase 1: Paketgenerator (ohne Verkauf, ohne Öffentlichkeit)

- **Umfang:**
  - Aufnahmeregel mit Bestätigung
  - Evidenzleser mit Zustand
  - Statusabbildung
  - Schablonentexte
  - Filter für verbotene Formulierungen
  - Regelwerk-Prüfung (2.4)
  - PDF, Bereinigungsfilter
  - Vorlagen je Paketversion
  - Katalog
  - Aufnahmeprotokoll
- **Prüfweg:**
  1. **Einheitstests mit synthetischen Beispieldateien.** Sie sind als solche gekennzeichnet und enthalten keine echten Werte. Abgedeckt: jedes Kriterium erfüllt, nicht erfüllt und Feld fehlt; unbekannte Statuswerte; defekte JSONL-Zeilen.
  2. **Determinismus:** Zwei Läufe auf demselben Stand ergeben identische SHA-256 aller Vorlagen und des Katalogs.
  3. **Negativtests Bereinigung:** Eine Testdatei mit `/opt/…`, `pnl_eur`, „€“ und einer Kursreihe wird zuverlässig abgewiesen.
  4. **Negativtests Texte:** Jede Gruppe aus 4.2 wird erkannt.
  5. **Lauf gegen die echten Evidenzdateien auf dem Server.** Der Betreiber liest das Aufnahmeprotokoll und drei erzeugte Pakete vollständig, darunter mindestens einen Baustein und eine Familie mit gescheitertem Mitglied, und bestätigt die Regel (3.1).
  6. **Vollständigkeit:** Zahl der Katalogeinträge = Zahl der Ergebnisdateien. Kein Fund fehlt, auch kein gescheiterter.

### Phase 2: Schaufenster (öffentlich, Shop aus)

- **Umfang:**
  - Übersichtsseite, Paketseite, Danke-Seite
  - Leser für den Katalog mit Zustandsrückgabe
  - Kaufbereich nur bei `shop_aktiv = true`; in dieser Phase immer aus
- **Prüfweg:**
  1. **Zählung und Filter:** Die Zählzeile stimmt mit dem Katalog überein. Filter „gescheitert“ zeigt alle gescheiterten Funde.
  2. **Automatische Seitenprüfung** auf verbotene Formulierungen, „€“/„EUR“, Parameterwerte angebotener Funde und Zahlen ohne n/Einstufung/Klasse (über `data-`-Attribute der Kennzahl-Komponente).
  3. **Unbekannte Paket-ID** liefert HTTP 404, nicht 200.
  4. **Darstellung:** Barrierefreiheit (WCAG 2.2 AA) und 390 px Breite nach dem Muster von `review-barrierefreiheit.md`. Status nie nur über Farbe.
  5. **Fehlerfall:** Katalog fehlt oder ist defekt → Hinweis, keine Kaufknöpfe.

### Phase 3: Kaufanbindung im DS24-Testmodus

- **Umfang:**
  - Shop-Dienst: IPN, Bestellbuch, Lizenz, Personalisierung, Tokens, Download über nginx, Nachlieferung
  - Anbindung Mail-Dienst mit Bounce-Webhook
  - Sonderfälle 6.1–6.10
  - Löschlauf
- **Zuerst:** alle **[DS24 prüfen]**-Punkte mit Doku und Testmodus festlegen und hier nachtragen (Feldnamen, Ereignisnamen, Antwortformat, Wiederholungen, Durchreich-Parameter, Danke-Seiten-Signatur).
- **Prüfweg:**
  1. **Signatur-Testvektoren aus dem DS24-Testmodus:**
     - gültige Signatur → angenommen
     - ein Zeichen geändert → abgewiesen
     - Parameter entfernt → abgewiesen
     - falsche Passphrase → abgewiesen
  2. **Idempotenz:** Dieselbe IPN zehnmal nacheinander und parallel ergibt genau eine Bestellung, eine Lizenz und eine Mail.
  3. **Fachprüfung:** Falsches Produkt, falsche Währung, manipulierte Paket-ID und ein Paket außerhalb des Angebots führen jeweils zu „prüfen“ ohne Zustellung.
  4. **Ende-zu-Ende im Testmodus:** Mail kommt in einem Test-Postfach an, mit Anhang, Link und Download. Die Lizenzkennung steht in PDF-Fuß, Code-Kopf und Manifest.
  5. **Token:** Ein abgelaufener Token und das Überschreiten der Abrufzahl ergeben 410 bzw. 403. Die Nachlieferung liefert einen neuen Link nur an die gespeicherte Adresse, und die Antwort ist unabhängig davon, ob die Bestellung existiert.
  6. **Sperre:** Rückerstattung und Rückbuchung sperren alle Tokens sofort.
  7. **Befund-Mail:** Ein Statuswechsel auf „gescheitert“ im Testbestand erzeugt je Lizenz genau eine Befund-Mail.
  8. **Hauptschalter aus:** Kaufknöpfe weg, laufende Zustellung weiter.
  9. **Sicherheit:**
     - Kein Geheimnis im Repository (Prüfskript).
     - Der Dienst ist von außen nur über die drei nginx-Pfade erreichbar.
     - Das ZIP ist nicht direkt per URL abrufbar.
  10. **Löschlauf:** Er löscht nach verkürzten Testfristen korrekt.

### Phase 4: Pilotbetrieb

- **Umfang:**
  - Shop an, mit einem Strategie-Paket und einem Familien-Paket, sofern die Aufnahmeregel welche aufnimmt. Sonst bleibt der Shop aus, und das ist ein gültiges Ergebnis.
  - Der Betreiber kauft beide selbst mit echter Zahlung und erstattet eines davon über DS24.
- **Prüfweg:**
  - Bestellbuch, Zustellprotokoll und DS24-Übersicht stimmen überein.
  - Die Erstattung sperrt die Lizenz.
  - Der Preis-Prüfjob meldet keine Abweichung.
  - Zwei Wochen ohne Betreiber-Meldungen, die nicht erklärt sind.

### Phase 5 (später): Aktualisierungen im Betrieb und Pine-Fassung

- Aktualisierungs-Mails (6.6) nach vier Wochen stabilem Betrieb.
- Pine-Fassung nach 2.5, erst nach Klärung der Kerzendaten-Frage.

---

## 11. Entscheidungen und offene Punkte

| ID | Art | Frage | Vorschlag |
|---|---|---|---|
| E1 | Betreiber | Bleiben Parameterwerte angebotener Funde auf der öffentlichen Dossierseite sichtbar? | ja (Forschung bleibt vollständig öffentlich) |
| E2 | Betreiber | Wie lange sind Aktualisierungen und neue Familienmitglieder im Kaufpreis enthalten? | 12 Monate |
| E3 | Betreiber | Zwei DS24-Produkte mit Durchreich-Parameter oder ein DS24-Produkt je Paket? | zwei Produkte, falls der Parameter signiert ist |
| E4 | Betreiber | Welcher Transaktions-Mail-Dienst? | Anforderungen 9.3 |
| E5 | Betreiber | Schwellen A6, A8, A13, W4 und Download-Werte bestätigen oder ändern | Tabelle 3.2 |
| E6 | Betreiber | Gescheiterte Funde: Wird auch das vollständige Regelwerk kostenlos, oder nur der Befund? | offen |
| O1 | Plattform | Wie ist die vorregistrierte Konfiguration in der Ergebnisdatei gekennzeichnet? | – |
| O2 | Plattform | Welche Werte nehmen `decision.status` und `decision.effect` an? | – |
| O3 | Plattform | Welche Obduktions- und Registerdateien (Pfad, Format) sind maßgeblich? Das Portal liest sie heute nicht. | – |
| O4 | DS24 | alle **[DS24 prüfen]**-Punkte | vor Phase 3 |
