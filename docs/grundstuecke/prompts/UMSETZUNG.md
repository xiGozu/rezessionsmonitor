# Prompt: Umsetzung des Grundstücks-Monitors (für Claude Code), Version 1.1

> **So wird er verwendet:**
> 1. Phase 0 der Spezifikation erledigen, vor allem echte Alarm-Mails als `.eml` unter `grundstuecke/tests/fixtures/mails/<portal>/` ablegen.
> 2. Die Platzhalter in `[eckigen Klammern]` mit den Entscheidungen aus Kapitel 14 füllen.
> 3. Den Block unten vollständig in eine neue Claude-Code-Sitzung im Ziel-Repository kopieren.
>
> Der Prompt baut zuerst das **MVP (Phase 1)** und danach den **Ausbau (Phase 2)**. Phase 3 folgt in einem eigenen Auftrag. Nach **jedem** Bauschritt kommt eine verpflichtende Gegenprüfung.

---

```text
# Auftrag

Baue den „Grundstücks-Monitor“: einen automatisierten Agenten, der mehrmals täglich
Grundstücke (mit oder ohne Haus) ab 1.000 m² Grundstücksfläche rund um Dresden,
Pirna, Meißen (Sachsen) und Schleiz, Neustadt an der Orla, Weimar, Gera (Thüringen)
sammelt, jedes Inserat unabhängig gegenprüft und alles übersichtlich auf einer
statischen Webseite unter [grundstuecke.warchhold.de] mit Direktlink zum Inserat
auflistet.

Verbindliche Grundlagen. Lies alle Dateien vollständig, bevor du beginnst:
- docs/grundstuecke/SPEZIFIKATION.md (v1.1)
- docs/grundstuecke/prompts/extraktion.md, gegenpruefung.md, discovery.md

Vorrang bei Widersprüchen:
1. diese Entscheidungen
2. SPEZIFIKATION.md (insbesondere Kapitel 6.3 Enums und Kapitel 8 Gegenprüfung)
3. Laufzeit-Prompts
4. der Rest dieses Prompts

Jeden gefundenen Widerspruch meldest du im Abschlussbericht.

Umfang dieses Auftrags: Spezifikation Kapitel 13, Phase 1 (MVP) und Phase 2 (Ausbau).
Phase 3 (Discovery, Bodenrichtwerte, Benachrichtigungen, Stichproben-Issue) baust du
NICHT. Lass dafür nur saubere Erweiterungspunkte.

# Entscheidungen (ausgefüllt vom Betreiber)

- Subdomain: [grundstuecke.warchhold.de]
- Zugriff: [öffentlich: GitHub Pages | geschützt: Cloudflare Pages + Access, privates Repo]
- Repository: [Unterordner grundstuecke/ in xigozu/rezessionsmonitor | eigenes Repo …]
- Modellprofil: [Ausgewogen]   Monatsbudget: [70] USD   lauf_max_usd: [3]   tag_max_inserate: [60]
- Radius: [20] km je Zentrum   Jena/Erfurt/Nachbarkreise im Radius: [ja]
- Land-/Forst-/Freizeitflächen: [nein]   Zwangsversteigerungen: [ja, gekennzeichnet]
- Stufe D (Portal-Scraping / Exposé-Abruf): [ausgeschlossen]
- Mail-Portale im MVP: [ImmoScout24, Immowelt]; echte .eml-Dateien liegen in
  grundstuecke/tests/fixtures/mails/: [ja | nein]
- Suchauftrag-Filter je Portal (so auf den Portalen eingerichtet):
  [immowelt: Grundstück+Haus, Grundstücksfläche ≥ 1000 m², Umkreis …]
  [immoscout24: …]
- Quellen Phase 2: zvg-portal.de (SN, TH) [ja], BVVG [ja], statische Quellen:
  [Liste von URLs der Gemeinde-Börsen/Sparkassen/Makler oder „noch keine“]

Fehlen echte .eml-Dateien, baue die Mail-Parser gegen selbst erstellte Beispiele,
kennzeichne sie im Code und in der README deutlich als „VORLÄUFIG – mit echten Mails
validieren“ und nenne das im Abschlussbericht als offenen Punkt.

# Technische Leitplanken

- Python 3.11, eigene gepinnte grundstuecke/requirements.txt, ruff, pytest, pytest-cov.
- Den bestehenden Rezessionsmonitor (app.py, update_data.py, data/,
  .github/workflows/update.yml, requirements.txt im Wurzelverzeichnis) NICHT ändern.
- Claude-API:
  - offizielles `anthropic`-SDK
  - Structured Outputs über `output_config.format` (nicht `output_format`)
  - Modell-IDs exakt wie in der Spezifikation, ohne Datums-Suffix
  - Bei claude-opus-5-5 / claude-haiku-5-5 den Parameter `thinking` weglassen und die
    Tiefe nur über `output_config.effort` steuern. Bei claude-sonnet-5-5 ebenso
    `thinking` weglassen.
  - Kein erzwungenes `tool_choice`.
  - Vor dem Lesen von Inhalten immer `stop_reason` auswerten (refusal, max_tokens,
    pause_turn).
  - Verifiziere jede SDK-Verwendung gegen die offizielle Doku bzw. das installierte
    SDK. Rate keine Methoden- oder Parameternamen. Prüfe, welche JSON-Schema-Features
    Structured Outputs unterstützt, und passe die Schemata gegebenenfalls an
    (Prüfung dann im Code).
- Prompts nach grundstuecke/prompts/ übernehmen, mit Versionskennung, zur Laufzeit
  laden. System-Prompts statisch (Prompt-Caching), variable Inhalte nur in der
  User-Nachricht.
- Alle Claude-Aufrufe laufen über eine gemeinsame Funktion, die vor dem Aufruf die
  Worst-Case-Budgetprüfung macht und danach usage (inkl. Cache-Tokens und
  Websuchen) ins Ledger schreibt. Kein Claude-Aufruf an budget.py vorbei.
- Inseratstexte sind nicht vertrauenswürdig: <inserat>-Tags, keine Tools bei
  Extraktion/Gegenprüfung, alle sicherheitsrelevanten Entscheidungen (Fläche,
  Gebiet, Links, Status) trifft der Code.
- Keine Inseratstexte, Originaltitel oder Fotos veröffentlichen. Keine
  personenbezogenen Daten speichern.
- Abruf: robots.txt, ≤ 1 Anfrage / 3 s je Domain, User-Agent mit KONTAKT_EMAIL.
  Kein Abruf von immobilienscout24.de, immowelt.de, kleinanzeigen.de außer dem
  Auflösen von Weiterleitungslinks aus Mails (nur Redirect folgen, keinen Inhalt
  auswerten).
- Secrets nur aus Umgebungsvariablen bzw. GitHub-Secrets. Niemals loggen.
- Webseite:
  - Jinja2 mit Autoescape, Vanilla-JS ausschließlich mit textContent
  - href nur https:// auf Allowlist-Domains, rel="noopener noreferrer"
  - CSP-Meta-Tag, keine Inline-Skripte
  - Leaflet vendored (feste Version), Karte erst nach Klick laden
  - noindex, mobil zuerst, Hell- und Dunkelmodus
- Daten auf Branch `grundstuecke-data` (im Workflow per `git worktree`). Commits
  dorthin, nie auf main.
- Workflow:
  - Cron `15 4-20/2 * * *` + workflow_dispatch (Inputs: nur_sammeln,
    discovery_erzwingen)
  - concurrency-Gruppe ohne Abbruch laufender Läufe, timeout-minutes
  - permissions minimal (contents, issues, ggf. pages, id-token)
  - acceptance.py als Gate vor dem Deployment

# Vorgehen

Arbeite die Bauschritte nacheinander ab. Nach JEDEM Bauschritt folgt die
„Gegenprüfung des Bauschritts“ (unten). Erst danach geht es weiter. Committe nach
jedem bestandenen Bauschritt.

Bauschritt 1 (Phase 1): Gerüst, Konfiguration, Geodaten
  - Struktur laut Spezifikation 5.3; config.yaml mit allen Parametern (Zentren,
    Radius, ausgeschlossene Kreise, Kriterien, Quellen mit liveness und
    Suchauftrag-Filter, Domain-Allowlist, Budget, Modellprofile, Effort).
  - geo.py: AGS-/Gemeindeverzeichnis, PLZ-Mittelpunkte, Haversine,
    Toleranzband, Kreiszuordnung. Ein Skript erzeugt data/kreise_im_gebiet.json
    aus den BKG-VG250-Grenzen. Quellen und Lizenzen in der README dokumentieren.
  - normalize.py inkl. NFKC, Zahlen-/Einheiten-/Preisformate, kanonische URLs.

Bauschritt 2 (Phase 1): Mail-Quelle
  - email_alerts.py: IMAP mit UID-Wasserzeichen aus state.json. Das Wasserzeichen
    wird erst nach erfolgreichem Daten-Commit fortgeschrieben. Je Portal ein
    Parser. Weiterleitungslinks werden zu kanonischen URLs aufgelöst, externe_id
    ist die Portal-ID. Weitergeleitete „Inserat teilen“-Mails (Anfangsbestand)
    werden unterstützt.

Bauschritt 3 (Phase 1): Vorfilter, Budget, Extraktion, Gegenprüfung
  - prefilter.py, pending.json, budget.py (Prognoseformel, Stufen
    Normal/Gedrosselt/Sparmodus, Worst-Case-Prüfung je Aufruf), extract.py,
    verify.py mit Ebene 1, 2 und 3 exakt nach Spezifikation 8.1–8.3 und
    gegenpruefung.md (inkl. belegt_durch_filter, Zahlenabgleich,
    Wohnflächen-Sperrbegriffe, erneute Ebene 1 nach Korrektur).
  - Eine Ablehnung durch den Extraktor ohne deterministische Bestätigung geht in
    die Gegenprüfung.

Bauschritt 4 (Phase 1): Speicher, Seite, Gate, Workflow
  - store.py (listings.json als einzige führende Datei, atomar, ID-Regel 6.1,
    Neubewertung abgelehnter Inserate bei geändertem Hash oder geänderter
    Config-Version).
  - render.py + einfache Kachelansicht mit allen Pflichtangaben aus 3.3 und
    XSS-Schutz. acceptance.py (A3, A4, A5, A12). Workflow.
  - Kostenmessung: Skript `measure_costs.py`, das N Inserate aus den Fixtures durch
    Extraktion und Gegenprüfung schickt und Tokens/Kosten je Schritt ausgibt.
    Ohne API-Key läuft es mit aufgezeichneten Antworten. Die echte Messung macht
    der Betreiber.

Bauschritt 5 (Phase 2): Weitere Quellen
  - zvg.py, bvvg.py, static_sites.py (konfigurierbare Selektoren je Quelle), jeweils
    mit liveness und Fixtures. ZVG-Daten: Aktenzeichen und Straße nur im
    geschützten Modus, Löschung nach dem Termin.

Bauschritt 6 (Phase 2): Dubletten, Lebenszyklus, Plausibilität, Overrides
  - dedup.py (nie innerhalb eines Portals, Konfliktregel, danach Ebene 1),
    lifecycle.py (Übergänge laut 3.4, 403/429 = unbekannt, fehlgeschlagene Quelle
    zählt nicht), Lauf-Plausibilität 8.5 mit Mindestanzahlen (bei Fehler nur
    Laufprotokoll + pending committen, Issue anlegen bzw. aktualisieren),
    overrides.yaml laut 8.6.

Bauschritt 7 (Phase 2): Webseite vollständig
  - Kennzahlen, Filter (Zustand in der URL), Sortierung, Tabelle, Karte, Reiter
    Aktiv / Zu prüfen (mit kopierbarer overrides-Zeile) / Offline-Archiv,
    Preisverlauf, Dubletten-Links, Prüfprotokoll, Fußzeile mit Modus und
    Monatskosten.
  - README für den Betreiber: Postfach + App-Passwort, Suchaufträge je Portal
    (exakte Filter) und Eintrag in config.yaml, DNS, Secrets, erster manueller
    Lauf, Kostenmessung, Overrides benutzen.

# Gegenprüfung des Bauschritts (nach JEDEM Bauschritt verpflichtend)

1. Tests: pytest vollständig grün. Neue Logik ist durch Tests abgedeckt, inklusive
   dieser Grenzfälle:
   - Flächenangaben: 999 / 1.000 m² · „ca. 0,1 ha“ · „10 a“
   - Wohnfläche 1.200 m² bei Grundstück 800 m² · Teilfläche aus größerem Flurstück
   - Vermarktung: Miete · Mietkauf · Verrentung · Gesuch · Bieterverfahren
   - Gebiet: Ort 21 km (Toleranzband) und 25 km entfernt · nicht geokodierbarer Ort
     · gleichnamige Orte („Neustadt“)
   - Preis: Preis auf Anfrage · Zwangsversteigerung mit Verkehrswert
   - Alarm-Mail ohne Grundstücksfläche, mit und ohne hinterlegten Portalfilter
   - Belegprüfung: erfundenes Belegzitat · Zitat „Wohnfläche ca. 1.200 m²“ als Beleg
     für grundstueck_m2
   - Prüfer meldet „bestaetigt“ trotz Feld „falsch“
   - Korrektur auf 950 m² · Dublette mit 975 m² aus zweitem Portal
   - Budget: Worst-Case-Prüfung bricht sauber ab · Sparmodus ohne LLM-Aufrufe
   - Lauf: Absturz nach dem Mail-Lesen (keine Mail verloren) · Plausibilitätsfehler
     (keine Statusänderung committet)
   - Sicherheit: XSS-Inserat · refusal / max_tokens
2. Lint/Format: `ruff check` und `ruff format --check` ohne Befunde.
3. Probelauf: `python -m agent.run --dry-run` gegen die Fixtures. Claude-Aufrufe
   kommen aus aufgezeichneten Antworten in tests/fixtures/llm/. Sieh dir die
   erzeugte listings.json und die gerenderte Seite inhaltlich an (Seite per
   Playwright bei 360 px und 1280 px Breite aufnehmen und anschauen), nicht nur den
   Exit-Code.
4. Abgleich mit der Spezifikation: Tabelle
   „Anforderung (Kapitel) | erfüllt? | Nachweis (Datei:Zeile oder Testname)“ für
   alle Abschnitte, die dieser Bauschritt berührt.
5. Adversariales Review des eigenen Diffs: Lies den Diff so, als wolltest du ihn
   ablehnen. Suche gezielt nach:
   - Wegen, auf denen ein Inserat < 1.000 m², ohne Flächennachweis, außerhalb des
     Gebiets oder mit ausgeschlossener Vermarktung in die Hauptliste gelangt
   - Claude-Aufrufen an budget.py vorbei, Schleifen ohne Kostenobergrenze
   - Datenverlust (Mails, pending) und nicht-atomaren Schreibvorgängen
   - None-, Zahlenformat- und Zeitzonen-Fehlern
   - Abstürzen bei refusal, leerer Antwort oder Schemafehler
   - Secrets in Logs, XSS-, Prompt-Injection-Lücken
   - Verstößen gegen die Leitplanken
6. Unabhängiger Reviewer: Wenn du Sub-Agenten starten kannst, lass einen
   separaten Reviewer-Agenten mit frischem Kontext NUR mit Spezifikation und Diff
   (ohne deine Begründungen) den Bauschritt prüfen. Bearbeite jeden Befund: beheben
   oder schriftlich begründen, warum er nicht zutrifft.
7. Befunde beheben und die Schritte 1–3 wiederholen, bis alles sauber ist. Melde
   ehrlich, was nicht erfüllt ist. Schönfärben ist ausdrücklich unerwünscht.

# Abschluss-Gegenprüfung

- Abnahmekriterien A1–A12 (Spezifikation Kapitel 12) einzeln bewerten: automatisch
  geprüft (Testname) / vom Betreiber zu prüfen (wie) / nicht erfüllt (Grund).
- Simulation in Tests (mit Nachweis):
  - (a) kaputte Quelle
  - (b) geändertes Mailformat, Pflichtfelder sind nicht lesbar
  - (c) Budget > 90 % und ≥ 100 %
  - (d) Absturz mitten im Lauf
  Zeige jeweils: letzte gute Seite bleibt online, keine Mail geht verloren, die
  Drosselung bzw. der Sparmodus greift.
- Kostenhochrechnung mit den Token-Zahlen aus den aufgezeichneten bzw. gemessenen
  Antworten für 25 Inserate/Tag, Vergleich mit Spezifikation 9.3.

# Abschlussbericht (letzte Ausgabe)

1. Was gebaut wurde (mit Dateiverweisen)
2. Tabelle Abnahmekriterien A1–A12 mit Status und Nachweis
3. Ergebnisse der Gegenprüfungen: gefundene und behobene Probleme, offene Punkte
4. Widersprüche in den Grundlagen und wie du sie aufgelöst hast
5. Schritt-für-Schritt-Liste für den Betreiber (Postfach, Suchaufträge, DNS,
   Secrets, Kostenmessung mit 20 echten Inseraten, erster Lauf)
6. Kosten pro Lauf und Monat (gemessen bzw. hochgerechnet)
7. Vorbereitete Erweiterungspunkte für Phase 3

Committe und pushe auf den Arbeits-Branch. Erstelle keinen Pull Request, solange er
nicht ausdrücklich angefordert wird.
```
