# Spezifikation: Grundstücks-Monitor („grundstuecke.warchhold.de“)

| | |
|---|---|
| **Version** | 1.0 (Entwurf zur Freigabe) |
| **Stand** | 08.10.2026 |
| **Projekt** | Neue Subdomain im Umfeld `warchhold`, neben dem bestehenden Rezessionsmonitor |
| **Ziel** | Ein KI-gestützter Agent sucht mehrmals täglich Grundstücke (mit oder ohne Bestandsgebäude) ab 1.000 m² in festgelegten Regionen Sachsens und Thüringens. Er prüft jedes Inserat gegen und listet es übersichtlich auf einer eigenen Webseite. |

---

## 1. Zusammenfassung

Der Grundstücks-Monitor besteht aus drei Teilen:

1. **Such-Agent (Pipeline):** Läuft automatisch mehrmals am Tag. Er sammelt Inserate aus mehreren Quellen, extrahiert die relevanten Daten mit Claude, prüft sie in einem **unabhängigen zweiten Durchgang (Gegenprüfung)**, entfernt Dubletten und verfolgt Preisänderungen sowie Inserate, die offline gehen.
2. **Datenspeicher:** Versionierte JSON-Dateien mit allen Inseraten, Preishistorie, Laufprotokollen und Token-Verbrauch.
3. **Webseite:** Statische, schnelle und mobilfähige Übersichtsseite unter `grundstuecke.warchhold.de` mit Filter-, Karten- und Tabellenansicht. Jedes Inserat ist direkt verlinkt.

Die Laufhäufigkeit richtet sich nach einem **festen Token-/Kostenbudget**. Der Agent drosselt sich selbst, wenn das Budget knapp wird (siehe Kapitel 9).

---

## 2. Namensvorschlag Subdomain

| Option | Bewertung |
|---|---|
| **`grundstuecke.warchhold.de`** | **Empfehlung.** Klar, ohne Umlaut, keine Punycode-Probleme |
| `grundstueck.warchhold.de` | Ebenfalls gut, Singular |
| `bauland.warchhold.de` | Kurz, trifft bebaute Grundstücke aber nicht ganz |
| `flaechen.warchhold.de` | Neutral, weniger selbsterklärend |
| `grundstück.warchhold.de` | **Nicht empfohlen**: Umlaut-Domain (IDN, `xn--…`) macht Probleme bei Zertifikaten, E-Mail und beim Teilen von Links |

> Annahme: Die Hauptdomain ist `warchhold.de`. Bei einer anderen TLD wird sie entsprechend ersetzt.

---

## 3. Fachliche Anforderungen

### 3.1 Suchgebiet

Das Gebiet ist über **Zentren mit Radius** definiert und in `config.yaml` konfigurierbar. Ergänzend gibt es eine Liste zulässiger Landkreise als zweite Absicherung.

| Bundesland | Zentrum | ca. Koordinaten (lat, lon) | Standard-Radius |
|---|---|---|---|
| Sachsen | Dresden | 51.050, 13.737 | 20 km |
| Sachsen | Pirna | 50.962, 13.940 | 20 km |
| Sachsen | Meißen | 51.164, 13.478 | 20 km |
| Thüringen | Schleiz | 50.579, 11.811 | 20 km |
| Thüringen | Neustadt an der Orla | 50.736, 11.746 | 20 km |
| Thüringen | Weimar | 50.980, 11.324 | 20 km |
| Thüringen | Gera | 50.880, 12.082 | 20 km |

**Zulässige Landkreise / kreisfreie Städte (Plausibilitätsprüfung):**
- Sachsen: Dresden, Landkreis Meißen, Landkreis Sächsische Schweiz-Osterzgebirge, Landkreis Bautzen (nur Randgebiete im Radius)
- Thüringen: Saale-Orla-Kreis, Weimar, Weimarer Land, Gera, Landkreis Greiz, Saale-Holzland-Kreis (nur Randgebiete im Radius)

**Geokodierung:** primär über eine Offline-Tabelle mit PLZ-Mittelpunkten (aus OpenStreetMap-Daten), ersatzweise Nominatim mit höchstens 1 Anfrage pro Sekunde und eindeutigem User-Agent. Die Entfernung zum nächstgelegenen Zentrum wird gespeichert und angezeigt.

### 3.2 Suchkriterien

| Kriterium | Regel |
|---|---|
| **Grundstücksfläche** | **≥ 1.000 m²** (Pflicht). Gemeint ist die **Grundstücksfläche, nicht die Wohnfläche**. Das ist die wichtigste Prüfregel. |
| Bebauung | **unbebaut** (Baugrundstück, Bauerwartungsland) **oder bebaut** (EFH, ZFH, MFH, Bauernhaus/Resthof, Villa, Abrissobjekt) |
| Vermarktungsart | Kauf (keine Miete/Pacht). Erbbaurecht und Zwangsversteigerung werden aufgenommen, aber **deutlich gekennzeichnet**. |
| Optional (Schalter in der Konfiguration, Standard: aus) | Reine Land-/Forstwirtschafts-, Garten- und Freizeitflächen ohne Bebaubarkeit. Sie werden, falls aktiviert, als eigene Kategorie geführt. |
| Preis | keine Obergrenze (Filter auf der Webseite). „Preis auf Anfrage“ ist zulässig. |

### 3.3 Anzuzeigende Informationen pro Inserat

**Pflicht (immer sichtbar auf der Karte/Kachel):**
- Titel (gekürzt), Badge(s): `NEU` (< 24 h), `PREIS GESENKT`, `ZWANGSVERSTEIGERUNG`, `ERBBAURECHT`, `GEPRÜFT ✔`
- Ort, PLZ, Landkreis, Bundesland, Entfernung zum nächsten Zentrum („12 km bis Pirna“)
- Grundstücksfläche (m²), Kaufpreis (€), **Preis pro m² Grundstück**
- Typ: unbebaut / bebaut (+ Gebäudeart)
- Bei Bebauung: Wohnfläche, Zimmer, Baujahr, Zustand (sofern angegeben)
- Quelle (Portal) + **Direktlink „Zum Inserat“** (öffnet in neuem Tab)
- Erstmals gesehen / zuletzt bestätigt

**Zusätzlich (aufklappbare Details):**
- Erschließung (voll/teil/unerschlossen/unbekannt), Bebaubarkeit (B-Plan, § 34, § 35 Außenbereich, unbekannt)
- Provision / Anbieter-Typ (privat, Makler, Bank, Amtsgericht, öffentliche Hand)
- Bei Zwangsversteigerung: Verkehrswert, Termin, Amtsgericht, Aktenzeichen
- Risiko-Hinweise (Flags): Außenbereich, Denkmalschutz, Hochwassergebiet (Elbe!), Erbbaurecht, Altlastenverdacht, Teilfläche, „Fläche unklar“
- **KI-Kurzfazit** (max. 2 Sätze, sachlich, ohne Werbesprache)
- Preisverlauf (Datum → Preis)
- „Auch gelistet bei …“ (Dubletten auf anderen Portalen, mit Links)
- Prüfstatus der Gegenprüfung mit Zeitstempel

### 3.4 Lebenszyklus eines Inserats

```
entdeckt → extrahiert → gegengeprüft ─┬─ bestätigt ─→ aktiv ─┬─ Preis geändert → aktiv
                                      ├─ korrigiert → aktiv  └─ 2 Läufe in Folge nicht auffindbar
                                      ├─ unsicher → „Zu prüfen“-Liste (nicht in Hauptliste)    → offline (30 Tage sichtbar, ausgegraut, dann Archiv)
                                      └─ abgelehnt → Ausschlussliste (mit Grund, nicht angezeigt)
```

---

## 4. Datenquellen

### 4.1 Grundsatz

Die großen Immobilienportale (ImmoScout24, Immowelt, Kleinanzeigen) **untersagen in ihren AGB das automatisierte Auslesen** und setzen Bot-Schutz ein. Ihre Datenbanken sind zudem nach § 87b UrhG geschützt (Recht des Datenbankherstellers). Deshalb ist der Zugang **gestaffelt**:

| Stufe | Quelle / Zugangsweg | Status |
|---|---|---|
| **A: E-Mail-Suchaufträge** | Auf ImmoScout24, Immowelt, Kleinanzeigen, Ohne-Makler, ImmoBörse der Sparkassen usw. werden Suchaufträge mit den Kriterien angelegt. Die Benachrichtigungen gehen an ein **eigenes Postfach** (z. B. `grundstuecke@warchhold.de`). Der Agent liest das Postfach per IMAP und wertet die Alarm-Mails aus (Titel, Preis, Fläche, Ort, Link). | **Kernquelle**, rechtlich sauber, robust |
| **B: Öffentliche Quellen direkt** | `zvg-portal.de` (Zwangsversteigerungen, Länder SN/TH), BVVG-Ausschreibungen, Landgesellschaften der Länder, kommunale Bauplatzbörsen / Amtsblätter der Gemeinden im Gebiet, Immobilienseiten regionaler Sparkassen und Volksbanken, ausgewählte regionale Makler | Direkter Abruf unter Beachtung von `robots.txt`, Rate-Limit ≤ 1 Anfrage / 3 s je Domain |
| **C: KI-Websuche (Discovery)** | Claude mit Web-Search-Tool, 1× täglich, findet Inserate aus Stufe-B-artigen Quellen, die noch nicht angebunden sind (Makler-Webseiten, Gemeinden). Neue ergiebige Quellen werden im Laufbericht vorgeschlagen. | Ergänzend, budgetbegrenzt |
| **D: Direktes Portal-Scraping** | Nur nach ausdrücklicher Freigabe durch den Betreiber (rechtliche Abwägung). **Standard: deaktiviert.** | Nicht Teil von v1 |

### 4.2 Umgang mit Inhalten
- Es werden **nur Fakten** (Zahlen, Ort, Typ) und ein **eigenes** KI-Kurzfazit gespeichert. Inseratstexte und **Fotos werden nicht kopiert** (Urheberrecht). Die Webseite verlinkt auf das Original.
- **Keine personenbezogenen Daten** speichern: keine Namen oder Telefonnummern privater Anbieter (DSGVO).
- Hinweis auf der Webseite: „Alle Angaben ohne Gewähr. Maßgeblich ist das Originalinserat.“

---

## 5. Systemarchitektur

### 5.1 Überblick

```
          ┌─────────────── GitHub Actions (Cron, 4×/Tag) ───────────────┐
          │                                                             │
 IMAP ──► │ 1 Sammeln ─► 2 Normalisieren ─► 3 Vorfilter (deterministisch)│
 ZVG  ──► │                                    │                        │
 BVVG ──► │                                    ▼                        │
 Web  ──► │        4 Extraktion (Claude, nur neue/geänderte Inserate)    │
          │                                    │                        │
          │                                    ▼                        │
          │        5 GEGENPRÜFUNG (deterministisch + Claude, unabhängig) │
          │                                    │                        │
          │                                    ▼                        │
          │ 6 Dubletten ─► 7 Lebenszyklus ─► 8 Speichern ─► 9 Rendern    │
          └──────────────────────────────────────┬──────────────────────┘
                                                 ▼
                      GitHub Pages / Cloudflare Pages ─► grundstuecke.warchhold.de
```

### 5.2 Technologie

| Baustein | Wahl | Begründung |
|---|---|---|
| Sprache | Python 3.11 | Wie im bestehenden Repo |
| LLM | Claude API über das offizielle `anthropic`-SDK, **Structured Outputs** (`output_config.format` mit JSON-Schema) | Garantiert valides JSON, keine Parse-Fehler |
| Scheduler | GitHub Actions `schedule` (Cron) + `workflow_dispatch` | Wie der bestehende Workflow `update.yml`, kostenlos |
| Speicher | JSON-Dateien auf eigenem Branch `grundstuecke-data` | Versioniert, nachvollziehbar, keine Datenbank nötig |
| Webseite | Statisches HTML + Vanilla-JS, gerendert mit Jinja2; Karte mit Leaflet + OpenStreetMap | Schnell, kostenlos zu hosten, keine Server-Wartung |
| Hosting | GitHub Pages mit Custom Domain (CNAME) **oder** Cloudflare Pages (falls Zugriffsschutz gewünscht, siehe 10.2) | |
| HTML-Abruf | `httpx` + `selectolax`/`BeautifulSoup`, kein Headless-Browser in v1 | Schlank |

> **Warum nicht Streamlit wie beim Rezessionsmonitor?** Streamlit Community Cloud unterstützt eigene Subdomains nur eingeschränkt, startet nach Inaktivität langsam und ist für eine reine Listenansicht überdimensioniert. Eine statische Seite lädt sofort und funktioniert gut auf dem Handy.

### 5.3 Verzeichnisstruktur (Vorschlag)

```
grundstuecke/
├── config.yaml                 # Regionen, Kriterien, Quellen, Budget, Modelle, Zeitplan
├── agent/
│   ├── run.py                  # Orchestrierung eines Laufs
│   ├── sources/
│   │   ├── email_alerts.py     # IMAP + Parser je Portal-Mailformat
│   │   ├── zvg.py              # zvg-portal.de
│   │   ├── bvvg.py
│   │   ├── static_sites.py     # generischer Parser für Gemeinden/Sparkassen/Makler (Liste in config)
│   │   └── web_discovery.py    # Claude Web Search
│   ├── normalize.py            # Einheiten, Zahlen („1.250 m²“, „ca. 0,3 ha“), Adressen
│   ├── geo.py                  # PLZ-Tabelle, Haversine, Landkreis-Check
│   ├── prefilter.py            # deterministische Regeln vor jedem LLM-Aufruf
│   ├── extract.py              # Claude-Extraktion (Prompt: prompts/extraktion.md)
│   ├── verify.py               # Gegenprüfung (Prompt: prompts/gegenpruefung.md)
│   ├── dedup.py
│   ├── lifecycle.py            # neu / Preisänderung / offline / Archiv
│   ├── budget.py               # Token-Ledger, Drosselung
│   ├── store.py
│   └── render.py               # erzeugt site/ aus templates/
├── prompts/                    # versionierte Prompts (siehe docs/grundstuecke/prompts/)
├── templates/                  # index.html.j2, Partials, CSS, JS
├── data/plz_centroids.csv
└── tests/
    ├── fixtures/               # echte, anonymisierte Beispiel-Mails & HTML-Seiten
    └── test_*.py
.github/workflows/grundstuecke.yml
```

---

## 6. Datenmodell

### 6.1 `listings.json` (ein Eintrag pro realem Objekt, nach Dubletten-Zusammenführung)

```json
{
  "id": "gs_7f3a9c",
  "status": "aktiv",                       // aktiv | offline | archiv | zu_pruefen | abgelehnt
  "pruefung": {
    "ergebnis": "bestaetigt",              // bestaetigt | korrigiert | unsicher | abgelehnt
    "zeitpunkt": "2026-10-08T07:14:02Z",
    "hinweise": ["Fläche im Text 1.250 m², in Mail 1.200 m² – Text maßgeblich"],
    "modell": "claude-opus-5-5"
  },
  "titel": "Großes Baugrundstück in ruhiger Ortsrandlage",
  "typ": "unbebaut",                       // unbebaut | bebaut | land_forst
  "gebaeudeart": null,                     // efh | zfh | mfh | bauernhaus | villa | abriss | sonstiges
  "vermarktung": "kauf",                   // kauf | zwangsversteigerung | erbbaurecht
  "ort": "Dohma", "plz": "01796", "landkreis": "Sächsische Schweiz-Osterzgebirge", "bundesland": "SN",
  "geo": { "lat": 50.93, "lon": 13.92, "genauigkeit": "plz" },
  "naechstes_zentrum": { "name": "Pirna", "km": 4.1 },
  "grundstueck_m2": 1250,
  "wohnflaeche_m2": null, "zimmer": null, "baujahr": null, "zustand": null,
  "preis_eur": 89000, "preis_auf_anfrage": false, "preis_pro_m2": 71.2,
  "verkehrswert_eur": null, "versteigerungstermin": null, "amtsgericht": null, "aktenzeichen": null,
  "erschliessung": "voll", "bebaubarkeit": "bplan",
  "provision": "3,57 % inkl. MwSt.", "anbieter_typ": "makler",
  "flags": ["hochwasser_pruefen"],
  "kurzfazit": "Voll erschlossenes Baugrundstück im B-Plan-Gebiet, rund 4 km von Pirna. Preis pro m² im regionalen Mittelfeld.",
  "quellen": [
    { "portal": "immowelt", "url": "https://…", "externe_id": "2abc…", "zuletzt_gesehen": "2026-10-08T07:10:00Z" }
  ],
  "preisverlauf": [ { "datum": "2026-10-01", "preis_eur": 95000 }, { "datum": "2026-10-08", "preis_eur": 89000 } ],
  "erstmals_gesehen": "2026-10-01T05:12:00Z",
  "zuletzt_bestaetigt": "2026-10-08T07:14:02Z",
  "nicht_gefunden_in_folge": 0
}
```

### 6.2 Weitere Dateien
- `runs/<datum>_<uhrzeit>.json`: Laufprotokoll (Quellen, Anzahl gefunden/neu/geprüft/abgelehnt, Fehler, Token, Kosten, Dauer)
- `usage.json`: Token- und Kosten-Ledger je Tag/Monat und Modell
- `rejected.json`: abgelehnte Inserate mit Grund (verhindert wiederholte LLM-Kosten für dasselbe Inserat)
- `review_queue.json`: Inserate mit Prüfergebnis „unsicher“

---

## 7. Verarbeitungsschritte im Detail

1. **Sammeln:** Jede Quelle liefert Rohdatensätze `{quelle, url, externe_id, rohtext, abgerufen_am}`. Fehler einer Quelle brechen den Lauf **nicht** ab, sondern werden protokolliert.
2. **Normalisieren:** Zahlen (`1.250 m²`, `ca. 0,3 ha` → 3.000, `1.200qm`), Preise (`VB`, `auf Anfrage`), PLZ/Ort.
3. **Vorfilter (deterministisch, ohne LLM):** bereits bekannt und unverändert → nur `zuletzt_gesehen` aktualisieren; in `rejected.json` → überspringen; erkennbare Fläche < 1.000 m² oder PLZ außerhalb des Gebiets → ablehnen; Miete/Pacht → ablehnen.
4. **Extraktion (Claude):** nur für neue oder inhaltlich geänderte Inserate. Ausgabe streng nach JSON-Schema. Prompt siehe `prompts/extraktion.md`.
5. **Gegenprüfung:** siehe Kapitel 8.
6. **Dubletten:** gleiche PLZ **und** Fläche ±3 % **und** (Preis ±5 % **oder** Titel-Ähnlichkeit ≥ 0,8). Grenzfälle entscheidet die Gegenprüfung mit. Dubletten werden zu einem Objekt mit mehreren `quellen` zusammengeführt.
7. **Lebenszyklus:** Preisänderung → Eintrag im `preisverlauf` + Badge. Ist ein Inserat in zwei aufeinanderfolgenden Läufen bei keiner Quelle auffindbar und meldet der Link einen HTTP-Status von 404/410 oder eine Weiterleitung auf eine Suchseite, wird es auf **offline** gesetzt. Bei Quellen, die keine Linkprüfung erlauben (Bot-Schutz), wird es nach **14 Tagen** ohne neue Alarm-Mail als „vermutlich offline“ markiert.
8. **Speichern:** atomar schreiben (temporäre Datei, dann umbenennen) und auf den Daten-Branch committen.
9. **Rendern & Veröffentlichen:** Seite bauen und deployen. **Sicherheitsregel:** Liefert ein Lauf auffällige Ergebnisse (siehe 8.4), wird **nicht** veröffentlicht. Die letzte gute Version bleibt online.

---

## 8. Gegenprüfung (Qualitätssicherung)

Kein Inserat erscheint in der Hauptliste, ohne die Gegenprüfung bestanden zu haben. Sie hat vier Ebenen.

### 8.1 Ebene 1: Deterministische Regeln (Code, kostenlos)
| Prüfung | Regel |
|---|---|
| Pflichtfelder | `url`, `ort` oder `plz`, `grundstueck_m2`, `typ` vorhanden |
| Fläche | `grundstueck_m2 >= 1000`. Ist nur eine Wohnfläche erkennbar → `unsicher` |
| Fläche vs. Wohnfläche | bebaut und `grundstueck_m2 <= wohnflaeche_m2 × 1,2` → Verdacht auf Verwechslung → `unsicher` |
| Gebiet | Entfernung zum nächsten Zentrum ≤ Radius **und** Landkreis in der Positivliste |
| Plausibilität Preis | Preis/m² zwischen 3 € und 1.500 €; sonst `unsicher` (Ausreißer werden nicht verworfen, sondern manuell geprüft) |
| Link | URL syntaktisch gültig, Domain passt zur Quelle; HTTP-Status, wo erlaubt |
| Kauf | keine Begriffe wie „Miete“, „Pacht“, „zu vermieten“ als Vermarktungsart |

### 8.2 Ebene 2: Unabhängige KI-Gegenprüfung (Claude)
- Ein **zweiter, separater Claude-Aufruf** mit **eigenem Prompt** (`prompts/gegenpruefung.md`) erhält den **Rohtext der Quelle** und das **extrahierte JSON**.
- Er prüft jedes Kernfeld (Fläche, Preis, Ort, Typ, Vermarktungsart, Bebauung) gegen den Rohtext und muss für jedes Feld ein **wörtliches Belegzitat** aus der Quelle liefern. Ohne Beleg gilt das Feld als „nicht belegt“.
- Ergebnis: `bestaetigt` | `korrigiert` (mit korrigierten Werten und Begründung) | `unsicher` | `abgelehnt` (mit Grund).
- **Unabhängigkeit:** Der Prüfer sieht weder den Extraktions-Prompt noch dessen Begründung. Empfohlen ist ein **stärkeres oder anderes Modell** als bei der Extraktion (siehe 9.2).
- Die Code-Ebene prüft anschließend, ob jedes Belegzitat **tatsächlich wörtlich im Rohtext vorkommt** (Normalisierung von Leerzeichen erlaubt). Ein erfundenes Zitat führt zu `unsicher`.

### 8.3 Ebene 3: Laufende Nachprüfung
- Aktive Inserate werden alle 24 h auf Erreichbarkeit geprüft (ohne LLM, wo technisch und rechtlich erlaubt).
- Ändert sich der Inhalt (Preis, Fläche), laufen Extraktion und Gegenprüfung erneut.

### 8.4 Ebene 4: Lauf-Plausibilität (Schutz vor Fehlveröffentlichung)
Ein Lauf wird **nicht veröffentlicht** und als Fehler gemeldet, wenn:
- mehr als 30 % der Quellen fehlschlagen, **oder**
- die Zahl aktiver Inserate gegenüber dem letzten Lauf um mehr als 40 % fällt, **oder**
- mehr als 50 % der neuen Inserate in der Gegenprüfung `abgelehnt`/`unsicher` sind (Hinweis auf geänderte Mailformate oder Parser-Fehler).

### 8.5 Sichtbarkeit
- Hauptliste: nur `bestaetigt` und `korrigiert` (bei `korrigiert` mit Info-Symbol und Hinweis)
- Reiter „Zu prüfen“: `unsicher`, mit Grund und Link zur manuellen Prüfung
- Abgelehnte Inserate werden nicht angezeigt, sind aber in `rejected.json` nachvollziehbar
- Wöchentlicher **Stichproben-Bericht** im Laufprotokoll: 5 zufällige bestätigte Inserate mit Rohtext-Auszug und extrahierten Werten zur menschlichen Kontrolle

---

## 9. Zeitplan, Token-Budget und Kosten

### 9.1 Zeitplan
- **Standard: 4 Läufe pro Tag**, ca. 07:00, 12:00, 17:00 und 21:00 Uhr deutscher Zeit. GitHub-Cron läuft in UTC, daher z. B. `15 5,10,15,19 * * *`. Die Zeiten verschieben sich bei der Sommer-/Winterzeitumstellung um eine Stunde, das ist akzeptabel.
- **Discovery per Websuche:** 1× täglich (im Morgenlauf).
- Manuell auslösbar über `workflow_dispatch`.

### 9.2 Modellwahl (konfigurierbar in `config.yaml`)

Preise laut Anthropic-Preisliste (Stand Oktober 2026, pro 1 Mio. Tokens Input/Output): Claude Opus 5.5 4 $ / 20 $, Claude Sonnet 5.5 2 $ / 10 $, Claude Haiku 5.5 0,10 $ / 0,50 $. Websuche: 10 $ pro 1.000 Suchen.

| Profil | Extraktion | Gegenprüfung | Discovery | Bewertung |
|---|---|---|---|---|
| **Qualität** | Opus 5.5 | Opus 5.5 | Opus 5.5 | Höchste Genauigkeit |
| **Ausgewogen (Empfehlung)** | Haiku 5.5 | Opus 5.5 | Opus 5.5 | Günstige Extraktion, starke und **unabhängige** Gegenprüfung durch ein anderes Modell |
| **Sparsam** | Haiku 5.5 | Sonnet 5.5 | Sonnet 5.5 | Sehr günstig, Gegenprüfung immer noch mit anderem Modell |

Einstellungen: Extraktion mit `effort: low`, Gegenprüfung mit `effort: medium`. Für den statischen Teil (System-Prompt + Schema) wird Prompt-Caching genutzt.

### 9.3 Kostenschätzung

**Annahmen:** 25 neue oder geänderte Inserate pro Tag nach dem Vorfilter. Extraktion ca. 4.000 Input- und 600 Output-Tokens je Inserat, Gegenprüfung ca. 5.000 Input- und 400 Output-Tokens. Discovery 1×/Tag mit 8 Suchen und ca. 60.000 Input- und 3.000 Output-Tokens. Dazu ein Aufschlag von 30 % für Thinking-Tokens und Wiederholungen.

| Profil | ca. pro Tag | ca. pro Monat |
|---|---|---|
| Qualität | 2,30 $ | **≈ 70 $** |
| Ausgewogen | 1,40 $ | **≈ 45 $** |
| Sparsam | 0,80 $ | **≈ 23 $** |

Rechenweg (Qualität, pro Tag): Extraktion 25 × (4.000 × 4 $ + 600 × 20 $) / 1 Mio. = 0,70 $ · Gegenprüfung 25 × (5.000 × 4 $ + 400 × 20 $) / 1 Mio. = 0,70 $ · Discovery 0,08 $ Suchen + 0,24 $ Input + 0,06 $ Output = 0,38 $ → 1,78 $ × 1,3 ≈ 2,30 $.

> Die tatsächlichen Werte werden ab dem ersten Lauf in `usage.json` gemessen. Nach 2 Wochen wird die Schätzung überprüft.

### 9.4 Budget-Steuerung („so oft, wie das Budget erlaubt“)
- `config.yaml`: `budget.monat_usd` (z. B. 50), `budget.lauf_max_usd` (z. B. 3).
- Vor jedem Lauf wird die Hochrechnung aus `usage.json` gebildet:
  - **< 80 % des anteiligen Monatsbudgets:** normaler Betrieb
  - **80–100 %:** Drosselung. Nur noch 2 Läufe/Tag, keine Discovery, Extraktion mit dem günstigsten Modell
  - **> 100 %:** **Sparmodus.** Keine LLM-Aufrufe. Neue Inserate aus Mail-Alarmen werden nur per Regex erfasst und als „ungeprüft“ in „Zu prüfen“ geführt.
- Innerhalb eines Laufs wird bei Erreichen von `lauf_max_usd` sauber abgebrochen. Nicht verarbeitete Inserate kommen in die Warteschlange für den nächsten Lauf.
- Die Webseite zeigt in der Fußzeile den Betriebsmodus („Normalbetrieb“ / „Gedrosselt“ / „Sparmodus“).

---

## 10. Webseite

### 10.1 Aufbau
1. **Kopfbereich:** Titel „Grundstücks-Monitor Sachsen & Thüringen“, Stand der letzten Aktualisierung, Betriebsmodus
2. **Kennzahlen-Leiste:** aktive Inserate · neu (24 h) · Preis gesenkt (7 Tage) · Ø €/m² je Region
3. **Filterleiste** (bleibt beim Scrollen sichtbar, auf dem Handy als ausklappbares Panel):
   Region/Zentrum (Mehrfachauswahl) · Bundesland · bebaut/unbebaut · Fläche von–bis · Preis bis · €/m² bis · nur neue · Quelle · Zwangsversteigerungen ein/aus · Umkreis-km
4. **Sortierung:** neueste zuerst (Standard) · Preis ↑↓ · €/m² ↑ · Fläche ↓ · Entfernung ↑
5. **Ansichten (umschaltbar):** **Kacheln** (Standard, mobil) · **Tabelle** (Desktop, kompakt, sortierbar) · **Karte** (Leaflet, Marker nach Typ eingefärbt, Kreise für die Suchradien)
6. **Reiter:** „Aktiv“ · „Zu prüfen“ · „Offline / Archiv“
7. **Fußzeile:** Quellenliste, Haftungshinweis, Impressum-Link (Hauptdomain), Betriebsmodus und Kosten des laufenden Monats (optional)

### 10.2 Nicht-funktionale Anforderungen
- Ladezeit < 1,5 s auf dem Handy (4G). Alle Daten liegen in einer gerenderten `listings.json`, gefiltert wird im Browser.
- Barrierearm: ausreichende Kontraste, Tastaturbedienung, Hell- und Dunkelmodus
- `noindex, nofollow` (privates Werkzeug, keine Suchmaschinen-Indexierung)
- **Zugriffsschutz (Entscheidung offen):** öffentlich (GitHub Pages) **oder** geschützt (Cloudflare Pages + Cloudflare Access mit E-Mail-Login, kostenlos für kleine Teams). Aus rechtlicher Sicht wird eine **nicht öffentliche** Nutzung empfohlen.
- Filterzustand in der URL (teilbare Links, z. B. `?zentrum=pirna&typ=bebaut`)

---

## 11. Betrieb, Sicherheit, Monitoring

- **Secrets (GitHub Actions):** `ANTHROPIC_API_KEY`, `IMAP_HOST`, `IMAP_USER`, `IMAP_PASSWORD`. Keine Secrets im Code oder in Logs.
- **Prompt-Injection-Schutz:** Inseratstexte sind **nicht vertrauenswürdige Daten**. Sie werden in Prompts klar abgegrenzt (`<inserat>…</inserat>`). Anweisungen darin werden ignoriert. Das LLM hat **keine Tools** außer bei Discovery (nur Websuche).
- **Fehlermeldung:** Schlägt ein Lauf fehl oder greift die Sicherheitsregel aus 8.4, erstellt der Workflow ein GitHub-Issue (bzw. aktualisiert ein bestehendes, um Duplikate zu vermeiden).
- **DNS:** `CNAME grundstuecke → <github-user>.github.io` (bzw. Cloudflare-Pages-Ziel), HTTPS erzwingen.
- **Respektvoller Abruf:** `robots.txt` beachten, eindeutiger User-Agent mit Kontaktadresse, Rate-Limit, Caching (ETag/Last-Modified).

---

## 12. Abnahmekriterien

| # | Kriterium | Messung |
|---|---|---|
| A1 | Die Seite ist unter der Subdomain per HTTPS erreichbar | manuell |
| A2 | Mindestens 4 automatische Läufe pro Tag im Normalbetrieb | Laufprotokolle |
| A3 | **0 Inserate mit Grundstücksfläche < 1.000 m²** in der Hauptliste | automatischer Test über `listings.json` |
| A4 | **0 Inserate außerhalb des Suchgebiets** in der Hauptliste | automatischer Test |
| A5 | Jedes Inserat der Hauptliste hat einen funktionierenden Link und Prüfstatus `bestaetigt`/`korrigiert` | automatischer Test |
| A6 | Stichprobe von 20 Inseraten: ≥ 95 % der Kernfelder (Fläche, Preis, Ort, Typ) korrekt | manuelle Kontrolle gegen das Original |
| A7 | Dubletten über Portale hinweg werden zusammengeführt (Stichprobe ≥ 90 %) | manuell |
| A8 | Monatskosten ≤ konfiguriertes Budget; Drosselung nachweislich wirksam | `usage.json`, Test mit simuliertem Budget |
| A9 | Fehlerhafter Lauf (z. B. Parser kaputt) führt **nicht** zu einer leeren oder fehlerhaften Seite | Test mit manipulierten Fixtures |
| A10 | Seite lädt auf dem Handy < 1,5 s und ist ohne horizontales Scrollen bedienbar | Lighthouse ≥ 90 (Performance, Accessibility) |
| A11 | Testabdeckung der Kernmodule (`normalize`, `prefilter`, `geo`, `dedup`, `lifecycle`, `budget`, `verify`-Code-Ebene) ≥ 80 % | `pytest --cov` |

---

## 13. Umsetzungsphasen

| Phase | Inhalt | Ergebnis |
|---|---|---|
| **0: Vorbereitung** (Betreiber) | Postfach anlegen, Suchaufträge auf den Portalen einrichten, Subdomain-DNS, API-Key, Entscheidungen aus Kapitel 14 | Zugänge & Entscheidungen |
| **1: MVP** | Mail-Alarm-Quelle (2 Portale), Normalisierung, Vorfilter, Extraktion, **Gegenprüfung Ebene 1+2**, JSON-Speicher, einfache Kachel-Seite, Cron 4×/Tag, Budget-Ledger | Seite live mit echten, geprüften Daten |
| **2: Ausbau** | Weitere Portale, ZVG, BVVG, Dubletten, Lebenszyklus, Preisverlauf, Karten- und Tabellenansicht, Gegenprüfung Ebene 3+4, Drosselung | Abnahmekriterien A1–A11 |
| **3: Komfort** | Discovery per Websuche, Bodenrichtwert-Vergleich (BORIS Sachsen / Geoportal Thüringen), Hochwasser-Layer, E-Mail- oder Telegram-Zusammenfassung neuer Top-Treffer, wöchentlicher Stichproben-Bericht | Mehrwert & Komfort |

---

## 14. Offene Entscheidungen (bitte vor Umsetzung klären)

1. **Subdomain-Name:** `grundstuecke.warchhold.de`? Wie lautet die genaue Hauptdomain?
2. **Öffentlich oder geschützt?** (GitHub Pages vs. Cloudflare Access)
3. **Modellprofil / Monatsbudget:** Qualität (~70 $), Ausgewogen (~45 $, Empfehlung) oder Sparsam (~23 $)?
4. **Radius je Zentrum:** 20 km für alle oder individuell?
5. **Land-/Forst-/Freizeitflächen** zusätzlich aufnehmen (eigene Kategorie)?
6. **Zwangsversteigerungen** anzeigen (Standard: ja, gekennzeichnet)?
7. **Portal-Scraping (Stufe D)** ausdrücklich ausschließen (Empfehlung) oder rechtlich prüfen lassen?
8. **Benachrichtigungen** bei neuen Treffern gewünscht (E-Mail/Telegram)? Ab welchen Kriterien?
9. **Repository:** in diesem Repo (`grundstuecke/`) oder eigenes Repo (sauberere Trennung, eigene Pages-Domain)?

---

## Anhang

- `prompts/UMSETZUNG.md`: Prompt für Claude Code zur Implementierung, mit Selbst- und Gegenprüfung
- `prompts/extraktion.md`: Laufzeit-Prompt für die Datenextraktion
- `prompts/gegenpruefung.md`: Laufzeit-Prompt für die unabhängige Gegenprüfung
- `prompts/discovery.md`: Laufzeit-Prompt für die Websuche
