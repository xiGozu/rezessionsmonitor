# Spezifikation: Grundstücks-Monitor („grundstuecke.warchhold.de“)

| | |
|---|---|
| **Version** | 1.1 (gegengeprüft, zur Freigabe) |
| **Stand** | 08.10.2026 |
| **Projekt** | Neue Subdomain im Umfeld `warchhold`, neben dem bestehenden Rezessionsmonitor |
| **Ziel** | Ein KI-gestützter Agent sucht mehrmals täglich Grundstücke (mit oder ohne Bestandsgebäude) ab 1.000 m² in festgelegten Regionen Sachsens und Thüringens. Er prüft jedes Inserat gegen und listet es übersichtlich auf einer eigenen Webseite. |

**Änderungen gegenüber 1.0 (Ergebnis der unabhängigen Gegenprüfung):**
- Grundstücksfläche kann über einen Portal-Suchfilter belegt werden (Alarm-Mails enthalten sie oft nicht).
- Offline-Erkennung je Quellentyp.
- Gebiet über Kreisgrenzen statt fester Landkreisliste.
- Belegprüfung mit festen Feldern und Zahlenabgleich.
- Realistischere Kostenschätzung inkl. Thinking-Tokens.
- Budget-Formel.
- Manuelle Freigabe über `overrides.yaml`.
- Anfangsbestand.
- XSS-Schutz.
- Zentrale Enums (Kapitel 6.3).
- Lauf-Integrität (IMAP-Wasserzeichen, kein Commit bei Fehllauf).
- Datenschutz bei Karten und Zwangsversteigerungen.

---

## 1. Zusammenfassung

Der Grundstücks-Monitor besteht aus drei Teilen:

1. **Such-Agent (Pipeline):** Läuft automatisch mehrmals täglich. Er sammelt Inserate aus mehreren Quellen, extrahiert die relevanten Daten mit Claude und prüft sie in einem **unabhängigen zweiten Durchgang (Gegenprüfung)**. Außerdem entfernt er Dubletten und verfolgt Preisänderungen sowie Inserate, die offline gehen.
2. **Datenspeicher:** Versionierte JSON-Dateien mit allen Inseraten, Preishistorie, Laufprotokollen und Kosten-Ledger.
3. **Webseite:** Statische, schnelle, mobilfähige Übersichtsseite unter `grundstuecke.warchhold.de` mit Filter-, Karten- und Tabellenansicht. Jedes Inserat ist direkt verlinkt.

Das Sammeln (IMAP, öffentliche Quellen) kostet nichts und läuft häufig. Die **KI-Verarbeitung richtet sich nach einem festen Monatsbudget** (Kapitel 9).

---

## 2. Namensvorschlag Subdomain

| Option | Bewertung |
|---|---|
| **`grundstuecke.warchhold.de`** | **Empfehlung.** Klar, ohne Umlaut |
| `grundstueck.warchhold.de` | Ebenfalls gut |
| `bauland.warchhold.de` | Kurz, trifft bebaute Grundstücke aber nicht ganz |
| `grundstück.warchhold.de` | **Nicht empfohlen**: Umlaut-Domain (IDN, `xn--…`) macht Probleme bei Zertifikaten und beim Teilen von Links |

> Annahme: Die Hauptdomain ist `warchhold.de`.

---

## 3. Fachliche Anforderungen

### 3.1 Suchgebiet

Das Gebiet ist über **Zentren mit Radius** definiert (in `config.yaml`). Die Entscheidung trifft **ausschließlich der Code**, nicht das LLM.

| Bundesland | Zentrum | ca. Koordinaten (lat, lon) | Standard-Radius |
|---|---|---|---|
| Sachsen | Dresden | 51.050, 13.737 | 20 km |
| Sachsen | Pirna | 50.962, 13.940 | 20 km |
| Sachsen | Meißen | 51.164, 13.478 | 20 km |
| Thüringen | Schleiz | 50.579, 11.811 | 20 km |
| Thüringen | Neustadt an der Orla | 50.736, 11.746 | 20 km |
| Thüringen | Weimar | 50.980, 11.324 | 20 km |
| Thüringen | Gera | 50.880, 12.082 | 20 km |

**Zulässige Kreise:** Sie werden **automatisch erzeugt**. Aufgenommen wird jeder Kreis, dessen Fläche einen der Radius-Kreise schneidet. Grundlage sind die Verwaltungsgrenzen des BKG (VG250, Open Data). Die erzeugte Liste wird als `data/kreise_im_gebiet.json` versioniert.

Bei 20 km Radius gehören dazu nachweislich auch:
- **Jena** und **Erfurt** (Rand bei Weimar)
- **LK Sömmerda**, **Altenburger Land**, **Burgenlandkreis** (Sachsen-Anhalt, Raum Zeitz bei Gera)
- **Vogtlandkreis** (Raum Pausa/Mühltroff bei Schleiz)
- **LK Mittelsachsen** (Rand bei Meißen)
- eventuell auch LK Zwickau, LK Hof (Bayern) und Ilm-Kreis

Der Radius um Pirna reicht knapp nach Tschechien. Ausländische Objekte werden ausgeschlossen. Ob Großstadt-Ränder (Jena, Erfurt) gewünscht sind, ist eine offene Entscheidung (Kapitel 14). Über `ausgeschlossene_kreise` in der Config lassen sie sich ausnehmen.

**Geokodierung (Reihenfolge):**
1. Gemeinde oder Ortsteil aus dem Inserat → amtlicher Gemeindeschlüssel (AGS) über das Gemeindeverzeichnis von Destatis → Gemeindemittelpunkt und Kreis
2. Nur PLZ → PLZ-Mittelpunkt (OSM-Daten, ODbL, Namensnennung) → Kreis über die PLZ-Gemeinde-Zuordnung
3. Ersatzweise Nominatim, auf eine Viewbox Sachsen/Thüringen/Umgebung begrenzt, höchstens 1 Anfrage/s, Ergebnisse gecacht
4. Nicht geokodierbar → Prüfstatus `unsicher` (nie stillschweigend verwerfen)

**Toleranzband:** Bei Entfernung ≤ Radius wird das Inserat aufgenommen. Zwischen Radius und Radius + 3 km wird es `unsicher`, weil PLZ-Mittelpunkte ungenau sind. Über Radius + 3 km wird es abgelehnt.

### 3.2 Suchkriterien

| Kriterium | Regel |
|---|---|
| **Grundstücksfläche** | **≥ 1.000 m²** (Pflicht). Gemeint ist die **Grundstücksfläche, nicht die Wohnfläche.** Als Nachweis gilt entweder ein belegter Wert aus dem Inseratstext **oder** der Suchauftrag-Filter des Portals (Kapitel 4.3). |
| Bebauung | **unbebaut** (Baugrundstück, Bauerwartungsland) **oder bebaut** (EFH, ZFH, MFH, Bauernhaus/Resthof, Villa, Abrissobjekt) |
| Vermarktungsart | **kauf**, **zwangsversteigerung**, **erbbaurecht** (die letzten beiden deutlich gekennzeichnet). Ausgeschlossen: Miete, Pacht, Tausch, Mietkauf, Leibrente/Verrentung, Nießbrauch, Gesuche. Bieterverfahren gilt als Kauf mit Hinweis. |
| Teilflächen | Nur aufnehmen, wenn die **verkaufte** Fläche belegt ≥ 1.000 m² ist |
| Land-/Forst-/Freizeitflächen | Schalter `land_forst_aufnehmen` (Standard: **aus**). Bei „aus“ werden sie hart abgelehnt. |
| Preis | Keine Obergrenze (Filter auf der Webseite). „Preis auf Anfrage“ ist zulässig. |
| Land | Nur Deutschland |

### 3.3 Anzuzeigende Informationen pro Inserat

**Immer sichtbar (Kachel/Zeile):**
- **Kurztitel**: wird **selbst formuliert** (z. B. „Baugrundstück 1.250 m², Ortsrandlage“). Der Inseratstitel wird nicht kopiert.
- Badges: `NEU` (< 24 h), `PREIS GESENKT`, `ZWANGSVERSTEIGERUNG`, `ERBBAURECHT`, `KORRIGIERT` (Gegenprüfung hat Werte korrigiert), `FLÄCHE LAUT PORTALFILTER`
- Ort, Landkreis, Bundesland, Entfernung zum nächsten Zentrum („12 km bis Pirna“)
- Grundstücksfläche (m²) bzw. „≥ 1.000 m² (laut Portalfilter)“
- Kaufpreis (€) bzw. „Preis auf Anfrage“ bzw. Verkehrswert bei Zwangsversteigerung
- **€/m² Grundstück**: bei `unbebaut` als echter Bodenpreis, bei `bebaut` mit Hinweis „inkl. Gebäude“
- Typ: unbebaut / bebaut (+ Gebäudeart). Bei Bebauung zusätzlich Wohnfläche, Zimmer, Baujahr, Zustand.
- Quelle (Portal) + **Direktlink „Zum Inserat“** (neuer Tab)
- „Erstmals gesehen“ und „zuletzt bestätigt vor X Tagen“

**Aufklappbare Details:**
- Erschließung, Bebaubarkeit (B-Plan / § 34 / § 35 Außenbereich). **Nur belegte Werte**, sonst „unbekannt“.
- Provision, Anbieter-Typ
- Bei Zwangsversteigerung: Verkehrswert, Termin, Amtsgericht (Aktenzeichen nur im geschützten Modus, siehe 4.2)
- Risiko-Hinweise (Flags): Außenbereich, Denkmalschutz, Hochwasser, Erbbaurecht, Altlasten, Teilfläche, Fläche unklar, sanierungsbedürftig
- **KI-Kurzfazit** (max. 2 Sätze, sachlich, gegengeprüft)
- Preisverlauf, „Auch gelistet bei …“ (Dubletten mit Links), Prüfprotokoll (Ergebnis, Zeitpunkt, Hinweise)

### 3.4 Lebenszyklus eines Inserats

```
entdeckt → Vorfilter → Extraktion → Gegenprüfung ─┬─ bestaetigt/korrigiert → aktiv
                                                  ├─ unsicher → zu_pruefen ─(overrides.yaml)→ aktiv | abgelehnt
                                                  └─ abgelehnt → abgelehnt (bei Inhalts- oder Config-Änderung erneute Bewertung)

aktiv ── Preis geändert ──────────────→ aktiv (+ Preisverlauf, erneute Prüfung)
aktiv ── Quelle „vollliste“: 2 erfolgreiche Läufe in Folge nicht gelistet ──→ offline
aktiv ── Quelle „linkcheck“: Link 2× in Folge 404/410/Weiterleitung auf Suchseite ──→ offline
aktiv ── Quelle „keine“ (Alarm-Mails): 45 Tage ohne Bestätigung ──→ archiv_unbestaetigt
offline ── nach 30 Tagen ──→ archiv
```

Wichtig:
- HTTP 403/429, Captcha oder Timeout bedeuten **„unbekannt“**, nie „offline“.
- Fehlgeschlagene Quellen zählen nicht als „nicht gelistet“.

---

## 4. Datenquellen

### 4.1 Grundsatz und Stufen

ImmoScout24, Immowelt und Kleinanzeigen **untersagen in ihren AGB das automatisierte Auslesen** und setzen Bot-Schutz ein. Ihre Datenbanken sind nach § 87b UrhG geschützt. Deshalb ist der Zugang gestaffelt:

| Stufe | Quelle / Zugangsweg | Lebendprüfung (`liveness`) | Status |
|---|---|---|---|
| **A: E-Mail-Suchaufträge** | Suchaufträge auf ImmoScout24, Immowelt, Kleinanzeigen, Ohne-Makler, Sparkassen-Immobilienportal usw. Die Benachrichtigungen gehen an ein **eigenes Postfach**, der Agent liest sie per IMAP. | `keine` | **Kernquelle** |
| **B: Öffentliche Quellen direkt** | `zvg-portal.de` (SN/TH), BVVG-Ausschreibungen, Landgesellschaften, Grundstücksbörsen und Amtsblätter der Gemeinden, Immobilienseiten regionaler Sparkassen/Volksbanken und Makler | `vollliste` oder `linkcheck` (je Quelle in der Config) | Unter Beachtung von `robots.txt` und Nutzungsbedingungen, ≤ 1 Anfrage / 3 s je Domain |
| **C: KI-Websuche (Discovery)** | Claude mit Web-Search-Tool, 1× täglich, für Quellen, die die Portale nicht abdecken | wie gefundene Quelle | Ergänzend, budgetbegrenzt |
| **D: Portal-Scraping / Exposé-Abruf** | Nur nach ausdrücklicher rechtlicher Freigabe | — | **Standard: deaktiviert** |

### 4.2 Umgang mit Inhalten (Recht und Datenschutz)
- Gespeichert werden nur **Fakten** und **eigene Formulierungen** (Kurztitel, Kurzfazit). Inseratstexte und **Fotos werden nicht übernommen**.
- **Rohtexte** werden nur intern zur Prüfung gespeichert, nicht veröffentlicht und nach 90 Tagen gelöscht.
- **Keine personenbezogenen Daten:** keine Namen oder Telefonnummern privater Anbieter.
- **Zwangsversteigerungen:** Aktenzeichen und Straße sind schuldnerbezogene Daten. Sie werden nur im geschützten Modus angezeigt und nach dem Versteigerungstermin gelöscht. Die **Nutzungsbedingungen des ZVG-Portals** werden in Phase 0 geprüft.
- Hinweis auf der Webseite: „Alle Angaben ohne Gewähr. Maßgeblich ist das Originalinserat.“

### 4.3 Besonderheit Alarm-Mails (Stufe A)

Alarm-Mails enthalten je nach Portal oft **nur Titel, Preis, Wohnfläche, Zimmer und Ort**. Die **Grundstücksfläche fehlt dann häufig**, besonders bei Häusern. Lösung:

1. **Phase 0:** Der Betreiber legt je Portal **echte Alarm-Mails als `.eml`** ab (`tests/fixtures/mails/`). Pro Portal wird dokumentiert, welche Felder die Mail enthält. Ohne echte Mails gelten die Parser als „vorläufig“.
2. Jeder Suchauftrag wird in `config.yaml` mit seinem **tatsächlich gesetzten Filter** hinterlegt, z. B. `{portal: immowelt, absender: "...", filter: {grundstueck_min_m2: 1000, objektarten: [haus, grundstueck]}}`. Den Filter auf dem Portal zu setzen und zu pflegen, liegt beim Betreiber.
3. Enthält die Mail keine Grundstücksfläche, gilt die Fläche als **„≥ 1.000 m² laut Portalfilter“**. In diesem Fall:
   - Prüfstatus des Feldes: `belegt_durch_filter`
   - Anzeige mit Badge `FLÄCHE LAUT PORTALFILTER`
   - Das Inserat zählt **nicht** als unsicher.
4. Nennt die Mail eine Fläche, gilt immer der **belegte Wert** (auch wenn er < 1.000 m² ist → ablehnen).
5. **Tracking- und Weiterleitungslinks** in Mails werden zur **kanonischen Exposé-URL** aufgelöst (ohne Tracking-Parameter). Dafür wird nur der Redirect gefolgt (HEAD/GET ohne Body-Auswertung). Wo das blockiert ist, wird die URL aus der Portal-ID gebildet. `externe_id` = Portal-ID.

### 4.4 Anfangsbestand

Alarm-Mails melden nur **Neuzugänge**. Für den Start gilt: Der Betreiber nutzt auf den Portalen die Funktion „Inserat per E-Mail teilen/weiterleiten“ für aktuell interessante Inserate und schickt diese Mails an das Postfach. Sie werden wie Alarm-Mails verarbeitet. Zusätzlich holt ein **einmaliger Discovery-Lauf** mit erhöhtem Suchlimit Inserate aus Stufe-B-Quellen.

---

## 5. Systemarchitektur

### 5.1 Überblick

```
          ┌──────────── GitHub Actions (Cron alle 2 h, 06–22 Uhr) ─────────────┐
 IMAP ──► │ 1 Sammeln ─► 2 Normalisieren ─► 3 Vorfilter (Code) ─► pending.json │
 ZVG  ──► │                                                        │           │
 BVVG ──► │      ┌──────── Budget-Freigabe (budget.py) ◄───────────┘           │
 Web  ──► │      ▼                                                             │
          │ 4 Extraktion (Claude) ─► 5 GEGENPRÜFUNG (Code + Claude + Code)     │
          │      ▼                                                             │
          │ 6 Dubletten ─► 7 Lebenszyklus ─► 8 Lauf-Plausibilität              │
          │      ├─ ok:     9 Commit Daten ─► 10 Abnahme-Test ─► 11 Rendern/Deploy
          │      └─ Fehler: nur Laufprotokoll committen, GitHub-Issue, alte Seite bleibt
          └────────────────────────────────────────────────────────────────────┘
```

### 5.2 Technologie

| Baustein | Wahl |
|---|---|
| Sprache | Python 3.11 (eigener Workflow, eigene gepinnte `grundstuecke/requirements.txt`; der bestehende Rezessionsmonitor bleibt unverändert) |
| LLM | Claude API über das offizielle `anthropic`-SDK, Structured Outputs (`output_config.format`) |
| Scheduler | GitHub Actions `schedule` + `workflow_dispatch`, `concurrency`-Gruppe (keine parallelen Läufe) |
| Speicher | JSON-Dateien auf Branch `grundstuecke-data` (im Workflow per `git worktree` ausgecheckt) |
| Webseite | Statisches HTML + Vanilla-JS (Jinja2 mit **Autoescape**). Leaflet **selbst gehostet** im Seitenbundle. Kartenkacheln von OpenStreetMap mit Namensnennung (siehe 10.2 Datenschutz). |
| Hosting | GitHub Pages **oder** Cloudflare Pages + Cloudflare Access (Entscheidung, siehe 10.2/14) |
| HTML-Abruf | `httpx` + `selectolax`, kein Headless-Browser |

> **Warum nicht Streamlit wie beim Rezessionsmonitor?** Für eine reine Listenansicht ist eine statische Seite schneller, kostenlos zu hosten und unterstützt eigene Subdomains problemlos.

### 5.3 Verzeichnisstruktur

```
grundstuecke/
├── config.yaml                 # Zentren, Radius, Kriterien, Quellen + Suchauftrag-Filter, Budget, Modelle
├── overrides.yaml              # manuelle Freigaben/Ablehnungen/Korrekturen (Kapitel 8.6)
├── requirements.txt            # gepinnt
├── agent/
│   ├── run.py                  # Orchestrierung, CLI (--dry-run, --nur-sammeln, --discovery-erzwingen)
│   ├── sources/  email_alerts.py · zvg.py · bvvg.py · static_sites.py · web_discovery.py
│   ├── normalize.py · geo.py · prefilter.py · extract.py · verify.py
│   ├── dedup.py · lifecycle.py · budget.py · store.py · render.py · acceptance.py
├── prompts/                    # Laufzeit-Prompts (aus docs/grundstuecke/prompts/), mit Versionskennung
├── templates/                  # Jinja2, CSS, JS, Leaflet (vendored)
├── data/  plz_centroids.csv · gemeinden.csv · kreise_im_gebiet.json
└── tests/ fixtures/{mails,html,llm}/ · test_*.py
.github/workflows/grundstuecke.yml
```

Daten-Branch `grundstuecke-data`: `listings.json`, `pending.json`, `usage.json`, `state.json` (IMAP-UID-Wasserzeichen, letzte Discovery), `runs/…`.

---

## 6. Datenmodell

### 6.1 `listings.json`: einzige führende Datei

Ein Eintrag pro realem Objekt, **in allen Status** (auch `abgelehnt` und `zu_pruefen`). Weitere Sichten (Liste „Zu prüfen“, Abgelehnt-Bericht) werden daraus erzeugt.

```json
{
  "id": "gs_7f3a9c12",
  "status": "aktiv",
  "pruefung": {
    "ergebnis": "korrigiert",
    "zeitpunkt": "2026-10-08T07:14:02Z",
    "felder": { "grundstueck_m2": "korrekt", "preis_eur": "korrekt", "ort": "korrekt", "...": "..." },
    "hinweise": ["Fläche im Text 1.250 m², Extraktion 1.200 m² – korrigiert"],
    "modell_extraktion": "claude-haiku-5-5",
    "modell_pruefung": "claude-opus-5-5",
    "prompt_version": "extraktion@1.1 / gegenpruefung@1.1"
  },
  "kurztitel": "Baugrundstück 1.250 m², Ortsrandlage",
  "typ": "unbebaut",
  "gebaeudeart": null,
  "vermarktung": "kauf",
  "ort": "Dohma", "ortsteil": null, "plz": "01796", "ags": "14628080",
  "kreis": "Sächsische Schweiz-Osterzgebirge", "bundesland": "SN",
  "geo": { "lat": 50.93, "lon": 13.92, "genauigkeit": "gemeinde" },
  "naechstes_zentrum": { "name": "Pirna", "km": 4.1 },
  "grundstueck_m2": 1250, "grundstueck_nachweis": "beleg",
  "wohnflaeche_m2": null, "zimmer": null, "baujahr": null, "zustand": null,
  "preis_eur": 89000, "preis_auf_anfrage": false, "preis_pro_m2": 71.2,
  "verkehrswert_eur": null, "versteigerungstermin": null, "amtsgericht": null, "aktenzeichen": null,
  "erschliessung": "voll", "bebaubarkeit": "bplan",
  "provision": "3,57 % inkl. MwSt.", "anbieter_typ": "makler",
  "flags": ["hochwasser"],
  "kurzfazit": "Voll erschlossenes Baugrundstück im B-Plan-Gebiet, rund 4 km von Pirna.",
  "quellen": [
    { "portal": "immowelt", "url": "https://…", "externe_id": "2abc…", "liveness": "keine",
      "zuletzt_bestaetigt": "2026-10-08T07:10:00Z", "inhalt_hash": "sha256:…" }
  ],
  "preisverlauf": [ { "datum": "2026-10-01", "preis_eur": 95000 }, { "datum": "2026-10-08", "preis_eur": 89000 } ],
  "erstmals_gesehen": "2026-10-01T05:12:00Z",
  "nicht_gelistet_in_folge": 0,
  "ablehnung": null,
  "config_version": "2026-10-08a"
}
```

- **ID-Bildung:** `gs_` + die ersten 8 Hex-Zeichen von `sha256(portal + ":" + externe_id)` der **ersten** Quelle. Die ID bleibt bei Zusammenführungen stabil.
- `ablehnung`: `{grund, regel, inhalt_hash, config_version}`. Ein abgelehntes Inserat wird **neu bewertet**, wenn sich `inhalt_hash` oder `config_version` ändert.

### 6.2 Weitere Dateien

| Datei | Inhalt |
|---|---|
| `pending.json` | Warteschlange: gesammelte, noch nicht KI-verarbeitete Rohdatensätze (bei Budget-Drosselung oder Laufabbruch) |
| `usage.json` | Ledger je Aufruf: Datum, Zweck, Modell, Input-/Output-/Cache-Write-/Cache-Read-Tokens, Websuchen, Kosten in USD |
| `state.json` | IMAP-UID-Wasserzeichen je Postfachordner, Datum der letzten Discovery, letzter erfolgreicher Lauf |
| `runs/<zeitstempel>.json` | Laufprotokoll: Quellen (ok/Fehler), Anzahlen je Stufe, Prüfergebnisse, Kosten, Dauer, Plausibilitätsergebnis |

### 6.3 Zentrale Enums (verbindlich für Code und alle Prompts)

| Feld | Werte |
|---|---|
| `status` | `aktiv`, `zu_pruefen`, `abgelehnt`, `offline`, `archiv_unbestaetigt`, `archiv` |
| `pruefung.ergebnis` | `bestaetigt`, `korrigiert`, `unsicher`, `abgelehnt`, `ungeprueft` (nur im Sparmodus) |
| Feld-Prüfstatus | `korrekt`, `falsch`, `nicht_belegt`, `leer_korrekt`, `belegt_durch_filter` |
| `typ` | `unbebaut`, `bebaut`, `land_forst` |
| `gebaeudeart` | `efh`, `zfh`, `mfh`, `bauernhaus`, `villa`, `abriss`, `sonstiges`, `null` |
| `vermarktung` | `kauf`, `zwangsversteigerung`, `erbbaurecht`, `ausgeschlossen` (Miete, Pacht, Tausch, Mietkauf, Verrentung, Nießbrauch, Gesuch) |
| `grundstueck_nachweis` | `beleg`, `portalfilter`, `override` |
| `erschliessung` | `voll`, `teil`, `unerschlossen`, `unbekannt` |
| `bebaubarkeit` | `bplan`, `paragraph34`, `aussenbereich`, `unbekannt` |
| `anbieter_typ` | `privat`, `makler`, `bank`, `amtsgericht`, `oeffentliche_hand`, `unbekannt` |
| `flags` | `aussenbereich`, `denkmalschutz`, `hochwasser`, `erbbaurecht`, `altlasten`, `teilflaeche`, `flaeche_unklar`, `sanierungsbeduerftig`, `abrissobjekt` |
| `bundesland` | `SN`, `TH`, `ST`, `BY` |
| `liveness` | `vollliste`, `linkcheck`, `keine` |
| `geo.genauigkeit` | `adresse`, `ortsteil`, `gemeinde`, `plz`, `keine` |

---

## 7. Verarbeitungsschritte

1. **Sammeln:** Jede Quelle liefert Rohdatensätze `{quelle, url_kanonisch, externe_id, rohtext, abgerufen_am, suchauftrag_filter}`. Fehler einer Quelle brechen den Lauf nicht ab, die Quelle gilt in diesem Lauf als `fehlgeschlagen`. IMAP liest alle Mails mit **UID > Wasserzeichen**. Das Wasserzeichen wird **erst nach erfolgreichem Daten-Commit** fortgeschrieben, damit bei einem Abbruch keine Mails verloren gehen.
2. **Normalisieren:** Unicode NFKC (geschützte Leerzeichen, „²“, weiche Trennzeichen), Zahlen (`1.250 m²`, `ca. 0,3 ha` → 3.000, `1.200qm`), Preise (`VB`, `auf Anfrage`), Ort/PLZ, kanonische URL.
3. **Vorfilter (Code, ohne LLM):**
   - Identität über `(portal, externe_id)`: bekannt und `inhalt_hash` unverändert → nur `zuletzt_bestaetigt` aktualisieren.
   - Abgelehnt mit gleichem Hash und gleicher Config-Version → überspringen.
   - Eindeutige Ausschlüsse **per Regel**: Fläche im Text eindeutig < 1.000 m², Ausschlussbegriffe (Miete, Pacht, Mietkauf, Verrentung, Gesuch), Ort außerhalb Radius + 3 km.
   - Alles Übrige kommt nach `pending.json`.
4. **Budget-Freigabe:** `budget.py` entscheidet, wie viele Einträge aus `pending.json` verarbeitet werden (Kapitel 9.4). Reihenfolge: älteste zuerst.
5. **Extraktion (Claude):** nur für neue oder inhaltlich geänderte Inserate. Prompt: `prompts/extraktion.md`. Setzt der Extraktor `ist_relevant = false`, wird **nur dann direkt abgelehnt, wenn eine deterministische Regel den Grund bestätigt**. Sonst geht das Inserat in die Gegenprüfung.
6. **Gegenprüfung:** Kapitel 8.
7. **Dubletten** (nur über Portale hinweg, **nie innerhalb desselben Portals**):
   - **Kandidat:** gleiche Gemeinde (AGS), Fläche ±3 %, Preis ±5 %, gleicher `typ`. Fehlt die Fläche, braucht es zusätzlich Ortsteil- bzw. Straßengleichheit.
   - **Konfliktregel:** Der kleinste belegte Flächenwert und der niedrigste Preis gewinnen. Bei Flächenabweichung > 1 % wird das Flag `flaeche_unklar` gesetzt.
   - Danach läuft **Ebene 1 der Gegenprüfung erneut**.
8. **Lebenszyklus:** laut 3.4. Preisänderung → Preisverlauf + erneute Gegenprüfung.
9. **Lauf-Plausibilität:** Kapitel 8.5. Bei Fehler: **keine Lebenszyklus-Übergänge und keine Listing-Änderungen committen.** Nur Laufprotokoll und `pending.json` werden gespeichert, das IMAP-Wasserzeichen bleibt unverändert, ein GitHub-Issue wird angelegt.
10. **Speichern:** atomar schreiben, Commit auf `grundstuecke-data`, dann Wasserzeichen fortschreiben (zweiter Commit).
11. **Abnahme-Test vor Deployment** (`acceptance.py`, Kapitel 12 A3–A5). Bei Verstoß kein Deployment.
12. **Rendern & Deployen.**

---

## 8. Gegenprüfung (Qualitätssicherung)

Kein Inserat erscheint in der Hauptliste, ohne die Gegenprüfung bestanden zu haben.

### 8.1 Ebene 1: Deterministische Regeln (Code, kostenlos, vor **und** nach der KI-Prüfung)

| Prüfung | Regel → Ergebnis |
|---|---|
| Grundstücksfläche | `grundstueck_m2 >= 1000` mit Nachweis `beleg`, **oder** kein Wert + Suchauftrag-Filter ≥ 1.000 (`portalfilter`). Belegter Wert < 1.000 → **abgelehnt**. Kein Wert und kein Filter → **unsicher**. |
| Fläche vs. Wohnfläche | bebaut und `grundstueck_m2 <= wohnflaeche_m2 × 1,2` → **unsicher** |
| Teilfläche | Flag `teilflaeche` → **unsicher**, außer die verkaufte Fläche ist belegt ≥ 1.000 m² |
| Typ | `land_forst` und Schalter aus → **abgelehnt** |
| Vermarktung | `ausgeschlossen` → **abgelehnt**. Ausschlussbegriffe im Rohtext ohne klare Kaufangabe → **unsicher** |
| Gebiet | ≤ Radius → ok; ≤ Radius + 3 km → **unsicher**; darüber oder Ausland → **abgelehnt**; nicht geokodierbar → **unsicher** |
| Preis-Plausibilität | Nur bei `unbebaut` mit Preis: €/m² zwischen 3 und 1.500, sonst **unsicher**. Bei `bebaut` keine €/m²-Regel. |
| Preis fehlt | Erlaubt, wenn `preis_auf_anfrage` belegt ist oder es sich um eine Zwangsversteigerung mit belegtem Verkehrswert handelt |
| Link | `https://`, Domain gehört zur Quelle (Allowlist in Config), keine Tracking-Parameter |
| Pflichtfelder | `url`, Ort (Gemeinde oder PLZ), `typ`, `vermarktung` |

### 8.2 Ebene 2: Unabhängige KI-Gegenprüfung (Claude)

- **Separater** Claude-Aufruf mit eigenem Prompt (`prompts/gegenpruefung.md`). Er erhält **nur** den Rohtext, das extrahierte JSON und gegebenenfalls den Suchauftrag-Filter, **nicht** den Extraktions-Prompt.
- Es soll ein **anderes, mindestens gleich starkes Modell** als bei der Extraktion sein.
- Geprüft wird über ein **Objekt mit festen Schlüsseln** (alle Pflicht):
  - **Kernfelder:** `grundstueck_m2`, `preis_eur`, `preis_auf_anfrage`, `verkehrswert_eur`, `ort`, `typ`, `vermarktung`, `wohnflaeche_m2`
  - **Nebenfelder:** `gebaeudeart`, `erschliessung`, `bebaubarkeit`, `flags_kritisch` (aussenbereich, erbbaurecht, teilflaeche)
- Jedes Feld bekommt einen Status und ein **wörtliches Belegzitat** (bei `leer_korrekt` ist kein Zitat nötig).
- Das Gebiet beurteilt der Prüfer **nicht**. Das macht ausschließlich der Code.
- Der Prüfer prüft zusätzlich Kurztitel und Kurzfazit auf erfundene Aussagen und personenbezogene Daten.

### 8.3 Ebene 3: Code-Prüfung der KI-Gegenprüfung

1. **Zitat wörtlich vorhanden:** Das Zitat muss nach Normalisierung (NFKC, Leerzeichen zusammenfassen, Groß-/Kleinschreibung ignorieren) im Rohtext vorkommen.
2. **Zahl passt:** Die im Zitat genannte Zahl (normalisiert inkl. ha/a/Tausenderpunkt) entspricht dem (korrigierten) Feldwert, Toleranz 0,5 %.
3. **Richtige Fläche:** Ein Zitat für `grundstueck_m2` darf keine Begriffe wie „Wohnfl“, „Wfl“, „Nutzfl“ oder „Gewerbefl“ enthalten.
4. **Konsistenz:** Meldet der Prüfer `bestaetigt`, obwohl ein Feld `falsch` oder `nicht_belegt` ist, wird das Ergebnis auf `unsicher` gesetzt.
5. **Korrigierte Werte** werden gegen Typen und Enums validiert. Danach läuft Ebene 1 erneut.

**Einheitliche Folgen:**

| Fall | Folge |
|---|---|
| **Kernfeld** ohne gültigen Beleg (Status `nicht_belegt`, falsches oder fehlendes Zitat, Zahl passt nicht) | Ergebnis mindestens **unsicher**. Ausnahme: `grundstueck_m2` mit `belegt_durch_filter`. |
| **Nebenfeld** ohne gültigen Beleg | Wert wird auf `unbekannt` zurückgesetzt bzw. das Flag entfernt. Das Inserat bleibt veröffentlichbar, die Angabe wird nicht angezeigt. |
| Kurztitel/Kurzfazit beanstandet | Wird leer gelassen, nicht veröffentlicht |

### 8.4 Ebene 4: Laufende Nachprüfung
- Quellen mit `vollliste`: Ist ein Inserat bei der Quelle nicht mehr gelistet (nur in erfolgreichen Läufen gezählt), steigt `nicht_gelistet_in_folge`.
- Quellen mit `linkcheck`: Link 1× täglich prüfen. Nur 404/410 bzw. eine Weiterleitung auf eine Such- oder Startseite zählt als „weg“.
- Quellen mit `keine`: Anzeige „zuletzt bestätigt vor X Tagen“, nach 45 Tagen `archiv_unbestaetigt`.
- Ändert sich der Inhalt (Hash), laufen Extraktion und Gegenprüfung erneut.

### 8.5 Lauf-Plausibilität (Schutz vor Fehlveröffentlichung)

Ein Lauf gilt als fehlerhaft (Folgen siehe Kapitel 7, Schritt 9), wenn:

- **Quellen fallen aus:** Mindestens 3 Quellen sind konfiguriert, und mehr als 50 % davon schlagen fehl. Discovery zählt dabei nicht mit.
- **Aktiver Bestand bricht ein:** Mindestens 20 Inserate sind aktiv, und davon würden in diesem Lauf mehr als 40 % auf offline oder Archiv wechseln.
- **Parser liefern keine Daten:** Mindestens 5 neue Datensätze kommen aus einer Quelle, und bei mehr als 50 % davon sind die **Pflichtfelder nicht lesbar**. Das deutet auf ein geändertes Mailformat hin.

Hinweis: Viele `unsicher`-Ergebnisse allein sind **kein** Fehlergrund.

### 8.6 Manuelle Prüfung: `overrides.yaml`

```yaml
- id: gs_7f3a9c12
  aktion: freigeben           # freigeben | ablehnen | korrigieren
  werte: { grundstueck_m2: 1400 }   # nur bei korrigieren
  begruendung: "Exposé geprüft, Fläche 1.400 m² laut Flurkarte"
  datum: 2026-10-09
```

- Die Pipeline übernimmt Overrides in jedem Lauf.
- Auch freigegebene oder korrigierte Inserate durchlaufen **Ebene 1**. Ein Override kann also z. B. kein Inserat mit 800 m² freigeben.
- Eine manuelle Korrektur der Fläche setzt `grundstueck_nachweis: override`.

### 8.7 Sichtbarkeit
- **Hauptliste:** `aktiv` (Ergebnis `bestaetigt` oder `korrigiert`, Letzteres mit Badge)
- **Reiter „Zu prüfen“:** `zu_pruefen`, mit Grund, Direktlink und der fertigen `overrides.yaml`-Zeile zum Kopieren
- **Reiter „Offline / Archiv“:** `offline`, `archiv_unbestaetigt`, `archiv`
- **Abgelehnte Inserate:** nicht auf der Seite, aber im Laufprotokoll mit Grund
- **Wöchentlicher Stichproben-Bericht** (als GitHub-Issue): 5 zufällige aktive Inserate mit Rohtext-Auszug und Werten, dazu 5 zufällige abgelehnte Inserate mit Grund (prüft, ob gute Treffer verloren gehen)

---

## 9. Zeitplan, Token-Budget und Kosten

### 9.1 Zeitplan
- **Cron alle 2 Stunden von ca. 06 bis 22 Uhr** (UTC: `15 4-20/2 * * *`, 9 Läufe/Tag). Sammeln kostet nichts. Ob und wie viel KI-Verarbeitung stattfindet, entscheidet das Budget. So läuft der Agent „so oft, wie die Tokens es erlauben“.
- Die GitHub-Actions-Minuten liegen bei ca. 9 × 3 min × 30 ≈ 800 min/Monat. Das ist bei öffentlichen Repos kostenlos und liegt bei privaten Repos innerhalb von 2.000 Freiminuten.
- **Discovery:** 1× täglich, beim ersten Lauf des Tages. Das Kriterium ist das Datum der letzten Discovery in `state.json`, nicht die Uhrzeit.
- Manuell über `workflow_dispatch` (mit Optionen `nur_sammeln`, `discovery_erzwingen`).
- Achtung: In öffentlichen Repos deaktiviert GitHub geplante Workflows nach 60 Tagen ohne Repository-Aktivität. Der Workflow erkennt das und meldet es im Stichproben-Issue.

### 9.2 Modellprofile (in `config.yaml` umschaltbar)

Preise laut Anthropic-Preisliste (Stand Oktober 2026, pro 1 Mio. Tokens Input / Output):

| Modell | Input | Output |
|---|---|---|
| Claude Opus 5.5 | 4 $ | 20 $ |
| Claude Sonnet 5.5 | 2 $ | 10 $ |
| Claude Haiku 5.5 | 0,10 $ | 0,50 $ |

Websuche: 10 $ pro 1.000 Suchen. Hinweise: Opus 5.5 denkt immer mit (Thinking ist nicht abschaltbar, Steuerung über `effort`), und Thinking wird als Output abgerechnet. Die Websuche (`web_search_20260209`) läuft nicht mit Haiku, Discovery braucht daher Opus oder Sonnet.

| Profil | Extraktion | Gegenprüfung | Discovery |
|---|---|---|---|
| **Qualität** | Opus 5.5 (`low`) | Opus 5.5 (`medium`) | Opus 5.5 (`medium`) |
| **Ausgewogen** | Haiku 5.5 (`low`) | Opus 5.5 (`medium`) | Opus 5.5 (`medium`) |
| **Sparsam** | Haiku 5.5 (`low`) | Sonnet 5.5 (`medium`) | Sonnet 5.5 (`medium`) |

Alle Profile erfüllen das Unabhängigkeitsprinzip (anderes oder stärkeres Prüfmodell, eigener Prompt). Ausnahme ist „Qualität“: Dort prüft dasselbe Modell, aber mit eigenem Prompt und höherem `effort`.

### 9.3 Kostenschätzung

**Annahmen** (konservativ, inkl. Thinking-Tokens auf der Output-Seite):

| Schritt | Menge | Input | Output |
|---|---|---|---|
| Extraktion | 25 neue oder geänderte Inserate pro Tag | 4.000 Tokens je Inserat | 1.000 Tokens je Inserat |
| Gegenprüfung | 25 Inserate pro Tag | 5.000 Tokens je Inserat | 2.000 Tokens je Inserat |
| Discovery | 1× täglich, 8 Suchen | 60.000 Tokens | 6.000 Tokens |

Dazu kommt ein Aufschlag von 10 % für Wiederholungen.

| Profil | Rechnung pro Tag | ≈ pro Tag | **≈ pro Monat** |
|---|---|---|---|
| Qualität | Extraktion 0,90 + Gegenprüfung 1,50 + Discovery 0,44 = 2,84 × 1,1 | 3,12 $ | **≈ 95 $** |
| Ausgewogen | 0,02 + 1,50 + 0,44 = 1,96 × 1,1 | 2,16 $ | **≈ 65 $** |
| Sparsam | 0,02 + 0,75 + 0,26 = 1,03 × 1,1 | 1,13 $ | **≈ 35 $** |

Einzelwerte:
- Extraktion mit Opus: 25 × (4.000 × 4 + 1.000 × 20) / 1 Mio. = 0,90 $
- Extraktion mit Haiku: 25 × (4.000 × 0,1 + 1.000 × 0,5) / 1 Mio. ≈ 0,02 $
- Gegenprüfung mit Opus: 25 × (5.000 × 4 + 2.000 × 20) / 1 Mio. = 1,50 $
- Gegenprüfung mit Sonnet: 25 × (5.000 × 2 + 2.000 × 10) / 1 Mio. = 0,75 $
- Discovery mit Opus: 0,08 (Suchen) + 0,24 (Input) + 0,12 (Output) = 0,44 $
- Discovery mit Sonnet: 0,08 + 0,12 + 0,06 = 0,26 $

Prompt-Caching des statischen System-Prompts senkt die Input-Kosten zusätzlich etwas. Die Schätzung ist bewusst ohne Caching gerechnet.

> **Pflicht vor der Budget-Festlegung:** In Phase 1 werden 20 echte Inserate durch Extraktion und Gegenprüfung geschickt. Die gemessenen Token-Zahlen ersetzen die Annahmen oben. Zudem schwankt die Zahl neuer Inserate pro Tag. Sie wird in den ersten 2 Wochen gemessen.

### 9.4 Budget-Steuerung

**Parameter (`config.yaml`):** `monat_usd`, `lauf_max_usd`, `tag_max_inserate`.

**Prognose:** `prognose = ausgegeben_monat + Ø_Tageskosten_letzte_7_Tage × verbleibende_Tage`. In den ersten 7 Tagen wird stattdessen die Schätzung aus 9.3 verwendet.

| Stufe | Bedingung | Verhalten |
|---|---|---|
| **Normal** | prognose ≤ 90 % von `monat_usd` | volles Profil, Discovery an |
| **Gedrosselt** | prognose > 90 % | Discovery aus, Gegenprüfung mit dem Prüfmodell des Profils „Sparsam“ und `effort: low`, höchstens `tag_max_inserate`/2 pro Tag. Überschüssige Inserate bleiben in `pending.json`. |
| **Sparmodus** | **ausgegeben** ≥ `monat_usd` (harte Sperre) | **keine** LLM-Aufrufe. Neue Inserate werden per Regex erfasst, als `ungeprueft` in „Zu prüfen“ geführt und im nächsten Monat nachverarbeitet. |

- **Vor jedem einzelnen Aufruf** wird eine Worst-Case-Prüfung gemacht: `ausgegeben_lauf + (geschätzte Input-Tokens × Input-Preis + max_tokens × Output-Preis)` muss ≤ `lauf_max_usd` sein. Sonst endet der Lauf sauber, der Rest bleibt in `pending.json`.
- Das Ledger erfasst `usage.input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens` und `server_tool_use.web_search_requests`.
- Die Webseite zeigt in der Fußzeile den Modus und die Kosten des laufenden Monats.

---

## 10. Webseite

### 10.1 Aufbau
1. **Kopf:** „Grundstücks-Monitor Sachsen & Thüringen“, Stand der letzten Aktualisierung, Betriebsmodus
2. **Kennzahlen:** aktive Inserate · neu (24 h) · Preis gesenkt (7 Tage) · Ø €/m² **unbebauter** Grundstücke je Region
3. **Filter** (bleibt beim Scrollen sichtbar, mobil als ausklappbares Panel): Zentrum (Mehrfachauswahl), Bundesland, bebaut/unbebaut, Fläche von–bis, Preis bis, €/m² bis, nur neue, Quelle, Zwangsversteigerungen ein/aus, max. Entfernung
4. **Sortierung:** neueste zuerst (Standard), Preis ↑↓, €/m² ↑, Fläche ↓, Entfernung ↑
5. **Ansichten:** **Kacheln** (Standard, mobil), **Tabelle** (Desktop, sortierbar), **Karte** (Leaflet, Marker nach Typ, Radius-Kreise)
6. **Reiter:** Aktiv · Zu prüfen · Offline/Archiv
7. **Fußzeile:** Quellen, Haftungshinweis, Impressum/Datenschutz (Hauptdomain), OSM-Namensnennung, Modus und Monatskosten

### 10.2 Nicht-funktionale Anforderungen
- Ladezeit < 1,5 s auf dem Handy (4G). Daten als eine JSON-Datei, gefiltert wird im Browser.
- Mobil zuerst, keine horizontale Scrollleiste ab 360 px, Hell- und Dunkelmodus, Tastaturbedienung, ausreichende Kontraste
- `noindex, nofollow`; Filterzustand in der URL
- **Sicherheit (XSS):** Alle Inseratsdaten gelten als nicht vertrauenswürdig.
  - Jinja2 läuft mit Autoescape, im JS wird ausschließlich `textContent` verwendet (kein `innerHTML` mit Daten).
  - `href` nur `https://` auf Domains der Quellen-Allowlist, mit `rel="noopener noreferrer"`.
  - Content-Security-Policy per `<meta>`: keine Inline-Skripte, nur eigene Skripte und das Kachel-Host-Ziel.
- **Datenschutz:** Leaflet wird selbst gehostet. OSM-Kacheln übertragen die IP-Adresse der Besucher an den Kachelserver. Das wird in der Datenschutzerklärung erwähnt, alternativ wird die Karte erst nach Klick geladen („Karte laden“). Die OSM Tile Usage Policy wird beachtet.
- **Zugriff (Entscheidung):**
  - **Öffentlich:** GitHub Pages; Aktenzeichen und Straßen bei Zwangsversteigerungen werden ausgeblendet.
  - **Geschützt:** Cloudflare Pages + Cloudflare Access (E-Mail-Login). Dann muss auch das **Repository privat** sein, sonst sind die Daten über den Daten-Branch trotzdem öffentlich. Empfehlung dafür: eigenes privates Repo.

---

## 11. Betrieb und Sicherheit

- **Secrets:**
  - immer: `ANTHROPIC_API_KEY`, `IMAP_HOST`, `IMAP_USER`, `IMAP_PASSWORD` (App-Passwort eines **eigenen** Postfachs, nur Lesezugriff, wenn der Anbieter das unterstützt), `KONTAKT_EMAIL` (für User-Agent und Nominatim)
  - bei Cloudflare: zusätzlich `CLOUDFLARE_API_TOKEN` und `CLOUDFLARE_ACCOUNT_ID`
  - Secrets nie loggen.
- **Workflow-Rechte:** `contents: write`, `issues: write`; bei GitHub Pages zusätzlich `pages: write` und `id-token: write`.
- **Prompt-Injection-Schutz:**
  - Inseratstexte stehen in `<inserat>`-Tags und gelten als Daten.
  - Extraktion und Gegenprüfung laufen ohne Tools; Discovery nur mit Websuche.
  - LLM-Ausgaben werden schema-validiert, und der Code prüft alle sicherheitsrelevanten Entscheidungen (Fläche, Gebiet, Links).
- **API-Robustheit:**
  - `stop_reason` prüfen: `refusal` → Inserat `unsicher`; `max_tokens` → ein Wiederholungsversuch mit höherem Limit, sonst `unsicher`; `pause_turn` bei Discovery → fortsetzen.
  - Die SDK-Retries für 429/5xx sind aktiv.
- **Fehlermeldungen:** fehlgeschlagener Lauf oder Plausibilitätsfehler → GitHub-Issue (ein offenes Issue pro Fehlerart wird aktualisiert, keine Duplikate).
- **DNS:** `CNAME grundstuecke → <ziel>` (GitHub Pages bzw. Cloudflare Pages), HTTPS erzwingen.
- **Respektvoller Abruf:** `robots.txt`, User-Agent mit Kontaktadresse, Rate-Limit, ETag/Last-Modified.

---

## 12. Abnahmekriterien

| # | Kriterium | Messung |
|---|---|---|
| A1 | Seite unter der Subdomain per HTTPS erreichbar | manuell |
| A2 | Mindestens 4 erfolgreiche Läufe mit KI-Verarbeitung pro Tag im Normalbetrieb | Laufprotokolle |
| A3 | **0 Inserate in der Hauptliste mit belegter Grundstücksfläche < 1.000 m² oder ohne Flächennachweis** | `acceptance.py`, Deploy-Gate |
| A4 | **0 Inserate in der Hauptliste außerhalb des Radius** | `acceptance.py`, Deploy-Gate |
| A5 | Jedes Inserat der Hauptliste hat eine gültige kanonische `https`-URL einer zugelassenen Domain und das Prüfergebnis `bestaetigt`/`korrigiert`. Erreichbarkeit wird nur bei `linkcheck`-Quellen geprüft. | `acceptance.py`, Deploy-Gate |
| A6 | Stichprobe von 20 Inseraten: ≥ 95 % der Kernfelder korrekt | manuell gegen Original |
| A7 | Dubletten über Portale hinweg werden zusammengeführt, keine falschen Zusammenführungen (Stichprobe ≥ 90 %) | manuell |
| A8 | Monatskosten ≤ `monat_usd`, Drosselung und Sparmodus nachweislich wirksam | Tests mit simuliertem Ledger |
| A9 | Fehlerhafter Lauf (kaputte Quelle, geändertes Mailformat) führt nicht zu leerer oder falscher Seite und nicht zu verlorenen Mails | Tests mit manipulierten Fixtures |
| A10 | Mobil < 1,5 s, keine horizontale Scrollleiste, Lighthouse ≥ 90 (Performance, Accessibility) | Lighthouse |
| A11 | Testabdeckung der Kernmodule (`normalize`, `geo`, `prefilter`, `verify`, `dedup`, `lifecycle`, `budget`, `acceptance`) ≥ 80 % | `pytest --cov` |
| A12 | XSS-Test: ein Inserat mit `<script>`, `javascript:`-URL und HTML im Titel wird harmlos dargestellt | automatischer Test |

---

## 13. Umsetzungsphasen

| Phase | Inhalt |
|---|---|
| **0: Vorbereitung (Betreiber)** | Eigenes Postfach + App-Passwort; Suchaufträge je Portal mit **Grundstücksfläche ≥ 1.000 m²** und Region anlegen und in der Config dokumentieren; **mindestens 3 echte Alarm-Mails je Portal als `.eml`** bereitstellen; DNS; API-Key; Nutzungsbedingungen ZVG-Portal prüfen; Entscheidungen aus Kapitel 14 |
| **1: MVP** | Mail-Quelle (2 Portale), Normalisierung, Geokodierung, Vorfilter, Extraktion, **Gegenprüfung Ebene 1–3**, Budget-Ledger mit harter Sperre, `listings.json`, `pending.json`, einfache Kachel-Seite mit XSS-Schutz, Abnahme-Gate, Cron, **Kostenmessung mit 20 echten Inseraten** |
| **2: Ausbau** | Weitere Portale, ZVG, BVVG, statische Quellen, Dubletten, Lebenszyklus, Preisverlauf, Tabelle + Karte, Lauf-Plausibilität, Drosselstufen, `overrides.yaml`, Reiter „Zu prüfen“ |
| **3: Komfort** | Discovery, Anfangsbestand-Lauf, Bodenrichtwert-Vergleich (BORIS Sachsen / Geoportal Thüringen), Hochwasser-Hinweise, Benachrichtigungen, wöchentlicher Stichproben-Bericht |

---

## 14. Offene Entscheidungen (vor der Umsetzung klären)

1. **Subdomain:** `grundstuecke.warchhold.de`? Genaue Hauptdomain?
2. **Öffentlich oder geschützt?** Bei „geschützt“: eigenes privates Repo + Cloudflare (Empfehlung aus rechtlicher Sicht)
3. **Modellprofil und Monatsbudget:** Qualität (~95 $), Ausgewogen (~65 $) oder Sparsam (~35 $)? Endgültig erst nach der Kostenmessung in Phase 1.
4. **Radius je Zentrum:** 20 km? **Sollen Jena, Erfurt und die Nachbarkreise in Sachsen-Anhalt und Bayern** einbezogen werden, soweit sie im Radius liegen?
5. **Land-/Forst-/Freizeitflächen** aufnehmen?
6. **Zwangsversteigerungen** anzeigen? (Standard: ja, gekennzeichnet)
7. **Portal-Scraping / Exposé-Abruf (Stufe D)** ausschließen? (Empfehlung: ja)
8. **Benachrichtigungen** bei neuen Treffern (E-Mail/Telegram)? Ab welchen Kriterien?
9. **Repository:** Unterordner hier oder eigenes Repo? (Bei geschütztem Betrieb: eigenes privates Repo)
10. **Anfangsbestand:** Weiterleitung bestehender Inserate durch den Betreiber (Kapitel 4.4) oder verzichten?

---

## Anhang

| Datei | Inhalt | Version |
|---|---|---|
| `prompts/UMSETZUNG.md` | Prompt für Claude Code zur Implementierung, mit Selbst- und Gegenprüfung | 1.1 |
| `prompts/extraktion.md` | Laufzeit-Prompt Datenextraktion | `extraktion@1.1` |
| `prompts/gegenpruefung.md` | Laufzeit-Prompt Gegenprüfung | `gegenpruefung@1.1` |
| `prompts/discovery.md` | Laufzeit-Prompt Websuche | `discovery@1.1` |

Bei Widersprüchen gilt diese Spezifikation vorrangig vor den Prompts, insbesondere Kapitel 6.3 (Enums).
