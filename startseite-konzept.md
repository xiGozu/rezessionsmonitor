# Öffentliche Startseite für warchhold.com: Konzept

Begleitdokument zu `startseite-prototyp.html`. Stand der übernommenen Zahlen: 28.09.2026 (Abruf 19:30–20:20 Uhr, Europe/Berlin).

## 1. Ausgangslage

`https://warchhold.com/` ist heute das Bedienpult des Handelssystems (Seitentitel „Edge Lab“). Ein Besucher sieht beim Aufruf:

- **Kopfzeile:** Live-Kurs, „BID“, „ASK“, „SPREAD“, „SYSTEM: OK“, „MARKT: OFFEN“, einen Umschalter „DAX 40 / JAPAN 225“ und „EINLOGGEN“.
- **Performance-Kasten:** „Heute -14.4p · Gesamt -12.3p · Win Rate 31.2% · Profit Factor 1.00 · 80 Trades“ mit Reitern „LIVE / ALLE / SHADOW / LOG“.
- **„STRATEGIE-AUSWERTUNG“:** interne Kürzel („IEF“, „REF“, „ABAE“, „FHRG“, „RNRSE“, „PPB“, „RNREL“, „PPFVS“, „EMAG“, „PLR“, „SAB“, „TO“) mit Punkten je Tag, Woche, Monat und Jahr.
- **Chart:** 5-Minuten-DAX-Chart mit EMA5/8/13, Bollinger-Bändern und BUY/SELL-Markern.
- **„SIGNAL-MATRIX“:** „TREND / STRUKTUR / VOLA / LONG / SHORT“ × „M1…H1“ mit Werten wie „MID“, „RNG“, „WCH“, „NO“.
- **„ORDERS“:** Zeit, Richtung, Modus „LIVE“, Lot „0.5“, Entry, SL/Trail und P&L in Punkten **und Euro** (z. B. „-91.4p / -45.70€“), Aktion „BROKER_SL_TP“.
- **Weitere Kästen:** „LIVE RUNNER V2“ mit 14 Strategie-IDs, „DAILY BRIEFING“, „KI-ANALYSE“, „WIRTSCHAFTSKALENDER“, „SYSTEM-MONITOR“ (Host, CPU, RAM, Disk, DB-WAL, Kern-Prozesse, „Stale aufraeumen“).

Das Werkzeug ist für den Betreiber gebaut, nicht für Besucher. Wer die Seite aufruft, versteht weder, was das Projekt ist, noch welche Zahlen belastbar sind. Die Orderliste mit „LIVE“ und Euro-Beträgen legt außerdem Echtgeldhandel nahe. Das widerspricht „Ein Konto mit echtem Geld besteht nicht“ auf `/algostrategien/ueber-mich` (siehe `review-hub-extern.md`, Befund 1).

**Ziel:** eine eigene öffentliche Startseite, die in einem Satz erklärt, worum es geht, ehrlich mit Unsicherheit umgeht und in die bestehende Forschungsdokumentation führt. Das Bedienpult bleibt unverändert, zieht aber hinter eine Anmeldung.

---

## 2. Seitenaufbau

| # | Bereich | Zweck | Inhalt im Prototyp |
|---|---|---|---|
| 0 | Prototyp-Hinweis | nur im Prototyp | Erklärt die Etiketten „Beispielwert“ und die Quellenangaben. Entfällt im Produktivbetrieb. |
| 1 | Kopfzeile | Orientierung | Wortmarke „Warchhold Research“, fünf Navigationspunkte (Methodik, Ergebnisse, Evidenzstand, Strategien, Über das Projekt), Umschalter Hell/Dunkel. **Kein** Login-Knopf, **keine** Kursleiste. |
| 2 | Kopfbereich | „Was ist das hier?“ in einem Satz | H1 als ganzer Satz: „Eine offene, autonome Forschungsplattform für algorithmischen DAX-Handel, die ihre Fehlschläge genauso zeigt wie ihre Funde.“ Darunter zwei Sätze zum Vorgehen und drei Lesegrundsätze (nur DEMO; n und Einstufung; Backtest ist kein Vorteil). Rechts ein kleiner DAX-Chart, sichtbar als **Platzhalter** markiert. |
| 3 | Drei Einstiege | Hauptwege | Karten „Wie wir forschen“, „Was wir gefunden haben (und was nicht)“, „Aktueller Evidenzstand“, jeweils mit einer belegten Kennzahl und Quelle. |
| 4 | Spielgeldbetrieb (DEMO) | Handelsergebnisse, nüchtern | Vier Kennzahlen (Trades, Gesamtbilanz in Punkten, aktive Slots, Strategien im Katalog) und eine Tabelle der vier aktiven Strategien in Klartext: Name, Idee in einem Satz, Status, n, Profit-Faktor mit Einstufung bzw. 90-%-Bereich, Bilanz in Punkten. |
| 5 | Was wir gefunden haben (und was nicht) | Negatives gleichrangig | Vier Kennzahlen der Gesamtbilanz (195 / 77 / 18 / 1) und die fünf jüngsten Versuche mit ihrem Urteil in Klartext. Die negativen Urteile stehen oben. |
| 6 | Aktueller Evidenzstand | Was ist belegt, woraus | Tabelle Hypothesen je Evidenzklasse, dazu drei Hinweise: „0 von 9“ vorwärts überlebt, „2 von 79“ entscheidbar, Kalibrierung „−0,61“ mit Intervall. |
| 7 | Wie wir forschen | Methode in vier Schritten | Vorher festlegen · realistisch rechnen · gegen Zufall prüfen · auf neuen Daten bestehen. Jeder Schritt mit einem wörtlichen Zitat aus der Methodik. |
| 8 | Fußzeile | Vertrauen, Recht, Betreiber | Haftungshinweis, Links zu Impressum und Datenschutz (anzulegen), Unterstützen und Über das Projekt. Unauffälliger Link „Bedienpult (nur für den Betreiber, Anmeldung erforderlich)“. |

Die Reihenfolge folgt der Frage, die ein Fremder stellt: *Was ist das? → Wo steige ich ein? → Was kommt heraus? → Was ist gescheitert? → Wie belastbar ist das? → Wie wird gearbeitet?* Handelsergebnisse stehen bewusst erst an dritter Stelle, nach der Einordnung.

---

## 3. Designentscheidungen und Begründung

**Ton und Typografie**
- **Serifen-Überschriften (Source Serif 4), serifenlose Grundschrift (Source Sans 3).** Das erinnert an einen Forschungsbericht, nicht an ein Trading-Terminal. Beide Schriften stammen aus einer Familie, laden als einzige externe Abhängigkeit über Google Fonts und haben ausgebaute deutsche Glyphen und Tabellenziffern.
- **Zahlen immer in der serifenlosen Schrift**, in Tabellen mit Tabellenziffern (`tabular-nums`), damit Spalten fluchten. Große Einzelwerte nutzen proportionale Ziffern.
- **Die H1 ist der verlangte Satz selbst**, kein Slogan. Wer nur die Überschrift liest, weiß, was die Seite ist.

**Farbe**
- **Warmes Papierweiß (#fbfaf7) bzw. warmes Fast-Schwarz (#121211)** statt reinem Weiß oder Terminal-Schwarz. Das wirkt ruhig und unterscheidet die Seite sichtbar vom Bedienpult.
- **Genau eine Akzentfarbe (Blau)** für Datenlinien und Links. Kein Grün für Gewinne, kein Rot-Grün-Kontrast. Positive Ergebnisse sollen nicht „feiern“.
- **Rot nur für Negativbefunde**, und nie allein: immer mit Text („DEMO läuft, bisher negativ“) und, in der Versuchsliste, mit eigener Form (Raute = kein Vorteil, Kreis = kein Mehrwert, Quadrat = in-sample besser). So ist nichts nur über Farbe codiert.
- **Die Farben der Datenmarken sind geprüft.** Grundlage ist das Palettenprüfskript der Dataviz-Richtlinien. Hell: #1c5cab / #c0392b auf #fbfaf7. Dunkel: #3987e5 / #e66767 auf #161614. Alle Prüfungen bestanden (Helligkeitsband, Farbsättigung, Unterscheidbarkeit bei Farbsehschwäche, Kontrast ≥ 3:1).
- **Dunkelmodus mit eigenen Werten**, nicht automatisch invertiert. Er folgt der Systemeinstellung und lässt sich per Knopf umschalten (die Wahl wird lokal gespeichert, die Seite funktioniert auch ohne Speicher).
- **Keine Farbverläufe, keine Emojis, keine Illustrationen.**

**Zahlen und Ehrlichkeit**
- **DEMO-Ergebnisse nur in Punkten.** Punkte hängen nicht von der Positionsgröße ab und werden nicht als Kontostand gelesen. Euro-Beträge erscheinen nirgends.
- **Jede Handelskennzahl trägt n und Einstufung.** Explorative Werte (n < 20) werden wie auf `/strategien` durchgestrichen und grau gezeigt, mit dem Satz „kein Bereich berechnet“. Vorläufige Werte (n < 50) bekommen das Etikett „vorläufig“ und eine kleine Intervallgrafik mit Bezugslinie bei 1. Ein Leser sieht sofort, ob der Bereich die 1 einschließt.
- **Jede übernommene Zahl hat eine sichtbare Quelle** (Seitenlink und Stand). Nicht belegte Zahlen tragen das gestrichelte Etikett „Beispielwert“.
- **Kein Podium, keine Rangliste, keine Superlative.** Die aktiven Strategien stehen in der Reihenfolge der Quellseite (Status, dann Profit-Faktor), ohne Platzierungen. Der Text vermeidet Wörter wie „beste“, „stärkste“, „robust“ und „bestätigt“ ohne Evidenzklasse.
- **Die unbequemen Zahlen stehen vorn:** „0 von 9“, „2 von 79“ und die Kalibrierung „−0,61“ (Backtests lagen systematisch zu hoch). Im heutigen Hub stehen sie nur auf `/research/evidenz` (siehe `audit-zahlen.md`, Rang 7).
- **Die Kalibrierung ist als „explorativ, n = 8“ gekennzeichnet.** So misst die Seite sich selbst mit ihrem eigenen Maßstab.
- **Keine In-sample-Profit-Faktoren auf der Startseite.** Die Versuchsliste zeigt Urteile, keine „bestes Trial: PF 1.97“-Werte. Solche Zahlen gehören auf die Detailseiten, wo sie mit Bereich und Mehrfachtest-Kontext stehen.

**Chart**
- **Ein einzelner, kleiner Linienchart** statt Kerzen, Indikatoren und Signalen. Er gibt Kontext („das ist ein DAX-Projekt“), ist aber kein Arbeitswerkzeug.
- **Deutlich als Platzhalter markiert**, dreifach: gestricheltes Etikett „Platzhalter“, schraffierter Hintergrund mit Wasserzeichen „PLATZHALTER · KEINE KURSDATEN“ und eine Beschriftung darunter. Der Tooltip beim Überfahren nennt die Uhrzeit und „Platzhalter, kein Kurs“.
- **Keine Preisskala.** Solange keine echten Daten fließen, sollen keine Kursniveaus suggeriert werden.
- **Später:** verzögerte Kurse (z. B. 15 Minuten) aus einer öffentlichen Quelle, ohne Orders, Signale oder Positionen.

**Layout und Zugänglichkeit**
- Maximale Breite 1.120 px, großzügige Abstände, dünne Trennlinien statt schwerer Kästen.
- **Ab 390 px nutzbar** (geprüft bei 390 und 1.440 px, hell und dunkel, ohne horizontales Scrollen). Die Navigation bricht in eine zweite Zeile um. Die Strategietabelle wird zu Karten mit einem 2×2-Raster (Status, n, Profit-Faktor, Bilanz). Die Kennzahlkacheln bleiben zweispaltig.
- Sprunglink „Zum Inhalt springen“, sichtbare Fokusrahmen, `aria-label`/`desc` für alle Grafiken, `prefers-reduced-motion` beachtet.
- **Kein JavaScript für Inhalte.** Nur der Umschalter und der Platzhalter-Chart nutzen JavaScript. Alle Zahlen stehen im HTML und sind ohne Skript lesbar.

---

## 4. Datenherkunft

### 4.1 Wörtlich übernommene Werte

| Element auf der Startseite | Wert im Prototyp | Quelle (Seite, Abschnitt) | Wörtlich dort |
|---|---|---|---|
| Einstieg „Wie wir forschen“ | 21 Tage | `/algostrategien/research/methodik`, „Test-Setup“ | „Research-Backtests enden 21 Tage vor heute (Daten-Embargo)“ |
| Einstieg „Was wir gefunden haben“ | 77 von 195 | `/algostrategien/ueber-mich`, „Bisherige Bilanz“ | „195 untersuchte Hypothesen“, „77 davon ohne nachweisbaren Vorteil“ |
| Einstieg „Evidenzstand“ und Hinweis | 0 von 9 | `/algostrategien/research/evidenz` | „Vorwärts mehrmonatig geprüft: 9 Kandidaten, davon überlebt: 0.“ |
| DEMO-Kachel Trades | 1.006 | `/algostrategien/strategien`, Kopfzahlen | „Abgeschlossene DEMO-Trades 1.006“ |
| DEMO-Kachel aktive Slots | 4 | `/algostrategien/strategien` | „Aktive DEMO-Slots 4“ |
| DEMO-Kachel Katalog | 17 | `/algostrategien/strategien` | „Strategien im Strategy Vault 17“ |
| Aktive Strategien: Status, n, PF | 14 / 7,62 explorativ · 18 / 0,65 explorativ · 12 / 0,62 explorativ · 24 / 0,76 (0,22–1,63) vorläufig | `/algostrategien/strategien`, Tabelle „Strategien durchsuchen“ | Zeilen „DAX Initial-Balance-Extension Fade … Beobachtung … 14 … 7,62“, „DAX Timely Opening-Range Breakout … 18 … 0,65“, „DAX Floor-Trader-Pivot Level Reversal … 12 … 0,62“, „DAX Sigma-Adaptive Open-Anchored Breakout … Läuft · negativ … 24 … 0,76 · 0,22–1,63“ |
| Gesamtbilanz-Kacheln | 195 · 77 · 18 · 1 | `/algostrategien/ueber-mich` | „18 schwaches Signal, unbestätigt“, „1 als Edge-Kandidat eingestuft, vorbehaltlich Walk-Forward“ |
| Versuchsliste (Titel und Urteil, in Klartext umformuliert) | fünf Versuche vom 28.09.2026 | `/algostrategien/research`, „Zuletzt dokumentierte Versuche“ | z. B. „DAX Flaggen-Fortsetzungsmuster … → Kein Edge nachgewiesen“, „Anker-Verschiebungs-Placebo … → Baustein ohne Mehrwert“, „Pre-Open-Lern-Gate … → Baustein verbessert Träger (in-sample)“ |
| Evidenztabelle | 14 · 36 · 5 / 0 · 0 · 0 / 0 · 0 · 0 | `/algostrategien/research/evidenz`, „Hypothesen je Evidenzklasse“ | wie dort |
| Hinweis Entscheidbarkeit | 2 von 79 (günstigster Fall 31) | `/algostrategien/research/evidenz`, „Entscheidbarkeit“ | „Von 79 berechenbaren Kandidaten sind realistisch 2 und selbst im günstigsten Fall 31 in unter 24 Monaten entscheidbar.“ |
| Hinweis Kalibrierung | −0,61, Intervall −0,89 bis −0,48, n = 8 | `/algostrategien/research/evidenz`, „Kalibrierung der eigenen Hürden“ | „Median(DEMO-PF − Einlass-PF) · -0,61 · [-0,89; -0,48] · 8“ |
| Methodenschritte (Zitate) | vier Zitate | `/algostrategien/research/methodik` | „Ohne Registrierung läuft kein Backtest“, „Spread real aus Ticks + 0.5 Pt Slippage pro Seite“, „gegen 20 Zufalls-Replikationen … getestet“, „Research-Backtests enden 21 Tage vor heute“ |
| Lesegrundsätze im Kopfbereich | Schwellen 20 / 50 | `/algostrategien/research/lesehilfe`, „Drei Regeln“ | „Unter 20 Trades ist jedes Ergebnis explorativ … Unter 50 ist es vorläufig.“ |
| Fußzeile Haftungshinweis | Text | Fußzeile aller Hub-Seiten, `/ueber-mich` „Hinweis zum Angebot“ | „… keine Anlageberatung …“, „Es werden weder Signale noch Strategien verkauft.“ |

### 4.2 Beispielwerte (im Prototyp gekennzeichnet)

| Element | Beispielwert | Warum kein echter Wert | Künftige Quelle |
|---|---|---|---|
| Gesamtbilanz DEMO in Punkten | −250 Pkt | `/strategien` weist die DEMO-Bilanz nur in **Euro** aus („−300,72 €“ usw.). Das Bedienpult zeigt zwar „Gesamt -12.3p“ bei „80 Trades“, aber unter dem Reiter „LIVE“, ohne Zeitraum und unvereinbar mit den 1.006 DEMO-Trades. Die Zahl ist daher nicht zuordenbar. | Öffentliche API: Summe `pnl_pts` über alle abgeschlossenen DEMO-Trades, mit n. |
| Bilanz je aktiver Strategie | +120 / −90 / −30 / −140 Pkt | wie oben, je Strategie nur Euro veröffentlicht | Öffentliche API: `pnl_pts` je Strategieversion, dazu n, PF, 90-%-Bereich. |
| DAX-Linie | synthetisch | bewusst kein Kurs | verzögerte öffentliche Kursdaten |

### 4.3 Redaktionelle Texte (vom Betreiber freizugeben)

Die Klartextnamen und die Ein-Satz-Ideen der vier aktiven Strategien habe ich aus den Katalognamen und Untertiteln auf `/strategien` („Initial-Balance Range-Day Extension Fade“, „Time-Windowed Volatility Breakout“, „Pivot-Level Barrier“, „Intraday Volatility Breakout“) und der Hypothese im Dossier `2026-08-12_dax_ib_extension_fade` abgeleitet. Das gilt auch für die Klartext-Titel der fünf Versuche. Sie sind inhaltlich zu prüfen. Empfehlung: Klartextname und Ein-Satz-Idee als Pflichtfelder in den Strategiekatalog aufnehmen, damit die Startseite sie automatisch übernimmt.

---

## 5. Was von der alten Startseite ersatzlos entfallen sollte

Das Bedienpult bleibt vollständig erhalten, aber **nicht mehr öffentlich unter `/`**. Vorschlag: `/bedienpult` (oder eine eigene Subdomain) hinter Anmeldung. Die Anmeldung besteht bereits („EINLOGGEN“; eine Ressource antwortet heute schon mit 401). Auf der öffentlichen Startseite entfallen ersatzlos:

| Element heute | Warum es entfällt |
|---|---|
| Kursleiste „BID / ASK / SPREAD“, „SYSTEM: OK“, „MARKT: OFFEN“ | Betriebsinformation ohne Forschungsbezug. Wirkt wie ein Handelsangebot. |
| Reiter „DAX 40 / JAPAN 225“ | Nikkei spielt in der öffentlichen Forschungsdarstellung kaum eine Rolle. Zwei Märkte ohne Einordnung verwirren. |
| Performance-Kasten mit „LIVE / ALLE / SHADOW / LOG“ | Unklare Grundgesamtheit (80 Trades vs. 1.006), keine Einstufung, kein Bereich. „LIVE“ legt Echtgeld nahe. |
| „STRATEGIE-AUSWERTUNG“ mit Kürzeln (IEF, REF, ABAE, FHRG, RNRSE, PPB, RNREL, PPFVS, EMAG, PLR, SAB, TO) und Tag/Woche/Monat/Jahr | Für Fremde unlesbar. Kurze Zeitfenster laden zum Fehlschluss aus kleinen Stichproben ein. Wird durch die Klartexttabelle ersetzt. |
| Kerzenchart mit EMA5/8/13, Bollinger-Bändern, BUY/SELL-Markern | Signalhafte Darstellung, die als Handelsempfehlung missverstanden werden kann. Wird durch den ruhigen Platzhalter-/Verlaufschart ersetzt. |
| „SIGNAL-MATRIX“ (TREND/STRUKTUR/VOLA/LONG/SHORT × M1…H1) | Interne Entscheidungslogik. Liest sich wie ein Signaldienst, was die Seite ausdrücklich nicht ist. |
| „ORDERS“ mit Lot, Entry, SL/Trail, P&L in **Euro**, Modus „LIVE“, „BROKER_SL_TP“ | Euro-Beträge und „LIVE“ widersprechen der Aussage, es gebe kein Echtgeldkonto. Positionsdetails sind Betriebsinterna. |
| „LIVE RUNNER V2“ (Liste von 14 Strategie-IDs, „DAX DEMO FREI“, „NKY FREI“) | Interne Slot-Verwaltung. Widerspricht außerdem den „4 aktiven DEMO-Slots“ auf `/strategien`. |
| „DAILY BRIEFING“, „KI-ANALYSE“ | Tagesaktuelle Markteinschätzungen sind inhaltlich eine Marktmeinung und rechtlich heikel. Kein Bezug zur Forschung. |
| „WIRTSCHAFTSKALENDER“ | Allgemeiner Marktservice, nicht Kern des Projekts. |
| „SYSTEM-MONITOR“ (Host, CPU, RAM, Disk, DB-WAL, Kern-Prozesse, „Stale aufraeumen“) | Infrastrukturdetails sollten aus Sicherheitsgründen nicht öffentlich sein. Ein öffentlicher Aufräum-Knopf gehört nicht auf eine Besucherseite. |
| Seitentitel „Edge Lab“ | Interner Name. Neuer Titel: „Warchhold Research – offene Forschung zum DAX-Handel“. |

**Was konzeptionell weiterlebt:** Der DAX-Verlauf (als ruhige, verzögerte Linie ohne Signale) und eine DEMO-Bilanz, jetzt in Punkten, mit n, Einstufung und Bereich.

---

## 6. Voraussetzungen vor dem Livegang

1. **Öffentliche Datenschnittstelle in Punkten** je Strategie und gesamt: `n`, `pf`, `pf_ci90`, `pnl_pts`, Einstufung, Stand. Die bestehende API hinter `/strategien` liefert Euro.
2. **Echtgeld-Aussage vereinheitlichen** (Über mich, Wissenskarte, Betriebsregister, Bedienpult „LIVE“). Solange das offen ist, sollte die Startseite keine Aussage zum Echtgeld machen. Der Prototyp nennt deshalb nur „Alle veröffentlichten Handelsergebnisse stammen aus dem Spielgeldbetrieb“.
3. **Zählgrößen angleichen** (195 Hypothesen vs. andere Zählungen; Evidenz „Vorwärts 0 · 0 · 0“ vs. „9 geprüft“). Die Startseite übernimmt heute bewusst die Zahlen, wie sie stehen, und nennt die Quelle (siehe `review-hub-extern.md` Nr. 3 und 6, `audit-zahlen.md` Rang 3).
4. **Impressum, Datenschutz und Kontakt anlegen.** Die Fußzeile verlinkt `/impressum` und `/datenschutz`, beide antworten heute mit 404.
5. **Routing:** `/` zeigt die neue Seite, das Bedienpult zieht nach `/bedienpult` hinter die Anmeldung, und alte Lesezeichen leiten dorthin weiter. Die Hub-Pfade unter `/algostrategien/…` bleiben unverändert.
6. **Meta- und OG-Texte** der neuen Startseite (im Prototyp enthalten) übernehmen. Auf „Live geprüft“ und „echte AUTO-Trades“ verzichten.
