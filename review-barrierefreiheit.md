# Barrierefreiheit und Nutzbarkeit bei 390 px

**Maßstab:** WCAG 2.2, Stufe AA; zusätzlich die Nutzbarkeit bei 390 px Breite.
**Geprüft am:** 28.09.2026, ca. 20:15–20:45 Uhr (Europe/Berlin).
**Seiten:**
- `https://warchhold.com/algostrategien`
- `…/algostrategien/strategien`
- `…/algostrategien/research`
- `…/research/evidenz`
- `…/research/methodik`
- `…/research/bestenliste`
- `…/research/kalender`
- `…/research/gehirn/karte`
- `https://warchhold.com/` (Bedienpult)

**Vorgehen**
- Die Seiten wurden in Chromium mit JavaScript gerendert, bei 390 × 844 px (hell und mit `prefers-color-scheme: dark`) sowie bei 1.280 px.
- Je Seite wurde erfasst:
  - Überschriftenfolge und Landmarken,
  - sichtbare Linktexte und zugängliche Namen (Accessibility-Baum),
  - Tabellen samt Scrollcontainer,
  - Elemente außerhalb des Viewports und abgeschnittene Texte (`text-overflow`),
  - Textkontrast (Vordergrund auf der tatsächlich darunterliegenden Hintergrundfarbe),
  - Zielgrößen nach 2.5.8 einschließlich der Abstandsausnahme,
  - Tab-Reihenfolge und Fokusdarstellung über 40 Tabstopps.
- Die interaktive Karte wurde zusätzlich per Tastatur bedient (Suche, Tour, Schaltflächen).
- Farbcodierung, Fokus und Abschneidungen wurden an Screenshots beurteilt.

**Grenzen**
- Kein echter Screenreader-Durchlauf. Aussagen zu Namen und Rollen stammen aus dem Accessibility-Baum von Chromium.
- Die Kontrastmessung überspringt Text auf Hintergrundbildern und Verläufen.
- Die Hub-Seiten haben **keinen Dunkelmodus**: Mit `prefers-color-scheme: dark` bleibt der Hintergrund `rgb(246, 247, 249)`. „Dunkel“ ließ sich dort deshalb nicht getrennt prüfen. Das Bedienpult ist umgekehrt nur dunkel.

**Schwere**
- **hoch:** Inhalte oder Funktionen sind für eine Nutzergruppe nicht erreichbar, oder ein AA-Kriterium ist klar verletzt.
- **mittel:** AA-Verstoß mit Umweg oder deutliche Erschwernis.
- **gering:** Best Practice oder kleine Hürde.

---

## Befunde (gravierendste zuerst)

| Nr | Seite | Stelle | Kriterium | Schwere | Vorschlag |
|---|---|---|---|---|---|
| 1 | …/research/gehirn/karte | **Graph-Fläche (`<canvas>`, 934 × 608 px bei 1.280 px; 344 × 608 px bei 390 px).** Das Canvas hat weder `role` noch `aria-label` noch `tabindex` (`tabIndex = -1`), die Elterncontainer haben keine Rolle. Laut Seitentext „trägt jeder Punkt und jede Linie eine Erklärung … anklicken genügt“. Knoten sind über die Suche („Wissensgraph durchsuchen“) per Tastatur erreichbar: Treffer sind Buttons, Enter öffnet den Inspektor. **Verbindungen (Linien) dagegen sind nur per Maus anklickbar.** Die Bedienhinweise lauten „oder Knoten anklicken · ziehen dreht · Rad zoomt“. | 2.1.1 Tastatur (A); 1.1.1 Nicht-Text-Inhalt (A) | hoch | Dem Canvas `role="img"` und eine Kurzbeschreibung geben. Eine gleichwertige Textansicht anbieten („Als Liste anzeigen“: Knoten nach Typ, je Knoten die Verbindungen als Links). Verbindungen im Inspektor eines Knotens als fokussierbare Liste zeigen. Pfeiltasten-Navigation zwischen Nachbarknoten ergänzen. |
| 2 | …/research/gehirn/karte | **Knotentypen im Graphen nur über Farbe unterscheidbar.** Die Legende „KNOTENTYP“ ordnet 13 Typen je einen Farbpunkt zu (z. B. Instrument grau, Handelsplatz blau, Strategie orange, Version grün, Regime-Regel rosa, Nachweis dunkelgrün). Im Canvas sind alle Knoten gleich geformte Punkte, und Beschriftungen erscheinen nur an wenigen Knoten in sehr kleiner Schrift (Screenshot bei 1.280 px). | 1.4.1 Benutzung von Farbe (A); 1.4.11 Nicht-Text-Kontrast (AA) | hoch | Je Typ eine eigene Form oder ein Symbol verwenden (Kreis, Quadrat, Raute …) und Beschriftungen beim Überfahren oder Fokussieren zeigen. Die Textansicht aus Nr. 1 löst das mit. |
| 3 | https://warchhold.com/ | **Keine einzige Überschrift** (0 × h1–h6), kein Sprunglink. Landmarken nur `header`, `aside`, `main`. Bereiche wie „PERFORMANCE“, „STRATEGIE-AUSWERTUNG“, „SIGNAL-MATRIX“, „ORDERS“ und „SYSTEM-MONITOR“ sind nur optisch als Überschrift gestaltet. Seitentitel: „Edge Lab“. | 1.3.1 Info und Beziehungen (A); 2.4.1 Blöcke umgehen (A); 2.4.6 Überschriften (AA); 2.4.2 Seitentitel (A) | hoch | Bereichstitel als `h2` auszeichnen, einen `h1` und einen Sprunglink ergänzen. Aussagekräftiger Titel, z. B. „Bedienpult – Warchhold Research“. (Nach der Trennung Startseite/Bedienpult betrifft das nur noch den Betreiber.) |
| 4 | https://warchhold.com/ | **Charts als `<canvas>` ohne Textalternative:** drei Canvas-Elemente (u. a. 388 × 492 px Kurschart bei 1.280 px, 390 × 240 px bei 390 px), jeweils ohne `role`/`aria-label`. Kursniveau, O/H/L/C, EMA- und BB-Linien sowie BUY/SELL-Marker sind für Screenreader unsichtbar. | 1.1.1 Nicht-Text-Inhalt (A) | hoch | `role="img"` mit Kurzbeschreibung („DAX 5 Minuten, letzter Kurs …“) und die Kerndaten als Text oder Tabelle daneben. |
| 5 | https://warchhold.com/ | **Zu geringer Kontrast bei sehr kleiner Schrift** (gemessen bei 1.280 px): „Power“, „V2“, „Kern“ 2,9:1 bei 9 px; „80 Trades“, „OK“, „12“, „16“, Uhrzeit „18:36:24“ 4,23:1 bei 10–11 px; Euro-Beträge in der Orderliste („-8.85€“, „-45.70€“ …) 3,92:1 bei 8 px (`rgba(255,51,85,.85)` auf `rgb(20,22,25)`). | 1.4.3 Kontrast Minimum (AA) | hoch | Sekundärtext mindestens 4,5:1 (z. B. `#9aa3b0` statt `rgb(111,120,133)`), Mindestschriftgröße 12 px, Rot ohne Transparenz. |
| 6 | https://warchhold.com/ | **Bei 390 px breiter als der Bildschirm:** Der Vollseiten-Screenshot ist 431 px breit. Die Kopfzeile wird abgeschnitten („SYSTEM: OK“ endet bei x = 434 px, sichtbar nur „SYS“). Rechts ragt der Kasten „SYSTEM-MONITOR“ (bis x = 830 px) mit „HO…/CP…/RA…/DI…“ in den Bildschirm. | 1.4.10 Umfluss (AA) | hoch | Kopfzeile bei schmalen Breiten umbrechen oder Werte in ein Menü verlagern. Den Seitenkasten bei < 768 px vollständig ausblenden (`display:none`) statt ihn aus dem Bild zu schieben. |
| 7 | /algostrategien | **Keine sichtbare Fokusanzeige** auf den beiden Hauptschaltflächen „So funktioniert die Forschung“ und „Strategien & Ergebnisse“. Mit Tastaturfokus (`:focus-visible` = true) sind `outline` transparent und `box-shadow` identisch zum unfokussierten Zustand. Der Screenshot zeigt keinen Unterschied. Alle anderen Hub-Links haben einen deutlichen 2-px-Rahmen `rgb(21,112,117)`. | 2.4.7 Fokus sichtbar (AA) | hoch | Für diese Schaltflächen denselben Fokusrahmen wie im Rest des Hubs setzen, z. B. `outline: 2px solid #157075; outline-offset: 3px`. |
| 8 | …/research/gehirn/karte | **Tour „▸ Von der Idee zum echten Trade“:** Nach Aktivierung per Enter springt der Fokus auf `<body>`, er geht verloren. Der Tourtext („Schritt 1 von 7 … Alles beginnt hier: …“) steht in keiner Live-Region und wird nicht angesagt. Die Tour-Schaltflächen „beenden“ und „weiter →“ werden erst mit dem nächsten Tab erreicht. Das Tourfenster überdeckt bei 1.280 px zudem den unteren Teil des Inspektors („EINGANG …“ wird abgeschnitten). | 2.4.3 Fokus-Reihenfolge (A); 4.1.3 Statusmeldungen (AA); 2.4.11 Fokus nicht verdeckt (AA) | mittel | Beim Start den Fokus in das Tourfenster setzen (Überschrift oder „weiter“), den Tourtext in `aria-live="polite"` ausgeben und Tourfenster und Inspektor nicht überlappen lassen. |
| 9 | …/research/gehirn/karte | **Drehen und Verschieben nur durch Ziehen:** „ziehen dreht“. Für Zoomen gibt es „+“/„−“, für Drehen oder Verschieben ist keine Einzelzeiger-Alternative ohne Ziehen sichtbar. | 2.5.7 Ziehbewegungen (AA, neu in 2.2) | mittel | Schaltflächen oder Pfeiltasten für Drehen und Verschieben ergänzen. „Ansicht zurücksetzen“ ist bereits vorhanden. |
| 10 | …/research/gehirn/karte | **Schaltfläche „Unverbundene ausblenden“ heißt bei 390 px nur „6“:** Der sichtbare Text „Unverbundene ausblenden“ wird ausgeblendet. Übrig bleibt der zugängliche Name „6“, die Erklärung steht nur im `title` („6 Knoten ohne Verbindung ausblenden …“). | 4.1.2 Name, Rolle, Wert (A); 2.4.6 (AA) | mittel | `aria-label="Unverbundene Knoten ausblenden (6)"` setzen oder den Text visuell verstecken statt mit `display:none`. |
| 11 | …/research/gehirn/karte | **Zu geringer Kontrast in der Kartenleiste:** Zahl „88“ neben „Trading-Plattform“ 3,12:1 (blau `rgb(57,135,229)` auf `rgb(7,18,39)`), „154“ neben „Research-Plattform“ 3,56:1 (orange `rgb(201,133,0)`), jeweils 10 px. | 1.4.3 (AA) | mittel | Hellere Stufen (z. B. `#86b6ef`, `#f0b429`) oder Zahlen in der Textfarbe mit farbigem Punkt davor. |
| 12 | …/research/bestenliste | **Tabelle mit 11 Spalten bei 390 px:** 992 px breit in einem 344 px breiten Scrollbereich. Die erste Spalte „Kandidat“ ist **nicht fixiert** (`position: static`). Beim Wischen nach rechts verschwindet der Name, und Werte wie „0.98–2.1“ oder „22%“ sind keiner Zeile mehr zuzuordnen (Screenshot nach 400 px Scroll). Bei 1.280 px wird die letzte Spalte „Verdict“ am rechten Rand abgeschnitten („Baustein verbess… Träger (… sample)“). | 1.3.2 Bedeutungstragende Reihenfolge (A); Nutzbarkeit mobil | mittel | Erste Spalte `position: sticky; left: 0` mit Hintergrund. Auf dem Handy besser eine Kartenansicht wie auf `/strategien` (Name, Urteil, Score, PF, n, Bereich) und die Detailspalten aufklappbar. |
| 13 | …/research/bestenliste | **Score-Einstufung nur über Farbe:** Das Score-Etikett ist bei 66,5 und 65,7 gelb hinterlegt mit brauner Schrift (`rgb(113,63,18)`), bei 37,3 rot (`rgb(153,27,27)`). Die Einleitung sagt, unter 45 sei ein Ergebnis „eine Anekdote“. Die Grenze ist nur farblich markiert. | 1.4.1 Benutzung von Farbe (A) | mittel | Ein Textetikett ergänzen („Anekdote“ / „belastbar gemessen“) oder ein Symbol neben der Zahl. |
| 14 | …/strategien | **Zehn Links mit gleichem Text und verschiedenen Zielen:** bei 1.280 px zehnmal „Öffnen“ (Spalte „Dossier“), bei 390 px zehnmal „Dossier“, jeweils zu `/research/build/…`. In einer Linkliste des Screenreaders sind sie nicht unterscheidbar. | 2.4.4 Linkzweck im Kontext (A; knapp erfüllt über die Tabellenzeile) / 2.4.9 (AAA) | mittel | `aria-label="Dossier: DAX Initial-Balance-Extension Fade"` bzw. visuell versteckten Zusatz im Linktext. |
| 15 | …/strategien | **55 fokussierbare `<abbr>`-Elemente bei 390 px** (je Karte „DEMO n“, „PF“, „PnL“ mit `tabindex="0"`). Die Erklärung steht nur im `title` (z. B. „Profit Factor: Bruttogewinn geteilt durch Bruttoverlust …“). Tastaturnutzer brauchen 55 zusätzliche Tabstopps, Touch-Nutzer sehen `title`-Texte nie. | 2.4.3 (A, Erschwernis); Nutzbarkeit Touch (`title` ist nicht per Touch erreichbar) | mittel | `tabindex` entfernen. Die Erklärungen einmal über der Liste als kurze Legende oder als aufklappbares „Was bedeuten die Spalten?“ zeigen. |
| 16 | …/strategien | **Tabelle bei 1.280 px breiter als die Seite:** Die Tabelle reicht bis x = 1.411 px, die Seite scrollt horizontal (`scrollWidth` 1.411 bei 1.280 px Fenster). Spalten „DEMO PnL“ und „Dossier“ liegen außerhalb, ohne Scrollcontainer. | 1.4.10 gilt ab 320 px; hier Nutzbarkeit Desktop | mittel | Tabelle in einen beschrifteten Scrollcontainer setzen (wie auf `/research/evidenz`) oder die Kartenansicht bis ~1.400 px verwenden. |
| 17 | …/strategien | **Sortier-Schaltflächen im Tabellenkopf 16 px hoch** („Strategie“ 63 × 16, „Status“ 44 × 16, „Markt“ 40 × 16, „Logik“ 37 × 16, „TF“ 16 × 16, „Version“ 51 × 16, „DEMO PF“ 61 × 16). Die Abstandsausnahme greift nicht, weil benachbarte fokussierbare Elemente im Kopf zu nah liegen. | 2.5.8 Zielgröße Minimum (AA) | mittel | Die Schaltflächen auf die volle Kopfzellenhöhe (≥ 24 px) vergrößern. |
| 18 | …/strategien | **Sprunglinks bei 390 px** „Strategiekatalog“ (108 × 20), „Dossiers“ (55 × 20), „Evidenzstufen“ (92 × 20) stehen zu dicht nebeneinander. | 2.5.8 (AA) | gering | `min-height: 24px` und Abstand ≥ 8 px. |
| 19 | https://warchhold.com/ | **„SIGNAL-MATRIX“ aus Schaltflächen ohne verständlichen Namen:** 25 Buttons heißen „MID“, „RNG“, „NORM“, „WCH“, „NO“, „UP“, „BAL“, „LOW“, „GO“. Zeile (TREND/STRUKTUR/VOLA/LONG/SHORT) und Spalte (M1…H1) sind nicht im Namen. Der Screenreader liest z. B. 15× „MID, Schaltfläche“. | 1.3.1 (A); 4.1.2 (A); 2.4.6 (AA) | mittel | Als Tabelle mit `th` für Zeilen und Spalten auszeichnen und die Kürzel ausschreiben (`aria-label="Trend, 1 Minute: mittel"`). Buttons nur, wenn eine Aktion folgt. |
| 20 | https://warchhold.com/ | **Abgeschnittene Strategienamen in „ORDERS“:** 15 Einträge mit `text-overflow: ellipsis` (z. B. „DAX ROUND NUMBER REVERSAL SL15 EXIT LAB“, „DAX EXTREME MOVE ACTIVITY GATE“). Der volle Name steht nur im `title`. | Nutzbarkeit; 1.3.1 (A, vollständige Information nur im `title`) | mittel | Zeilenumbruch zulassen oder den Namen in einer zweiten Zeile zeigen. `title` ist per Tastatur und Touch nicht erreichbar. |
| 21 | https://warchhold.com/ | **Legende „BUY“/„SELL“ nur über Linienfarbe** (grün/rot) über dem Kurschart. Gleiches gilt für BID (grün) und ASK (rot) in der Kopfzeile, dort allerdings mit Textlabel. | 1.4.1 (A) | mittel | BUY/SELL-Marker mit Form (Dreieck auf/ab) und Beschriftung, Legende mit Symbol statt nur Farbe. |
| 22 | https://warchhold.com/ | **Emojis in Schaltflächennamen:** „📈 PERF“, „📊 STRATS“, „☰ ORDERS“, „🕑 HISTORY“, „🔒 ANMELDEN“, „🔬 RESEARCH“, „📊 SYSTEM-MONITOR“. Screenreader lesen die Emoji-Namen mit („Diagramm mit Aufwärtstrend PERF“). Dazu kommen englische Abkürzungen in einer deutschen Seite (`lang="de"`). | 1.3.1 / 3.1.2 Sprache von Teilen (AA) | gering | Emojis mit `aria-hidden="true"` in ein eigenes `span` setzen und Beschriftungen ausschreiben („Leistung“, „Strategien“, „Orders“, „Verlauf“). |
| 23 | …/research/methodik | **Linktext widerspricht dem Ziel:** „Zurück zur Strategie-Datenbank“ führt auf `/algostrategien` (Überblick), nicht auf `/algostrategien/strategien`. | 2.4.4 Linkzweck (A) | mittel | Text „Zurück zum Überblick“ oder das Ziel auf den Strategiekatalog ändern. |
| 24 | …/research/methodik | **Agententabelle bei 390 px:** 7 Spalten, 950 px breit, in einem 344 px breiten Scrollbereich (gut: `tabindex="0"`, `aria-label="Agentenstatus"`). Lange Beschreibungstexte in Spalte 2 machen jede Zeile sehr hoch. Die Statusspalte (z. B. „error“) ist die letzte und erst nach ca. 600 px Wischen sichtbar. | Nutzbarkeit mobil; 1.3.2 | mittel | Auf dem Handy Karten je Agent (Name, Status, Takt, letzter Lauf), die Beschreibung aufklappbar. Status in die erste oder zweite Spalte. |
| 25 | …/research/kalender | **Dritte Tabelle ohne Kopfzellen:** Die Tabelle unter „DIENSTAG, 29.09.“ (Zeilen wie „00:05 · research · Suche · Warchhold Literatur-Research …“) hat 0 `th`. Die beiden anderen Tabellen haben Kopfzellen. | 1.3.1 Info und Beziehungen (A) | mittel | Dieselben Spaltenköpfe wie bei „Feste Timer heute“ (Zeit, Lauf, letzter Ausgang, zuletzt/Dauer) als `th scope="col"`. |
| 26 | …/research/kalender · …/research/bestenliste | **Scrollbereiche der Tabellen mit Allerweltsnamen:** `aria-label="Datentabelle, horizontal scrollbar"` (alle drei Kalender-Tabellen und die Bestenliste). Auf `/research/evidenz` und `/research/methodik` sind die Bereiche dagegen inhaltlich benannt („Hypothesen je Evidenzklasse“, „Agentenstatus“). | 2.4.6 Überschriften und Beschriftungen (AA) | gering | Nach Inhalt benennen („Tabelle: Feste Timer heute“, „Tabelle: Kandidatenvergleich“). Zusätzlich `<caption>` setzen (heute auf keiner der geprüften Tabellen vorhanden). |
| 27 | …/research/kalender · …/research/gehirn/karte | **Umlaute als ae/oe/ue in generiertem Text:** „Vollstaendig generiert … Nicht von Hand pflegen — Aenderungen gehoeren in die .timer-Units“ (Kalender), „Takt: taeglich 21:10“, im Karten-Inspektor „Laeuft taeglich um 00:05. Betrieben ueber: Claude-Abo“. Screenreader sprechen diese Wörter falsch aus. | 3.1 Lesbarkeit (Best Practice) | gering | In der Ausgabe echte Umlaute verwenden. |
| 28 | …/research/gehirn/karte | **Überschriftenebene uneinheitlich:** „Was die Verbindungen bedeuten“ ist `h3` und steht damit als Unterpunkt unter den Knotentypen (`h3` „Instrument“ … „KI-Anbieter“), ist aber ein eigener Abschnitt neben „Die Bauteile und ihre Funktion“ (`h2`). | 1.3.1 (A) / 2.4.6 (AA) | gering | Auf `h2` anheben. |
| 29 | /algostrategien · …/research | **Sehr lange Linknamen durch ganze Karten als Link:** z. B. „Neu hier? In fünf Minuten verstehen, wie die Zahlen gemeint sind — mit Glossar. Lesehilfe öffnen“. Auf `/research` erscheinen die fünf jüngsten Versuche zweimal als Link (Liste und „Forschungsereignisse“), der zweite mit „→ Baustein verbessert Träger (in-sample) bestes Trial: PF 1.48, n=227 · ENTWURF“. | 2.4.4 (Erschwernis) | gering | Nur die Kartenüberschrift verlinken, den Rest als Beschreibung per `aria-describedby`. Doppelte Linkziele zusammenfassen. |
| 30 | /algostrategien · …/research | **Pfeile „→“ im Linktext** („Alle Quellen & Funde →“, „Alle Forschungsfelder und Strategiebausteine →“, „Arbeitspakete & Nachweise ansehen →“) werden als „Pfeil nach rechts“ vorgelesen. | 1.1.1 / Best Practice | gering | Pfeil per CSS (`::after`) oder in `span aria-hidden="true"`. |
| 31 | …/research/bestenliste | **Nur eine Überschrift** („Kandidatenvergleich“, `h1`). Die 57-zeilige Tabelle hat weder Zwischenüberschrift noch `caption`, der Einleitungsabsatz ist der einzige Kontext. | 2.4.6 (AA, knapp) | gering | `caption` bzw. `h2` „Alle Kandidaten nach Robustheits-Score“. |
| 32 | alle Hub-Seiten | **Kein Dunkelmodus:** Die Hub-Seiten ignorieren `prefers-color-scheme: dark` (Hintergrund bleibt `rgb(246, 247, 249)`). Das ist kein WCAG-Verstoß, aber Nutzer mit Blendempfindlichkeit, die systemweit dunkel eingestellt haben, bekommen hier eine helle Fläche. Die hellen Seiten selbst bestehen die Kontrastmessung. | Nutzbarkeit (kein AA-Kriterium) | gering | Optional einen Dunkelmodus mit eigenen, geprüften Farbwerten anbieten. |

---

## Was gut funktioniert (bitte beibehalten)

- **Sprunglink „Zum Inhalt springen“** auf allen Hub-Seiten, Ziel `#hub-content` mit `tabindex="-1"`.
- **Mobilmenü:** `aria-label="Menü"`, `aria-expanded` wechselt korrekt, `aria-controls="hub-mobile-menu"`. Escape schließt das Menü, der Fokus bleibt auf der Schaltfläche.
- **Fokusrahmen im Hub:** einheitlich 2 px `rgb(21,112,117)` auf fast allen Links und Schaltflächen (Ausnahme: Nr. 7).
- **Kontrast der hellen Hub-Seiten:** Auf den acht geprüften Hub-Seiten fand die Messung außerhalb der dunklen Kartenfläche (Nr. 11) keinen Text unter 4,5:1 (bzw. 3:1 bei großer Schrift).
- **Kein horizontales Scrollen der Seite bei 390 px** auf allen Hub-Seiten. Breite Tabellen liegen in **fokussierbaren, benannten Scrollbereichen** (`/research/evidenz`: „Hypothesen je Evidenzklasse“, „Kalibrierung der eigenen Hürden“, „Parallele Methodikfassungen“; `/research/methodik`: „Agentenstatus“).
- **`/strategien` bei 390 px:** Die breite Tabelle wird durch eine Kartenliste ersetzt. Explorative Werte sind **durchgestrichen und** mit dem Wort „explorativ“ versehen, also nicht nur farblich markiert. Status steht als Text („Beobachtung“, „Läuft · negativ“, „Kein DEMO-Slot“).
- **`/research/kalender`:** Laufstatus als Textetikett („gelaufen“, „offener Befund“), der farbige Punkt ist nur Zusatz.
- **Karte:** Die Suche ist ein beschriftetes Eingabefeld (`aria-label="Wissensgraph durchsuchen"`). Die Treffer sind echte, fokussierbare Buttons, und Enter öffnet den Inspektor. Die Legendenschaltflächen tragen `aria-pressed` und einen `title` („Instrument ausblenden“).
- **Sprache:** `lang="de"` auf allen Seiten gesetzt.

---

## Zusammenfassung nach Prüfschwerpunkt

| Schwerpunkt | Ergebnis |
|---|---|
| Verständliche Linktexte | Überwiegend gut. Probleme: „Öffnen“/„Dossier“ ×10 (Nr. 14), irreführender Rücklink (Nr. 23), überlange Kartenlinks (Nr. 29). |
| Überschriftenhierarchie | Hub sauber (ein `h1`, logische `h2`/`h3`). Das Bedienpult hat gar keine Überschriften (Nr. 3). Kleinigkeit auf der Karte (Nr. 28). |
| Tabellen auf dem Handy | Scrollbereiche vorhanden und fokussierbar. Es fehlen eine fixierte erste Spalte (Nr. 12), Kartenalternativen für sehr breite Tabellen (Nr. 12, 24), Kopfzellen (Nr. 25), inhaltliche Beschriftungen und `caption` (Nr. 26). |
| Tastaturbedienung der Karte | Knoten über die Suche erreichbar. Graph, Verbindungen, Drehen und Verschieben sind nicht per Tastatur bedienbar (Nr. 1, 9). Fokusverlust bei der Tour (Nr. 8). |
| Farbe als einziger Bedeutungsträger | Knotentypen der Karte (Nr. 2), Score-Grenze der Bestenliste (Nr. 13), BUY/SELL im Bedienpult (Nr. 21). |
| Kontrast hell/dunkel | Hub hell bestanden, kein Dunkelmodus (Nr. 32). Dunkle Kartenleiste (Nr. 11) und Bedienpult (Nr. 5) unter 4,5:1. |
| Abgeschnittene Texte | Bedienpult: Kopfzeile und Seitenkasten bei 390 px (Nr. 6), Strategienamen in Orders (Nr. 20). Bestenliste: letzte Spalte bei 1.280 px (Nr. 12). Karte: Inspektor unter dem Tourfenster (Nr. 8). |
