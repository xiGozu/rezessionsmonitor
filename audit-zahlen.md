# Zahlen-Audit: Warchhold Algo Research Hub

**Perspektive:** skeptischer Statistiker ohne Serverzugang, nur öffentliche Seiten
**Abruf:** 28.09.2026, 19:40–20:00 Uhr (Europe/Berlin), gerendert mit JavaScript (Headless-Chromium). Ein zweiter Abruf um 19:58 Uhr ergab identische Inhalte.
**Geprüfte Seiten** (alle unter `https://warchhold.com/algostrategien`): `/`, `/strategien`, `/hall-of-fame`, `/research`, `/research/evidenz`, `/research/bestenliste`, `/research/kandidaten`, `/research/validierung`, `/research/funde`, `/research/vermessung`, dazu die Build-Dossiers
- `/research/build/2026-09-28_dax_round_number_reversal_sl15_exit_lab` (Rang 2 der Bestenliste),
- `/research/build/2026-09-28_dax_flag_continuation` (jüngster Negativbefund).

**Vorgehen und Grenzen**
- Zitate sind wörtlich (Dezimaltrennzeichen wie auf der Seite).
- Es wurde nichts neu berechnet. Wo zwei Zahlen verglichen werden, stehen beide wörtlich im Text.
- Seiten mit Dutzenden gleich aufgebauter Einträge (Kandidaten: 50 Auto-Build-Entwürfe und 7 kuratierte Untersuchungen; Bestenliste: 57 Zeilen) werden **je Zeilenformat einmal** bewertet, mit Beispielzitaten. Auffällige Einzelfälle stehen in eigenen Zeilen.
- Auf `/research/funde` ist nur der jüngste Lauf aufgeklappt. Einträge aus zugeklappten Läufen sind mit „(aufklappbar)“ markiert.
- „n sichtbar?“ heißt: Die Stichprobe steht an derselben Stelle wie die Zahl. „Unsicherheit?“ heißt: ein Intervall, eine Power-Angabe oder eine Einstufung (explorativ/vorläufig) an derselben Stelle.

---

## Die 10 riskantesten Stellen

| Rang | Seite | Stelle (wörtlich) | Warum riskant |
|---|---|---|---|
| 1 | /hall-of-fame | Siegerpodest mit großen Karten: „#1 DEMO (Spielgeld) hull_suite_v3_final · PF-Bereich (90 %) 0,88–2,53 · Punktwert 1.52 · n=83“, „#2 … dax_round_number_reversal · 0,81–4,09 · Punktwert 1.76 · vorläufig · n=20“, „#3 Vorwärtstest · Time-of-Day Momentum (16:30) · PF-Bereich (90 %) — · Punktwert 1.43 · vorläufig · n=20“ | Die Seite heißt „Hall of Fame“ und zeigt ein Podium, obwohl alle drei Bereiche die 1 einschließen oder gar keinen Bereich haben. Die Regel „Rangfolge … nach der unteren Grenze des 90-%-Vertrauensbereichs“ lässt sich auf Platz 3 nicht anwenden, er steht trotzdem vor Platz 4 „0,65–1,21“. Ein Leser sieht drei Gewinner, wo die Daten keinen Gewinner stützen. |
| 2 | /hall-of-fame vs. /research/validierung | Ergebnisvergleich: „Time-of-Day Momentum (16:30) … PnL 103 EUR“, „Volatility Breakout (Baseline) … 6 EUR“, „VWAP-Reversion (Baseline) … -179 EUR“, „ORB 15min … -359 EUR“, „Baustein: Session-Loss-Limit … -347 EUR“ · Validierung (dieselben Zeilen, Monat 2026-08): „+103.28 Pkt“, „+5.84 Pkt“, „-179.09 Pkt“, „-359.28 Pkt“, „-347.09 Pkt“ | Simulierte Vorwärtstest-Punkte erscheinen als Euro und stehen in derselben Tabelle wie echte DEMO-Kontoergebnisse in EUR. Einheit und Evidenzklasse werden vermischt. |
| 3 | /research/evidenz vs. /research/validierung · /research/bestenliste · /hall-of-fame | Evidenz: „Vorwärts (neue Monate) 0 · 0 · 0“, „Live / DEMO 0 · 0 · 0“, direkt darunter „Vorwärts mehrmonatig geprüft: 9 Kandidaten, davon überlebt: 0“ · Validierung: „14 Kandidaten fuer 2026-08 gemessen“ · Bestenliste: „DAX Time-of-Day Momentum (Late-Day) … 2/2 Monate (+231 Pkt) · Kein Edge nachgewiesen“ · Ergebnisvergleich: dieselbe Idee auf Platz 3 | Ausgerechnet die Seite, die Evidenzklassen trennen soll, meldet null Vorwärts- und DEMO-Urteile. Gleichzeitig stehen „9 geprüft, 0 überlebt“ und ein Kandidat mit „2/2 Monate“ positiv, der auf einer Seite Podiumsplatz ist und auf der anderen „kein Edge“ hat. „Überlebt“ ist nirgends definiert. |
| 4 | /research/kandidaten · /research/bestenliste · /research | Urteil „Der Baustein verbesserte die Ausgangsstrategie“ bzw. „Baustein verbessert Träger (in-sample)“ bei: „(A/B: PF 1.78->2.00, PnL +361->+362)“, „(A/B: PF 0.70->0.72, PnL -259->-240)“, „(A/B: PF 1.00->1.04, PnL -22->+79)“, „(A/B: PF 0.96->1.01, PnL -132->+17)“, „(A/B: PF 0.90->0.96, PnL -784->-268)“, „(A/B: PF 0.67->0.86, PnL -4488->-2364)“ | Das Etikett „verbessert“ wird ohne Unsicherheit der gepaarten Differenz vergeben, auch bei +1 Punkt Gesamt-PnL und bei weiter verlustbringendem PF < 1. Die Evidenzseite sagt selbst: „Typisch sind wenige Punkte Ertrag je Trade bei rund hundert Punkten Streuung.“ |
| 5 | /research/kandidaten (kuratierte Untersuchungen) | Befundtexte widersprechen den Tabellen darunter: ORB „grenzwertig positiv (PF 1.10)“, „Long-Seite trägt (PF 1.36)“ vs. Tabelle „OR 15 Min 59 … 1,01“, „OR 15 Min, nur Long 30 … 0,89“ · Noise-Area „N=1.0 ergibt PF 0.63 (-1046 Pkt) bei n=101 — … belastbar“ vs. „Band 1.0x (Baseline) 76 … 0,78 -464 Pkt“ · Volatility „Baseline PF 0.57 bei n=153“ vs. „Squeeze 3.0xATR (Baseline) 132 … 0,67“ · VWAP „Bei n=171 … statistisch belastbar“ vs. drei Zeilen mit je „141“ · PDH/PDL „TP-Verlängerung auf 2.2R zerstört den Edge (PF 1.09)“ vs. „TP 2.2R 24 … 1,30“ | Die Aussage „belastbar“ stützt sich auf ein n, das in der Tabelle nicht vorkommt. Text und Tabelle stammen offenbar aus verschiedenen Läufen. Ein Prüfer kann keine der Zahlen reproduzieren. |
| 6 | /research/kandidaten vs. /strategien | Kandidaten: „Mean Reversion · etabliert · Eigene verfeinerte Variante (HMR) mit Regime-Filtern etabliert“, „Die hauseigene, stark verfeinerte Mean-Reversion (HMR, mit Regime-Filtern) trägt“ · Strategien: „dax_hyper_mean_reversion · Kein DEMO-Slot · … · 306 · 0,75 · 0,59–0,92 · −300,72 €“ | Die einzige Vault-Strategie, deren sichtbarer 90-%-Bereich vollständig unter 1 liegt, wird als „etabliert“ und „trägt“ beschrieben. Das ist eine Erfolgsbehauptung ohne Evidenzklasse gegen das eigene DEMO-Ergebnis. |
| 7 | /research/evidenz vs. alle Seiten mit Backtest-PF | Evidenz, „Kalibrierung der eigenen Hürden“: „Rangkorrelation Einlass-Kennzahl 'pf' gegen DEMO-Ertrag je Trade (EUR) · -0,07 · [-0,8; 0,63] · 8“, „Median(DEMO-PF − Einlass-PF) · -0,61 · [-0,89; -0,48] · 8“ · gleichzeitig prominent: /research „bestes Trial: PF 1.97, n=41“, Bestenliste „2,17“, „2,32“, „2,60“ | Die Plattform misst selbst, dass der Backtest-PF das DEMO-Ergebnis nicht vorhersagt und systematisch darüber liegt. Diese Warnung steht nur auf der Evidenzseite. Neben den In-sample-PF-Werten fehlt jeder Hinweis darauf. |
| 8 | /research/bestenliste | Einleitung: „ein hoher PF mit Score unter 45 ist eine Anekdote, kein Befund“ · Spitze: „Round-Number-Reversal Exit-Lab-A/B … Score 66.5/100 · 1,50 · 214 · 0.98–2.1“ und „… Exit-Lab-A/B II … Score 65.7/100 · 1,48 · 227 · 0.96–2.06“ | Die Formulierung legt nahe, ab 45 liege ein Befund vor. In allen sichtbaren Zeilen mit Score ≥ 45 schließt der angezeigte PF-Bereich jedoch Werte unter 1 ein. Die zwei Spitzenplätze sind zwei Exit-Varianten desselben Trägers (Mehrfachtest nicht ausgewiesen). Das Dossier nennt „UNTERPOWERT“, die Liste nicht. |
| 9 | /strategien (Abschnitt „Strategiedossiers“) | „Profit Factor 1,58 · Sharpe 0,13 · Trades 19 · Warchhold Score 51 / 100 · Backtest 2026-03-09 bis 2026-05-29 · Kosten noch nicht enthalten · Out-of-Sample offen“ | Eine Backtest-Kennzahl ohne Kosten steht auf der DEMO-Ergebnisseite, mit n = 19, aber ohne die sonst übliche Einstufung „explorativ“ und ohne Bereich. Der „Warchhold Score“ ist nirgends definiert. |
| 10 | /research/build/2026-09-28_dax_flag_continuation · /research · /research/bestenliste | Trials: „Baseline (Quelle: 15m-Flagge …) 1 · 0,0% · 0,00 · -74“, „5m-Signal-Bars … 0 · 0,0% · 0,00 · +0“, „Placebo: reiner 10-Bar-Bruch … 56 · 28,6% · 0,65 · -849“ → Kopfzahlen „Robustheit (bestes Trial) 0.34 – 1.1“, „Score 29/100“; /research: „bestes Trial: PF 0.65, n=56“ | Die Hypothese hat einen einzigen Trade erzeugt. Sämtliche Kopfzahlen gehören zum Placebo-Kontrollarm. Das Urteil „Kein Edge nachgewiesen“ ist so nicht belegt, korrekt wäre „nicht prüfbar“. Der Placebo wird zudem gegen Placebos getestet („Perzentil-Rang 0%“). |

---

## Vollständige Tabelle

Risiko-Stufe: **H** = hoch, **M** = mittel, **G** = gering, **+** = vorbildlich (als Positivbeispiel aufgeführt).

### /algostrategien (Startseite)

| Seite | Zahl (wörtlich) | n sichtbar? | Unsicherheit? | Risiko der Fehldeutung | Vorschlag |
|---|---|---|---|---|---|
| / | „11 / 17 Arbeitspakete im Abschlussaudit formal abgenommen.“ (Überschrift „Plattform mit Nachweis“) | – (Zählung) | – | **G.** Die Überschrift „mit Nachweis“ kann als Forschungsnachweis gelesen werden. Es handelt sich aber um technische Abnahmen. | Durch eine Forschungskennzahl ersetzen, z. B. „geprüfte Hypothesen / davon verworfen / im Vorwärtstest“. |
| / | Ereignisliste „2026-09-28 AUTO-BUILD … → Baustein verbessert Träger (in-sample)“ (3×), „→ Baustein ohne Mehrwert“ (1×), „→ Kein Edge nachgewiesen“ (1×) | nein | Evidenzklasse „in-sample“ sichtbar | **G.** Die Evidenzklasse steht dabei. Die letzten fünf Ereignisse wirken aber überwiegend positiv, eine Gesamtbilanz fehlt. | Eine Bilanzzeile daneben: „seit Start: x Builds, y ohne Edge“. |

### /algostrategien/strategien

| Seite | Zahl (wörtlich) | n sichtbar? | Unsicherheit? | Risiko der Fehldeutung | Vorschlag |
|---|---|---|---|---|---|
| /strategien | „Strategien im Strategy Vault 17 · Aktive DEMO-Slots 4 · Abgeschlossene DEMO-Trades 1.006 · Veröffentlichte Dossiers 1“ | – | – | **G.** „1 Dossier“, obwohl die Tabellenspalte „Dossier“ elf „Öffnen“-Links enthält (Build-Dossiers). | Begriffe trennen: „1 Strategiedossier · 11 Versuchsprotokolle“. |
| /strategien | „DAX Initial-Balance-Extension Fade … Beobachtung · 14 · 7,62EXPLORATIV · +438,65 €“ | ja (14) | Einstufung ja, PF durchgestrichen; kein Bereich | **M.** Die Zeile steht in der Standardsortierung („Status, dann DEMO-PF“) ganz oben. Die PnL „+438,65 €“ ist nicht gekennzeichnet und springt ins Auge. | Explorative Zeilen ans Ende sortieren oder PnL ebenfalls ausgrauen. Bei n < 20 „kein Bereich berechnet“ schreiben. |
| /strategien | „DAX Sigma-Adaptive Open-Anchored Breakout · Läuft · negativ · 24 · 0,76 · 0,22–1,63 · vorläufig · −136,15 €“ | ja | ja (Bereich und Einstufung) | **+** Vorbildliches Format: Zahl, n, Bereich, Einstufung. | Als Standard für alle Seiten übernehmen. |
| /strategien | „DAX Round-Number Barrier Reversal … Kein DEMO-Slot · 20 · 1,76 · 0,81–4,09 · vorläufig · +95,30 €“ | ja | ja | **M.** Die Tabelle selbst ist korrekt. Dieselbe Zahl wird auf /hall-of-fame zu Podiumsplatz 2 (siehe Top 10, Rang 1). „Kein DEMO-Slot“ ist unerklärt. | Den Status erklären (beendet? Grund?). |
| /strategien | „Hull Suite v3 Final · Kein DEMO-Slot · 83 · 1,52 · 0,88–2,53 · +219,10 €“ | ja | ja | **M.** Nach der Seitenregel (<20/<50) ohne Einstufung. Die Methodik verlangt aber „Mindestens 100 Trades … Darunter … nur mit ausdrücklichem Vorbehalt“. Die Schwellen sind nicht abgestimmt. | Eine Schwellenstaffel an einer Stelle definieren (z. B. <20/<50/<100) und überall gleich kennzeichnen. |
| /strategien | „dax_hyper_mean_reversion · Kein DEMO-Slot · 306 · 0,75 · 0,59–0,92 · −300,72 €“ | ja | ja | **H.** Hier ist die Zahl selbst korrekt. Sie widerspricht aber „HMR … etabliert“ und „trägt“ auf /research/kandidaten (Top 10, Rang 6). | Den Kandidaten-Text an diese Zahl koppeln. |
| /strategien | Kartenansicht: „PF 0,76vorläufig“, „PF 1,76vorläufig“, „PF 1,52“ (ohne Bereich) | ja („DEMO n“) | nur Einstufung, **kein Bereich** | **M.** Die Fußnote verspricht „darunter steht der 90-%-Vertrauensbereich“, in der Kartenansicht fehlt er. Die Karten sind die Darstellung auf schmalen Bildschirmen. | Den Bereich auch in den Karten zeigen. |
| /strategien | Strategiedossier-Karte: „Profit Factor 1,58 · Sharpe 0,13 · Trades 19 · Warchhold Score 51 / 100 … Kosten noch nicht enthalten · Out-of-Sample offen“ | ja (19) | nein; keine Einstufung „explorativ“ | **H.** Ein kostenfreier Backtest steht auf der Seite „Ergebniszahlen stammen ausschließlich aus dem Spielgeldbetrieb“. Die Regel „<20 explorativ“ wird hier nicht angewendet. | „Backtest ohne Kosten, n = 19, explorativ“ als Etikett. Score definieren oder entfernen. |
| /strategien | „Level 5 · WARCHHOLD VERIFIED“, „Level 6 · LIVE VERIFIED · Mit echten Trades im Live-Betrieb bestätigt.“ | – | – | **M.** Die Stufen „VERIFIED“ klingen wie Gütesiegel. Die Seite zeigt nur Level 1 („1 Dossier“), was nicht auf den ersten Blick klar wird. | Neben jeder Stufe die Anzahl erreichter Strategien zeigen (heute überwiegend 0). |

### /algostrategien/hall-of-fame

| Seite | Zahl (wörtlich) | n sichtbar? | Unsicherheit? | Risiko der Fehldeutung | Vorschlag |
|---|---|---|---|---|---|
| /hall-of-fame | Podiumskarten „#1 … 0,88–2,53 · Punktwert 1.52 · n=83 · WR 47.0% · PnL 219 EUR“, „#2 … 0,81–4,09 · Punktwert 1.76 · vorläufig · n=20 · WR 60.0% · PnL 95 EUR“, „#3 Vorwärtstest … — · Punktwert 1.43 · vorläufig · n=20 · WR 30.0% · PnL 103 EUR“ | ja | Bereich ja (außer #3), Einstufung ja | **H.** Seitenname „Hall of Fame“ plus Podium bei Bereichen, die die 1 einschließen (siehe Top 10, Rang 1). | Podium entfernen und neutrale Tabelle „Vergleich“ zeigen. Oben einen Satz ergänzen: „Keiner der Bereiche liegt vollständig über 1.“ |
| /hall-of-fame | Sortierregel „Rangfolge nach der unteren Grenze des 90-%-Vertrauensbereichs“ vs. Zeilen „3 … — · 1.43“, „6 Volatility Breakout (Baseline) … 50 · — · 1.01“, „7 VWAP-Reversion (Baseline) … 63 · —“, „11 ORB 15min … 29 · —“, „13 Baustein: Session-Loss-Limit … 63 · —“ | ja | bei Vorwärtstests nein | **H.** Die Sortierregel ist für fünf Zeilen nicht anwendbar, die Zeilen stehen trotzdem mitten im Ranking. | Bereiche auch für Vorwärtstests berechnen oder diese getrennt listen. |
| /hall-of-fame | Vorwärtstest-Zeilen „PnL 103 EUR“, „6 EUR“, „-179 EUR“, „-359 EUR“, „-347 EUR“ vs. Validierung „+103.28 Pkt“, „+5.84 Pkt“, „-179.09 Pkt“, „-359.28 Pkt“, „-347.09 Pkt“ | ja | nein | **H.** Punkte werden als Euro ausgewiesen (Top 10, Rang 2). | Einheit „Pkt“ übernehmen, Vorwärtstest und DEMO-Konto in getrennten Tabellen. |
| /hall-of-fame | „WR 47.0%“, „WR 60.0%“ (n=20), „WR 30.0%“ | ja | nein | **M.** Trefferquote bei n = 20 ohne Bereich. Eine auf die Nachkommastelle genaue Angabe suggeriert Präzision. | Trefferquote ohne Nachkomma und mit Bereich oder nur in den Details. |
| /hall-of-fame | „Beweisquelle: AUTO-DEMO (Spielgeld) seit 2026-05-13“ neben „Vorwaertstest 2026-08 (simuliert auf neuen Daten)“ in **einer** Rangliste | – | – | **M.** Die Evidenzklasse ist beschriftet, aber zwei Klassen werden in einer Rangfolge gemischt. Die Lesehilfe sagt, DEMO wiege „am meisten“. | Nach Evidenzklasse getrennt ranken. |
| /hall-of-fame | Platz 3 „Time-of-Day Momentum (16:30) … 1.43“ vs. /research/bestenliste „DAX Time-of-Day Momentum (Late-Day) … Kein Edge nachgewiesen“ vs. /research/kandidaten „Die 16:30-Variante endet bei PF 1.07 — Rauschen um Break-even“ | – | – | **H.** Dieselbe Idee ist hier Podium, anderswo „kein Edge“ bzw. „Rauschen“. | Die Zeile mit dem Kandidaten-Eintrag verlinken und die Statusdifferenz in einem Satz erklären. |
| /hall-of-fame | „Punktwert 1.52“ neben „0,88–2,53“ | – | – | **G.** Dezimalpunkt und -komma in derselben Karte. | Einheitlich de-DE. |

### /algostrategien/research

| Seite | Zahl (wörtlich) | n sichtbar? | Unsicherheit? | Risiko der Fehldeutung | Vorschlag |
|---|---|---|---|---|---|
| /research | „bestes Trial: PF 1.48, n=227 · ENTWURF“ (Round-Number Exit-Lab II) | ja | nein; „in-sample“ nur im Urteilstext davor | **M.** „Bestes Trial“ ist das Maximum aus drei Varianten, ohne Bereich und ohne Auswahlkorrektur. Die Evidenzseite misst, dass solche PF-Werte im DEMO-Betrieb im Median um „-0,61“ niedriger liegen. | Bereich ergänzen, „bestes von 3 Trials, in-sample“ an die Zahl schreiben. |
| /research | „bestes Trial: PF 1.97, n=41 · ENTWURF“ (Pre-Open-Gate) | ja | nein; keine Einstufung „vorläufig“ trotz n < 50 | **H.** Die höchste Zahl der Ereignisliste steht ohne Einstufung, obwohl sie unter die eigene Schwelle fällt. Das Dossier nennt den Bereich „0.99–3.41“ und „Top-3-Trades 40% des Gewinns“. | „PF 1,97 (vorläufig, n = 41, 90 %: 0,99–3,41, in-sample)“. |
| /research | „bestes Trial: PF 0.96, n=380 · ENTWURF“ bei „→ Baustein verbessert Träger (in-sample)“ | ja | nein | **H.** Positives Urteil bei PF < 1 (siehe Top 10, Rang 4). | Urteil ergänzen: „… bleibt verlustbringend“. |
| /research | „bestes Trial: PF 0.65, n=56“ (Flaggen-Fortsetzung) | ja | nein | **H.** Die Zahl gehört zum Placebo-Arm (siehe Top 10, Rang 10). | Kontrollarme bei der Wahl des „besten Trials“ ausschließen. |

### /research/evidenz

| Seite | Zahl (wörtlich) | n sichtbar? | Unsicherheit? | Risiko der Fehldeutung | Vorschlag |
|---|---|---|---|---|---|
| /research/evidenz | „In-sample (Bauhistorie) 14 · 36 · 5“ mit Erläuterung „„Bestätigt“ in-sample heißt: … Das ist die schwächste Form.“ | – (Zählung) | Evidenzklasse ja | **+** „Bestätigt“ trägt hier eine Evidenzklasse, so wie gefordert. | So beibehalten. |
| /research/evidenz | „Vorwärts (neue Monate) 0 · 0 · 0“, „Live / DEMO 0 · 0 · 0“ vs. „Vorwärts mehrmonatig geprüft: 9 Kandidaten, davon überlebt: 0“ | – | – | **H.** Widerspruch innerhalb der Seite und zu Validierung, Bestenliste und Ergebnisvergleich (Top 10, Rang 3). | Zählweise definieren. Was heißt „überlebt“, was „entscheidbar“? Die Tabelle aus denselben Daten speisen wie die Vorwärtsberichte. |
| /research/evidenz | „Differenz zur Kontrolle: 0,23 [0,1809; 0,31116]“ (hawkes kurssprung, M2) | nein | Intervall ja | **M.** Kein n, keine Einheit (Anteil? Rate?). Fünf Nachkommastellen suggerieren Präzision. | n und Einheit nennen, Intervall auf zwei Stellen runden. |
| /research/evidenz | „Differenz zur Kontrolle: -0,43 [-0,44314; -0,41284]“ (rauigkeit volatilitaet, M2) | nein | Intervall ja | **M.** wie oben. Außerdem unklar, ob -0,43 eine Differenz des Hurst-Exponenten ist (Text: „H etwa 0,1“). | Messgröße benennen. |
| /research/evidenz | „Differenz zur Kontrolle: 0,00052 [-0,00008; 0,00152]“ (M1, „Kontrolle nicht geschlagen“) | nein | Intervall ja | **G.** Das Urteil passt zum Intervall. n und Einheit fehlen. | n und Einheit ergänzen. |
| /research/evidenz | „Von 79 berechenbaren Kandidaten sind realistisch 2 und selbst im günstigsten Fall 31 in unter 24 Monaten entscheidbar.“ | ja (79) | ist selbst eine Power-Aussage | **+** Das ist die wichtigste ehrliche Zahl der Plattform, aber sie steht nur hier. Sie steht im Konflikt mit „Holdout … 20 Handelstage — ausreichend“ auf /research/funde. | Auf Startseite, Bestenliste und Ergebnisvergleich zitieren. |
| /research/evidenz | „Rangkorrelation Einlass-Kennzahl 'pf' gegen DEMO-Ertrag je Trade (EUR) · -0,07 · [-0,8; 0,63] · 8“ | ja | ja | **+ / H in der Konsequenz.** Vorbildlich angegeben: Der Backtest-PF hat keine erkennbare Vorhersagekraft. Die übrigen Seiten zeigen den Backtest-PF aber als Hauptkennzahl ohne diesen Hinweis (Top 10, Rang 7). | Unter jedem In-sample-PF verlinken: „Backtest-PF sagt DEMO-Ergebnis bisher nicht voraus (Kalibrierung).“ |
| /research/evidenz | „Rangkorrelation Einlass-Kennzahl 'placebo_rang' gegen DEMO-Ertrag je Trade (EUR) · 0,62 · [-0,31; 1] · 6“ | ja (6) | ja | **M.** Bei n = 6 wäre die Zahl nach eigener Regel „explorativ“, sie ist aber nicht so gekennzeichnet. Der Punktwert 0,62 wirkt wie ein Beleg, obwohl der Bereich die 0 einschließt. | Als „explorativ (n = 6)“ kennzeichnen. |
| /research/evidenz | „Median(DEMO-PF − Einlass-PF) · -0,61 · [-0,89; -0,48] · 8“ | ja (8) | ja | **M.** Eine zentrale und klare Aussage (DEMO liegt systematisch unter Backtest), bei n = 8 aber ohne Einstufung. Im Rest des Hubs ist sie unsichtbar. | Einstufung ergänzen. Die Aussage prominent auf Strategien und Bestenliste zeigen. |
| /research/evidenz | „einlass_v1_kosten_streng · DEMO · 3 (0) · -0,49 € [-4,77; 3,78] · offen“ | Monate ja, Trades nein | ja | **G.** Korrekt als „offen“ markiert. n (Trades) fehlt. | Trades-n ergänzen. |
| /research/evidenz | „Übernommen: 4 von 28 Läufen; Rücknahmequote 0,00.“ | ja (4/28) | nein | **G.** Eine Quote von 0,00 bei vier Übernahmen klingt nach Qualitätsnachweis. | „0 von 4 zurückgenommen“ schreiben. |

### /research/bestenliste

| Seite | Zahl (wörtlich) | n sichtbar? | Unsicherheit? | Risiko der Fehldeutung | Vorschlag |
|---|---|---|---|---|---|
| /research/bestenliste | Einleitung „ein hoher PF mit Score unter 45 ist eine Anekdote, kein Befund“ | – | – | **H.** Der Umkehrschluss (ab 45 = Befund) wird durch keine Zeile gedeckt. In allen Zeilen mit Score ≥ 45 schließt der angezeigte „PF-90%-CI“ Werte unter 1 ein (z. B. „66.5 … 0.98–2.1“, „58.6 … 0.8–2.64“, „55.2 … 0.99–3.41“). | Klar sagen: „Derzeit liegt bei keinem Kandidaten der Bereich vollständig über 1.“ |
| /research/bestenliste | Zeilenformat „Kandidat · Typ · Score · PF (m.Slip) · PF (o.Slip) · n · PF-90%-CI · Top-3-Anteil · PF +1 Slip · Forward-Bilanz · Verdict“, z. B. „Baustein: Round-Number-Reversal Exit-Lab-A/B II … Entwurf · Score 65.7/100 · 1,48 · — · 227 · 0.96–2.06 · 24% · 1.31 · — · Baustein verbessert Träger (in-sample)“ | ja | Bereich ja; Einstufung fehlt als Spalte | **M.** Das Format ist gut und reichhaltig. Es fehlen die Kennzeichnung „bestes von 3 Trials“ und die Einstufung explorativ/vorläufig. | Spalte „Einstufung“ und Hinweis „bestes Trial“ ergänzen. |
| /research/bestenliste | Plätze 1 und 2: „Round-Number-Reversal Exit-Lab-A/B … Score 66.5/100 · 1,50 · 214“ und „… Exit-Lab-A/B II … Score 65.7/100 · 1,48 · 227“ | ja | ja | **H.** Zwei Exit-Tests desselben Trägers belegen die Spitze. Die Mehrfachtest-Last der Familie ist nicht sichtbar. Das Dossier meldet „UNTERPOWERT“, die Liste nicht. | Varianten eines Trägers gruppieren und eine Spalte „Power“ (unterpowert ja/nein) ergänzen. |
| /research/bestenliste | Urteil „Baustein verbessert Träger (in-sample)“ bei „PF (m.Slip)“ „0,86“ (Hull-Suite Give-Back), „0,96“ (Cootner), „0,79“ (Forward-Vola-Skalierer), „0,72“ (No-Progress), „1,01“ (Closed-Loop, 0.53–1.68) | ja | ja | **H.** Positives Etikett bei verlustbringendem oder nicht unterscheidbarem Ergebnis. | Siehe Formulierungsvorschlag 3. |
| /research/bestenliste | „Baustein: VaR-Sprung-Exit … Score 56/100 · 1,17 · 229 · 0.87–1.59 · Baustein ohne Mehrwert“; „Sigma-Adaptive-Breakout Exit-Lab … 1,59 · 95 · 0.8–2.64 · Baustein ohne Mehrwert“ | ja | ja | **M.** Die angezeigte PF-Zahl gehört zum verworfenen Overlay und ist höher als die des Trägers („Traeger PF 1.33 vs Overlay 1.59“). Das Verwerfungskriterium (offenbar PnL: „+1.379“ vs. „+1.282 Pkt“) steht nirgends. | Kriterium nennen und bei „ohne Mehrwert“ den Träger-PF zeigen. |
| /research/bestenliste | „Afternoon Move Reversal … Score 26.5/100 · 2,60 · 6 · 0.19–66.03 · 100%“; „DAX Range-Scaled Stretch-Breakout … 2,04 · 10 · 0.45–7.33 · 91%“ | ja | Bereich ja, Einstufung nein | **M.** PF 2,60 bzw. 2,04 steht groß da. Nur Bereich und Top-3-Anteil zeigen, dass die Zahlen wertlos sind. | Bei n < 20 den PF durchstreichen wie auf /strategien. |
| /research/bestenliste | „DAX Time-of-Day Momentum (Late-Day) · kuratiert · Score 21/100 · 0,82 · 0,86 · 40 · — · … · 2/2 Monate (+231 Pkt) · Kein Edge nachgewiesen“ | ja (Backtest) | Forward ohne n und ohne Bereich | **M.** Die Forward-Bilanz ist positiv, das Urteil negativ. Die Forward-Bilanz hat kein n. | Forward-n und Bereich ergänzen und das Urteil datieren („Urteil vom 10.06., vor Forward“). |
| /research/bestenliste | „DAX PDH/PDL Fade · kuratiert · Score 31/100 · 2,49 · 2,42 · 26 · — · 2/4 Monate (-145 Pkt) · Schwaches Signal“ | ja | nein (kuratiert: „—“) | **M.** Der dritthöchste PF der Liste steht ohne Bereich. Die Forward-Bilanz ist negativ, das wird aber nur im Kleingedruckten deutlich. | Für kuratierte Zeilen Bereiche nachrechnen oder „ohne Bereich (Altdaten)“ schreiben. |
| /research/bestenliste | Einleitung „die Forward-Bilanz füllt sich ab Juli automatisch“; alle 50 Entwurf-Zeilen „—“ | – | – | **G.** Überholter Text, leere Spalte ohne Grund. | Den Grund nennen („erster voller Monat nach Freeze noch offen“). |
| /research/bestenliste | „1,50“ (PF) neben „0.98–2.1“ (CI) und „1.35“ (PF +1 Slip) | – | – | **G.** Gemischte Dezimalzeichen in einer Zeile. | de-DE durchgehend. |

### /research/kandidaten (50 Entwürfe, 7 kuratierte Untersuchungen)

| Seite | Zahl (wörtlich) | n sichtbar? | Unsicherheit? | Risiko der Fehldeutung | Vorschlag |
|---|---|---|---|---|---|
| /research/kandidaten | Format „(A/B: PF 1.05->1.48, PnL +250->+1615) — Forward/Review noetig“ + „Robustheit (bestes Trial): PF-90%-CI 0.96–2.06 · Top-3-Trades 24% des Gewinns · PF bei +1 Pkt Slippage 1.31“ | in der Variantentabelle („267“, „227“) | Bereich des besten Trials, **nicht** der A/B-Differenz | **M.** Der Leser sieht eine Verbesserung „1.05->1.48“, aber keine Unsicherheit der Differenz. Der Bereich gehört zum besten von drei Trials, ohne Auswahlkorrektur. | Differenz mit gepaartem Bereich zeigen (die Evidenzseite empfiehlt gepaarte Vergleiche selbst). |
| /research/kandidaten | „Der Baustein verbesserte die Ausgangsstrategie auf der Bauhistorie.“ bei „(A/B: PF 1.78->2.00, PnL +361->+362)“ (US-Makro-Gate), „(A/B: PF 0.70->0.72, PnL -259->-240)“ (No-Progress), „(A/B: PF 1.00->1.04, PnL -22->+79)“ (Turnover-Cap), „(A/B: PF 0.96->1.01, PnL -132->+17)“ (Closed-Loop) | ja (Tabelle) | nein | **H.** „Verbessert“ bei Unterschieden, die laut eigener Streuungsangabe („rund hundert Punkten Streuung“ je Trade, /research/evidenz) nicht von Zufall trennbar sind. | Urteil nur bei gepaartem Bereich über 0, sonst „kein messbarer Unterschied“. |
| /research/kandidaten | „(A/B: Traeger PF 1.33 vs Overlay 1.59) — Baustein verworfen, Traeger unveraendert“; „(A/B: Traeger PF 1.11 vs Overlay 1.17) — Baustein verworfen“ | ja | nein | **M.** Ein höherer PF führt zu „verworfen“, weil das Kriterium unsichtbar ist. Das wirkt willkürlich. | Das Entscheidungskriterium in einem Satz ausweisen. |
| /research/kandidaten | „(PF 1.40 bei n=29 < 30 — Stichprobe zu klein)“, „(PF 2.60 bei n=6 < 30 — Stichprobe zu klein)“, „(PF 2.04 bei n=10 < 30 …)“ | ja | Einstufung (eigene Schwelle 30) | **M.** Eine vierte Schwelle (30) neben 20/50/100 ohne Begründung. | Eine Schwellenstaffel für den ganzen Hub. |
| /research/kandidaten | VaR-Eintrag: „Träger ist dax_acd_timed_breakout (PF 1,30, n=153, Robustheit 65, höchste unter den ORB-Bauten)“ vs. Tabelle desselben Eintrags „Träger pur (dax_acd_timed_breakout, Original-Exits) 224 · 48,7% · 1,11“ vs. /strategien „DAX ACD Time-Confirmed Opening-Range Breakout … 29 · 0,56 · 0,19–1,12 · −278,25 €“ vs. /research/funde „dax_acd_timed_breakout 1810 Pkt gegen Zufalls-Median 5198“ (Placebo-Rang 0.0) | ja | nein | **H.** Superlativ („höchste“ Robustheit) auf alter In-sample-Basis. Dieselbe Strategie ist im Placebo-Test auf Rang 0 und im DEMO-Betrieb negativ. Drei unterschiedliche PF-Werte ohne Evidenzklasse. | Superlative streichen und immer „(in-sample, Datum)“ an die Zahl schreiben. |
| /research/kandidaten | „den bereits als tragend bestaetigten fixen Zeit-Cap (Exit-Lab time120_sl25 +712 Pkt)“ | nein | nein | **M.** „Bestätigt“ ohne Evidenzklasse. Es handelt sich um ein In-sample-Exit-Lab. | „im Exit-Lab (in-sample) bisher beste Policy“. |
| /research/kandidaten | „dax_atr_band_trend, den stärksten Trend-Träger (PF 1,41–2,37 in den bisherigen A/Bs)“ | nein | nein | **M.** Superlativ auf der Spanne der besten A/B-Werte, ohne n und ohne Klasse. | Superlativ streichen und n und Klasse ergänzen. |
| /research/kandidaten | Archetyp-Katalog: „Mean Reversion · etabliert · Eigene verfeinerte Variante (HMR) mit Regime-Filtern etabliert“; „Trend / Momentum · etabliert · Eigene Hull-Suite-Variante etabliert“ | nein | nein | **H.** Siehe Top 10, Rang 6. Laut /strategien hat Hull einen Bereich von 0,88–2,53, und beide Strategien sind „Kein DEMO-Slot“. | „Etabliert“ ersetzen durch den aktuellen DEMO-Stand mit Bereich. |
| /research/kandidaten | Archetyp-Katalog: „PDH/PDL-Fade · untersucht · Starkes In-Sample (PF 2.11-2.27), aber Walk-Forward-OOS unbestätigt (PF 0.92)“ | nein | nein | **G.** Ehrlich formuliert, aber „Starkes“ bei n = 24 (siehe unten). | „In-sample hoch (n = 24), OOS nicht bestätigt“. |
| /research/kandidaten | ORB (kuratiert): „Die 15-Minuten-Variante war im Gesamtfenster grenzwertig positiv (PF 1.10) … IS-optimiert PF 1.34, Out-of-Sample PF 0.77 (-229 Pkt)“, „Long-Seite trägt (PF 1.36)“ vs. Tabelle „OR 15 Min · 59 · 30,5% · 1,01“, „OR 15 Min, nur Long · 30 · 26,7% · 0,89“ | Tabelle ja, Text nein | nein | **H.** Die Textzahlen finden sich nicht in der Tabelle, die Long-Aussage widerspricht ihr. | Den Text aus der Tabelle generieren oder datieren („Befund vom 10.06., Tabelle neu gerechnet am …“). |
| /research/kandidaten | Volatility Compression: „Baseline PF 0.57 bei n=153 (über der 100-Trade-Schwelle)“, „Winrate 20%“, „reduziert die Signale auf n=8“ vs. Tabelle „Squeeze 3.0xATR (Baseline) · 132 · 22,7% · 0,67“, „Squeeze 2.0xATR (enger) · 7“ | beides, aber verschieden | nein | **H.** Die Aussage „über der 100-Trade-Schwelle“ stützt sich auf n = 153, die Tabelle zeigt 132. | wie oben. |
| /research/kandidaten | PDH/PDL Fade: „Stärkster In-Sample-Befund des Katalogs“, „Kleinste Drawdowns aller untersuchten Familien (257 Pkt Baseline)“, „Beide Richtungen tragen: Long (PDL-Fade) PF 2.7-2.8, Short (PDH-Fade) PF 1.6-1.8 — kein Drift-Artefakt“, „TP-Verlängerung auf 2.2R zerstört den Edge (PF 1.09)“, „range_day-Tagen (n=29, PF 3.36, +830 Pkt …) ABER +652 Pkt von +771 Pkt stammen aus nur 3 März-Trades … bei n=36“ vs. Tabelle „TP 1.5R + Reaktionsbar (Baseline) · 24“, „TP 2.2R · 24 · 33,3% · 1,30“ | uneinheitlich (24/29/36) | nein | **H.** Drei Superlative und eine Richtungsaussage bei n = 24. „PF 1.09“ steht im Widerspruch zur Tabelle („1,30“). Drei n-Werte ohne Erklärung. Der ehrliche Satz über die drei März-Trades ist vorhanden, wird aber durch die Superlative davor entwertet. | Superlative streichen und mit dem Konzentrationsbefund beginnen. |
| /research/kandidaten | Gap Fade/Follow: „Fade-Baseline klar negativ (PF 0.65)“, „nur 20-24 qualifizierende Gaps in 63 Handelstagen“ vs. Tabelle „min Gap 25 Pkt (Baseline) · 17 · 35,3% · 0,70“ | abweichend | nein | **M.** „Klar negativ“ bei n = 17, eine Zahl, die nicht in der Tabelle steht. | Bei n < 20 kein „klar“ verwenden. |
| /research/kandidaten | Time-of-Day: „Die 16:30-Variante endet bei PF 1.07“, „Selbst die beste Variante (16:30) trägt nur auf der Long-Seite (PF 1.56)“, „15:30: PF 0.36“ vs. Tabelle mit nur „Drift 25 / 35 / 15 Pkt“ | nein | nein | **M.** Die im Text genannten Varianten fehlen in der „vollständig offengelegten“ Trialtabelle. | Alle genannten Varianten in die Tabelle aufnehmen. |
| /research/kandidaten | Noise-Area: „Paper-Default N=1.0 ergibt PF 0.63 (-1046 Pkt) bei n=101 — über der 100-Trade-Schwelle, der Negativ-Befund ist belastbar“, „(N=0.5: PF 0.97, N=1.0: 0.63, N=1.5: 0.89)“, „78 von 101 Trades“ vs. Tabelle „Band 1.0x (Baseline) · 76 · 23,7% · 0,78 · -464 Pkt“, „Band 1.5x (enger) · 52 · … · 0,97“, „Band 0.75x (lockerer) · 90 · … · 0,61“ | abweichend | nein | **H.** „Belastbar“ wird mit n = 101 begründet, die Tabelle zeigt 76. Die Varianten sind anders benannt, und die PF-Werte sind anderen Varianten zugeordnet. | wie ORB. |
| /research/kandidaten | VWAP: „Bei n=171 (über der 100-Trade-Schwelle) ist dieser Negativ-Befund statistisch belastbar.“ vs. Tabelle „Dev 2.0xATR (Baseline) · 141“, „Dev 2.5xATR · 141“, „Dev 1.5xATR · 141“ | abweichend | nein | **M.** „Statistisch belastbar“ ohne Bereich, mit einem n, das nicht zur Tabelle passt. | Bereich angeben und n angleichen. |
| /research/kandidaten | Kuratierte Kopfzeilen: „Robustheits-Score 26.9/100 (… Fenster 2026-03-17..2026-05-24, n=43)“ direkt gefolgt von „Baseline Slippage-Vergleich (n=44, Fenster 2026-03-17..2026-05-27)“; analog 128/132, 16/17, 38/40, 71/76, 135/141 | ja | – | **G.** Zwei Fenster mit drei Tagen Unterschied direkt nebeneinander irritieren. | Ein Fenster pro Untersuchung. |
| /research/kandidaten | „Score 65.7/100“, „Score 29/100“, „Score 1.8/100“ | – | – | **G.** Es bleibt unklar, dass es der Robustheits-Score ist (auf /strategien heißt ein anderer Score „Warchhold Score“). | „Robustheits-Score“ ausschreiben. |

### /research/validierung

| Seite | Zahl (wörtlich) | n sichtbar? | Unsicherheit? | Risiko der Fehldeutung | Vorschlag |
|---|---|---|---|---|---|
| /research/validierung | „Regel: bestätigt nur wenn OOS-PF >= 1.1 UND OOS-n >= 10 UND optimiert >= Baseline im OOS.“ | – | – | **H.** „Bestätigt“ ist ab n = 10 möglich, also unter der eigenen Explorativ-Schwelle (20). Außerdem fehlt eine Evidenzklasse im Wort. | Schwelle auf ≥ 50 anheben oder das Wort „bestätigt“ durch „OOS-Hürde bestanden (explorativ)“ ersetzen. |
| /research/validierung | FOMC-Optimizer: „IS-Auswahl: … (Expectancy 55.4591, PF 2.9803, n=23)“, „Nachbar-Stabilitaet: 4/4 Nachbarn positiv, Nachbar-Ø/Optimum = 0.76 (PLATEAU — robust)“ | ja | nein | **M.** „Robust“ bei n = 23 in-sample. Die OOS-Zeilen danach („Optimiert · 14 · 7.14% · 1.1682 · +71 Pkt“, „Baseline (Anker) · 22 · 9.09% · 1.1977 · +90 Pkt“) zeigen keinen Vorteil. Vier Nachkommastellen suggerieren Präzision. | „Plateau (in-sample)“ statt „robust“. PF auf zwei Stellen runden. |
| /research/validierung | „In-Sample Top 10 (von 24)“ mit „{"max_even_week": 0, "sl_min_points": 30.0, "tp_points": 0.0} · 14 · 14.29% · 3.323 · +933 Pkt“ | ja | nein | **M.** Die höchsten Zahlen (PF 3,323) stehen oben und prominent, obwohl sie in-sample und ausgewählt sind. | OOS-Ergebnis zuerst zeigen, IS-Top-10 zuklappen. |
| /research/validierung | Forward 2026-08: „Time-of-Day Momentum (16:30) · 20 · 30.0% · 1.4323 · +103.28 Pkt“, „Overnight-Gap-Fade ohne Close-Location-Gate (A/B-Kontrolle) · 17 · 41.18% · 1.5189“, „Open-Fade plus Trend-Tag-Block … · 9 · 33.33% · 0.411“ | ja | **nein** (kein Bereich, keine Einstufung) | **M.** Echte Out-of-Sample-Ergebnisse, aber ohne explorativ/vorläufig-Etikett (n = 9, 17, 20) und mit vier Nachkommastellen. | Einstufung und Bereich wie auf /strategien. |
| /research/validierung | „Stand 2026-09-28T18:36:09+02:00: 14 Kandidaten fuer 2026-08 gemessen“ | ja | – | **M.** Widerspricht „Vorwärts … 0 · 0 · 0“ auf /research/evidenz (siehe dort). | Zählweise angleichen. |

### /research/funde

| Seite | Zahl (wörtlich) | n sichtbar? | Unsicherheit? | Risiko der Fehldeutung | Vorschlag |
|---|---|---|---|---|---|
| /research/funde | Lauf 2026-09-28: „Die ORB-Familie steht bei 62 Trials in 6 Builds … Die PF 1.2–1.3 liegen im Band, in dem der Digest „~57 Zufallstreffer“ erwartet“, „UNTERPOWERT (EV 8–19 gegen MDE 20–36 Pkt)“ | ja (Trials) | ja (Power) | **+** Mehrfachtest-Last und Power offen benannt. | Diese Angaben auch in Bestenliste und Dossiers führen. |
| /research/funde | Lauf 2026-09-28: „Deine DAX-Backtest-Prognose: Anker-Arme B/C erreichen einen PF innerhalb von ±0.1 um Arm A.“ Ergebnis auf /research/kandidaten: „Träger pur … 1,03“, „Fetna fester Offset-Anker +90 Bars … 0,60“, „+180 Bars … 0,99“ | – | – | **M.** Die vorab registrierte Prognose wird nirgends gegen das Ergebnis ausgewertet. Arm „+90 Bars“ liegt laut den beiden zitierten Zahlen außerhalb von ±0,1. | Prognose und Treffer im Dossier gegenüberstellen („Prognose verfehlt“). |
| /research/funde | Lauf 2026-09-28: „dax_acd_timed_breakout 1810 Pkt gegen Zufalls-Median 5198, dax_timely_orb 795 gegen 4211, dax_sigma_adaptive_breakout 1459 gegen 4574“ | nein | Placebo-Rang ja („Rang 0.0“) | **G.** Eine klare Negativaussage. Sie widerspricht aber dem Superlativ auf /research/kandidaten (siehe VaR-Eintrag). | Querverweis setzen. |
| /research/funde (aufklappbar) | Signal-Mining 2026-09-27: „2 Fund(e) ueber der Evidenzschwelle (Walk-Forward bestanden, OOS n >= 25, OOS PF >= 1.15, IS n >= 50)“, „Holdout 2026-08-08 bis 2026-09-06 = 20 Handelstage — ausreichend“, „4896 Kombinationen“, Trials 1–3 jeweils „IS n=63 PF 1.41, OOS n=28 PF 1.45 WR 60.7 % EV +3.17 Pkt“, „Stabilitaet 1.00, Composite 0.547“ | ja | nein (ein Hinweis „ein Hinweis, kein Beweis“ steht weiter unten) | **H.** „Bestanden“ und „ausreichend“ nach der Auswahl aus 4896 Kombinationen bei OOS n = 28. Die Evidenzseite sagt, realistisch seien nur 2 von 79 Kandidaten in unter 24 Monaten entscheidbar. „Stabilitaet 1.00“ entsteht, weil die drei Trials identische Ergebnisse haben. Das ist kein Stabilitätsbeleg. | „Walk-Forward-Hürde erreicht (Auswahl aus 4.896, explorativ)“. „Ausreichend“ streichen, identische Trials als einen zählen. |
| /research/funde (aufklappbar) | Empirie 2026-09-27: „Das stärkste Muster ist F2-Montag (Eröffnungsstunde, n=19, +42.4 Pkt, t=2.89) … Wahrscheinlichkeit für mindestens ein \|t\| ≥ 2.89 durch Zufall bei rund 35–40 %. … Ergebnis: 0 Funde.“ | ja | ja (Mehrfachtest-Rechnung) | **+** Vorbildlich: Bonferroni-Kontext, Zufallserwartung, n. | Als Muster für alle Befundtexte nehmen. |
| /research/funde (aufklappbar) | Klassik 2026-09-28: „Flagge = dax_flag_continuation (kein Edge, PF 0.65); Pre-Open-Gate = (baustein_verbessert, PF 1.20 → 1.97, unterpowert)“ | nein | „unterpowert“ ja | **M.** „PF 0.65“ ist der Placebo-Arm (siehe Flaggen-Dossier). Beim Pre-Open-Gate ist „unterpowert“ vorbildlich angegeben. | Wie Dossier korrigieren. |

### /research/vermessung

| Seite | Zahl (wörtlich) | n sichtbar? | Unsicherheit? | Risiko der Fehldeutung | Vorschlag |
|---|---|---|---|---|---|
| /research/vermessung | „runde marke 100 · 55.0 % · 52.0 % · +2.9 Pp · 7.0 % · 1,000 · 362 / 111 · zu grob gemessen: auflösbar erst ab 7.0 % Unterschied“ (analog 8 Ebenenarten) | ja | ja (auflösbarer Mindestunterschied) | **+ / G.** Das Konzept ist vorbildlich: Kontrolle, Auflösungsgrenze, Nicht-Fund ≠ Ergebnis. Allerdings steht p in allen 8 Zeilen bei „1,000“ (mit Komma als Tausender lesbar), und Prozent werden mit Punkt geschrieben. | p als „1,00“ bzw. „p = 1“ schreiben, Zahlen in de-DE. Erläutern, dass p korrigiert ist. |
| /research/vermessung | „Für 7 von 8 Ebenenarten war die Messung zu grob, um einen Unterschied von 4 % überhaupt von null zu trennen.“ | ja | ja | **+** Eine ehrliche Power-Aussage. | – |
| /research/vermessung | Fehlerberichte „Gemessen wurden 56,6 % Abpraller — ein reiner Zufallslauf liefert bei dieser Geometrie 58,3 %“, „91 % Abpraller gegen 55 % in der Kontrolle, mit p = 0,000“ | nein | – | **+** Eigene Messfehler werden mit Zahlen offengelegt. „p = 0,000“ sollte „p < 0,001“ heißen. | „p < 0,001“. |

### Build-Dossier `2026-09-28_dax_round_number_reversal_sl15_exit_lab`

| Seite | Zahl (wörtlich) | n sichtbar? | Unsicherheit? | Risiko der Fehldeutung | Vorschlag |
|---|---|---|---|---|---|
| Dossier RN II | Kopf: „Score 65.7/100 · Baustein verbessert Träger (in-sample) · Tier 1“ | – | – | **M.** „Tier 1“ ist unerklärt und klingt wie eine Qualitätsstufe. Gemeint ist offenbar die Quellenstufe. | „Quelle: Tier 1“ ausschreiben. |
| Dossier RN II | Trials: „Baseline … 267 · 43,1% · 1,05 · +250 · 971 · 1,10“, „time120_sl25 … 259 · 25,1% · 1,32 · +1.695“, „sl15_notp … 227 · 13,2% · 1,48 · +1.615“ | ja | nein (je Trial) | **M.** Das „beste“ Trial (höchster PF) hat weniger PnL als das zweite. Die Auswahlregel ist unklar. | Auswahlkriterium nennen. |
| Dossier RN II | „PF-90%-Konfidenzintervall 0.96 – 2.06“ und darunter „Block-Bootstrap-CI (Tages-Blöcke, ehrlicher als iid): PF 0.95–2.15 über 117 Handelstage“ | ja | zwei Bereiche | **M.** Groß gezeigt wird der engere Bereich, den die Seite selbst als weniger ehrlich einstuft. Bestenliste und Kandidaten übernehmen ebenfalls den engeren. | Überall den Block-Bootstrap zeigen. |
| Dossier RN II | „EV 7.12 Pkt vs. minimal nachweisbarer Edge 9.99 Pkt — UNTERPOWERT (Stichprobe zu klein für diesen Edge).“ | ja | ja | **+ / M.** Vorbildlich, aber nur hier. Auf der Bestenliste steht derselbe Kandidat auf Platz 2 ohne diesen Hinweis. | Auf Bestenliste und /research mitführen. |
| Dossier RN II | „Kandidat 1615.3 Pkt vs. Placebo-Median 1170.6 / p95 1654.3 Pkt · Perzentil-Rang 90% · Kandidat zwischen p80 und p95 — schwacher Timing-Beitrag“ | nein | ja (Perzentil) | **G.** Korrekt eingeordnet. | – |
| Dossier RN II | Regime: „range_day · 169 · 1.61 · +1479“, „volatile_day · 25 · 3.18 · +744“, „mixed_day · 6 · 0 · -117“, „trend_down_day · 10 · 0 · -187“, „trend_up_day · 17 · 0 · -304“ | ja | nein | **M.** Nachträgliche Regime-Aufteilung ohne Bereich. „PF 3.18“ bei n = 25 lädt zum Regime-Filtern ein. „PF 0“ heißt offenbar „kein Gewinner“. | Als „nachträgliche Aufteilung, explorativ“ kennzeichnen, „PF 0“ als „0 Gewinner“ schreiben. |
| Dossier RN II | Monate „2026-05: +781“, „2026-07: +732“, „2026-08: -178“ | nein | nein | **G.** Einheit fehlt (Pkt). Das Ergebnis konzentriert sich auf zwei Monate, das wird nicht kommentiert. | Einheit ergänzen, Leave-one-month-out-Satz wie bei anderen Dossiers. |
| Dossier RN II | Forward-Kasten: „Eingefroren – erster Messmonat August 2026“ vs. „2026-10 · 2026-11 · 2026-12 · 2027-01“ und „Freeze 2026-09-28“; Kriterien „mindestens 30 Trades, PF ≥ 1,20“ | – | – | **M.** Der erste Messmonat liegt vor dem Freeze. Die Freigabeschwelle (30 Trades, PF 1,20) ist eine andere als bei der Validierungsregel (n ≥ 10, PF ≥ 1,1). | Monat korrigieren und die Schwellen hubweit angleichen. |
| Dossier RN II | „Nikkei-Cross-Check … n=? · PF ? · PnL ? Pkt“ | Platzhalter | – | **G.** Platzhalter statt Zahl. | Ausblenden, bis Werte vorliegen. |

### Build-Dossier `2026-09-28_dax_flag_continuation`

| Seite | Zahl (wörtlich) | n sichtbar? | Unsicherheit? | Risiko der Fehldeutung | Vorschlag |
|---|---|---|---|---|---|
| Dossier Flagge | Trials: „Baseline (Quelle: 15m-Flagge …) · 1 · 0,0% · 0,00 · -74“, „5m-Signal-Bars … · 0 · 0,0% · 0,00 · +0“, „Placebo: reiner 10-Bar-Bruch … · 56 · 28,6% · 0,65 · -849“ | ja | – | **H.** Die Hypothese wurde praktisch nicht getestet (n = 1 bzw. 0). | Urteil „nicht prüfbar – zu wenige Signale“. |
| Dossier Flagge | „Robustheit (bestes Trial) · PF-90%-Konfidenzintervall 0.34 – 1.1 · Top-3-Trades 29% · PF bei +1.0 Pkt Slippage 0.62“, „Score 29/100“ | ja | ja | **H.** Alle Kopfzahlen stammen aus dem Placebo-Arm. | Kontrollarme von der Wahl des „besten Trials“ ausschließen. |
| Dossier Flagge | „Kandidat -849.2 Pkt vs. Placebo-Median 1074.2 / p95 1541.5 Pkt · Perzentil-Rang 0%“ | nein | ja | **H.** Hier wird ein Placebo gegen Placebos getestet. Das ergibt keine Aussage über die Flagge. | Placebo-Kontrolle nur für Hypothesenarme. |
| Dossier Flagge | „Block-Bootstrap-CI … PF 0.35–1.1 über 44 Handelstage. EV -15.16 Pkt vs. minimal nachweisbarer Edge 22.07 Pkt — UNTERPOWERT“ | ja | ja | **M.** Das ist formal korrekt, beschreibt aber ebenfalls den Placebo-Arm. | wie oben. |

---

## Formulierungsvorschläge für die fünf schlimmsten Fälle

**1. Ergebnisvergleich: Podium** (`/hall-of-fame`)

- **Vorher:** Seitentitel „Hall of Fame“; Karten „#1 DEMO (Spielgeld) hull_suite_v3_final · PF-Bereich (90 %) 0,88–2,53 · Punktwert 1.52“, „#2 …“, „#3 Vorwärtstest Time-of-Day Momentum (16:30) · PF-Bereich (90 %) — · Punktwert 1.43 · vorläufig“.
- **Nachher:** Seitentitel „Ergebnisvergleich (DEMO und Vorwärtstest)“. Statt Karten ein Hinweis und zwei getrennte Tabellen:
  > „Derzeit liegt bei **keiner** Strategie der 90-%-Bereich des Profit-Faktors vollständig über 1. Die Reihenfolge unten ist deshalb keine Rangliste von Gewinnern, sondern eine Sortierung nach der unteren Bereichsgrenze.
  > Tabelle A – DEMO-Konto (Euro): Hull Suite v3 Final · n = 83 · PF 1,52 (90 %: 0,88–2,53) · +219 €
  > Tabelle B – Vorwärtstest, simuliert (Punkte): Time-of-Day Momentum 16:30 · n = 20 · PF 1,43 (vorläufig, Bereich noch nicht berechnet) · +103 Pkt · Hinweis: im Katalog als ‚kein Edge‘ eingestuft (Backtest vom 10.06.).“

**2. Einheiten Punkte und Euro** (`/hall-of-fame`, Vorwärtstest-Zeilen)

- **Vorher:** „Time-of-Day Momentum (16:30) · Vorwaertstest 2026-08 (simuliert auf neuen Daten) · 20 · — · 1.43 (vorläufig) · 30.0 · 103 EUR“
- **Nachher:** „Time-of-Day Momentum (16:30) · Vorwärtstest 08/2026, simuliert · n = 20 (vorläufig) · PF 1,43 · Trefferquote 30 % · **+103 Punkte** (kein Kontoergebnis)“

**3. Urteil „Baustein verbessert Träger“** (`/research/kandidaten`, `/research/bestenliste`, `/research`)

- **Vorher:** „Regelbasiertes Urteil: Der Baustein verbesserte die Ausgangsstrategie auf der Bauhistorie. … (A/B: PF 1.78->2.00, PnL +361->+362) — Forward/Review noetig“ bzw. in der Bestenliste „Baustein: Hull-Suite Give-Back Exit-Lab … 0,86 … Baustein verbessert Träger (in-sample)“.
- **Nachher (je nach Fall):**
  > „In-sample kein messbarer Unterschied: PnL +361 → +362 Pkt; gepaarte Differenz je Trade [Bereich], schließt 0 ein. Baustein bleibt im Vorwärtstest, ist aber nicht belegt.“
  > „In-sample weniger Verlust als der Träger (PF 0,67 → 0,86), bleibt aber verlustbringend (PF < 1). Kein Kandidat.“

**4. „HMR … etabliert / trägt“** (`/research/kandidaten`)

- **Vorher:** „Mean Reversion · etabliert · Eigene verfeinerte Variante (HMR) mit Regime-Filtern etabliert.“ und „Die hauseigene, stark verfeinerte Mean-Reversion (HMR, mit Regime-Filtern) trägt — die nackte VWAP-Form nicht. Mean Reversion auf DAX ist offenbar nur MIT Regime-/Filterlogik handelbar.“
- **Nachher:**
  > „Mean Reversion · untersucht · Die hauseigene Variante mit Regime-Filtern (dax_hyper_mean_reversion) war im Backtest positiv. Im DEMO-Betrieb liegt sie bei PF 0,75 (n = 306, 90 %: 0,59–0,92) und ist inzwischen ohne DEMO-Slot. Die Filterlogik hat den Verlust verringert, einen Vorteil aber nicht belegt.“

**5. Ereignisliste mit „bestes Trial“** (`/research`)

- **Vorher:** „Baustein: Pre-Open-Lern-Gate … → Baustein verbessert Träger (in-sample) · bestes Trial: PF 1.97, n=41 · ENTWURF“
- **Nachher:**
  > „Pre-Open-Lern-Gate auf Overnight-Gap-Fade → im Backtest besser als der Träger · bestes von 3 Varianten: PF 1,97 (**vorläufig**, n = 41, 90 %: 0,99–3,41, 40 % des Gewinns aus 3 Trades) · **unterpowert** · Hinweis: Backtest-PF lag im DEMO bisher im Median 0,61 niedriger (Kalibrierung, n = 8).“

---

## Werden Negativbefunde fair gezeigt?

**Kurzurteil: Negativbefunde sind vollständig vorhanden, aber weniger prominent als Erfolge. Die Verfügbarkeit ist fair, die Gewichtung nicht.**

**Was fair ist (mit Fundstelle):**
- `/strategien` zeigt alle 17 Vault-Strategien einschließlich der negativen, mit n, Bereich und Einstufung. Status „Läuft · negativ“ ist ein eigener Filter, und der Text sagt: „Negative und leere Werte sind Absicht“.
- `/research/kandidaten` zeigt 50 Entwürfe im gleichen Format, davon laut Urteilszeilen 21× „Der Test fand keinen Vorteil gegenüber Zufall und Kosten“, 10× „Der getestete Baustein verbesserte die Ausgangsstrategie nicht“ und 4× „Ein schwaches, unbestätigtes Signal“. Dazu kommen 15× „verbesserte … auf der Bauhistorie“.
- `/research/evidenz` weist aus: „widerlegt 36“ gegenüber „bestätigt 14“ (in-sample), „9 Kandidaten, davon überlebt: 0“, die Selbstkalibrierung „Median(DEMO-PF − Einlass-PF) -0,61“ und „realistisch 2 … in unter 24 Monaten entscheidbar“.
- `/research/vermessung` legt eigene Messfehler mit Zahlen offen („91 % Abpraller gegen 55 % … der ganze Wert war Aufbau“).
- `/research/funde` enthält vorbildliche Nullbefunde („Ergebnis: 0 Funde“, Mehrfachtest-Rechnung).

**Wo die Gewichtung kippt:**
1. **Die Bühne gehört den Erfolgen.** `/hall-of-fame` hebt die drei höchsten Plätze in großen Karten hervor. Die negativen Zeilen stehen nur klein in der Tabelle. Eine vergleichbare Seite „Was nicht funktioniert hat“ gibt es nicht.
2. **Die Urteilssprache ist asymmetrisch.** Positive Etiketten („verbessert Träger“, „etabliert“, „trägt“, „stärkster“, „robust“) werden großzügig vergeben, auch bei PF < 1 oder +1 Punkt Differenz. Negative Etiketten sind nüchtern.
3. **Die stärksten Warnungen stehen abseits.** Die Kalibrierung (Backtest-PF sagt DEMO nicht voraus, liegt im Median 0,61 höher) und die Entscheidbarkeit (2 von 79) stehen nur auf `/research/evidenz`. Dagegen stehen In-sample-PF-Werte wie „1.97“, „2,17“, „2,32“ auf Start-nahen Seiten ohne diesen Kontext.
4. **Die Bilanz fehlt am Einstieg.** Startseite und `/research` zeigen die letzten fünf Ereignisse (heute drei „verbessert“, zwei negativ), aber keine Gesamtbilanz. Ein Erstleser bekommt so einen positiveren Eindruck als die Evidenzseite hergibt.
5. **Die Mehrfachtest-Last ist unsichtbar, wo gerankt wird.** Die Bestenliste zeigt zwei Exit-Varianten desselben Trägers auf Platz 1 und 2, ohne Trial-Zahl der Familie und ohne „unterpowert“. In `/research/funde` ist beides vorbildlich angegeben.

**Empfehlung in einem Satz:** Die vorhandenen ehrlichen Zahlen (Kalibrierung, Entscheidbarkeit, „0 überlebt“, Mehrfachtest-Last) gehören an die Stellen, an denen Erfolge gezeigt werden. Das Podium sollte durch eine nüchterne Vergleichstabelle ersetzt werden, und für „verbessert“ und „bestätigt“ braucht es eine Mindestanforderung: gepaarter Bereich ohne 0, n ≥ 50, Evidenzklasse im selben Satz.
