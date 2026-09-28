# Methodik-Gutachten: autonome Forschungsplattform für algorithmischen DAX-Handel

**Gutachter:** unabhängig; Schwerpunkt empirische Finanzmarktforschung und Statistik (Mehrfachtests, Überanpassung, Vorwärtsvalidierung)
**Datum:** 28.09.2026
**Grundlage:** Exportpaket `warchhold-methodik-20260928T172136Z.zip` (Format `warchhold-cloud-review-export-v1`, Paket „methodik“) mit neun Dateien. Die SHA-256-Werte stehen in `EXPORT_MANIFEST.json`.

| Kurzname im Gutachten | Datei |
|---|---|
| **ABLAUF** | `docs/spezifikation-forschungsablauf-v1.0.md` (Stand 30.08.) |
| **VA03** | `docs/spezifikation-forschungsablauf-va03-placebo-und-folgeversion-2026-08-31.md` |
| **AUSSTIEG** | `docs/spezifikation-ausstiegsforschung-v1.0.md` (Stand 18.09., „noch nicht gebaut“) |
| **WERKZEUGE** | `docs/forschungsmethodik-werkzeuge-2026-09-03.md` |
| **LERNKREIS** | `docs/umsetzungskette-lernkreis-2026-09-23.md` |
| **ANALYSE** | `docs/analyse-weiterentwicklung-autonomes-forschungssystem-2026-09-23.md` |
| **RUA** | `docs/spezifikation-korrektur-unabhaengigkeit-r-u-a-2026-09-08.md` |
| **AUDIT** | `docs/audit-umsetzung-research-hub-2026-09-26.md` |
| **AGENTEN** | `CLAUDE_research_agents.md` |

**Was geprüft wurde und was nicht:** Geprüft wurden ausschließlich die Spezifikationen. Code, Daten, Ledger und Ergebnisdateien lagen nicht vor. Aussagen über die Umsetzung stützen sich darauf, wie die Dokumente sie beschreiben. Alle Zahlen im Gutachten stammen aus den Dateien; wo ich schlussfolgere, ist das als Schlussfolgerung formuliert.

---

## 1. Gesamturteil

**Die Plattform ist gegen die klassischen handwerklichen Fehler ungewöhnlich gut geschützt. Ihre Entscheidungsstatistik hält mit diesem Schutz aber nicht Schritt.**

**Stärken, die ausdrücklich zu würdigen sind:**
- Das Trial-Ledger wird vor der Ausführung erzwungen (AGENTEN, „Governance-Paket P1–P3“).
- Das Embargo von 21 Tagen ist technisch durchgesetzt (ebd.).
- Ergebnisparität wird getrennt von Datenparität geprüft (ABLAUF, Stufe 4).
- Die Look-Ahead-Lektion aus der Reaktionskarte wurde als eigene Sperre verankert (ABLAUF §5).
- Das Hypothesenregister kennt die Urteile „undecidable“ und `post_hoc` (WERKZEUGE §2).
- Die Plattform kalibriert ihre eigenen Hürden (LERNKREIS, LK-2).
- Die Power-Rechnung wird ehrlich offengelegt (LERNKREIS, LK-4.3).
- Die eigenen Fehler sind offen dokumentiert, samt Korrekturen der Korrekturen (RUA §3a, LERNKREIS LK-0.1).

**Die methodischen Schwächen liegen in vier Bereichen:**

1. **Die Entscheidungsregeln sind überwiegend feste Schwellen, keine Inferenz unter Mehrfachtests.**
   - Das Verdict-Gate „PF ≥ 1,3 & n ≥ 100“ wird ausdrücklich als „Multiple-Testing-Schutz“ bezeichnet (AGENTEN), wächst aber nicht mit der Zahl der Versuche.
   - Holm, Benjamini-Hochberg oder Verfahren der Data-Snooping-Literatur sind in keiner der neun Dateien spezifiziert. Wo eine „Mehrfachtest-Korrektur“ verlangt wird (ABLAUF Stufe 0; AUSSTIEG §5), bleibt das Verfahren unbenannt.
2. **Fehlende Messbarkeit wird oft als Abwesenheit gelesen.**
   - Die eigene Power-Rechnung sagt, dass kein Kandidat realistisch unter 24 Monaten entscheidbar ist (LERNKREIS LK-4.3).
   - Trotzdem werden „0 von 9 überlebt“, „kein Edge“ und „widerlegt“ als Befunde über Existenz geführt, und Stufe 0 schließt aus zu grober Auflösung auf „nichts Handelbares“ (WERKZEUGE §5).
3. **Die Vorregistrierung ist formal sauber, aber epistemisch durchlässig.**
   - Rückkopplungsschleifen (Digest, gemeinsames Agentengedächtnis, „Situations-Hinweise“) formen neue Hypothesen aus derselben kurzen Historie, auf der sie danach getestet werden.
   - Eine versiegelte Reserve gibt es nicht (ANALYSE §15.2; Entscheidung E3 offen).
4. **Die Messlatte ist nicht gegen die eigene Autonomie gesichert.** Nach der Betreiberentscheidung E1 darf der autonome Entwickler alles übernehmen außer einer kurzen Sperrliste. Einlassschwellen, Score-Formel und Urteilslogik stehen nicht darauf; geschützt sind sie nur durch eine Prompt-Regel (LERNKREIS LK-3, Stand 26.09.).

**Beurteilung:** Der Ablauf (Vermessen → Vermuten → Bauen → Justieren → Prüfen → Vorschlagen) ist als Gerüst tragfähig. Als **Beweisverfahren** trägt er bei der gegebenen Datenmenge (rund 104 bis 136 Handelstage je nach Dokument, ein Instrument) nur, wenn drei Dinge geschehen:
- die Urteile werden konsequent nach Power dreigeteilt,
- die Mehrfachtest-Kontrolle wird auf Familienebene statistisch statt per Schwelle geführt,
- das Forschungsbudget wird von vielen Einzelstrategien auf wenige, informationsreiche Messungen umgeschichtet.

Die Ausstiegsforschung ist im Kern richtig begründet (Kaminski & Lo, 2014). Ihre Entscheidungsregel kann die Suche aber aus dem falschen Grund beenden (Befund H6).

---

## 2. Befunde nach Schwere

Schweregrade:
- **H (hoch):** kann zu falschen Forschungsschlüssen oder falschen Einsatzentscheidungen führen.
- **M (mittel):** verzerrt Aussagen oder verschwendet Budget.
- **G (gering):** Klarheit und Konsistenz.

### 2.1 Hoch

| Nr | Datei · Abschnitt | Befund | Begründung | Vorschlag |
|---|---|---|---|---|
| **H1** | WERKZEUGE §4 (Kennzahl 1: „belastbar geprüft (≥ 3 Monate und kumuliert ≥ 60 Trades): 7 Kandidaten, 0 überlebt“); WERKZEUGE §7; ANALYSE §0 („0 von 9“); LERNKREIS LK-4.3 | **„0 überlebt“ wird als Beleg gegen Existenz gelesen, obwohl die eigene Power-Rechnung Entscheidbarkeit ausschließt.** | LK-4.3: „Typische Kandidaten haben +5 bis +10 Punkte Erwartungswert bei rund 100 Punkten Streuung je Trade. Ein Nachweis braucht über tausend Trades.“ Eine Klasse „belastbar geprüft“ ab 60 Trades liegt eine Größenordnung darunter. Wenn alle Kandidaten einen echten kleinen Vorteil hätten, wäre „0 von 9 überleben“ bei dieser Power **ebenfalls** das erwartbare Ergebnis. Die Zahl unterscheidet also kaum zwischen „kein Vorteil“ und „Vorteil zu klein zum Messen“. Ioannidis (2005) zeigt, dass bei geringer Power und vielen Tests schon positive Befunde wenig verlässlich sind. Negative Befunde sind bei geringer Power ohnehin kaum aussagekräftig. | Kennzahl 1 dreiteilen: „bestanden / gescheitert **mit ausreichender Power** / nicht entscheidbar“. „Belastbar geprüft“ erst ab der Stichprobe vergeben, die der Power-Plan für die jeweilige Effektgröße verlangt. Die Bezeichnung „belastbar“ für ≥ 60 Trades streichen. |
| **H2** | WERKZEUGE §5 („Die Auflösung von 7 bis 27 Punkten je halbe Stunde sagt, dass alles, was handelbar gewesen wäre, sichtbar gewesen wäre.“) gegen LERNKREIS LK-4.3 (+5 bis +10 Punkte Erwartungswert) | **Nicht-Messbarkeit wird als Abwesenheit berichtet**, entgegen der eigenen Regel 4 der Stufe 0 (ABLAUF §3, „Ein Nicht-Fund bei zu grober Auflösung ist kein Ergebnis“). | Die Plattform selbst beziffert typische Kandidaten-Effekte auf 5 bis 10 Punkte. Eine Auflösung von 7 bis 27 Punkten hätte solche Effekte in weiten Teilen **nicht** gesehen. Die Einheiten unterscheiden sich (Halbstunden-Bucket gegen Trade), und genau deshalb ist der Schluss „alles Handelbare wäre sichtbar gewesen“ nicht belegt. Die Stufe-0-Spezifikation formuliert das Prinzip korrekt, der Bericht verletzt es. | Den Satz in WERKZEUGE §5 zurücknehmen. Jede Nullmeldung der Stufe 0 bekommt drei Felder: Mindestgröße, erreichte Auflösung und die Aussage „kleinere Effekte nicht ausgeschlossen“. Die kostenrelevante Mindestgröße wird aus dem Kostenmodell abgeleitet und vorab registriert. |
| **H3** | AGENTEN, „Research-Vorgehen“ („Das Verdict-Gate (PF>=1.3 & n>=100) bleibt der Multiple-Testing-Schutz“); ABLAUF §3 Stufe 0, Übergang Punkt 3; AUSSTIEG §5; LERNKREIS LK-5 Stufenregel (M2); AGENTEN „Empirie-Research“ (Bonferroni-Kopf) | **Keine spezifizierte Mehrfachtest-Kontrolle auf Familienebene.** Holm oder Benjamini-Hochberg sind nirgends festgelegt. Wo „Korrektur über den ganzen Lauf“ verlangt wird, fehlt das Verfahren. Mechanismen erreichen M2, sobald ein 90-%-Intervall 0 ausschließt, ohne Korrektur über die gleichzeitig geprüften Mechanismen. | Eine feste PF-Schwelle ist keine Mehrfachtest-Kontrolle: Die Wahrscheinlichkeit, dass *irgendein* Trial sie zufällig überschreitet, wächst mit jedem Versuch. Das Ledger zählt die Versuche (22.449 laut ANALYSE §0), die Entscheidungsregel nutzt diese Zahl aber nicht. Das Hochsetzen von Hürden mit der Zahl der Versuche ist in der Finanzliteratur Standard (Harvey, Liu & Zhu, 2016). Für die Auswahl der besten aus vielen Regeln gibt es eigene Tests (White, 2000; Hansen, 2005; Romano & Wolf, 2005; empirisch für technische Regeln Sullivan, Timmermann & White, 1999). | Eine Entscheidungsschicht mit benannten Verfahren vorregistrieren: **(a) Entdeckung** (Stufe 0, Signal Mining, Mechanismen) mit FDR-Kontrolle nach Benjamini & Hochberg (1995), bei abhängigen Tests nach Benjamini & Yekutieli (2001) oder mit resamplingbasiertem max-t nach Westfall & Young (1993). **(b) Bestätigung** (Einlass, Vorwärtsurteil) mit FWER-Kontrolle nach Holm (1979) oder Romano & Wolf (2005) innerhalb der Familie. **(c) Auswahl der besten Regel** eines Builds oder Grids über Reality Check oder SPA. Die Familiendefinition (welche Versuche zählen zusammen) gehört ins Ledger. |
| **H4** | WERKZEUGE §4 (Kennzahl 4, `SIGNAL_ALPHA = 0,05` als „erklärte Annahme“); ANALYSE §0 (Signalrate 0,038 gegen 0,05) | **Die zentrale Aussage „Signalrate unter Zufallserwartung“ steht auf einer unkalibrierten Null-Rate und einer unklaren Familie.** | (1) Die Null-Wahrscheinlichkeit, dass ein Trial **nach Kosten** PF ≥ 1,1 bei n ≥ 30 erreicht, ist nicht 5 %, sondern hängt von Kosten, Haltedauer, Trefferprofil und n ab. Mit Kosten liegt sie für Regeln ohne Brutto-Vorteil vermutlich **unter** 5 %. Dann könnte 3,5 % sogar **über** der Null liegen. Die Richtung der Schlussfolgerung ist ohne Kalibrierung offen. (2) Die Trials sind stark abhängig (ANALYSE: 87 % Rasterpunkte aus vier Läufen; je Bau drei Varianten derselben Idee). Die Varianz der Signalzahl unter der Null ist dann viel größer als binomial. (3) Kennzahl 4 rechnet mit 424 Trials, das Ledger zählt 22.449, AUSSTIEG §8 nennt 16.963. Welche Familie gemeint ist, bleibt offen. | Die Null-Rate **empirisch** bestimmen: Dieselbe Bau-Pipeline (dieselben drei Varianten, dieselben Kosten, dieselbe Urteilslogik) läuft auf (a) Placebo-Einstiegen und (b) synthetischen Kursen ohne Vorhersagbarkeit mit echten Spreads. Das liefert die Null-Rate samt Streuung (Simulationsintervall). Kennzahl 4 nur gegen diese Null berichten und die Familie explizit definieren. |
| **H5** | AGENTEN, „Ergebnis-Digest — Rückkopplungs-Schleife“, „Digest-Baustein-Fallback“, „Agenten-Langzeitgedächtnis“; ANALYSE §15.2 und §15.4; AGENTEN „Empirie-Research“ (Fensterstaffelung) | **Vorregistrierung ohne Datenhygiene:** Hypothesen werden aus Ergebnissen derselben Historie abgeleitet (Situations-Hinweise, Regime-Splits, Gedächtnis) und dann auf dieser Historie „vorregistriert“ getestet. Eine versiegelte Reserve fehlt. Der Empirie-Holdout (heute−48 bis heute−21) rollt wöchentlich und wird dadurch mehrfach verwendet. | Vorregistrierung verhindert nur Anpassungen **nach** dem Test, nicht die Nutzung derselben Daten **vor** der Formulierung. Das ist der „Garden of Forking Paths“ (Gelman & Loken, 2014) auf Systemebene. Die Pipeline ist adaptiv: Jede Runde sieht die Ergebnisse der letzten. Ein mehrfach adaptiv befragter Holdout verliert seine Gültigkeit (Dwork et al., 2015, „The reusable holdout“, Science). Das Etikett „DIGEST-ABGELEITET (in-sample)“ ist richtig, aber das Register behandelt diese Hypothesen nicht als `post_hoc`. | (1) Das **Datenexpositionsbuch** (LERNKREIS LK-2) zur Pflicht machen: Jede Hypothese trägt die Fenster, deren Ergebnisse ihr Urheber nachweislich gesehen hat. Tests auf diesen Fenstern zählen als in-sample, egal wann registriert wurde. (2) Versiegelte **Wochenblöcke** (E3) sofort einführen, auch wenn der Vorwärtstest dadurch Tage verliert. (3) Den Kontext der Agenten (Digest, Gedächtnis) um die jüngsten 21 Tage und um DEMO-Ergebnisse bereinigen, sonst unterläuft der Kontext das Embargo (ANALYSE §15.4 benennt das selbst). |
| **H6** | AUSSTIEG §2, §3 (Messgröße 1), §4.1 („Berichtet wird ausschließlich die Differenz“), §5 (Entscheidungsregel) | **Die Entscheidungsregel der Ausstiegsforschung kann die Suche aus dem falschen Grund beenden.** Details in Abschnitt 3.5. | (a) Ob eine Ausstiegsregel den Erwartungswert verändern kann, hängt am **rohen** Pfad nach dem Einstieg, nicht an seiner Differenz zu Zufallseinstiegen. Gibt es allgemeines Intraday-Momentum, hilft ein Nachziehen des Stopps allen Einstiegen, und die Differenz zur Placebo-Kontrolle ist null. (b) Gepoolte lineare Autokorrelation (Messgröße 1) ist nur eine Form der Vorhersagbarkeit. Zustandsabhängige Drift (Zeit seit Einstieg, Abstand zu Kursmarken, Volatilitätszustand) kann bei Autokorrelation null bestehen. | Die Entscheidung an **Messgröße 3** (bedingte erwartete Restbewegung gegeben vorregistrierter Zustände) gegen eine Martingal-Null knüpfen, mit der Random-Walk-Kontrolle aus §4.3 als Nullmodell. Die Placebo-Differenz nur zur **Zuordnung** verwenden (strategiespezifisch oder marktweit), nicht als Abbruchgrund. |
| **H7** | AGENTEN, „Statistik-Härtung“ („Power-Check … MDE = 2*std/sqrt(n)“); LERNKREIS LK-4.3 (benötigte Trades = ((1,645 + 0,842)·s/μ)²); AUSSTIEG §4.5 („Maßgeblich ist der tagesgebundene Standardfehler, nie n“) | **Zwei unvereinbare Power-Definitionen, beide ohne Tagescluster.** | `MDE = 2·σ/√n` ist die Signifikanzschwelle (etwa 5 % zweiseitig) mit rund 50 % Power, also **keine** Mindest-Effektgröße im üblichen Sinn (80 % Power). Das Etikett „UNTERPOWERT“ ist damit zu nachsichtig. Der Power-Plan nutzt dagegen 80 % Power bei einseitigem Test. Beide rechnen mit Einzeltrade-Streuung, obwohl AUSSTIEG §4.5 den tagesgebundenen Standardfehler als bindend erklärt. Beide unterschätzen damit die nötige Stichprobe. Der Power-Plan setzt μ = Bau-Erwartungswert (Auswahl des besten Trials, also nach oben verzerrt) und „realistisch ½ μ“ als freie Annahme. Die eigene Kalibrierung zeigt stärkere Schrumpfung (DEMO-PF minus Einlass-PF im Median −0,63, LERNKREIS LK-2). | Eine Funktion für MDE und Power, überall verwendet: Tagescluster-Standardfehler (oder Block-Bootstrap), Power 80 %, α passend zur Familienkorrektur aus H3. μ nicht aus dem Bau übernehmen, sondern mit der **gemessenen** Schrumpfung aus der Kalibrierung schätzen. Unsicherheit über μ als Szenariobereich angeben. „UNTERPOWERT“ neu definieren. |
| **H8** | LERNKREIS LK-3, „Stand 2026-09-26“ (E1: übernommen wird alles außer Echtgeld-Code, Unit-Installation, Zugangsdaten, Datenbanken, `known_red_tests`, `pytest.ini`, `tests/conftest.py`, eigene Richtlinie, gelöschte oder abgeschwächte Tests) | **Die Messlatte liegt in Reichweite des autonomen Entwicklers.** Einlassschwellen (`admission.py`), Score-Formel (`robustness.py`), Urteilslogik (`verdict_for`), Placebo-Schwellen, Power-Plan und Bau-Prompt stehen nicht auf der Sperrliste. Die ursprüngliche Risikotabelle hatte „Schwellen, Evidenzregeln“ noch gesperrt; sie ist „überholt“. | Die Methodikfassung (LK-0.3) **erkennt** Änderungen, **verhindert** sie aber nicht. Eine Methodik, die sich zwischen Hypothese und Urteil ändern kann, ist ein weiterer Freiheitsgrad; die Plattform selbst nennt dieses Risiko („Ausführer verbessert die Messlatte“, ANALYSE §11). Die Absicherung durch eine Prompt-Regel („Forschungshürden nie rückwirkend lockern“) widerspricht dem Grundsatz „Enforcement im Code, nicht im Prompt“ (AGENTEN, Governance-Paket P1–P3). | Alle Dateien, die über Urteile entscheiden, auf die Liste „bleibt Vorschlag“ setzen: Schwellen, Scores, Urteilslogik, Placebo- und Power-Code, Kalibrierungsregeln. Änderungen sind nur als **neue, parallel laufende Methodikfassung** zulässig (LK-4.6), per Test erzwungen: Hash der Entscheidungsdateien gegen die aktive Fassung. |

### 2.2 Mittel

| Nr | Datei · Abschnitt | Befund | Begründung | Vorschlag |
|---|---|---|---|---|
| **M1** | AGENTEN, „Governance-Paket P4+P5“, Punkt 5 (Placebo: K = 20, Entry-Jitter ±60 Min, „läuft automatisch … für den besten Trial“, „INFO-ONLY“); LERNKREIS LK-4.1 (BESTANDEN ≥ p95) | **Die Placebo-Kontrolle ist grob, selektionsblind und misst nur einen Teil des Effekts.** | (1) Bei K = 20 ist der Rang auf 1/20 gerastert. „≥ p95“ heißt „höchstens eine Replikation darüber“, das ist ein sehr grober Test. (2) Verglichen wird der **beste** von drei Trials mit Placebos **einer** Konfiguration. Die Auswahl selbst wird nicht nachgebildet, deshalb ist der Rang zugunsten des Kandidaten verzerrt. (3) Tag, Richtung und Haltedauer bleiben erhalten. Getestet wird also nur das **Timing innerhalb des Tages**; ein Vorteil aus der **Auswahl der Tage** bleibt unentdeckt, ein Placebo-Scheitern widerlegt ihn nicht. | K ≥ 199 (feinere Ränge). Die **gesamte Auswahlprozedur** auf den Placebo-Einstiegen wiederholen (je Replikation drei Varianten, dann das Maximum), sonst misst die Kontrolle etwas anderes als der Kandidat. Zwei Placebo-Stufen berichten: Tageswahl (Einstieg an zufälligen Tagen gleicher Anzahl) und Timing (die heutige Kontrolle). |
| **M2** | VA03 §2.2, §2.5 („Nullkontrolle … 3/40 = 0,075 — bestanden“), §5.4; `MINIMUM_SHIFT = 3` | **Die Kalibrierung der kausalen Placebokontrolle ist mit 40 Läufen nicht nachgewiesen; einige Kandidatentypen sind unprüfbar.** | (1) Bei 40 Läufen und wahrem α = 0,05 liegt die Standardabweichung der beobachteten Rate bei rund 0,034. 0,075 ist also mit Kalibrierung vereinbar, aber auch mit deutlicher Übertreibung. „Bestanden“ ist nicht belegt. (2) Ist der Prädiktor konstant (reine Long-Strategie mit Einheitsposition), greift `degenerate_series`, und die Kontrolle bricht ab: Solche Kandidaten sind nie placebo-prüfbar. (3) Eine feste Mindestverschiebung von 3 Ereignissen ist nur dann „horizont-bewusst“, wenn sich Halteperioden über höchstens drei Ereignisse überlappen. Bei längeren Überlappungen korrelieren verschobene und echte Ausrichtung, und der Test wird konservativ, aber schwer deutbar. | Selbsttest mit ≥ 1.000 Nullläufen, dazu ein Binomialintervall für die Rate. Die Mindestverschiebung aus der gemessenen Überlappung der Halteperioden ableiten. Für Long-only-Kandidaten eine eigene Kontrolle festlegen (z. B. Zufallszeitpunkte statt Ausrichtung). |
| **M3** | AUDIT §5 (PBO nach CSCV mit 8 Blöcken und 70 Aufteilungen, DSR „über die Tagesreihen aller Varianten“, „PBO warnt erst ab 5 Varianten“); AUDIT „Grenzen“ | **PBO und DSR werden auf zu kleine Variantenmengen angewandt; die DSR-Deflation nutzt vermutlich nicht die volle Versuchszahl.** | PBO nach CSCV (Bailey, Borwein, López de Prado & Zhu, 2017) ist für die Auswahl aus **vielen** Konfigurationen gedacht; bei 3 bis 5 Varianten ist die Größe kaum aussagekräftig. Die Deflated Sharpe Ratio (Bailey & López de Prado, 2014) deflationiert mit der Zahl der Versuche und ihrer Streuung. Werden nur die Varianten eines Builds gezählt, ist die Deflation um Größenordnungen zu schwach, weil die Familie im Ledger Dutzende Builds umfasst (AUDIT: „die Familie aller Versuche erfasst weiterhin KPI 4“). | PBO nur auf Optimierer-Grids anwenden (bis zu 24 Kombinationen, AGENTEN „Trial-Budget“). Die DSR mit der **Ledger-Zahl der Familie** und der gemessenen Streuung der Trial-Sharpe-Werte rechnen. Beides als Signal behalten, aber mit dieser Familiendefinition. |
| **M4** | AGENTEN, „Robustheits-Metriken + Score“ (Bootstrap „Seed 42, 1000 Resamples“ über Einzeltrades); AGENTEN „Statistik-Härtung“ (Block-Bootstrap „ADDITIV“, iid-CI „dokumentiert optimistisch“); AUDIT §1 und „Grenzen“ | **Entscheidungsrelevante und öffentliche Intervalle beruhen auf dem Einzeltrade-Bootstrap**, obwohl dessen Optimismus bekannt ist. | Tagescluster (mehrere Trades je Tag, Volatilitätsregime) machen iid-Intervalle zu eng. Der Score (Signifikanzkomponente „CI-low ≥ 1,4“) und die öffentliche Rangfolge nach der unteren Grenze übernehmen diesen Fehler. | Für jede **Entscheidung** den Block- bzw. stationären Bootstrap verwenden (Politis & Romano, 1994). Den iid-Wert nur noch zur Kontinuität des eingefrorenen Scores mitführen und neben dem Block-Intervall klar als optimistisch kennzeichnen. Die Score-Formel v1.0 nicht ändern, sondern v1.1 als parallele Fassung führen (LK-4.6). |
| **M5** | WERKZEUGE §2 („fällt danach das Urteil gegen den besten Trial“); LERNKREIS LK-4.2 („10 Bestätigungen, alle in-sample“) | **Hypothesen werden am Maximum von drei Varianten bestätigt.** | Die beste von drei Varianten überschreitet eine Vorhersageschwelle öfter als eine einzelne, vorab bestimmte Variante. Das „Bestätigt“ ist damit doppelt geschönt: in-sample und selektiert. | Je Hypothese **eine** primäre Variante vorab festlegen, nur sie urteilt. Die übrigen Varianten sind Sensitivität. Alternativ das Urteil gegen die Placebo-Verteilung des Maximums (siehe M1). |
| **M6** | ABLAUF §3, Stufe 4 („alle fünf Prüfungen bestanden“); AGENTEN „Auto-Optimize“ („Bestätigt nur wenn OOS-PF ≥ 1,1 UND n ≥ 10“); LERNKREIS LK-4.1 (p95/p80); AGENTEN Score („CI-low ≥ 1,4 voll“) | **Die Prüfkette hat keine quantitativ festgelegten Bestehensregeln; die Einzelschwellen sind uneinheitlich und teils sehr niedrig.** | „Bestätigt“ ab 10 OOS-Trades liegt weit unter jeder Power-Anforderung aus H1/H7. Stufe 4 nennt fünf Prüfungen, aber nicht, was „bestanden“ je Prüfung heißt (Toleranz der Ergebnisparität, Placebo-Schwelle, Vorwärtskriterium). | Eine vorregistrierte **Entscheidungstabelle** je Stufe mit Größe, Test, α (familienkorrigiert), Mindest-Power und Toleranzen, versioniert als Teil der Methodikfassung. Das Optimierer-Urteil in „OOS-Hürde erreicht“ umbenennen und erst ab einer Power-basierten Mindestzahl „bestätigt“ nennen. |
| **M7** | ABLAUF §3 („Keine Stufe darf übersprungen werden“), §2 (sechs Lehrbuchfamilien, rund 85 % der Bauten), Stufe 5 („seit 2026-08-03 1978-mal blockiert“); AGENTEN „Erste Vault-Übernahmen“ (Übernahme eines „schwachen Signals“ auf Betreiberfreigabe) | **Der beschriebene Ablauf ist nicht der tatsächliche.** Die meisten Bauten beginnen bei Stufe 1 (Literatur) ohne Stufe-0-Befund. Stufe 5 ist seit Wochen blockiert. DEMO-Strategien kamen per Einzelfreigabe dorthin. | Für die **Kalibrierung** (LK-2: Einlass-Kennzahl gegen DEMO-Ertrag) heißt das: Die DEMO-Stichprobe ist nicht die Menge „Kandidaten, die die Kette bestanden haben“, sondern ein gemischter, handverlesener Satz. Die Kalibrierungsaussagen gelten für diese Selektion, nicht für die Pipeline. | Im Entscheidungsbuch (LK-1) je DEMO-Strategie den Zugangsweg (Kette oder Einzelfreigabe) führen und die Kalibrierung getrennt ausweisen. Die Spezifikation an die Praxis anpassen: Stufe 0 als eigene Spur mit eigenem Budget, Literatur-Bauten ausdrücklich als Spur 1 ohne Stufe-0-Vorbefund kennzeichnen. |
| **M8** | LERNKREIS LK-4.6 („monatlich … „führend“ nur mit ≥ 2 Monaten … revidierbar“); ANALYSE §8 Punkt 4 („Jeden Monat wird auf dem aufsummierten Stand entschieden“) | **Wiederholte Zwischenauswertungen ohne sequenzielles Design.** | Monatliches Nachsehen mit fester Schwelle und die Möglichkeit, bei günstigem Stand „führend“ zu erklären, erhöhen die Fehlerrate erster Art (Optional-Stopping-Problem). „Revidierbar“ heilt das nicht, es verschiebt nur den Zeitpunkt des Irrtums. | Ein gruppensequenzielles Design mit Alpha-Verbrauch (z. B. Grenzen nach O'Brien & Fleming, 1979) oder zeitlich gültige Verfahren. Die Plattform nennt e-Werte und sequenzielle Inferenz bereits als Suchauftrag (LK-4.4); das gehört vor die erste Führungsentscheidung. |
| **M9** | AGENTEN, „Punkte 3–5 umgesetzt“, Punkt (4) (Slippage: „Median 0.4 Pkt … p75 2.0, p90 3.8“; Annahme 0,5 „evidenzbasiert + leicht konservativ“) | **Das Kostenmodell stützt sich auf den Median einer rechtsschiefen Verteilung und nur auf Einstiege.** | Für den Erwartungswert zählt der **Mittelwert** der Slippage, nicht der Median. Er ist nicht angegeben; bei p75 = 2,0 und p90 = 3,8 Punkten liegt er vermutlich deutlich über 0,5. Gemessen wurde „zur Entry-Zeit“; Stopp-Ausführungen in schnellen Märkten rutschen typischerweise stärker. Bei Erwartungswerten von 5 bis 10 Punkten je Trade (LK-4.3) ist das entscheidungsrelevant. | Mittelwert und Mittelwert der oberen Tails berichten, Ausstiege getrennt kalibrieren (Stopp, Ziel, Zeit). Das Standard-Kostenmodell auf den Mittelwert setzen; die Kosten-Sensitivität (+1 Punkt) als Pflichtprüfung behalten. |
| **M10** | ABLAUF §6 (WP-1.1-MATH: 199 Permutationen, kleinstes roh-p 0,005, „harte Grenze von zehn entscheidbaren Tests je Familie“) | **Die Auflösung des eingefrorenen Hypothesen-Runners begrenzt die Familiengröße strukturell.** | Die Plattform erkennt das richtig. Die Folge ist aber, dass Familien mit mehr als zehn Tests grundsätzlich „nicht entscheidbar“ sind. Das muss so berichtet werden und darf nicht als Nullbefund in das Negativwissen (`math_negative_knowledge`) eingehen. | Eine Nachfolgefassung mit mehr Permutationen (die Laufzeit ist bei Circular Shifts gering) als parallele Methodikfassung. Bis dahin Familien > 10 im Negativwissen als „nicht entscheidbar“ führen. |
| **M11** | AGENTEN, „Warmup-Regel-Fix im BT“ („ALLE BT-Ergebnisse vor dem Fix … mit Vorsicht zu lesen … Re-Run der 14 Research-Builds steht aus“); AUSSTIEG §6b (fünf Juli-Exit-Lab-Berichte mit Überlappungs-Defekt) | **Bekannt defekte Alt-Ergebnisse speisen möglicherweise weiter Digest, Kennzahlen und Kalibrierung.** | Der Digest ist Pflicht-Input aller Agenten (AGENTEN, „Rückkopplungs-Schleife“). Ungültige Altstände lenken damit die Hypothesenwahl, und Kennzahl 4 zählt sie mit. Ob der Re-Run erfolgt ist, geht aus den Dateien nicht hervor. | Jede Ergebniszeile trägt den Engine-Fingerprint (vorhanden, AGENTEN P1–P3). Digest, Kennzahlen und Kalibrierung schließen Fingerprints vor dem Fix standardmäßig aus oder weisen sie getrennt aus. |
| **M12** | LERNKREIS LK-5, Tabelle „Entdeckung 26.09.“ (Rauigkeit: Hurst der log-Volatilität 0,05, Kontrolle 0,48, Differenz −0,43 [−0,44; −0,41]); Information: 0,00060 Bit | **Die Kontrollen der Mechanismusspur sind für zwei der drei Größen möglicherweise trivial.** | Liegt die Kontrolle für die Rauigkeit bei H ≈ 0,5, ist sie vermutlich eine permutierte oder unabhängige Reihe. Sie zu schlagen zeigt dann nur Abhängigkeit, nicht Rauigkeit im Sinne von Gatheral, Jaisson & Rosenbaum (2018). Mikrostrukturrauschen der CFD-Kurse drückt H nach unten (LK-5 nennt das selbst). Das sehr enge Intervall legt nahe, dass die Abhängigkeit zwischen Tagen oder die Schätzunsicherheit von H nicht voll eingeht; das ist aus der Datei nicht prüfbar. Die Plug-in-Schätzung von Transinformation ist nach oben verzerrt; die Kontrolle mindert das, aber nur, wenn sie dieselbe Stichprobengröße und Diskretisierung nutzt. | Für die Rauigkeit eine Kontrolle mit **bekanntem** H (z. B. simuliert mit gleichem Rauschmodell) und eine Positivkontrolle mit H = 0,1 im selben Codepfad. Das Intervall per Tagesblock-Bootstrap. Den Aufbau der Kontrollen je Messgröße in der Befundliste dokumentieren. |

### 2.3 Gering

| Nr | Datei · Abschnitt | Befund | Vorschlag |
|---|---|---|---|
| **G1** | ABLAUF §2 („104 Handelstage“, Stand 30.08.); AUSSTIEG §4.5 („rund 110“) und §8 („136 Handelstage“, Stand 18.09.); LERNKREIS LK-5 („115 Handelstage“, 26.09.); ANALYSE §0 („~136“, 23.09.) | Die Datenbasis wird uneinheitlich beziffert, teils innerhalb desselben Dokuments. Jede Power-Aussage hängt daran. | Eine Quelle (z. B. Datenstand-Datei) mit Zählregel (Handelstage mit vollständiger Session nach Qualitätsfilter), überall referenziert. |
| **G2** | AGENTEN, „Globale Invarianten“ Punkt 3 („Anzeige/Reports in PUNKTEN, nie EUR“) gegen LERNKREIS LK-2 und LK-4.6 („DEMO-Ertrag je Trade“, „−0,49 € je Trade“) und RUA §1 („−1.814,77 €“) | Kalibrierung und Methodik-Wettlauf rechnen in Euro, entgegen der eigenen Invariante; Euro hängt an der Positionsgröße. | Alle Kalibrierungen in Punkten oder in ATR-Einheiten (wie in AUSSTIEG §3). |
| **G3** | WERKZEUGE §2 („bei n < 30 automatisch `undecidable`“) | Die Unentscheidbarkeitsgrenze ist eine feste Trade-Zahl statt einer Power-Grenze. Bei n ≥ 30 wird geurteilt, auch wenn die Power gering ist. | Unentscheidbar, solange die Power für die registrierte Vorhersage unter 80 % liegt (H7). |
| **G4** | AGENTEN, „Exit-Lab“ („ohne Single-Position-Gate-Interferenz … lange Haltezeiten sind daher OPTIMISTISCH“) | Der Exit-Lab-Vergleich bevorzugt systematisch lange Haltedauern. Das ist bekannt und dokumentiert, geht aber in die Rangfolge ein. | Die Rangfolge nur aus dem Verifikations-Backtest über die echte Engine bilden, Lab-Werte als Vorauswahl kennzeichnen. |
| **G5** | RUA §3a | Die Korrektur „Erbauer darf den Nachweis erbringen, wenn die Latte extern gesetzt ist“ ist sachlich gut begründet. Für **statistische** Urteile bleibt aber die Monokultur-Frage (ANALYSE S4: Erzeugung, Bau und empirische Prüfung auf demselben Modell). | Für Methodikänderungen und Mechanismusdeutungen die in ANALYSE §3 S4 genannte Regel umsetzen („Modell A schlägt vor, Modell B greift an, deterministische Evidenz entscheidet“). Bis dahin deterministische Kontrollen (H4, M1) vorziehen. |

---

## 3. Antworten auf die sechs Prüffragen

### 3.1 Ist der Ablauf methodisch tragfähig? Wo sind Lücken?

**Tragfähig als Gerüst.** Die Reihenfolge Vermessen → Vermuten → Bauen → Justieren → Prüfen → Vorschlagen (ABLAUF §3) entspricht guter empirischer Praxis. Besonders richtig:
- Stufe 0 misst Mechanismen mit Tausenden Ereignissen statt Strategien mit zwanzig bis hundert Trades (ABLAUF Stufe 0, „Warum sie fehlt und warum sie zuerst kommt“).
- Stufe 1 verlangt eine benannte Mechanik, die auch sagt, wann der Effekt endet.

**Lücken:**
1. **Praxis ≠ Spezifikation (M7).** Rund 85 % der Bauten starten ohne Stufe-0-Befund bei der Literatur. Stufe 5 ist blockiert, DEMO-Zugänge kamen per Einzelfreigabe.
2. **Keine versiegelte Reserve (H5).** Das rollende Embargo schützt nur drei Wochen; danach wird jeder Tag Suchdatenbestand. Der Empirie-Holdout wird wöchentlich weitergeschoben und damit mehrfach verwendet.
3. **Keine quantitativen Bestehensregeln in Stufe 4 (M6).** „Alle fünf Prüfungen bestanden“ ist ohne Kriterien nicht prüfbar.
4. **Rückkopplung ohne Expositionsbuchhaltung (H5).** Der Digest macht aus Ergebnissen neue Hypothesen, die dann auf denselben Daten getestet werden.
5. **Ergebnisparität existiert, fließt aber nicht in die Bewertung zurück** (ANALYSE S8). Solange der Bau-PF zum DEMO-PF im Median um 0,63 bis 0,76 schrumpft (LERNKREIS LK-2; ANALYSE §0), müsste diese Schrumpfung in jede Einlass- und Power-Rechnung eingehen (H7).

### 3.2 Mehrfachtest-Kontrolle: richtig eingesetzt? Was fehlt?

| Instrument | Einsatz laut Dateien | Beurteilung |
|---|---|---|
| **Trial-Ledger** | vor Ausführung erzwungen, Familien gezählt (AGENTEN P1–P3) | **Richtig und vorbildlich** als Buchführung. Es fehlt die Nutzung in der Entscheidung (H3) und eine Familiendefinition, die Kennzahl 4 und DSR teilen (H4, M3). |
| **Holm / BH** | in keiner Datei spezifiziert; „Korrektur über den ganzen Lauf“ ohne Verfahren (ABLAUF Stufe 0, AUSSTIEG §5); nur ein Bonferroni-Kopf im Empirie-Snapshot | **Fehlt.** Vorschlag in H3: FDR für Entdeckung, FWER für Bestätigung. Bonferroni ist bei stark abhängigen Buckets (Wochentag × Uhrzeit × Regime) unnötig streng; max-t per Resampling (Westfall & Young, 1993) nutzt die Abhängigkeit. |
| **Placebo (Einstiegs-Jitter)** | K = 20, bester Trial, nur Information (AGENTEN P5) | **Richtig gedacht, zu grob und selektionsblind** (M1). |
| **Kausales Placebo (VA-03)** | zyklische Verschiebung, 199 Verschiebungen, versiegelt | **Methodisch elegant** (kleinste Intervention, gleicher Codepfad). Kalibrierung mit 40 Läufen nicht belegt, Long-only nicht prüfbar (M2). |
| **PBO / DSR** | je Build über 3 bis 5 Varianten, als Signal (AUDIT §5) | **Falsche Skala** (M3): PBO braucht viele Konfigurationen, die DSR die Versuchszahl der Familie. |
| **Verdict-Gate PF ≥ 1,3 & n ≥ 100** | ausdrücklich „Multiple-Testing-Schutz“ | **Kein Mehrfachtest-Schutz** (H3), weil es mit der Zahl der Versuche nicht wächst. |
| **Kennzahl 4** | Signalrate gegen angenommenes α = 0,05 | **Richtige Frage, unkalibrierte Null** (H4). |

**Was fehlt zusätzlich:** Ein Test für die **Auswahl der besten Regel** aus vielen. Genau das tut die Pipeline jede Nacht; dafür gibt es seit White (2000) etablierte Bootstrap-Tests (Hansen, 2005; Romano & Wolf, 2005). Für die Signal-Mining-Raster (19.584 Rasterpunkte laut ANALYSE §0) ist das die passende Kontrolle, nicht eine Schwelle je Rasterpunkt.

### 3.3 Vorwärtstest und Power: realistisch? Bessere Priorisierung?

**Die Planung ist ehrlich, aber zu optimistisch gerechnet** (H7), und die Konsequenz wird nicht voll gezogen. LK-4.3 kommt selbst zu dem Schluss, dass Einzelkandidaten auf dem DAX „in Jahren nicht entscheidbar“ sind. Trotzdem laufen laut ANALYSE §15.3 „37 Kandidaten parallel, meist als Einzelkandidaten“.

**Bessere Priorisierung bei so wenig Daten**, geordnet nach Informationsgewinn je Handelstag:
1. **Varianz senken statt Stichprobe vergrößern.**
   - Gepaarte A/B-Vergleiche mit gleichem Träger (LK-4.5, schon eingeführt, bitte ausbauen).
   - Kontrollvariablen, z. B. die Ergebnisse je Trade um die gleichzeitige Indexbewegung bereinigen.
   - Ergebnisgrößen mit fixem Horizont statt Stopp/Ziel-PnL: Ausrichtung mal Folgebewegung, wie die Größe S in VA03 §2.1. Stopp- und Ziel-Mechaniken erhöhen die Streuung der Einzelergebnisse und verringern die Information je Ereignis.
2. **Poolen über Familien.** Hierarchische Modelle über Strategiefamilien (ANALYSE §7 nennt sie selbst „passt genau zur Knappheit“). Eine Familienaussage („Eröffnungsausbrüche auf dem DAX-CFD haben im Mittel keinen Vorteil nach Kosten“) ist mit den vorhandenen Daten eher entscheidbar als jede Einzelaussage.
3. **Mechanismen vor Strategien.** Ereignisbasierte Messungen der Stufe 0 haben um Größenordnungen mehr Beobachtungen. Das Vorwärtsbudget sollte bevorzugt in die monatliche **Replikation** von M2-Befunden gehen (LK-5), nicht in weitere Einzelkandidaten.
4. **Wenige Kandidaten mit vorab berechneter Power**, der Rest wird eingefroren und nur mitgeschrieben. Kandidaten ohne realistische Aussicht auf Entscheidung unter 24 Monaten gehören nicht in die Vorwärtsliste, höchstens in ein Familien-Pooling.
5. **Sequenzielles Design** für alle monatlichen Entscheidungen (M8).
6. **Mehr Daten für Mechanismen**: Die Datenstufen A/B/C in ANALYSE §15.1 (externe Historie nur für Strukturbefunde, Handelsvalidierung nur auf echten IG-Ticks) sind methodisch schlüssig. Die Übertragbarkeit Future → CFD ist nicht garantiert, deshalb bleiben Stufe C und der Vorwärtstest entscheidend.

### 3.4 Kontrollen im selben Codepfad, Look-Ahead-Schutz, Null- und Positivkontrollen: ausreichend beschrieben?

**Für Messwerkzeuge gut, für die Strategie-Pipeline als Ganzes nicht.**

- **Gut beschrieben:**
  - die fünf Messregeln der Stufe 0 (WERKZEUGE §5) und ihre Übernahme in AUSSTIEG §4,
  - die Ankerregel „Ebene **und** Zeitpunkt, ab dem sie bekannt ist“ samt Falsifikationsprobe (ABLAUF §5),
  - Null- und Positivkontrollen für Kalibrierung (LK-2) und Mechanismuslabor (LK-5),
  - der Selbsttest des kausalen Placebos (VA03 §2.5),
  - die Lehre, dass Kontrollen Look-Ahead **in der Definition** nicht fangen (ABLAUF §5).
- **Lücken:**
  1. **Keine End-to-End-Nullkontrolle der Strategie-Pipeline.** Niemand lässt die komplette Nachtpipeline (Phase A bis Urteil) auf Daten ohne Vorhersagbarkeit laufen. Damit wären Kennzahl 4 (H4) und die Fehlalarmrate des Urteils kalibriert.
  2. **Keine End-to-End-Positivkontrolle.** Ein eingepflanzter Effekt bekannter Größe (z. B. eine synthetische Drift nach einem definierten Ereignis in einer Kopie der Tickdaten) müsste von Bau, Urteil und Vorwärtstest mit der vom Power-Plan versprochenen Wahrscheinlichkeit gefunden werden. Erst dann ist „nicht gefunden“ deutbar.
  3. **Look-Ahead-Schutz für Strategiecode** ist über Engine-Parität (Canary, ein Handelstag pro Woche) und Import-Allowlist beschrieben, nicht aber über einen **Verschiebungstest**: Dieselbe Strategie mit um einen Bar verzögerter Signalausführung darf nicht dramatisch schlechter werden, sonst lebt sie von Information aus demselben Bar. Der Warmup-Fehler (AGENTEN) zeigt, dass Engine-Differenzen real vorkommen.
  4. **Kontrollen der Mechanismusspur** möglicherweise trivial (M12).

### 3.5 Ausstiegsforschung (Kaminski/Lo, zustandsbedingte Ausstiege): stichhaltig?

**Im Kern ja, in der Operationalisierung nein.**

- **Richtig:**
  - Der MFE ist „das Maximum eines Pfades, den man zum Zeitpunkt der Entscheidung nicht kennt“ (AUSSTIEG §2); die Lücke von 6.345 Punkten zwischen MFE und Ergebnis ist kein abholbares Guthaben.
  - Die Zuspitzung auf die Vorfrage „Kann überhaupt eine Ausstiegsregel Wert schaffen?“ ist ökonomisch.
  - Kaminski & Lo (2014, Journal of Financial Markets) zeigen, dass einfache Stop-Loss-Regeln die erwartete Rendite unter Random Walk senken und nur bei Momentum Wert schaffen.
  - Die Normierung in ATR-Einheiten und der Tagescluster-Standardfehler (§4.5) sind richtig.
- **Präziser wäre:**
  - Für einen Intraday-Pfad ohne Drift (Martingal) folgt aus dem Optional-Stopping-Theorem, dass **keine** beschränkte Ausstiegsregel den Erwartungswert ändert, nur die Verteilung.
  - Die Bedingung für Mehrwert ist also: Der Erwartungswert der weiteren Bewegung, **bedingt auf die zum Zeitpunkt t bekannte Information**, ist nicht null.
  - Lineare Autokorrelation der Zuwächse (Messgröße 1) ist davon nur ein Spezialfall. Zustandsabhängige Drift kann bei Autokorrelation null bestehen, etwa nach Zeit seit Einstieg (Signalabbau), nach Abstand zu Kursmarken (der eigene Rundmarken-Mechanismus!) oder nach Volatilitätszustand.
- **Operative Mängel:**
  1. **Entscheidung an der falschen Größe (H6):** Abbruch, wenn „Persistenz nicht von null unterscheidbar“. Maßgeblich sollte Messgröße 3 (bedingte Restaussicht) mit vorregistrierten Zustandsvariablen sein.
  2. **Nur die Differenz zur Placebo-Kontrolle wird berichtet (§4.1).** Für die Wert-Frage zählt der rohe Pfad. Marktweite Persistenz macht Stopps für alle Einstiege wertvoll und fällt in der Differenz heraus. Beide Zahlen berichten: roh (für die Entscheidung) und Differenz (für die Zuordnung).
  3. **Nur der Erwartungswert wird betrachtet.** Ausstiege, die die **Streuung** senken, ohne den Erwartungswert zu ändern, sind für dieses Projekt wertvoll, weil sie die Power jedes späteren Vorwärtstests erhöhen (H7). Diese Zielgröße fehlt.
  4. **„Auflösung ausreichend“ (§5) ist nicht definiert.** Vorschlag: Power ≥ 80 % für die kleinste Persistenz, die nach Kosten einen Ausstiegsvorteil ergäbe. Diese Größe wird vorab aus einem einfachen Pfadmodell berechnet und registriert.
  5. Der Verzicht auf Reinforcement Learning und (ungepooltes) Meta-Labeling (§8) ist bei 65 bis 254 Einstiegen je Strategie richtig begründet.

### 3.6 Wo verwechselt die Plattform „nicht messbar“ mit „nicht vorhanden“?

Die Plattform kennt die Unterscheidung ausdrücklich (ABLAUF §3 Stufe 0 Regel 4; WERKZEUGE §2 Regel 3; AUSSTIEG §4.4 und §5; LERNKREIS LK-2 „nicht nachweisbar ist der Normalfall“). Sie wendet sie aber nicht überall an:

| Stelle | Aussage | Warum „nicht messbar“ zutreffender wäre |
|---|---|---|
| WERKZEUGE §5 | „alles, was handelbar gewesen wäre, [wäre] sichtbar gewesen“ | Auflösung 7 bis 27 Punkte gegen typische Effekte von 5 bis 10 Punkten (H2). |
| WERKZEUGE §4, §7; ANALYSE §0 | „0 von 9 überlebt“, „belastbar geprüft: 7, 0 überlebt“ | Ab 60 Trades bei nötigen > 1.000 (H1). |
| WERKZEUGE §4 | Kennzahl 4: „mit null echten Effekten vereinbar“, „nicht mehr Signale erzeugt, als reines Würfeln“ | Vereinbar auch mit vielen kleinen echten Effekten. Die Null-Rate ist unkalibriert (H4). |
| AGENTEN, Auto-Build-Urteil | „PF ≥ 1,1 → schwaches Signal; sonst kein Edge“ | „Kein Edge“ wird unabhängig von der Power vergeben. Das eigene Feld „UNTERPOWERT“ (AGENTEN Statistik-Härtung) fließt nicht ins Urteil ein. |
| WERKZEUGE §2, LERNKREIS LK-4.2 | Hypothesen „widerlegt“ ab n ≥ 30 | n ≥ 30 ist keine Power-Grenze (G3). Ein Teil der in-sample „widerlegten“ Hypothesen ist vermutlich unentscheidbar. |
| ABLAUF §6 | Familien > 10 Tests im eingefrorenen Runner | Strukturell unentscheidbar (M10); darf nicht als Negativwissen gelten. |
| AUSSTIEG §5 | Abbruch bei „Persistenz nicht von null unterscheidbar und Auflösung ausreichend“ | Die Regel selbst ist korrekt gebaut. Solange „ausreichend“ nicht per Power definiert ist, droht dieselbe Verwechslung (3.5, Punkt 4). |

**Positiv hervorzuheben:** LERNKREIS LK-2 führt „nicht nachweisbar“ als Normalfall, und die Empirie-Logs rechnen die Zufallserwartung bei Mehrfachtests aus (AGENTEN, Empirie-Research). Diese Sprache sollte die Vorlage für alle Urteile sein.

---

## 4. Die fünf wichtigsten Änderungen

1. **Eine vorregistrierte Inferenzschicht mit End-to-End-Kalibrierung.**
   - Benannte Verfahren je Stufe: FDR (Benjamini & Hochberg, 1995, bzw. Benjamini & Yekutieli, 2001) für Entdeckung; Holm (1979) oder Romano & Wolf (2005) für Bestätigung; Reality Check oder SPA für die Auswahl der besten Regel; DSR mit der Versuchszahl der Familie.
   - Überall Tagesblock-Intervalle (Politis & Romano, 1994).
   - Eine **Null- und eine Positivkontrolle der gesamten Pipeline** (synthetische Kurse ohne und mit eingepflanztem Effekt) liefert die Null-Signalrate für Kennzahl 4 und die tatsächliche Trefferwahrscheinlichkeit.
   - Behebt H3, H4, M3, M4 und Teile von 3.4.
2. **Urteile überall nach Power dreiteilen.**
   - Eine einzige MDE- und Power-Funktion (Tagescluster, 80 % Power, familienkorrigiertes α, μ mit gemessener Schrumpfung).
   - Jedes Urteil (Bau, Hypothese, Vorwärts, Stufe 0, Ausstieg) lautet „belegt“, „widerlegt mit ausreichender Power“ oder „nicht entscheidbar“.
   - „0 überlebt“, „kein Edge“ und „belastbar geprüft“ werden entsprechend umbenannt.
   - Behebt H1, H2, H7, G3.
3. **Daten und Messlatte schützen.**
   - Versiegelte Wochenblöcke (E3) jetzt einführen.
   - Das Datenexpositionsbuch ist Pflicht für jede Hypothese; Digest- und Gedächtnis-Kontext wird um die Embargozone und DEMO-Ergebnisse bereinigt.
   - Alle Entscheidungsdateien (Schwellen, Scores, Urteilslogik, Placebo, Power) kommen auf die Liste „bleibt Vorschlag“ des autonomen Entwicklers. Änderungen nur als parallele Methodikfassung, per Hash-Test erzwungen.
   - Behebt H5, H8, M11.
4. **Das Vorwärtsbudget umbauen.**
   - Wenige Kandidaten mit berechneter Power statt 37 Einzelkandidaten.
   - Vorrang für gepaarte Vergleiche, Familien-Pooling (hierarchisch), Replikation von Mechanismusbefunden und Ergebnisgrößen mit geringer Streuung.
   - Monatliche Entscheidungen gruppensequenziell (O'Brien & Fleming, 1979) oder zeitlich gültig.
   - Placebo über die gesamte Auswahlprozedur mit K ≥ 199.
   - Behebt 3.3, M1, M5, M8.
5. **Die Ausstiegsforschung auf die richtige Größe stellen.**
   - Entscheidung an der bedingten Restaussicht (Martingal-Test mit vorregistrierten Zuständen), nicht an gepoolter Autokorrelation.
   - Rohwert für die Entscheidung, Placebo-Differenz nur zur Zuordnung.
   - „Auflösung ausreichend“ per Power definieren.
   - Varianzsenkung als zweites Ziel aufnehmen.
   - Behebt H6 und 3.5.

---

## 5. Offene Fragen an die Betreiber

1. **Familiendefinition:** Welche Versuche bilden für Kennzahl 4, DSR und Einlass eine Familie? 424 Trials (WERKZEUGE §4), 16.963 (AUSSTIEG §8) oder 22.449 (ANALYSE §0)?
2. **„Überlebt“:** Welches Kriterium entscheidet in Kennzahl 1, ob ein Kandidat mehrmonatig „überlebt“ (PF-Schwelle, Intervall, Vorzeichen, Anzahl Monate)? Die Dateien nennen es nicht.
3. **Mehrfachtest-Verfahren:** Welches Verfahren steckt hinter „Mehrfachtest-Korrektur über den gesamten Suchlauf“ (ABLAUF Stufe 0, Übergang Punkt 3) und „Korrektur über den ganzen Lauf“ (AUSSTIEG §5)? Ist es im Code festgelegt oder im Prompt?
4. **Warmup-Fix:** Ist der angekündigte Re-Run der Builds vor dem Warmup-Fix erfolgt? Werden Ergebnisse mit altem Engine-Fingerprint in Digest, Kennzahlen und Kalibrierung noch mitgezählt?
5. **Slippage:** Wie hoch sind Mittelwert und Ausstiegs-Slippage (Stopps) in der Kalibrierung mit 614 Fills? Wird die 0,5-Punkte-Annahme gegen den Mittelwert geprüft?
6. **Placebo-Einsatz:** Läuft die Placebo-Kontrolle über die Auswahl der drei Varianten oder nur über den besten Trial? Gibt es Pläne für eine Tageswahl-Kontrolle?
7. **Kausales Placebo:** Wie wird mit Long-only-Kandidaten verfahren, bei denen `degenerate_series` greift? Wie groß ist die typische Überlappung der Halteperioden im Verhältnis zu `MINIMUM_SHIFT = 3`?
8. **Mechanismuslabor:** Wie sind die Kontrollen für Rauigkeit und Information konstruiert? Wie wird das Intervall der Hurst-Differenz berechnet (Tagesblöcke oder Einzelbeobachtungen)?
9. **Autonomer Entwickler:** Darf er nach E1 `admission.py`, `robustness.py`, `verdict_for`, `placebo.py`, `power_plan.py` oder den Bau-Prompt ohne Betreiberfreigabe ändern? Gab es solche Änderungen seit dem 26.09.?
10. **Datenexposition:** Ab wann ist das Datenexpositionsbuch für Hypothesen verbindlich? Werden Digest-abgeleitete Hypothesen im Register als `post_hoc` für die exponierten Fenster geführt?
11. **Versiegelte Reserve (E3) und externe Historie (E2):** Gibt es eine Entscheidung und einen Zeitplan? Beide bestimmen, ob die Mechanismusspur jemals über M4 hinauskommt.
12. **Forward-Liste:** Nach welchen Kriterien werden die aktuell rund 37 Vorwärtskandidaten (ANALYSE §15.3) ausgewählt oder aussortiert, wenn der Power-Plan für keinen eine Entscheidung unter 24 Monaten erwartet?
13. **DEMO-Zugang:** Welche der DEMO-Strategien, die in die Kalibrierung eingehen, haben die Kette durchlaufen, und welche kamen per Einzelfreigabe (vgl. AGENTEN, „Erste Vault-Übernahmen“)?
14. **Datenbasis:** Welche Zahl an Handelstagen gilt heute, nach welcher Zählregel (G1)?

---

## Zitierte Literatur

Ich nenne nur Arbeiten, die ich sicher kenne. Die Jahresangaben beziehen sich auf die Veröffentlichung in der genannten Zeitschrift.

- Bailey, D. H., & López de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management*.
- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). The Probability of Backtest Overfitting. *Journal of Computational Finance*.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate. *Journal of the Royal Statistical Society, Series B*.
- Benjamini, Y., & Yekutieli, D. (2001). The Control of the False Discovery Rate in Multiple Testing under Dependency. *Annals of Statistics*.
- Dwork, C., Feldman, V., Hardt, M., Pitassi, T., Reingold, O., & Roth, A. (2015). The Reusable Holdout: Preserving Validity in Adaptive Data Analysis. *Science*.
- Gatheral, J., Jaisson, T., & Rosenbaum, M. (2018). Volatility Is Rough. *Quantitative Finance*.
- Gelman, A., & Loken, E. (2014). The Statistical Crisis in Science. *American Scientist*.
- Hansen, P. R. (2005). A Test for Superior Predictive Ability. *Journal of Business & Economic Statistics*.
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). … and the Cross-Section of Expected Returns. *Review of Financial Studies*.
- Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics*.
- Ioannidis, J. P. A. (2005). Why Most Published Research Findings Are False. *PLoS Medicine*.
- Kaminski, K. M., & Lo, A. W. (2014). When Do Stop-Loss Rules Stop Losses? *Journal of Financial Markets*.
- O'Brien, P. C., & Fleming, T. R. (1979). A Multiple Testing Procedure for Clinical Trials. *Biometrics*.
- Politis, D. N., & Romano, J. P. (1994). The Stationary Bootstrap. *Journal of the American Statistical Association*.
- Romano, J. P., & Wolf, M. (2005). Stepwise Multiple Testing as Formalized Data Snooping. *Econometrica*.
- Sullivan, R., Timmermann, A., & White, H. (1999). Data-Snooping, Technical Trading Rule Performance, and the Bootstrap. *Journal of Finance*.
- Westfall, P. H., & Young, S. S. (1993). *Resampling-Based Multiple Testing*. Wiley.
- White, H. (2000). A Reality Check for Data Snooping. *Econometrica*.
