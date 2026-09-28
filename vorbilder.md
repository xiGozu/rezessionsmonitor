# Vorbilder für warchhold.com/algostrategien

**Frage:** Welche öffentlichen Quant-Research-Angebote stellen Ergebnisse, Methodik und Negativbefunde vorbildlich dar, und was davon lässt sich auf warchhold.com übertragen?
**Recherche:** 28.09.2026. Alle Quellen wurden abgerufen. Aussagen stützen sich auf den abgerufenen Seitentext oder das Original-PDF. Wo nur ein Suchergebnis vorlag, ist das vermerkt.
**Vergleichsbasis:** die öffentlichen Seiten unter `https://warchhold.com/algostrategien`, wie sie in `review-hub-extern.md`, `audit-zahlen.md` und `review-barrierefreiheit.md` dokumentiert sind.

Nicht aufgenommen, weil sie sich nicht belastbar prüfen ließen:
- **global-q.org** (Hou/Xue/Zhang): Die Seite zeigte nur eine Bot-Prüfung.
- **Macrosynergy, „Modern backtesting with integrity“**: HTTP 403.
- **Allocate Smartly**: abrufbar, aber ohne sichtbare Angaben zu gescheiterten oder eingestellten Strategien. Für die Frage nach Negativbefunden deshalb kein Vorbild.

---

## Überblick

| # | Vorbild | Typ | Stärke, die warchhold.com fehlt |
|---|---|---|---|
| 1 | Open Source Asset Pricing (Chen & Zimmermann) | akademisches Replikationsprojekt | Eigenes Ergebnis wird **systematisch gegen die Originalbehauptung** gestellt, inklusive „Placebos“. |
| 2 | JKP Global Factor Data (Jensen, Kelly, Pedersen) | akademisch mit Asset-Manager-Beteiligung | **Familien statt Einzelstrategien** bewerten, Bayes-Sicht auf Mehrfachtests, offener Wettbewerb mit gemeinsamem Maßstab. |
| 3 | AQR: Datensätze und „Fact, Fiction …“-Reihe | Forschung eines Asset Managers | **Laufend aktualisierte Daten** zu jedem Aufsatz, Mythen-und-Fakten-Format. |
| 4 | Newfound Research, „Timing Luck“ | Forschungsblog eines Asset Managers | **Glück durch Zeitpunktwahl** wird gemessen und als Streuung gezeigt. |
| 5 | Arnott, Harvey, Markowitz: „A Backtesting Protocol …“ | Methodik-Standard | **Sieben-Punkte-Checkliste**, die jede Untersuchung sichtbar beantworten kann. |
| 6 | Trading Strategy Finder (M. Fetna) | offenes Strategie-Projekt, agentengetrieben | **Maschinell prüfbares Behauptungsregister**, eingefrorener Track Record mit Datum, Versionen mit DOI. |
| 7 | Robot Wealth, „Brave New Backtest“ | Praktiker-Blog | **Mechanismus zuerst** („wer zahlt und warum?“), klare Haltung zu KI-erzeugten Backtests. |
| 8 | Institute for Replication (I4R) | Replikationsinstitut Ökonomie | **Replikationsbericht und Antwort der Autoren** werden gemeinsam veröffentlicht. |
| 9 | Gençay (2026): „What survives honest evaluation?“ | Fachaufsatz zu LLM-Strategiesuche | **Suchkorrektur automatisch** nach Trialzahl, **passive Benchmarks als Positivkontrolle**. |

### Vergleich nach Kriterien

| Kriterium | Beste Praxis bei den Vorbildern | warchhold.com heute |
|---|---|---|
| **Aufbau** | Wenige Einstiege: Daten, Code, Dokumentation, FAQ (OSAP, JKP). Ein Befund = eine zitierfähige Aussage mit ID (Fetna). | Über 20 Unterseiten mit überlappenden Zwecken, interne Betriebsseiten im Hauptmenü (siehe `review-hub-extern.md`, Nr. 11). |
| **Unsicherheit** | Vergleich Original-t-Wert gegen Replikation (OSAP), Streuung durch Zeitpunktwahl (Newfound), Deflation nach Trialzahl (Gençay), Power-Analyse bei jedem Negativbefund (Fetna). | Gute Bausteine vorhanden: 90-%-Bereich, Einstufung, Power-Angabe „UNTERPOWERT“, Kalibrierung „−0,61“. Aber uneinheitlich angewendet (Podium, „verbessert“ bei PF < 1; siehe `audit-zahlen.md`). |
| **Gescheiterte Ideen** | Placebos als eigene Kategorie (OSAP); Negativbefunde „mit derselben Strenge wie positive“, jeweils mit Power (Fetna); veröffentlichte Replikationsberichte (I4R). | Vollständig vorhanden, aber verstreut über sechs Seiten und weniger prominent als Erfolge. |
| **Vertrauenssignale** | Öffentlicher Code und Daten (OSAP, JKP), DOIs pro Version, reproduzierbar ohne Serverzugang (Fetna), Autorenantworten (I4R), Checkliste (Arnott/Harvey/Markowitz). | Trial-Ledger, Embargo, Placebo, Run-Manifest sind beschrieben. Code, Evidenzdateien und ein Weg zur externen Nachprüfung sind auf den geprüften Seiten nicht zugänglich („🔒 Export (Admin)“). |
| **Tonalität** | Nüchtern, oft selbstkritisch, eigene Fehler mit Datum (Fetna: „honest retraction“; Robot Wealth: der erste Gewinn war „pure luck“). | Ehrlich im Anspruch („Jeder Test zählt – auch wenn er scheitert“). In den Details aber voller interner Kürzel, teils mit Superlativen („stärkster In-Sample-Befund“, „Hall of Fame“). |

---

## 1. Open Source Asset Pricing (Chen & Zimmermann)

**Quellen:** https://www.openassetpricing.com/ · Code: https://github.com/OpenSourceAP/CrossSection · Aufsatz: https://www.federalreserve.gov/econres/feds/files/2021-037pap.pdf

**Was sie tun**
- Das Projekt stellt „test asset returns and signals replicated from the academic asset pricing literature“ bereit, mit Python- und R-Paket, Datenportal und Signal-Browser.
- Der Aufsatz vergleicht die eigenen t-Werte mit den Originalen: „For the 161 characteristics that were clearly significant in the original papers, 98% of our long-short portfolios find t-stats above 1.96. For the 44 characteristics that had mixed evidence, our reproductions find t-stats of 2 on average. A regression of reproduced t-stats on original long-short t-stats finds a slope of 0.90 and an R2 of 83%.“
- Die Signale sind nach der **Stärke der ursprünglichen Evidenz** eingeteilt: „clear predictors“, „likely predictors“ und 114 weitere, die „insignificant in the original papers“ oder Abwandlungen waren. Das Code-Repository führt sie in einem eigenen Ordner `Placebos/` („not predictors“ und „indirect evidence“).
- Eine Dokumentationstabelle `SignalDoc.csv` beschreibt jedes Signal. Der Code ist in drei Stufen aufgeteilt, damit man nur Teile nachrechnen kann.

**Vergleich:** warchhold.com liest ebenfalls Fachliteratur aus (Funde mit Quelle, Tier-Einstufung), stellt aber nie systematisch dar, **was die Quelle behauptet und was auf dem DAX herauskam**. Die Information liegt verstreut in Befundtexten.

**Übertragbare Ideen**
1. **Streudiagramm „Quelle vs. DAX“:** je Fund der in der Quelle berichtete Effekt (z. B. t-Wert oder PF) gegen das eigene Ergebnis, dazu eine Diagonale. Ein Bild erzählt die ganze Replikationsgeschichte.
2. **Einstufung nach Ausgangsevidenz:** Funde vor dem Test als „in der Quelle klar belegt / gemischt / ohne Beleg“ kennzeichnen (entspricht grob den Tier-Stufen, aber vor dem Ergebnis) und Ergebnisse je Stufe getrennt auswerten.
3. **Placebos als eigene, sichtbare Kategorie:** Die vorhandenen Placebo-Arme und Zufallskontrollen gesammelt zeigen, statt sie in Dossiers zu verstecken. Das zeigt, dass die Pipeline auch „nein“ sagen kann.
4. **Eine Tabelle, die alles dokumentiert:** eine öffentliche, herunterladbare Signal- und Strategieliste (CSV) mit Quelle, Regeln, Datum, Urteil.

## 2. JKP Global Factor Data (Jensen, Kelly, Pedersen)

**Quellen:** https://jkpfactors.com/ · Leaderboard: https://jkpfactors.com/ctf/leaderboard · Aufsatz: https://www.nber.org/papers/w28432

**Was sie tun**
- Die Seite bietet 153 Faktoren in 13 Themen-Clustern über 93 Länder, mit PDF-Dokumentation, Quellcode auf GitHub (MIT-Lizenz), Datenlizenz CC BY-NC 4.0 und einer Cluster-Grafik.
- Der Aufsatz argumentiert mit einem **Bayes-Modell der Faktorreplikation**: Die Mehrzahl der Faktoren „(i) can be replicated; (ii) can be clustered into 13 themes …; (iii) work out-of-sample in a new large data set covering 93 countries; and (iv) have evidence that is strengthened (not weakened) by the large number of observed factors.“ (Abstract laut NBER-Suchergebnis)
- Ein **„Common Task Framework“-Leaderboard** vergleicht eingereichte Modelle auf einem gemeinsamen Testzeitraum („Test period: 1990–2023“). Alle Modelle sind auf dieselbe Volatilität skaliert („standardized to have an ex-post volatility of 10%“).

**Vergleich:** warchhold.com bewertet jede Strategie einzeln, obwohl sie in Familien forscht („ORB-Familie steht bei 62 Trials in 6 Builds“). Ein gemeinsamer Vergleichsmaßstab fehlt. Die Bestenliste mischt Varianten desselben Trägers.

**Übertragbare Ideen**
1. **Familienansicht:** Ergebnisse nach Mechanismus-Familie bündeln (Eröffnungsausbruch, Lücken, Rückkehr in die Spanne, Tageszeit …). Je Familie: Zahl der Versuche, bestes und mittleres Ergebnis, Anteil verworfener Versuche.
2. **Bayes-Schrumpfung sichtbar machen:** neben dem Einzelergebnis ein zur Familie hin geschrumpfter Wert. Das macht den Mehrfachtest-Effekt anschaulich, statt nur „Score“ zu zeigen.
3. **Gleiche Skala für Vergleiche:** Punkteergebnisse auf eine gemeinsame Schwankung normieren, wenn Strategien nebeneinandergestellt werden.
4. **Lizenz und Zitierhinweis** für veröffentlichte Daten und Texte.

## 3. AQR: Datensätze und „Fact, Fiction …“

**Quellen:** https://www.aqr.com/Insights/Datasets · https://www.aqr.com/Insights/Research/Journal-Article/Fact-Fiction-and-Momentum-Investing-Supplement · Aufsatz-PDF: https://images.aqr.com/-/media/AQR/Documents/Journal-Articles/JPM-Fact-Fiction-and-Momentum-Investing.pdf

**Was sie tun**
- Die Datensatzseite listet Faktorzeitreihen (Momentum-Indizes, Betting Against Beta, Quality Minus Junk, HML Devil …), jeweils mit Bezug zum zugrunde liegenden Aufsatz. Alle tragen einen aktuellen Stand (Juli/August 2026), mit dem Hinweis, die Daten würden „update[d] monthly“.
- Die Reihe „Fact, Fiction and …“ ist nach **verbreiteten Behauptungen gegliedert**, die einzeln geprüft werden. Für Momentum nennt die Zusammenfassung u. a. „over 20 years of out-of-sample evidence from its original discovery“ (Formulierung laut Suchergebnis zum Aufsatz). Zum Aufsatz gibt es ein eigenes Datensupplement.

**Vergleich:** warchhold.com hat viel Negativwissen zu populären Intraday-Ideen (Eröffnungsausbruch, Lücken, Rückkehr zum VWAP), präsentiert es aber als Laufprotokolle statt als Antworten auf Fragen, die Leser tatsächlich haben.

**Übertragbare Ideen**
1. **Seite „Mythen und Befunde zum DAX-Intraday-Handel“:** pro verbreiteter Behauptung („Der Eröffnungsausbruch funktioniert“, „Lücken schließen sich“) eine Antwort mit Evidenzklasse, n und Link zum Protokoll.
2. **Jede Aussage mit Datensatz:** zu jedem Befund die zugrunde liegende Ergebnistabelle zum Herunterladen, mit Stand und Aktualisierungsrhythmus.
3. **„Seit Entdeckung“-Perspektive:** für Strategien aus der Literatur ausweisen, wie sie sich außerhalb des Zeitraums der Quelle verhalten haben. warchhold.com testet ohnehin auf einem späteren Zeitraum und einem anderen Markt.

## 4. Newfound Research: „Timing Luck“

**Quellen:** https://blog.thinknewfound.com/2018/01/quantifying-timing-luck/ · https://www.thinknewfound.com/rebalance-timing-luck

**Was sie tun**
- Newfound misst, wie stark Ergebnisse allein vom **Zeitpunkt** abhängen: Derselbe monatlich umgeschichtete Aktien/Cash-Ansatz, an 21 verschiedenen Handelstagen gestartet, lieferte 9,6 % bis 11,1 % Jahresrendite, also 150 Basispunkte Spanne bei identischer Logik.
- Dazu gibt es eine Faustformel für die Größe des Effekts (L = σ × √(T × f / 3)) und die pointierte Einordnung, diese Wahl könne „the difference between ‚hired‘ and ‚fired‘“ sein.
- Als Gegenmittel werden mehrere versetzte Teilportfolios empfohlen.

**Vergleich:** warchhold.com hat mit dem „Anker-Verschiebungs-Placebo“ und der zeitlich versetzten Placebo-Kontrolle (±60 Minuten) genau diese Frage schon im Werkzeugkasten. Gezeigt wird aber meist nur ein Punktwert („Perzentil-Rang 90%“), keine Streuung.

**Übertragbare Ideen**
1. **Band statt Punkt:** In jedem Dossier die Ergebnisse aller zeitversetzten Varianten als Streuband oder Histogramm zeigen, mit dem Kandidaten als Markierung. Die Daten der 20 Zufalls-Replikationen existieren bereits.
2. **Kennzahl „Zeitpunkt-Glück“:** eine Zahl je Strategie, wie viel des Ergebnisses allein durch die Startzeit schwankt, zusätzlich zum Vertrauensbereich.
3. **Klartext-Satz nach Newfound-Art:** „Dieselbe Regel, 30 Minuten später gestartet, hätte zwischen x und y Punkten erzielt.“

## 5. Arnott, Harvey, Markowitz: „A Backtesting Protocol in the Era of Machine Learning“

**Quelle:** https://people.duke.edu/~charvey/Research/Published_Papers/P138_A_backtesting_protocol.pdf (Journal of Financial Data Science, Winter 2019)

**Was sie tun**
- Der Aufsatz beginnt mit einer scheinbar hervorragenden Strategie, die sich als Zufallsfund aus Tickerbuchstaben entpuppt („This strategy might seem too good to be true. And it is.“).
- Daraus leiten die Autoren eine **Sieben-Punkte-Checkliste** ab (Exhibit 2): Research Motivation, Multiple Testing and Statistical Methods, Data and Sample Choice, Cross-Validation, Model Dynamics, Complexity, Research Culture. Beispiele für die Prüffragen:
  - „Did the economic foundation or hypothesis exist before the research was conducted?“
  - „Did the researcher keep track of all models and variables that were tried (both successful and unsuccessful) …?“
  - „Are the researchers aware that true out-of-sample tests are only possible in live trading?“
  - „Do the researchers and management understand that most tests will fail?“
- Ziel ist ausdrücklich Demut: „Hubris is our enemy.“

**Vergleich:** warchhold.com erfüllt viele Punkte bereits (Mechanismus-Pflicht, Trial Ledger, Embargo, Placebo, Einfrieren), sagt das aber nie in dieser prüfbaren Form. Leser müssen es aus der Methodik zusammensuchen.

**Übertragbare Ideen**
1. **Checkliste je Dossier:** die sieben Punkte als kurzer Kasten mit Ja / Nein / Teilweise und Beleg-Link. Das kostet wenig und ist ein starkes Vertrauenssignal.
2. **Eigene Einordnung „wahres Out-of-Sample nur im Live-Betrieb“:** Das würde die Evidenzklassen (in-sample, vorwärts, DEMO) begründen und die Verwirrung zwischen „Vorwärts“ und „DEMO“ auflösen.
3. **Aufhänger für die Startseite:** ein eigenes „zu gut, um wahr zu sein“-Beispiel aus dem eigenen Log (z. B. die 91-%-Abprallquote der Marktvermessung, die sich als Messfehler herausstellte).

## 6. Trading Strategy Finder (Mulham Fetna)

**Quelle:** https://github.com/mulhamfetna/trading-strategy-finder (README). Begleitaufsätze: SSRN 7428398 und 7428478; auf warchhold.com bereits als Fund zitiert.

**Was sie tun**
- Ein offenes, „primarily by AI agents“ entwickeltes Forschungsprojekt, also strukturell sehr ähnlich zu warchhold.com.
- **Behauptungsregister:** „replays the claims ledger (79 claims, each with three verifications that must fail for different reasons and a declared blind spot)“. Ein Selbsttest zeigt, dass das Prüftor scheitern kann („5 historical defects rejected“). Das Register „refuses any claim whose evidence is not in git“.
- **„Findings you can quote“:** Jede Kernaussage steht als Frage und Antwort mit Claim-ID. Beispiel: „Do optimized backtest champions hold up on genuinely fresh data? Mostly no — and we publish that.“
- **Symmetrie:** „Negative results are published with the same rigor as positive ones — each carries a power analysis, and each positive carries a dumb control and a noise check.“
- **Eingefrorener Track Record:** „Since 2026-08-31 the deployed parameter set and the 9-slot universe are hash-frozen under a signed, amendment-only protocol“. Außerdem: „no verdict before the pre-registered power threshold, and a negative outcome is a publishable result of the protocol, not a failure of it.“
- **Zwei Reproduzierbarkeitsstufen**, offen benannt: Jeder kann offline aus den veröffentlichten Evidenzdateien nachrechnen; die Neuberechnung aus Rohkursen geht nur auf dem Server.
- **Versionen mit DOI** (Zenodo) und **Positionierung** (`docs/POSITIONING.md`: „where this work sits in the field, rung by rung, each cell linked to its evidence“). Veraltete Dokumente werden als historisch gekennzeichnet, nicht gelöscht.

**Vergleich:** Methodisch liegen beide Projekte nah beieinander (Vorregistrierung, Placebo, Power, Einfrieren). Der Unterschied liegt in der **Außendarstellung**: Fetna liefert wenige, zitierfähige, prüfbare Sätze. warchhold.com liefert viele Seiten, auf denen sich dieselbe Zählgröße widerspricht (`review-hub-extern.md`, Nr. 6).

**Übertragbare Ideen**
1. **„Befunde, die man zitieren kann“:** fünf bis acht Kernaussagen als Frage und Antwort, jeweils mit ID, Evidenzklasse, n, Stand und Link. Diese Aussagen sind die einzige Stelle, an der Zählgrößen genannt werden.
2. **Prüftor für öffentliche Zahlen:** Jede Zahl auf der Website muss aus einer versionierten Evidenzdatei stammen. Ein Selbsttest zeigt öffentlich, dass das Tor Fehler findet. Die vorhandene „public-consistency“-Prüfung lässt sich dafür erweitern.
3. **Einfrier-Protokoll mit Datum und Hash:** öffentlich angeben, seit wann welche Parameter und Strategien unverändert laufen, und jede Änderung als datierte Ergänzung führen.
4. **Offline-Nachprüfung:** Ergebnis- und Evidenzdateien, also nicht die lizenzierten Kursdaten, öffentlich machen, dazu ein Skript, das die veröffentlichten Zahlen nachrechnet.
5. **Seite „Wo stehen wir?“:** eine ehrliche Einordnung gegenüber akademischer und professioneller Praxis, Punkt für Punkt mit Belegen.

## 7. Robot Wealth: „Brave New Backtest“

**Quellen:** https://robotwealth.com/brave-new-backtest/ (29.03.2026) · https://robotwealth.com/

**Was sie tun**
- Kris Longmore argumentiert, dass KI Backtests billig macht: „AI makes beautiful backtests trivially easy to produce, which means more false discoveries, more overfitting dressed up as research“.
- Die Kernfrage jeder Strategie sei „who pays you and why?“. Der empfohlene Arbeitsablauf: „human generates insight (theory of edge), then AI implements it (coding, data wrangling), then human evaluates results.“
- Auf der Startseite heißt es selbstkritisch: „a backtest can only tell you how a rule would have performed; it can't tell you why it should keep working“. Der erste Gewinn des Autors sei „pure luck“ gewesen.

**Vergleich:** warchhold.com verlangt bereits einen Mechanismus („Mechanismus-Pflicht“ im Empirie-Agenten) und lässt Agenten die Hypothesen bilden. Genau diese Rollenverteilung kritisiert Longmore. Das ist ein Punkt, den ein fachkundiger Leser sofort anspricht.

**Übertragbare Ideen**
1. **Pflichtfeld „Wer zahlt und warum?“** in jedem Versuchsprotokoll, sichtbar ganz oben, in einem Satz.
2. **Offene Auseinandersetzung mit dem KI-Einwand:** ein kurzer Abschnitt „Warum Agenten hier Ideen vorschlagen dürfen und wie wir verhindern, dass daraus Scheinfunde werden“ (Vorregistrierung, Suchkorrektur, Kalibrierung „−0,61“).
3. **Persönlicher, nüchterner Ton über eigene Fehlgriffe:** Die „Über mich“-Seite könnte ein konkretes eigenes Beispiel für einen Scheinfund nennen.

## 8. Institute for Replication (I4R)

**Quellen:** https://www.i4replication.org/ · Berichte: https://i4replication.org/reports/ · Meta-Studie (Suchergebnis): https://ideas.repec.org/p/zbw/i4rdps/287.html

**Was sie tun**
- I4R will „improving the credibility of science by systematically reproducing and replicating important empirical research in the social sciences“. Die Seite nennt „379 replication reports across 16 journals“ und „232+ discussion papers“.
- Ablauf laut Suchergebnis: Replikatoren schreiben einen Bericht nach Vorlage, ein Chair prüft ihn, die Originalautoren können antworten, und **Bericht und Antwort erscheinen gleichzeitig**.
- In einer Großstudie waren über 85 % der geprüften Aussagen rechnerisch reproduzierbar (laut Suchergebnis zur Meta-Studie).

**Vergleich:** warchhold.com prüft sich selbst (getrennte Agentenrollen, Kalibrierung), bietet aber keinen Weg für **externe** Nachprüfung und veröffentlicht keine Kritik von außen.

**Übertragbare Ideen**
1. **Einladung zur Nachprüfung:** eine Seite „Prüfen Sie uns“ mit Evidenzdateien, Vorlage für einen Prüfbericht und dem Versprechen, Befunde samt eigener Antwort zu veröffentlichen.
2. **Korrekturprotokoll:** öffentliche Liste von Korrekturen („was wir falsch veröffentlicht hatten, wann, warum“). Die Marktvermessung macht das bereits vorbildlich im Kleinen.
3. **Standardformat „reproduziert / nicht reproduziert / robust“** für die eigene Stufe „Reproduziert“, die heute unklar definiert ist.

## 9. Gençay (2026): „What survives honest evaluation?“

**Quelle:** https://arxiv.org/abs/2608.27734 (eingereicht am 27.08.2026)

**Was sie tun**
- Der Aufsatz beschreibt ein System für **LLM-getriebene Strategiesuche** mit zwei Schutzmechanismen:
  - „Registry-validated tools whose feature space excludes look-ahead by construction“,
  - eine Suchkorrektur, die alle Bewertungen mitzählt und das berichtete Ergebnis nach Trialzahl abwertet. Dabei zeigt sich: „the best in-sample Sharpe ratio climbs with each trial while the deflation threshold climbs faster“.
- Ergebnis: Das Verfahren bestätigt passive Benchmarks als signifikant und verwirft alle von LLMs gefundenen Strategien („honest evaluation certifies passive benchmarks“).
- Außerdem begründet der Aufsatz, warum „pre-registered hypotheses earn lower evidential bars than brute search“.

**Vergleich:** Das ist fast wörtlich die Lage von warchhold.com („0 von 9 vorwärts überlebt“, Kalibrierung −0,61). Der Aufsatz zeigt aber zwei Dinge, die warchhold.com fehlen: eine **Positivkontrolle**, die beweist, dass die Pipeline echte Effekte erkennen kann, und eine **sichtbare Deflationskurve**.

**Übertragbare Ideen**
1. **Positivkontrolle:** einen bekannten, einfachen Effekt (z. B. passives Halten des DAX oder einen gut belegten Kalendereffekt) regelmäßig durch dieselbe Pipeline schicken und das Ergebnis veröffentlichen. Wenn alles verworfen wird, muss man zeigen, dass die Hürde überhaupt passierbar ist.
2. **Deflationskurve je Familie:** bestes In-sample-Ergebnis und geforderte Schwelle über der Zahl der Versuche, als ein Diagramm. Die Daten (Trial Ledger) sind vorhanden.
3. **Hürden nach Herkunft staffeln:** vorregistrierte Literaturhypothesen mit niedrigerer Hürde als Ergebnisse aus Signal-Mining über „4896 Kombinationen“, und diese Staffel öffentlich begründen.

---

## Die 10 besten Ideen für warchhold.com

Geordnet nach dem Verhältnis von Nutzen zu Aufwand. Zuerst stehen die Ideen mit hohem Nutzen und wenig Aufwand, weil viele Daten bereits im System liegen.

| Rang | Idee | Vorbild | Aufwand | Nutzen | Wo auf warchhold.com |
|---|---|---|---|---|---|
| 1 | **Sieben-Punkte-Checkliste je Versuchsprotokoll** (Ja/Nein/Teilweise mit Beleg-Link). Die meisten Antworten liegen schon als Manifest-, Ledger- und Placebo-Daten vor. | Arnott/Harvey/Markowitz (5) | gering | hoch | Kopf jedes Build-Dossiers, Zusammenfassung in der Methodik |
| 2 | **„Befunde, die man zitieren kann“:** fünf bis acht Kernaussagen als Frage und Antwort mit ID, Evidenzklasse, n, Stand. Nur hier werden Zählgrößen genannt, alle anderen Seiten verlinken dorthin. | Fetna (6), AQR (3) | gering | hoch | Startseite und „Über mich“; beseitigt die widersprüchlichen Zählungen |
| 3 | **Einfrier-Protokoll mit Datum und Hash** für DEMO- und Vorwärtskandidaten, Änderungen nur als datierte Ergänzung. | Fetna (6) | gering | hoch | Strategien, Validierung |
| 4 | **Pflichtsatz „Wer zahlt und warum?“** ganz oben in jedem Protokoll, dazu ein kurzer Abschnitt zum KI-Einwand. | Robot Wealth (7) | gering | mittel | Build-Dossiers, Methodik |
| 5 | **Positivkontrolle:** ein bekannter Effekt bzw. passives Halten läuft regelmäßig durch dieselbe Pipeline, Ergebnis öffentlich. Das beweist, dass die Hürde passierbar ist. | Gençay (9) | mittel | hoch | Evidenzstand, Methodik |
| 6 | **Deflationskurve je Familie:** bestes In-sample-Ergebnis gegen die Schwelle über der Trialzahl. | Gençay (9), Arnott/Harvey/Markowitz (5) | mittel | hoch | Evidenzstand, Kandidatenvergleich |
| 7 | **Streudiagramm „Quelle vs. DAX“:** berichteter Effekt der Literatur gegen eigenes Ergebnis je Fund, mit Einstufung der Ausgangsevidenz. | Open Source Asset Pricing (1) | mittel | hoch | Hypothesen & Quellen, Startseite als Vorschau |
| 8 | **Familienansicht statt Einzelranking:** Versuche je Mechanismus-Familie mit Anzahl, Median, bestem Wert, Anteil verworfen, optional Bayes-geschrumpft. Ersetzt das Podium. | JKP (2) | mittel | hoch | ersetzt Ergebnisvergleich und Bestenliste |
| 9 | **Streuband „Zeitpunkt-Glück“** aus den vorhandenen Zufalls- und Anker-Replikationen in jedem Dossier, dazu ein Klartext-Satz. | Newfound (4) | mittel | mittel | Build-Dossiers |
| 10 | **Offline-Nachprüfung und externe Prüfung:** Ergebnis- und Evidenzdateien (ohne lizenzierte Kurse) plus Nachrechen-Skript öffentlich, Version mit DOI. Einladung zu Prüfberichten, die samt eigener Antwort erscheinen. | Fetna (6), I4R (8), JKP (2) | hoch | hoch | neue Seite „Prüfen Sie uns“, Fußzeile |

**Warum diese Reihenfolge:** Die Ränge 1 bis 4 sind überwiegend Darstellungsarbeit auf vorhandenen Daten und beheben die größten Glaubwürdigkeitsprobleme aus den bisherigen Reviews (widersprüchliche Zahlen, unklare Evidenz, Podium). Die Ränge 5 bis 9 brauchen neue Auswertungen, aber keine neue Infrastruktur. Rang 10 ist der größte Schritt (Veröffentlichung von Dateien, Rechte an Kursdaten, Prüfprozess), aber das stärkste Vertrauenssignal, das die Vorbilder gemeinsam haben.
