# Rechts- und Steuer-Checkliste: Forschungsshop

**Zweck:** Fragen für ein Gespräch mit Anwalt/Anwältin und Steuerberater/Steuerberaterin. Dieses Dokument enthält **keine Antworten** und keine rechtliche Einschätzung. Es beschreibt nur den Sachverhalt, damit die Fragen beantwortbar sind.

**Bezug:** `spezifikation-forschungsshop.md` (Entwurf 0.1).

**Kennzeichnung der Dringlichkeit:**
- **[vor Phase 2]:** vor der öffentlichen Schaufenster-Seite zu klären
- **[vor Phase 4]:** vor dem ersten echten Verkauf zu klären
- **[laufend]:** betrifft den Betrieb

---

## Sachverhalt in Kürze (zum Vorlegen)

**Wer und was:**
- **Betreiber:** Privatperson (Rechtsform und Hauptberuf bitte selbst ergänzen) betreibt die Website warchhold.com mit einer öffentlichen Forschungsplattform zu algorithmischen Handelsregeln auf den Deutschen Aktienindex (Index-CFD) und den Nikkei.
- **Angebot:** Digitale „Forschungspakete“ zu je 4,99 € (eine Strategie) bzw. 7,99 € (eine Familie aus Grundstrategie und Ergänzungen).
- **Inhalt eines Pakets:**
  - Regelwerk in Klartext mit allen Parametern
  - Pseudocode
  - Python-Referenzcode, nur mit der eigenen Backtest-Software lauffähig
  - Evidenzblatt als PDF mit Backtest-, Vorwärtstest- und DEMO-Ergebnissen in Indexpunkten
  - individuelle Lizenzkennung je Käufer
  - optional später eine TradingView-Fassung (Pine Script)

**Was nicht angeboten wird:**
- Kursdaten
- Handelssignale in Echtzeit
- Kontoverwaltung, Ertragsversprechen

**Wie es entsteht:**
- Eine KI wählt die angebotenen Pakete nach einer festen, vom Betreiber bestätigten Regel aus. Sie schreibt auch das Regelwerk in Klartext, das anschließend mechanisch gegen den Code geprüft wird.
- Gescheiterte Funde bleiben kostenlos sichtbar.
- Käufer erhalten Aktualisierungen und bei Scheitern automatisch den Befund.

**Wie verkauft wird:**
- Über Digistore24 als Merchant of Record (Zahlung, Rechnung, Umsatzsteuer, Widerruf).
- Zustellung durch eigenen Dienst per E-Mail mit befristetem Download-Link.
- Gespeichert wird nur die E-Mail-Adresse, verschlüsselt.

**Kursdaten und Kurse im Betrieb:**
- Die Kursdaten der Forschung stammen von einem Broker.
- Die Website zeigt heute DEMO-Ergebnisse (Spielgeld).
- Die interne Trading-Oberfläche zeigt Live-Kurse; ob und wo diese öffentlich erscheinen, bitte gesondert erfassen.

---

## 1. Gewerbe und Tätigkeit

1. **G-1 [vor Phase 4]:** Ist der Verkauf der Forschungspakete eine gewerbliche Tätigkeit, die angemeldet werden muss? Falls ja: als Haupt- oder Nebengewerbe, und mit welcher Tätigkeitsbeschreibung, damit sie nicht als Finanzdienstleistung missverstanden wird?
2. **G-2 [vor Phase 4]:** Ändert sich die Einordnung dadurch, dass eine KI die Auswahl und Texterstellung autonom übernimmt, der Betreiber aber Regel, Preise und Hinweistexte festlegt? Wer gilt als Anbieter und Verantwortlicher?
3. **G-3 [vor Phase 4]:** Falls der Betreiber angestellt ist: Ist eine Nebentätigkeitsanzeige oder -genehmigung beim Arbeitgeber nötig, insbesondere wenn der Arbeitgeber aus dem Finanzbereich kommt?
4. **G-4 [vor Phase 4]:** Braucht es für die Tätigkeit eine andere Rechtsform (z. B. UG/GmbH) zur Haftungsbegrenzung, oder genügt das Einzelunternehmen?

## 2. Steuern und Merchant of Record

1. **S-1 [vor Phase 4]:** Wie wirkt die Kleinunternehmerregelung (§ 19 UStG), wenn Digistore24 als Merchant of Record an Endkunden verkauft? Wer ist gegenüber dem Endkunden Leistender, und was genau stellt der Betreiber Digistore24 in Rechnung (Provision/Gutschrift)?
2. **S-2 [vor Phase 4]:** Muss der Betreiber gegenüber Digistore24 Rechnungen oder Gutschriften ausstellen bzw. annehmen, und mit welchen Pflichtangaben (mit oder ohne Umsatzsteuer)?
3. **S-3 [vor Phase 4]:** Wie werden die Auszahlungen einkommensteuerlich erfasst (Einkünfte aus Gewerbebetrieb bzw. selbständiger Arbeit), und welche Aufzeichnungen sind dafür nötig?
4. **S-4 [vor Phase 4]:** Hat es steuerliche Folgen, dass Käufer im EU-Ausland sitzen, wenn Digistore24 die Umsatzsteuer übernimmt?
5. **S-5 [laufend]:** Welche Kosten (Server, Mail-Dienst, KI-Nutzung, Broker-Datenzugang) sind dieser Tätigkeit zuzuordnen und absetzbar, wenn dieselbe Infrastruktur auch privat bzw. für eigenes Trading genutzt wird?
6. **S-6 [vor Phase 3]:** Welche Aufbewahrungsfristen gelten für das eigene Bestellbuch? Gespeichert werden Bestellnummer, Produkt, Betrag, Status, Lizenz und Zeitpunkte, aber keine Rechnungen, denn die stellt Digistore24 aus. Muss darüber hinaus etwas aufbewahrt werden?
7. **S-7 [vor Phase 4]:** Gilt die Preisangabe im Schaufenster (4,99 €, 7,99 €) als Endpreis im Sinne der Preisangabenverordnung? Wie ist sie zu formulieren, wenn der tatsächliche Endpreis im Digistore24-Checkout je nach Land abweichen kann?

## 3. Widerruf, Vertrag, Gewährleistung

1. **W-1 [vor Phase 4]:** Wie wird das Widerrufsrecht bei digitalen Inhalten (vorzeitiges Erlöschen nach ausdrücklicher Zustimmung und Bestätigung) korrekt umgesetzt, wenn Digistore24 den Checkout stellt? Welche Erklärungen müssen im Checkout erscheinen, und welche Bestätigung muss der Käufer danach erhalten? Wer liefert welchen Text?
2. **W-2 [vor Phase 4]:** Wer ist Vertragspartner des Käufers für den Inhalt des Pakets: Digistore24 als Wiederverkäufer oder der Betreiber? Welche Lizenzbedingungen (Nutzungsrechte am Code und an den Texten, keine Weitergabe) können wirksam vereinbart werden, und wo müssen sie dem Käufer vor dem Kauf vorliegen?
3. **W-3 [vor Phase 4]:** Welche Pflichten aus den Vorschriften über digitale Produkte (§§ 327 ff. BGB) treffen den Betreiber? Insbesondere:
   - Aktualisierungspflicht
   - Mangelbegriff bei einem Regelwerk, das „gescheitert“ ist
   - Dauer der Aktualisierungen

   Deckt die geplante Regel „Aktualisierungen 12 Monate, Befund bei Scheitern automatisch“ das ab, oder begründet sie selbst Pflichten?
4. **W-4 [vor Phase 4]:** Ist ein Paket, dessen Status nach dem Kauf auf „gescheitert“ wechselt, rechtlich mangelhaft, obwohl es ausdrücklich ohne Ertragsversprechen verkauft wurde?
5. **W-5 [vor Phase 4]:** Dürfen gescheiterte Funde kostenlos vollständig veröffentlicht werden, wenn inhaltsgleiche Pakete früher verkauft wurden? Gibt es Ansprüche früherer Käufer?
6. **W-6 [vor Phase 4]:** Wie ist mit Käufern umzugehen, die Unternehmer sind (kein Widerrufsrecht, andere Haftungsregeln)? Muss der Shop das unterscheiden?

## 4. Impressum, AGB, Datenschutz

1. **I-1 [vor Phase 2]:** Welche Impressumsangaben sind für die Website mit Shop nötig (§ 5 DDG), und wo müssen sie erreichbar sein? Im Quelltext des Portals liegt eine Impressumsvorlage (`lib/imprint.ts`), die derzeit nicht eingebunden ist.
2. **I-2 [vor Phase 4]:** Braucht der Betreiber neben den AGB von Digistore24 eigene AGB bzw. Lizenzbedingungen? Wie werden sie in den Digistore24-Checkout eingebunden?
3. **D-1 [vor Phase 4]:** Datenschutz-Rollen:
   - Ist Digistore24 für Käuferdaten eigener Verantwortlicher, gemeinsam Verantwortlicher oder Auftragsverarbeiter?
   - Welche Verträge sind nötig: mit Digistore24, mit dem Transaktions-Mail-Dienst (Auftragsverarbeitung), mit dem Hoster?
4. **D-2 [vor Phase 4]:** Rechtsgrundlagen und Fristen:
   - Genügt Vertragserfüllung als Rechtsgrundlage für Speicherung und Nutzung der E-Mail-Adresse, einschließlich der Mails „Aktualisierung verfügbar“ und „Befund zu Ihrem Paket“? Oder sind das Werbe-Mails, die eine Einwilligung brauchen?
   - Wie lange darf die Adresse gespeichert werden? Vorgeschlagen: Ende der Aktualisierungsfrist + 3 Monate.
5. **D-3 [vor Phase 4]:** Was muss die Datenschutzerklärung zusätzlich zum Shop enthalten? Genannt werden müssten: Mail-Dienst, Download-Protokoll ohne IP-Speicherung, Lizenzkennung in Dateien, Löschfristen.
6. **D-4 [vor Phase 4]:** Ist die Lizenzkennung in den Dateien (PDF-Fuß, Code-Kopf) ein personenbezogenes Datum, weil sie einer Bestellung zugeordnet werden kann? Muss darauf hingewiesen werden?
7. **D-5 [vor Phase 2]:** Die Texte und die Auswahl der Pakete werden von einer KI erzeugt. Bestehen Kennzeichnungspflichten (z. B. nach der KI-Verordnung der EU oder dem Wettbewerbsrecht)?

## 5. Nähe zu Anlageberatung, Finanzanalyse, Signaldiensten (BaFin)

1. **B-1 [vor Phase 2]:** Kann der Verkauf von Paketen mit konkreten Handelsregeln (Einstieg, Ausstieg, Stop, Parameter) für einen Index-CFD als Anlageberatung, Anlagevermittlung oder andere erlaubnispflichtige Tätigkeit nach KWG bzw. WpIG gelten? Welche Merkmale entscheiden das?
   - Merkmale auf der einen Seite: Regeln statt Einzelempfehlung, keine persönliche Ansprache, keine laufenden Signale.
   - Merkmale auf der anderen Seite: Das Paket ist „bestätigt“, und es gibt Aktualisierungen.
2. **B-2 [vor Phase 2]:** Sind die Evidenzblätter oder die Statusangaben („in Beobachtung“, „bestätigt“, „gescheitert“) eine Anlageempfehlung bzw. Finanzanalyse im Sinne der Marktmissbrauchsverordnung? Gibt es dann Offenlegungspflichten, z. B. zu eigenen Positionen des Betreibers in denselben Instrumenten?
3. **B-3 [vor Phase 4]:** Macht es einen Unterschied, dass der Betreiber dieselben Regeln selbst handelt? Heute nur DEMO öffentlich, intern möglicherweise mit Echtgeld. Wie ist dieser Interessenkonflikt offenzulegen?
4. **B-4 [vor Phase 4]:** Würde eine spätere Stufe mit automatischen Aktualisierungen (neue Parameter, neue Familienmitglieder) die Tätigkeit in Richtung „Signaldienst“ verschieben? Wo liegt die Grenze?
5. **B-5 [vor Phase 4]:** Ist eine Anfrage bei der BaFin (Negativtestat bzw. Auskunft zur Erlaubnispflicht) sinnvoll oder nötig, bevor der Verkauf startet?
6. **B-6 [vor Phase 2]:** Welche Aussagen dürfen Schaufenster und Pakete über vergangene Ergebnisse machen (Backtest, simulierter Vorwärtstest, DEMO), ohne irreführend im Sinne des Wettbewerbsrechts oder finanzaufsichtlich problematisch zu sein? Sind Pflichthinweise zu hypothetischen Ergebnissen vorgeschrieben, und mit welchem Wortlaut?

## 6. Haftung bei Nutzung mit Echtgeld

1. **H-1 [vor Phase 4]:** Kann der Betreiber haften, wenn ein Käufer ein Paket mit Echtgeld handelt und Verluste erleidet? Wie weit reicht ein Haftungsausschluss gegenüber Verbrauchern, und was lässt sich nicht ausschließen (Vorsatz, grobe Fahrlässigkeit)?
2. **H-2 [vor Phase 4]:** Kann der Betreiber haften, wenn Regelwerk und Referenzcode voneinander abweichen, obwohl eine mechanische Prüfung stattfindet? Ändert eine dokumentierte Prüfung die Haftungslage?
3. **H-3 [vor Phase 4]:** Welcher Hinweis- und Haftungstext ist an welcher Stelle nötig (Schaufenster, Checkout, Paket, Code-Kopf), und wer formuliert ihn? Die Spezifikation sieht einen einzigen, vom Betreiber festgelegten Text vor.
4. **H-4 [vor Phase 4]:** Ist eine Berufs- oder Vermögensschadenhaftpflichtversicherung für diese Tätigkeit sinnvoll oder üblich?
5. **H-5 [vor Phase 4]:** Welche Pflichten bestehen, wenn nachträglich ein Fehler im Backtest entdeckt wird, der verkaufte Evidenz betrifft? Genügen die Befund- und Aktualisierungs-Mails?

## 7. Marken und Namen

1. **M-1 [vor Phase 2]:** „DAX“ ist nach Kenntnis des Betreibers eine eingetragene Marke der Deutsche Börse AG. Das ist zu bestätigen.
   - Dürfen Produktbezeichnungen, Paketnamen, Dateinamen und Seitentitel den Indexnamen enthalten, z. B. „Forschungspaket DAX Overnight Gap Fade“?
   - Oder nur beschreibend im Text („für den deutschen Leitindex“)?
   - Ist ein Markenhinweis nötig?
2. **M-2 [vor Phase 2]:** Die Frage aus M-1 gilt entsprechend für „Nikkei“ bzw. „Nikkei 225“, für „TradingView“ und „Pine Script“ in der späteren Pine-Stufe sowie für Broker-Namen.
3. **M-3 [vor Phase 4]:** Sind die internen Strategie-Kennungen, die heute den Indexnamen tragen (z. B. `dax_…`), als Datei- oder Paketnamen unbedenklich, oder müssen sie für den Verkauf umbenannt werden?

## 8. Rechte an Kursdaten und an Quellen

1. **K-1 [vor Phase 2]:** Welche Nutzungsbedingungen des Brokers gelten für die Kursdaten (Ticks, Kerzen, Bid/Ask)? Sind darin kommerzielle Nutzung, Weitergabe und Veröffentlichung geregelt? Die Vertragsunterlagen des Brokers bitte mitbringen.
2. **K-2 [vor Phase 2]:** Darf die Website **Live-Kurse** öffentlich anzeigen, auch verzögert oder nur einzelne Werte wie Bid/Ask/Spread in einer Kopfzeile, wenn die Kurse vom Broker stammen? Gilt das auch für einen passwortgeschützten, aber aus dem Internet erreichbaren Bereich?
3. **K-3 [vor Phase 2]:** Sind **abgeleitete** Größen zulässig, die aus Broker-Kursen berechnet sind, aber selbst keine Kurse enthalten? Gemeint sind:
   - Ergebnisse in Punkten je Trade oder je Monat
   - Profit-Faktor, Trefferquote
   - Volatilitätskennzahlen im Evidenzblatt

   Wo liegt die Grenze zu einer Weitergabe der Daten, z. B. bei einer Tabelle einzelner Trades mit Zeitpunkt?
4. **K-4 [vor Phase 2]:** Braucht die Website für die Anzeige von Indexständen (auch historischer Werte) eine Lizenz des Indexanbieters, unabhängig von den Broker-Bedingungen?
5. **K-5 [vor Phase 5]:** Für die spätere Pine-Fassung ist eine Gleichheitsprüfung auf einer gemeinsamen Kerzenreihe geplant. Aus welcher Quelle dürfen diese Kerzen dafür stammen, und darf das Prüfprotokoll (ohne Kurse) im Paket stehen?
6. **Q-1 [vor Phase 4]:** Viele Funde gehen auf veröffentlichte Ideen zurück (Fachartikel, Blogs, Code-Repositorien; die Quelle steht je Fund in `spec.source_url`).
   - Darf die eigene Umsetzung verkauft werden?
   - Was gilt, wenn die Quelle Code unter einer Lizenz enthält (z. B. mit Weitergabe- oder Namensnennungspflicht)?
   - Muss die Quelle im Paket genannt werden?
7. **Q-2 [vor Phase 4]:** Wer hat Urheber- bzw. Nutzungsrechte an KI-erzeugten Texten und KI-erzeugtem Code, und welche Lizenz kann der Betreiber daran Käufern einräumen? Sind die Nutzungsbedingungen des verwendeten KI-Dienstes zu beachten?
8. **Q-3 [vor Phase 5]:** Erlauben die Bedingungen von TradingView den Verkauf von Pine-Script-Code außerhalb der Plattform?

## 9. Digistore24 selbst

1. **DS-1 [vor Phase 3]:** Lässt Digistore24 Produkte dieser Art (Handelsregeln für CFDs bzw. „Trading“) nach seinen Produktrichtlinien zu? Sind besondere Pflichtangaben oder Freigaben nötig?
2. **DS-2 [vor Phase 3]:** Welche Pflichten übernimmt Digistore24 als Merchant of Record vertraglich tatsächlich (Rechnung, Umsatzsteuer, Widerrufsabwicklung, Zahlungsausfall, Rückbuchungskosten)? Welche bleiben beim Betreiber?
3. **DS-3 [vor Phase 3]:** Darf der Betreiber die Zustellung selbst übernehmen (eigene Mail, eigener Download) statt über die Auslieferungsfunktion von Digistore24? Welche Vorgaben macht Digistore24 dazu, z. B. Zustellfristen oder Nachweis?
4. **DS-4 [vor Phase 3]:** Wie sind Provision und Auszahlung geregelt? Ist ein Preis von 4,99 € nach Gebühren wirtschaftlich? Das ist keine Rechtsfrage, gehört aber in dasselbe Gespräch mit dem Steuerberater.

---

## Mitzubringende Unterlagen

- `spezifikation-forschungsshop.md` (dieses Repository)
- geplanter Hinweis- und Haftungstext sowie Lizenzbedingungen (Entwurf, falls vorhanden)
- Vertrag bzw. Nutzungsbedingungen des Brokers einschließlich der Regeln zu Kursdaten
- Vertrag mit Digistore24 bzw. deren Verkäufer-AGB
- Nutzungsbedingungen des KI-Dienstes, der Texte und Code erzeugt
- ein Beispiel-Paket aus Phase 1, sobald vorhanden, mit geschwärzter Lizenzkennung
- Screenshots der öffentlichen Seiten mit Kursen und Ergebnisangaben (heutiger Stand)
