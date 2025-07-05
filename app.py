# app.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Rezessionsmonitor", layout="wide")
st.title("🔍 Rezessions-Frühwarnsystem mit Prognose")

# --- Hilfsfunktionen ---
def fetch_sample_data():
    today = datetime.date.today()
    dates = pd.date_range(end=today, periods=6, freq='M')
    data = pd.DataFrame({
        "Datum": dates,
        "EMI": [49, 48, 47, 46, 45, 44],
        "Arbeitslosenquote": [5.1, 5.3, 5.4, 5.6, 5.9, 6.1],
        "Zinskurve": [0.3, 0.1, -0.2, -0.4, -0.6, -0.8],
        "Industrieproduktion": [1.2, 0.8, 0.5, -0.3, -1.1, -2.0]
    })
    return data

# --- Prognoseberechnung nach Funktionsdefinition ---
df = fetch_sample_data()
df_model = df.copy()
df_model["Rezession"] = (df_model["Industrieproduktion"] < 0).astype(int)
features = ["EMI", "Arbeitslosenquote", "Zinskurve"]
X = df_model[features]
y = df_model["Rezession"]
model = LogisticRegression()
model.fit(X, y)
aktuell = df_model.iloc[-1][features].values.reshape(1, -1)
p_rezession = model.predict_proba(aktuell)[0][1]

heute = datetime.date.today()
if p_rezession > 0.6:
    ampel = "🔴 Hoch"
    prog_date = heute + datetime.timedelta(days=90)
    rez_text = f"Wahrscheinliche Rezession bis {prog_date.strftime('%B %Y')}"



color = "#d9534f" if "🔴" in ampel else ("#f0ad4e" if "🟡" in ampel else "#5cb85c")
st.markdown(f"""
<div style='display: flex; justify-content: space-between; align-items: center; padding: 1rem; background-color: #151B23; border-radius: 10px;'>
    <h2 style='margin: 0; color: white;'>Aktuelles Rezessionsrisiko: {ampel.replace('🚦 ', '')}</h2>
    <h4 style='margin: 0; color: white;'>{rez_text}</h4>
</div>
""", unsafe_allow_html=True)

# --- Maßnahmen gegen die Rezession ---
st.markdown("---")
st.subheader("🛠️ Wirtschaftspolitische Maßnahmen zur Abschwächung einer Rezession")

option = st.selectbox("🔍 Maßnahmen anzeigen nach ...", ["Alle Maßnahmen", "Nur hohe Priorität", "Nur mittlere Priorität", "Nur niedrige Priorität"])

maßnahmen = [
    {
        "titel": "Senkung der Leitzinsen (Geldpolitik)",
        "priorität": "Hoch",
        "beschreibung": "Zentralbanken können die Kreditkosten senken, um Investitionen und Konsum anzuregen.",
        "effekt": "Günstigeres Kapital fördert Unternehmensgewinne, Aktienkurse steigen oft.",
        "aktien": "Banken, Immobilien, Wachstumsaktien (z. B. Tech)"
    },
    {
        "titel": "Kurzarbeitergeld und Arbeitsmarktprogramme",
        "priorität": "Hoch",
        "beschreibung": "Sichern Beschäftigung und verhindern massive Kaufkraftverluste.",
        "effekt": "Stützt Konsumgüter- und Einzelhandelsunternehmen durch stabile Nachfrage.",
        "aktien": "Einzelhandel, Nahrungsmittel, Basiskonsum (z. B. Nestlé, Walmart)"
    },
    {
        "titel": "Unterstützung für Unternehmen",
        "priorität": "Hoch",
        "beschreibung": "Kredite, Bürgschaften oder Zuschüsse zur Stabilisierung gefährdeter Branchen.",
        "effekt": "Reduziert Insolvenzrisiken und stabilisiert besonders anfällige Branchen wie Tourismus oder Industrie.",
        "aktien": "Luftfahrt, Industrie, Logistik (z. B. Lufthansa, Siemens)"
    },
    {
        "titel": "Staatliche Investitionsprogramme",
        "priorität": "Mittel",
        "beschreibung": "Infrastrukturprojekte, Digitalisierung oder Energieprojekte schaffen kurzfristig Nachfrage und Arbeitsplätze.",
        "effekt": "Bau-, Maschinenbau-, Energie- und Rohstoffunternehmen können profitieren.",
        "aktien": "Bau, Solar, Wasserstoff (z. B. HeidelbergCement, Siemens Energy)"
    },
    {
        "titel": "Steuersenkungen",
        "priorität": "Mittel",
        "beschreibung": "Durch mehr verfügbares Einkommen können private Haushalte und Unternehmen mehr konsumieren oder investieren.",
        "effekt": "Positive Effekte auf Konsum- und Industriesektoren, insbesondere zyklische Aktien.",
        "aktien": "Auto, Konsum, Technologie (z. B. BMW, Adidas, Apple)"
    },
    {
        "titel": "Quantitative Lockerung",
        "priorität": "Niedrig",
        "beschreibung": "Zentralbanken kaufen Anleihen oder andere Wertpapiere, um Liquidität ins Finanzsystem zu pumpen.",
        "effekt": "Höhere Liquidität fließt häufig auch in Aktienmärkte – besonders wachstumsorientierte Titel profitieren.",
        "aktien": "Tech, Growth-ETFs, Nasdaq (z. B. Amazon, Nvidia)"
    }
]

df_ma = pd.DataFrame([m for m in maßnahmen if option == "Alle Maßnahmen" or m["priorität"] in option])
st.markdown("### 💡 Übersicht der Maßnahmen")
st.dataframe(df_ma.rename(columns={
    "titel": "Maßnahme",
    "priorität": "Priorität",
    "beschreibung": "Beschreibung",
    "effekt": "Wirkung auf Aktien",
    "aktien": "Aktienempfehlungen"
}))

# --- Frühwarn-Indikatoren ---
st.markdown("### 🧭 Frühwarn-Indikatoren")
st.markdown("""
**1. Einkaufsmanagerindex (EMI)**  
Ein Wert unter 50 deutet auf schrumpfende wirtschaftliche Aktivität hin.  
**Rezessionssignal:** Drei aufeinanderfolgende Monate unter 48 gelten als starkes Warnzeichen.

**2. Arbeitslosenquote**  
Ein ansteigender Trend über mehrere Monate zeigt eine Schwächung des Arbeitsmarktes.  
**Rezessionssignal:** Steigt die Quote um mehr als 0.5 Prozentpunkte innerhalb von 3-6 Monaten, gilt das als Warnzeichen.

**3. Zinskurve (10J - 2J)**  
Wenn die kurzfristigen Zinsen höher sind als die langfristigen, spricht man von einer inversen Zinskurve.  
**Rezessionssignal:** Inversion über einen Zeitraum von mehr als 2 Monaten ist historisch ein verlässliches Frühwarnsignal.

**Hinweis:** Aktuell basieren diese Indikatoren noch auf Beispielwerten. Eine Live-Anbindung ist geplant.
""")

# --- Aktuelle Werte der Indikatoren ---
st.markdown("### 📊 Frühwarn-Indikatoren – Verlauf & Bewertung")
verlauf_df = df.rename(columns={
    "Datum": "Datum",
    "EMI": "Einkaufsmanagerindex (EMI)",
    "Arbeitslosenquote": "Arbeitslosenquote (%)",
    "Zinskurve": "Zinskurve (10J - 2J)",
    "Industrieproduktion": "Industrieproduktion (%)"
}).set_index("Datum")
st.dataframe(verlauf_df.style.format("{:.2f}"))

# --- Indikatorbewertung ---
st.markdown("### 🧾 Bewertung der Frühwarn-Indikatoren")

# EMI-Auswertung
emi_wert = df["EMI"].tail(3).tolist()
emi_kritisch = all(emi < 48 for emi in emi_wert)
st.markdown(f"**Einkaufsmanagerindex (EMI)**")
st.markdown(f"Letzte 3 Werte: {emi_wert}")
st.markdown(f"Bewertung: {'⚠️ kritisch – 3 Werte unter 48' if emi_kritisch else '✅ stabil – kein durchgehendes Unterschreiten'}")

# Arbeitslosenquote-Auswertung
arbeitslosen_start = df["Arbeitslosenquote"].iloc[-4]
arbeitslosen_aktuell = df["Arbeitslosenquote"].iloc[-1]
diff_arbeitslos = arbeitslosen_aktuell - arbeitslosen_start
st.markdown(f"**Arbeitslosenquote**")
st.markdown(f"Anstieg in 3 Monaten: {arbeitslosen_start:.2f}% → {arbeitslosen_aktuell:.2f}% ({diff_arbeitslos:+.2f} Prozentpunkte)")
st.markdown(f"Bewertung: {'⚠️ steigend – Anstieg über 0.5 Punkte' if diff_arbeitslos > 0.5 else '✅ moderat – kein signifikanter Anstieg'}")

# Zinskurve-Auswertung
zins_wert = df["Zinskurve"].tail(3).tolist()
zins_invertiert = all(z < 0 for z in zins_wert)
st.markdown(f"**Zinskurve (10J - 2J)**")
st.markdown(f"Letzte 3 Werte: {zins_wert}")
st.markdown(f"Bewertung: {'⚠️ invers – alle 3 Werte negativ' if zins_invertiert else '✅ normal – keine dauerhafte Inversion'}")

# --- Legende und Hinweise ---
st.markdown("---")
st.caption("Frühwarn-Indikatoren basieren derzeit auf statischen Werten. Live-Integration folgt.")
