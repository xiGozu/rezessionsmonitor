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

# --- Legende und Hinweise ---
st.markdown("---")
st.caption("Frühwarn-Indikatoren basieren derzeit auf statischen Werten. Live-Integration folgt.")
