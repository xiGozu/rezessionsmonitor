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
    return pd.read_csv("data/indikatoren.csv", parse_dates=["Datum"])

# --- Prognoseberechnung ---
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
elif p_rezession > 0.3:
    ampel = "🟡 Mittel"
    prog_date = heute + datetime.timedelta(days=180)
    rez_text = f"Mögliche Rezession im Zeitraum bis {prog_date.strftime('%B %Y')}"
else:
    ampel = "🟢 Gering"
    rez_text = "Aktuell keine konkrete Rezession in Sicht – jedoch Beobachtung empfohlen."

color = "#d9534f" if "🔴" in ampel else ("#f0ad4e" if "🟡" in ampel else "#5cb85c")
st.markdown(f"""
<div style='display: flex; justify-content: space-between; align-items: center; padding: 1rem; background-color: #151B23; border-radius: 10px;'>
    <h2 style='margin: 0; color: white;'>Aktuelles Rezessionsrisiko: {ampel.replace('🚦 ', '')} ({p_rezession:.0%} Wahrscheinlichkeit)</h2>
    <h4 style='margin: 0; color: white;'>{rez_text}</h4>
</div>
""", unsafe_allow_html=True)

# --- Indikatorentabelle anzeigen ---
st.header("Frühwarnindikatoren – Zeitreihe")
st.dataframe(df.set_index("Datum"))

st.markdown("""
### Was bedeuten die Indikatoren?
- **Einkaufsmanagerindex (EMI):** Ein Wert unter 50 signalisiert Schrumpfung. Ein starker Rückgang ist oft ein Frühindikator.
- **Arbeitslosenquote:** Steigt sie innerhalb weniger Monate um 0,5 % oder mehr, deutet das oft auf eine bevorstehende Rezession hin.
- **Zinskurve (10y - 2y):** Eine Inversion (negativer Wert) ist in der Vergangenheit häufig ein zuverlässiger Frühindikator gewesen.
- **Industrieproduktion:** Rückläufige Produktionszahlen deuten auf eine konjunkturelle Schwächephase hin.
""")

# --- Bewertung der einzelnen Indikatoren ---
bewertungen = []

emi_change = df.iloc[-1]["EMI"] - df.iloc[-2]["EMI"]
emi_bewertung = "🔴 Sinkt deutlich" if emi_change < -1 else ("🟡 Leicht rückläufig" if emi_change < 0 else "🟢 Stabil")
bewertungen.append(("Einkaufsmanagerindex (EMI)", f"Veränderung: {emi_change:.2f}", emi_bewertung))

arbeitslosen_diff = df.iloc[-1]["Arbeitslosenquote"] - df.iloc[-2]["Arbeitslosenquote"]
arbeitslosen_bewertung = "🔴 Steigt deutlich" if arbeitslosen_diff > 0.3 else ("🟡 Steigt leicht" if arbeitslosen_diff > 0.1 else "🟢 Stabil")
bewertungen.append(("Arbeitslosenquote", f"Veränderung: {arbeitslosen_diff:.2f} %-Punkte", arbeitslosen_bewertung))

zins = df.iloc[-1]["Zinskurve"]
zins_bewertung = "🔴 Invertiert" if zins < 0 else ("🟡 Flach" if zins < 0.2 else "🟢 Normal")
bewertungen.append(("Zinskurve", f"Letzter Wert: {zins:.2f}%", zins_bewertung))

ip_diff = df.iloc[-1]["Industrieproduktion"] - df.iloc[-2]["Industrieproduktion"]
ip_bewertung = "🔴 Schrumpft" if ip_diff < -0.5 else ("🟡 Schwächer" if ip_diff < 0 else "🟢 Wächst")
bewertungen.append(("Industrieproduktion", f"Veränderung: {ip_diff:.2f}%", ip_bewertung))

bewertung_df = pd.DataFrame(bewertungen, columns=["Indikator", "Veränderung / Stand", "Bewertung"])
st.header("📊 Bewertung der Frühwarnindikatoren")
st.dataframe(bewertung_df)

# --- Maßnahmen gegen Rezession ---
st.header("🛠️ Mögliche staatliche Gegenmaßnahmen & Auswirkungen auf Aktien")
maßnahmen = [
    {
        "maßnahme": "📉 Zinssenkung durch die Zentralbank",
        "wirkung": "Kreditvergabe wird stimuliert, Konsum und Investitionen steigen",
        "aktien": "Wachstumsaktien, Immobilien, Tech"
    },
    {
        "maßnahme": "🏗️ Infrastrukturprogramme",
        "wirkung": "Staat investiert in Bau und öffentliche Projekte",
        "aktien": "Bau, Maschinenbau, Grundstoffe"
    },
    {
        "maßnahme": "💸 Steuererleichterungen",
        "wirkung": "Mehr verfügbares Einkommen für Haushalte und Unternehmen",
        "aktien": "Konsumgüter, Einzelhandel, Banken"
    },
    {
        "maßnahme": "🏦 Stützung von Banken und Kreditmärkten",
        "wirkung": "Finanzsystem bleibt stabil, Vertrauen wird gesichert",
        "aktien": "Banken, Versicherungen"
    }
]

maßnahmen_df = pd.DataFrame(maßnahmen)
st.dataframe(maßnahmen_df.rename(columns={
    "maßnahme": "Maßnahme",
    "wirkung": "Erwartete Wirkung",
    "aktien": "Mögliche Aktienprofiteure"
}))
