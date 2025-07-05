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
