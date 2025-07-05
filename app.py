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

# --- Layout: Zwei Spalten für Übersichtlichkeit ---
col1, col2 = st.columns([1, 1])

# --- Spalte 1: Indikatoren & Beschreibung ---
with col1:
    st.subheader("📊 Frühwarn-Indikatoren")
    df = fetch_sample_data()
    st.dataframe(df.set_index("Datum"))

    st.markdown("""
    #### 📘 Beschreibung der Frühwarn-Indikatoren

    **EMI (Einkaufsmanagerindex):**
    Ein zentraler Frühindikator für die wirtschaftliche Aktivität in der Industrie. Werte über 50 signalisieren Expansion, Werte unter 50 Schrumpfung.
    **Rezessionssignal:** Bei einem anhaltenden Rückgang unter 47 über mehrere Monate steigt die Rezessionswahrscheinlichkeit deutlich.

    **Arbeitslosenquote:**
    Gibt den prozentualen Anteil der arbeitslosen Personen an der Erwerbsbevölkerung an. Ein konstanter Anstieg über mehrere Monate signalisiert wirtschaftliche Schwäche.
    **Rezessionssignal:** Steigt die Quote um mehr als 0,5 Prozentpunkte innerhalb von 3–6 Monaten, gilt das als Warnzeichen.

    **Zinskurve (10J - 2J Staatsanleihen):**
    Differenz zwischen langfristigen und kurzfristigen Zinssätzen. Eine normale Kurve ist positiv (langfristige Zinsen höher). Eine inverse Zinskurve (negative Werte) zeigt, dass Investoren kurzfristig höhere Risiken sehen.
    **Rezessionssignal:** Eine invertierte Kurve über mehrere Wochen (z. B. < -0,25 %) war in der Vergangenheit ein sehr verlässlicher Frühindikator.

    **Industrieproduktion (Veränderung ggü. Vorjahr):**
    Misst die reale Produktion der Industrie im Vergleich zum Vorjahresmonat. Rückgänge deuten auf sinkende Nachfrage und reduzierte Wirtschaftstätigkeit hin.
    **Rezessionssignal:** Wenn der Wert drei Monate in Folge negativ ist (unter 0 %), ist dies ein starkes Alarmsignal.
    """)

    st.markdown("---")
    st.subheader("🧠 Einzelbewertung der Frühwarn-Indikatoren")
    latest = df.iloc[-1]
    
    def bewertung_emi(val):
        if val < 47:
            return "🔴 Kritisch (unter 47)"
        elif val < 50:
            return "🟡 Schwächephase (unter 50)"
        else:
            return "🟢 Stabil"

    def bewertung_arbeitslosenquote(series):
        delta = series.iloc[-1] - series.iloc[-4]  # Änderung über 3 Monate
        if delta > 0.5:
            return f"🔴 Anstieg um {delta:.2f} % → Warnsignal"
        elif delta > 0.2:
            return f"🟡 Leichter Anstieg ({delta:.2f} %)"
        else:
            return f"🟢 Stabil ({delta:.2f} %)"

    def bewertung_zinskurve(val):
        if val < -0.25:
            return "🔴 Invertiert (Rezessionssignal)"
        elif val < 0:
            return "🟡 Leicht negativ"
        else:
            return "🟢 Normal"

    def bewertung_industrieprod(series):
        negatives = (series < 0).tail(3).sum()
        if negatives == 3:
            return "🔴 Drei Monate negativ"
        elif negatives >= 1:
            return f"🟡 {negatives}x negativ"
        else:
            return "🟢 Stabil"

    st.markdown(f"**EMI:** {latest['EMI']} → {bewertung_emi(latest['EMI'])}")
    st.markdown(f"**Arbeitslosenquote:** {latest['Arbeitslosenquote']} % → {bewertung_arbeitslosenquote(df['Arbeitslosenquote'])}")
    st.markdown(f"**Zinskurve:** {latest['Zinskurve']} % → {bewertung_zinskurve(latest['Zinskurve'])}")
    st.markdown(f"**Industrieproduktion:** {latest['Industrieproduktion']} % → {bewertung_industrieprod(df['Industrieproduktion'])}")
