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

# --- Spalte 2: Prognose, Risikoampel, Rezessionstermin ---
with col2:
    # --- Prognosemodell ---
    df_model = df.copy()
    df_model["Rezession"] = (df_model["Industrieproduktion"] < 0).astype(int)
    features = ["EMI", "Arbeitslosenquote", "Zinskurve"]
    X = df_model[features]
    y = df_model["Rezession"]
    model = LogisticRegression()
    model.fit(X, y)
    aktuell = df_model.iloc[-1][features].values.reshape(1, -1)
    p_rezession = model.predict_proba(aktuell)[0][1]

    st.markdown("### 🔢 Rezessionswahrscheinlichkeit")
    st.metric(label="Deutschland / Eurozone", value=f"{p_rezession*100:.1f} %")

    st.markdown("### 🚨 Aktuelles Rezessionsrisiko")
    ampel = "🔴 **Hoch**" if p_rezession > 0.6 else ("🟡 **Mittel**" if p_rezession > 0.3 else "🟢 **Niedrig**")
    st.markdown(f"<div style='font-size: 24px; font-weight: bold;'>{ampel}</div>", unsafe_allow_html=True)

    st.markdown("### 📅 Erwarteter Rezessionszeitraum")
    heute = datetime.date.today()
    if p_rezession > 0.6:
        prog_date = heute + datetime.timedelta(days=90)
        st.markdown(f"Eine Rezession ist wahrscheinlich bis **{prog_date.strftime('%B %Y')}**.")
    elif p_rezession > 0.3:
        prog_date = heute + datetime.timedelta(days=180)
        st.markdown(f"Eine Rezession ist möglich bis **{prog_date.strftime('%B %Y')}**, falls sich der Trend verstärkt.")
    else:
        st.markdown("Aktuell keine konkrete Rezession in Sicht – jedoch Beobachtung empfohlen.")

# --- Empfehlungen für rezessionsresistente Sektoren ---
st.markdown("---")
st.subheader("📈 Sektor-Empfehlungen bei Rezessionsgefahr")
if p_rezession > 0.6:
    st.markdown("""
    Bei hohem Rezessionsrisiko gelten folgende Bereiche als relativ widerstandsfähig:

    - **Basiskonsum (Consumer Staples):** Lebensmittel, Haushaltswaren, Hygieneprodukte  
      *Beispiele:* Nestlé, Procter & Gamble, Unilever
    
    - **Gesundheitswesen (Healthcare):** Medikamente, Krankenhäuser, Medizintechnik  
      *Beispiele:* Pfizer, Roche, Johnson & Johnson

    - **Versorger (Utilities):** Strom, Wasser, Gas – stabile Einnahmen durch Grundversorgung  
      *Beispiele:* E.ON, RWE, NextEra Energy

    - **Gold & Edelmetalle:** Stabil in Krisenzeiten – profitieren von Unsicherheit und fallenden Realzinsen

    - **Hochqualitative Staatsanleihen:** Besonders bei erwarteten Zinssenkungen attraktiv
    """)
elif p_rezession > 0.3:
    st.markdown("""
    Es besteht ein moderates Risiko für eine wirtschaftliche Abschwächung. Folgende Sektoren könnten bereits stabilisierend wirken:

    - **Basiskonsum & Gesundheit:** Erste Umschichtungen in defensivere Titel sind möglich
    - **Cash & Geldmarkt-ETFs:** Erhöhte Liquidität sorgt für Flexibilität
    - **Große Technologieunternehmen mit stabilen Erträgen:** z. B. Microsoft, Apple
    """)
else:
    st.markdown("""
    Derzeit kein akuter Handlungsbedarf. Zyklische Branchen wie Industrie, Technologie und Konsumgüter profitieren bei Wachstum.
    Dennoch sollte ein schrittweiser Aufbau defensiver Positionen langfristig erwogen werden.
    """)

# --- Legende und Hinweise ---
st.markdown("---")
st.caption("Frühwarn-Indikatoren basieren derzeit auf statischen Werten. Live-Integration folgt.")
