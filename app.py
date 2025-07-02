# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Rezessionsmonitor", layout="wide")
st.title("🔍 Rezessions-Frühwarnsystem mit Prognose")

# --- Hilfsfunktionen ---
def fetch_index(ticker):
    df = yf.download(ticker, period="6mo", interval="1d")
    if not df.empty:
        df.index = pd.to_datetime(df.index)
        return df["Close"].rename(ticker)
    else:
        return pd.Series(dtype=float, name=ticker)

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

# --- Datenabruf und Anzeige ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Aktuelle Leitindizes")
    try:
        dax = fetch_index("^GDAXI")
        sp500 = fetch_index("^GSPC")
        chart_data = pd.concat([dax, sp500], axis=1)
        st.line_chart(chart_data)
    except Exception as e:
        st.warning(f"Leitindizes konnten nicht geladen werden: {str(e)}")

with col2:
    st.subheader("Frühwarn-Indikatoren")
    df = fetch_sample_data()
    st.dataframe(df.set_index("Datum"))
    with st.expander("ℹ️ Beschreibung der Indikatoren"):
        st.markdown("""
        **EMI (Einkaufsmanagerindex):** Frühindikator für die wirtschaftliche Aktivität. Werte unter 50 deuten auf eine Schrumpfung hin.
        
        **Arbeitslosenquote:** Steigende Arbeitslosigkeit ist häufig ein Zeichen für wirtschaftliche Abschwächung.
        
        **Zinskurve:** Differenz zwischen langfristigen und kurzfristigen Zinssätzen. Eine negative Zinskurve (invers) gilt als Rezessionssignal.

        **Industrieproduktion:** Veränderung der produzierten Gütermenge im Vergleich zum Vorjahr. Ein Rückgang deutet auf eine schwächere Wirtschaft hin.
        """)

# --- Prognosemodell (vereinfachtes Beispiel) ---
df_model = df.copy()
df_model["Rezession"] = (df_model["Industrieproduktion"] < 0).astype(int)
features = ["EMI", "Arbeitslosenquote", "Zinskurve"]
X = df_model[features]
y = df_model["Rezession"]
model = LogisticRegression()
model.fit(X, y)

# Aktuelle Werte zur Prognose
aktuell = df_model.iloc[-1][features].values.reshape(1, -1)
p_rezession = model.predict_proba(aktuell)[0][1]

st.subheader("🔢 Rezessionswahrscheinlichkeit")
st.metric(label="Deutschland / Eurozone (vereinfachtes Modell)", value=f"{p_rezession*100:.1f} %", delta=None)

# --- Ampelanzeige ---
st.subheader("🚨 Risikobewertung")
ampel = "🔴 Hoch" if p_rezession > 0.6 else ("🟡 Mittel" if p_rezession > 0.3 else "🟢 Niedrig")
st.markdown(f"**Aktuelles Rezessionsrisiko:** {ampel}")

# --- Zeitprognose für mögliche nächste Rezession ---
st.subheader("📅 Geschätzter Zeitpunkt einer möglichen Rezession")
heute = datetime.date.today()
if p_rezession > 0.6:
    prog_date = heute + datetime.timedelta(days=90)
    st.markdown(f"Basierend auf den aktuellen Daten ist eine Rezession bis **{prog_date.strftime('%B %Y')}** wahrscheinlich.")
elif p_rezession > 0.3:
    prog_date = heute + datetime.timedelta(days=180)
    st.markdown(f"Eine Rezession ist möglich bis **{prog_date.strftime('%B %Y')}**, falls sich der Trend verstärkt.")
else:
    st.markdown("Aktuell keine konkrete Rezession in Sicht – jedoch Beobachtung empfohlen.")

# --- Legende und Hinweise ---
st.markdown("---")
st.caption("Live-Daten für DAX, S&P 500 via Yahoo Finance. Andere Indikatoren basieren auf Beispieldaten.")
st.caption("Zukünftig werden echte Datenquellen wie Eurostat, FRED oder TradingEconomics integriert.")
