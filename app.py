# app.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

st.set_page_config(page_title="Rezessionsmonitor", layout="wide")
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
    rez_text = f"Mögliche Rezession bis {prog_date.strftime('%B %Y')}"
else:
    ampel = "🟢 Niedrig"
    rez_text = "Aktuell keine konkrete Rezession in Sicht"

st.markdown(f"""
<div style='display: flex; justify-content: space-between; align-items: center; padding: 1rem; background-color: #f5f5f5; border-radius: 10px;'>
    <h2 style='margin: 0;'>🚦 Aktuelles Rezessionsrisiko: {ampel}</h2>
    <h4 style='margin: 0;'>{rez_text}</h4>
</div>
""", unsafe_allow_html=True)

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

# --- Spalte 2: Prognose, Risikoampel, Rezessionstermin ---
df_model = df.copy()
df_model["Rezession"] = (df_model["Industrieproduktion"] < 0).astype(int)
features = ["EMI", "Arbeitslosenquote", "Zinskurve"]
X = df_model[features]
y = df_model["Rezession"]
model = LogisticRegression()
model.fit(X, y)
aktuell = df_model.iloc[-1][features].values.reshape(1, -1)
p_rezession = model.predict_proba(aktuell)[0][1]

st.markdown("## 🚦 Aktuelle Rezessionsprognose")
st.markdown("### 🔢 Rezessionswahrscheinlichkeit")
st.metric(label="Deutschland / Eurozone", value=f"{p_rezession*100:.1f} %")

st.markdown("### 🚨 Aktuelles Rezessionsrisiko")
ampel = "🔴 **Hoch**" if p_rezession > 0.6 else ("🟡 **Mittel**" if p_rezession > 0.3 else "🟢 **Niedrig**")
st.markdown(f"<div style='font-size: 32px; font-weight: bold;'>{ampel}</div>", unsafe_allow_html=True)

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
