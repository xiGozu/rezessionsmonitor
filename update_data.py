import pandas as pd
import datetime
import numpy as np

CSV_PATH = "data/indikatoren.csv"
df = pd.read_csv(CSV_PATH, parse_dates=["Datum"])
heute = datetime.date.today()
letzter_tag = df["Datum"].max().date()

if letzter_tag >= heute:
    print("🟢 Bereits aktuell.")
else:
    letzte = df.iloc[-1]
    neu = {
        "Datum": heute,
        "EMI": max(40, min(55, letzte["EMI"] + np.random.normal(0, 0.5))),
        "Arbeitslosenquote": max(3, letzte["Arbeitslosenquote"] + np.random.normal(0.05, 0.1)),
        "Zinskurve": letzte["Zinskurve"] + np.random.normal(0, 0.05),
        "Industrieproduktion": letzte["Industrieproduktion"] + np.random.normal(-0.1, 0.2)
    }
    df = pd.concat([df, pd.DataFrame([neu])], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)
    print(f"📈 Neue Daten für {heute} eingefügt.")
