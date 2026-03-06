import sqlite3
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

conn = sqlite3.connect("database.db")
df = pd.read_sql("SELECT soil, temperature, humidity, pump FROM sensor_data", conn)

df["pump"] = df["pump"].map({"ON": 1, "OFF": 0})

def label(row):
    if row["soil"] > 800:
        return "Poor"
    elif row["soil"] < 300:
        return "Overwatered"
    elif row["pump"] == 1:
        return "Good"
    else:
        return "Moderate"

df["health"] = df.apply(label, axis=1)

X = df[["soil", "temperature", "humidity", "pump"]]
y = df["health"]

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

joblib.dump(model, "plant_health_model.pkl")
print("✅ ML model trained & saved")
