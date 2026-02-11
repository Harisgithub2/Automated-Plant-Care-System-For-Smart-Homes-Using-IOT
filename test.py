import sqlite3
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

DB_PATH = "irrigation.db"
MODEL_PATH = "models/irrigation_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES_PATH = "models/feature_names.pkl"

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)


# -------------------------------------------------
# Load & prepare test data (MATCH TRAINING LOGIC)
# -------------------------------------------------
def load_test_data():
    conn = sqlite3.connect(DB_PATH)

    query = """
    SELECT
        timestamp,
        temperature,
        humidity,
        soil,
        CASE WHEN water='AVAILABLE' THEN 1 ELSE 0 END AS water_available,
        CASE WHEN pump='ON' THEN 1 ELSE 0 END AS actual
    FROM sensor_data
    WHERE temperature IS NOT NULL
      AND humidity IS NOT NULL
      AND soil IS NOT NULL
    ORDER BY timestamp
    LIMIT 500
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ---- Time features ----
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.weekday
    df["month"] = df["timestamp"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ---- Lag features ----
    df["temp_lag_1"] = df["temperature"].shift(1)
    df["humidity_lag_1"] = df["humidity"].shift(1)
    df["soil_lag_1"] = df["soil"].shift(1)

    # ---- Rolling averages ----
    df["temp_ma_3"] = df["temperature"].rolling(3).mean()
    df["humidity_ma_3"] = df["humidity"].rolling(3).mean()
    df["soil_ma_3"] = df["soil"].rolling(3).mean()

    # ---- Interaction features ----
    df["temp_humidity_interaction"] = df["temperature"] * df["humidity"] / 100
    df["humidity_soil_interaction"] = df["humidity"] * df["soil"] / 1000

    df.dropna(inplace=True)

    return df


# -------------------------------------------------
# Evaluate Model
# -------------------------------------------------
def evaluate_model(df):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)

    X = df[feature_names]
    y_true = df["actual"]

    X_scaled = scaler.transform(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_scaled)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_pred = (model.predict(X_scaled) >= 0.5).astype(int)

    metrics = {
        "Accuracy (%)": accuracy_score(y_true, y_pred) * 100,
        "Precision (%)": precision_score(y_true, y_pred, zero_division=0) * 100,
        "Recall (%)": recall_score(y_true, y_pred, zero_division=0) * 100,
        "F1-Score (%)": f1_score(y_true, y_pred, zero_division=0) * 100,
    }

    return y_true, y_pred, metrics


# -------------------------------------------------
# Plots & Tables
# -------------------------------------------------
def save_results(y_true, y_pred, metrics):
    # ---- Performance Table ----
    table = pd.DataFrame([metrics])
    table.to_csv(f"{OUT_DIR}/model_performance.csv", index=False)
    print("\n📊 MODEL PERFORMANCE TABLE")
    print(table.to_string(index=False))

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Pump OFF", "Pump ON"],
                yticklabels=["Pump OFF", "Pump ON"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=300)
    plt.close()

    # ---- Performance Bar Graph ----
    plt.figure(figsize=(6, 4))
    plt.bar(metrics.keys(), metrics.values())
    plt.ylim(0, 100)
    plt.ylabel("Percentage")
    plt.title("Model Performance Metrics")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/model_performance.png", dpi=300)
    plt.close()

    print("\n✅ Results saved in /results folder")


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    print("\n📊 Generating ML result figures...\n")

    df = load_test_data()
    if df.empty:
        print("❌ Not enough data")
        return

    y_true, y_pred, metrics = evaluate_model(df)
    save_results(y_true, y_pred, metrics)

    print("\n🎉 Model evaluation completed successfully!")


if __name__ == "__main__":
    main()
