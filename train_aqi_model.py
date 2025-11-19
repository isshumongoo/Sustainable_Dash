import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

def main():
    # 1. Load the merged AQI + weather dataset
    aqi_path = DATA_DIR / "aqi_daily.csv"
    if not aqi_path.exists():
        raise FileNotFoundError(f"Could not find {aqi_path}. Run fetch_data.py first.")

    df = pd.read_csv(aqi_path, parse_dates=["date"])
    df = df.sort_values("date")

    # 2. Create lag features and target (next day AQI)
    df["AQI_lag1"] = df["AQI"].shift(1)
    df["AQI_lag2"] = df["AQI"].shift(2)
    df["AQI_lag3"] = df["AQI"].shift(3)

    df["AQI_next"] = df["AQI"].shift(-1)

    # 3. Drop rows with missing values needed for training
    df = df.dropna(subset=["AQI_lag1", "AQI_lag2", "AQI_lag3", "AQI_next", "temp", "humidity", "wind"])

    feature_cols = ["AQI_lag1", "AQI_lag2", "AQI_lag3", "temp", "humidity", "wind"]
    X = df[feature_cols]
    y = df["AQI_next"]

    if len(df) < 20:
        print("Warning: very few rows available for training. Model quality may be poor.")

    # 4. Time based train-test split (first 80 percent train, last 20 percent test)
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 5. Train model
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    print(f"Test MAE:  {mae:.2f}")
    print(f"Test RMSE: {rmse:.2f}")

    # 7. Save model
    model_path = MODELS_DIR / "aqi_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved model to {model_path}")

    # Optional: save a small evaluation table for your report
    eval_df = pd.DataFrame(
        {
            "date": df["date"].iloc[split_idx:].reset_index(drop=True),
            "AQI_actual": y_test.reset_index(drop=True),
            "AQI_pred": y_pred,
        }
    )
    eval_df.to_csv(DATA_DIR / "aqi_model_eval.csv", index=False)
    print(f"Saved evaluation table to {DATA_DIR / 'aqi_model_eval.csv'}")

if __name__ == "__main__":
    main()
