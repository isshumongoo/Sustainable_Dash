import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from pathlib import Path

st.set_page_config(page_title="Air Quality Deep Dive", layout="wide")

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

@st.cache_data
def load_aqi():
    path = DATA_DIR / "aqi_daily.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"])

@st.cache_resource
def load_model():
    model_path = MODELS_DIR / "aqi_model.pkl"
    if not model_path.exists():
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)

st.title("Air Quality Deep Dive")

aqi = load_aqi()
model = load_model()

if aqi is None or aqi.empty:
    st.warning("AQI dataset not found. Add 'aqi_daily.csv' to data/processed.")
    st.stop()

aqi = aqi.sort_values("date")

st.subheader("Exploratory data analysis")
col1, col2 = st.columns(2)
with col1:
    fig = px.line(aqi, x="date", y="AQI", title="AQI over time")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    if {"temp", "humidity"}.issubset(aqi.columns):
        fig2 = px.scatter(
            aqi,
            x="temp",
            y="AQI",
            color="humidity",
            title="AQI vs temperature (colored by humidity)",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No temp and humidity columns found yet.")

st.subheader("Model evaluation (placeholder)")

if model is None:
    st.info("Model file 'aqi_model.pkl' not found. Train and save your model to show metrics here.")
else:
    st.write(
        "Here you can add code to compute MAE, RMSE, and a table of predictions "
        "vs actuals on a holdout set."
    )

st.markdown("---")
st.page_link("streamlit_app.py", label="⬅ Back to Home")

