import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Waste and Recycling Deep Dive", layout="wide")

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

@st.cache_data
def load_waste():
    path = DATA_DIR / "waste_by_area.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

st.title("Waste and Recycling Deep Dive")

waste = load_waste()

if waste is None or waste.empty:
    st.warning("Waste dataset not found. Add 'waste_by_area.csv' to data/processed.")
    st.stop()

st.subheader("Distribution of recycling rates")

if "recycle_rate" in waste.columns:
    fig = px.histogram(
        waste,
        x="recycle_rate",
        nbins=20,
        title="Distribution of recycling rates",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Column 'recycle_rate' not found in waste dataset.")

if {"area_name", "recycle_rate"}.issubset(waste.columns):
    st.subheader("Top and bottom performers")

    col1, col2 = st.columns(2)
    with col1:
        top = waste.sort_values("recycle_rate", ascending=False).head(15)
        fig_top = px.bar(
            top,
            x="area_name",
            y="recycle_rate",
            title="Top 15 areas",
        )
        st.plotly_chart(fig_top, use_container_width=True)

    with col2:
        bottom = waste.sort_values("recycle_rate", ascending=True).head(15)
        fig_bottom = px.bar(
            bottom,
            x="area_name",
            y="recycle_rate",
            title="Bottom 15 areas",
        )
        st.plotly_chart(fig_bottom, use_container_width=True)

st.markdown("---")
st.page_link("streamlit_app.py", label="⬅ Back to Home")
