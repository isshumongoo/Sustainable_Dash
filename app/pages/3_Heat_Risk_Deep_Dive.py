import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Heat Risk Deep Dive", layout="wide")

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

@st.cache_data
def load_heat():
    path = DATA_DIR / "heat_index_by_tract.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)

st.title("Heat Risk Deep Dive")

heat = load_heat()

if heat is None or heat.empty:
    st.warning("Heat index dataset not found. Add 'heat_index_by_tract.csv' to data/processed.")
    st.stop()

st.subheader("Heat index by area")

if {"area_name", "heat_index"}.issubset(heat.columns):
    fig = px.bar(
        heat.sort_values("heat_index", ascending=False).head(30),
        x="area_name",
        y="heat_index",
        title="Areas with highest heat index",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Expected columns 'area_name' and 'heat_index' not present.")

if {"heat_index", "tree_canopy", "median_income"}.issubset(heat.columns):
    st.subheader("Heat, canopy, and income")

    fig_scatter = px.scatter(
        heat,
        x="tree_canopy",
        y="heat_index",
        color="median_income",
        title="Heat index vs tree canopy (colored by income)",
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
else:
    st.caption(
        "Add columns 'tree_canopy' and 'median_income' to explore equity patterns."
    )

st.markdown("---")
st.page_link("streamlit_app.py", label="⬅ Back to Home")
