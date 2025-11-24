# app/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
from pathlib import Path

st.set_page_config(
    page_title="Sustainable City Dashboard",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

@st.cache_data
def load_dataset(filename: str, parse_dates=None):
    path = DATA_DIR / filename
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=parse_dates)

@st.cache_data
def load_all_data():
    aqi = load_dataset("aqi_daily.csv", parse_dates=["date"])
    waste = load_dataset("waste_by_area.csv")
    heat = load_dataset("heat_index_by_tract.csv")
    water = load_dataset("water_alerts.csv", parse_dates=["date"])
    return aqi, waste, heat, water

@st.cache_resource
def load_aqi_model():
    model_path = MODELS_DIR / "aqi_model.pkl"
    if not model_path.exists():
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

aqi, waste, heat, water = load_all_data()
aqi_model = load_aqi_model()

# After: aqi, waste, heat, water = load_all_data()

# Simple time range helper for AQI charts
TIME_RANGE_OPTIONS = {
    "Last 7 days": 7,
    "Last 30 days": 30,
    "Last 90 days": 90,
    "All data": None,
}

st.sidebar.markdown("### AQI View Range")
aqi_range_label = st.sidebar.selectbox(
    "Show AQI data for",
    list(TIME_RANGE_OPTIONS.keys()),
    index=1,
)
aqi_days = TIME_RANGE_OPTIONS[aqi_range_label]

if aqi is not None and not aqi.empty:
    aqi_sorted = aqi.sort_values("date")
    if aqi_days is not None:
        # filter by last N days
        max_date = aqi_sorted["date"].max()
        min_date = max_date - pd.Timedelta(days=aqi_days)
        aqi_view = aqi_sorted[aqi_sorted["date"].between(min_date, max_date)]
    else:
        aqi_view = aqi_sorted.copy()
else:
    aqi_view = None

# Sidebar filters (you can extend later)
st.sidebar.header("Filters")
city = st.sidebar.selectbox("City", ["Washington, DC"], index=0)
forecast_horizon = st.sidebar.slider("AQI forecast horizon (hours)", 1, 24, 6)

st.sidebar.markdown("---")
st.sidebar.page_link("streamlit_app.py", label="Home")
st.sidebar.page_link("pages/1_Air_Quality_Deep_Dive.py", label="Air Quality Deep Dive")
st.sidebar.page_link("pages/2_Waste_and_Recycling_Deep_Dive.py", label="Waste and Recycling Deep Dive")
st.sidebar.page_link("pages/3_Heat_Risk_Deep_Dive.py", label="Heat Risk Deep Dive")
st.sidebar.page_link("pages/4_Methods_and_Data.py", label="Methods and Data")
st.sidebar.page_link("pages/5_Water_Deep_Dive.py", label="Water Alerts Deep Dive")

# Title
st.title("🌍 Sustainable City Dashboard")
st.caption(f"City: {city}")
st.markdown("""
<div style="
    padding: 18px;
    border-radius: 10px;
    background: #E6EBF5;
    border-left: 6px solid #0C6CF2;
    font-size: 18px;">
This project explores environmental conditions in Washington DC through air quality, waste, urban heat, and water alerts using publicly available data.
</div>
""", unsafe_allow_html=True)


# KPIs
col1, col2, col3, col4 = st.columns(4)

if aqi is not None and not aqi.empty:
    aqi_sorted = aqi.sort_values("date")
    latest_row = aqi_sorted.iloc[-1]
    latest_aqi = latest_row.get("AQI", np.nan)

    weekly_series = aqi_sorted.set_index("date")["AQI"].last("7D")
    weekly_avg = weekly_series.mean() if not weekly_series.empty else np.nan

    max_temp = latest_row.get("temp_max", np.nan)
else:
    latest_aqi = np.nan
    weekly_avg = np.nan
    max_temp = np.nan

if waste is not None and "recycle_rate" in waste.columns and not waste.empty:
    recycle_pct = waste["recycle_rate"].mean()
else:
    recycle_pct = np.nan

col1.metric("Current AQI", "N/A" if np.isnan(latest_aqi) else f"{latest_aqi:.0f}")
col2.metric("7 Day Avg AQI", "N/A" if np.isnan(weekly_avg) else f"{weekly_avg:.1f}")
col3.metric("Max Temp (°F)", "N/A" if np.isnan(max_temp) else f"{max_temp:.1f}")
col4.metric("Avg Recycling Rate", "N/A" if np.isnan(recycle_pct) else f"{recycle_pct:.1f}%")

st.divider()

# Tabs
tab_overview, tab_air, tab_waste, tab_heat, tab_water = st.tabs(
    ["Overview", "Air Quality", "Waste", "Heat Risk", "Water"]
)

# ---------------- Overview tab ----------------
with tab_overview:
    st.subheader("City overview")

    if aqi_view is not None and not aqi_view.empty:
        # Two line charts side by side: AQI and temperature
        col_a, col_b = st.columns(2)

        with col_a:
            fig = px.line(
                aqi_view,
                x="date",
                y="AQI",
                title=f"AQI trend ({aqi_range_label.lower()})"
            )
            fig.update_traces(mode="lines+markers")
            fig.update_layout(margin=dict(l=0, r=0, t=40, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            if "temp" in aqi_view.columns:
                fig_temp = px.line(
                    aqi_view,
                    x="date",
                    y="temp",
                    title=f"Temperature trend ({aqi_range_label.lower()})"
                )
                fig_temp.update_traces(mode="lines+markers")
                fig_temp.update_layout(margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig_temp, use_container_width=True)
            else:
                st.info("No temperature column found in AQI dataset.")
    else:
        st.warning("AQI data not available. Place 'aqi_daily.csv' in data/processed.")

    st.markdown("### Waste and heat snapshots")

    col_a, col_b = st.columns(2)
    with col_a:
        if waste is not None and not waste.empty:
            if {"area_name", "recycle_rate"}.issubset(waste.columns):
                top = waste.sort_values("recycle_rate", ascending=False).head(8)
                fig_w = px.bar(
                    top,
                    x="area_name",
                    y="recycle_rate",
                    title="Top 8 areas by recycling rate",
                )
                fig_w.update_layout(xaxis_title="Area", yaxis_title="Recycle rate (%)")
                st.plotly_chart(fig_w, use_container_width=True)
        else:
            st.info("Waste data not available yet.")

    with col_b:
        if heat is not None and not heat.empty:
            # Treat heat_index or tree_canopy as "risk" metric depending on your file
            metric_col = "heat_index" if "heat_index" in heat.columns else "tree_canopy"
            title = "Areas with highest heat index" if metric_col == "heat_index" else "Areas with highest tree canopy"

            fig_h = px.bar(
                heat.sort_values(metric_col, ascending=False).head(8),
                x="area_name",
                y=metric_col,
                title=title,
            )
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.info("Heat data not available yet.")


# ---------------- Air tab ----------------
with tab_air:
    st.subheader("Air quality trends and forecast")

    if aqi_view is None or aqi_view.empty:
        st.warning("AQI data not available.")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            fig = px.line(
                aqi_view,
                x="date",
                y="AQI",
                title=f"AQI trend ({aqi_range_label.lower()})"
            )
            fig.update_traces(mode="lines+markers")
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig_hist = px.histogram(
                aqi_view,
                x="AQI",
                nbins=15,
                title=f"AQI distribution ({aqi_range_label.lower()})",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("#### Short term AQI forecast")

        if aqi_model is None:
            st.warning("AQI model not found. Run train_aqi_model.py to create models/aqi_model.pkl.")
        else:
            if len(aqi_sorted) < 3:
                st.warning("Not enough AQI history to generate a forecast.")
            else:
                last_row = aqi_sorted.iloc[-1]
                feature_row = {
                    "AQI_lag1": aqi_sorted["AQI"].iloc[-1],
                    "AQI_lag2": aqi_sorted["AQI"].iloc[-2],
                    "AQI_lag3": aqi_sorted["AQI"].iloc[-3],
                    "temp": last_row.get("temp", np.nan),
                    "humidity": last_row.get("humidity", np.nan),
                    "wind": last_row.get("wind", np.nan),
                }
                X_pred = pd.DataFrame([feature_row]).fillna(0)
                try:
                    y_hat = aqi_model.predict(X_pred)[0]
                    st.info(f"Predicted AQI for tomorrow: **{y_hat:.0f}**")
                except Exception as e:
                    st.error(f"Could not generate forecast: {e}")

        if {"AQI", "temp", "humidity"}.issubset(aqi_sorted.columns):
            st.markdown("#### Relationship between AQI and weather")
            fig_scatter = px.scatter(
                aqi_sorted,
                x="temp",
                y="AQI",
                color="humidity",
                title="AQI vs temperature (colored by humidity)",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

# ---------------- Waste tab ----------------
with tab_waste:
    st.subheader("Waste and recycling performance")

    if waste is None or waste.empty:
        st.warning("Waste data not available.")
    else:
        if {"area_name", "recycle_rate"}.issubset(waste.columns):
            col_left, col_right = st.columns(2)

            with col_left:
                top10 = waste.sort_values("recycle_rate", ascending=False).head(10)
                fig_top = px.bar(
                    top10,
                    x="area_name",
                    y="recycle_rate",
                    title="Top 10 areas by recycling rate",
                )
                fig_top.update_layout(
                    xaxis_title="Area",
                    yaxis_title="Recycle rate (%)",
                )
                st.plotly_chart(fig_top, use_container_width=True)

            with col_right:
                bottom10 = waste.sort_values("recycle_rate", ascending=True).head(10)
                fig_bottom = px.bar(
                    bottom10,
                    x="area_name",
                    y="recycle_rate",
                    title="Bottom 10 areas by recycling rate",
                )
                fig_bottom.update_layout(
                    xaxis_title="Area",
                    yaxis_title="Recycle rate (%)",
                )
                st.plotly_chart(fig_bottom, use_container_width=True)

            st.markdown("#### Detailed table")
            st.dataframe(
                waste.sort_values("recycle_rate", ascending=False),
                use_container_width=True,
            )
        else:
            st.info("Expected columns 'area_name' and 'recycle_rate' are not present.")

# ---------------- Heat tab ----------------
with tab_heat:
    st.subheader("Urban heat and canopy")

    if heat is None or heat.empty:
        st.warning("Heat or canopy data not available.")
    else:
        # Use heat_index if present, otherwise fall back to tree_canopy
        metric_col = "heat_index" if "heat_index" in heat.columns else "tree_canopy"
        title_metric = "Heat index" if metric_col == "heat_index" else "Tree canopy percent"

        col1, col2 = st.columns(2)

        with col1:
            fig_heat = px.bar(
                heat.sort_values(metric_col, ascending=False).head(15),
                x="area_name",
                y=metric_col,
                title=f"Top 15 areas by {title_metric.lower()}",
            )
            st.plotly_chart(fig_heat, use_container_width=True)

        with col2:
            if {"tree_canopy", metric_col}.issubset(heat.columns) and metric_col != "tree_canopy":
                fig_scatter = px.scatter(
                    heat,
                    x="tree_canopy",
                    y=metric_col,
                    title="Heat index vs tree canopy",
                    trendline="ols",
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info("Add both heat_index and tree_canopy columns to explore equity patterns.")

        st.caption("In future work this page can show a real choropleth map once tract shapes are added.")


# ---------------- Water tab ----------------
with tab_water:
    st.subheader("Water quality alerts")

    if water is None or water.empty:
        st.info("No water alerts dataset loaded yet.")
    else:
        water_sorted = water.sort_values("date", ascending=False)
        st.markdown("#### Recent alerts")
        st.dataframe(water_sorted.head(30), use_container_width=True)

        if "date" in water.columns:
            # Monthly count chart
            counts = (
                water.assign(month=water["date"].dt.to_period("M").astype(str))
                .groupby("month")
                .size()
                .reset_index(name="num_alerts")
            )
            fig_water = px.bar(
                counts,
                x="month",
                y="num_alerts",
                title="Number of water alerts per month",
            )
            st.plotly_chart(fig_water, use_container_width=True)

        if {"alert_type"}.issubset(water.columns):
            type_counts = (
                water.groupby("alert_type")
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            fig_type = px.bar(
                type_counts,
                x="alert_type",
                y="count",
                title="Alerts by type",
            )
            st.plotly_chart(fig_type, use_container_width=True)

st.markdown("---")
with st.expander("About this dashboard"):
    st.markdown(
        """
        **Sustainable City Dashboard**

        This app brings together air quality, waste and recycling, heat risk, and water alerts for a single city.
        The goal is to demonstrate applied data science techniques from data collection and cleaning to modeling and visualization.
        """
    )
