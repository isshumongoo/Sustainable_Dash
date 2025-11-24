import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Set paths relative to project root
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

st.set_page_config(page_title="Air Quality Deep Dive", layout="wide")

st.title("Air Quality Deep Dive")
st.caption("Detailed analysis of AQI trends and the AQI prediction model.")

# -------------------------------------------------------------------
# Helpers to load data and model
# -------------------------------------------------------------------


@st.cache_data
def load_aqi_data():
    aqi_path = DATA_DIR / "aqi_daily.csv"
    eval_path = DATA_DIR / "aqi_model_eval.csv"

    aqi_df = None
    eval_df = None

    if aqi_path.exists():
        aqi_df = pd.read_csv(aqi_path, parse_dates=["date"]).sort_values("date")

    if eval_path.exists():
        eval_df = pd.read_csv(eval_path, parse_dates=["date"]).sort_values("date")

    return aqi_df, eval_df



@st.cache_resource
def load_aqi_model():
    model_path = MODELS_DIR / "aqi_model.pkl"
    if model_path.exists():
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None


aqi_df, eval_df = load_aqi_data()
aqi_model = load_aqi_model()

# -------------------------------------------------------------------
# Section 1: Data overview
# -------------------------------------------------------------------

st.markdown("## 1. AQI data overview")

if aqi_df is None or aqi_df.empty:
    st.error("AQI dataset not found. Make sure `data/processed/aqi_daily.csv` exists.")
else:
    # Sidebar like controls local to this page
    st.sidebar.markdown("### Air deep dive filters")
    time_options = {
        "Last 7 days": 7,
        "Last 30 days": 30,
        "Last 90 days": 90,
        "All data": None,
    }
    range_label = st.sidebar.selectbox(
        "Show AQI for",
        list(time_options.keys()),
        index=1,
    )
    days = time_options[range_label]

    aqi_sorted = aqi_df.sort_values("date")
    if days is not None:
        max_date = aqi_sorted["date"].max()
        min_date = max_date - pd.Timedelta(days=days)
        aqi_view = aqi_sorted[aqi_sorted["date"].between(min_date, max_date)]
    else:
        aqi_view = aqi_sorted.copy()

    col1, col2 = st.columns(2)

    with col1:
        fig_aqi = px.line(
            aqi_view,
            x="date",
            y="AQI",
            title=f"AQI over time ({range_label.lower()})",
        )
        fig_aqi.update_traces(mode="lines+markers")
        st.plotly_chart(fig_aqi, use_container_width=True, key="deep_aqi_line")

    with col2:
        if {"temp", "humidity", "wind"}.issubset(aqi_view.columns):
            fig_temp = px.line(
                aqi_view,
                x="date",
                y="temp",
                title=f"Temperature over time ({range_label.lower()})",
            )
            st.plotly_chart(fig_temp, use_container_width=True, key="deep_temp_line")
        else:
            st.info("Temperature column not available in AQI dataset.")

    st.markdown("### AQI distribution")
    fig_hist = px.histogram(
        aqi_view,
        x="AQI",
        nbins=20,
        title=f"AQI distribution ({range_label.lower()})",
    )
    st.plotly_chart(fig_hist, use_container_width=True, key="deep_aqi_hist")

# -------------------------------------------------------------------
# Section 2: Model performance
# -------------------------------------------------------------------

st.markdown("## 2. AQI prediction model performance")

if eval_df is None or eval_df.empty or aqi_model is None:
    st.warning(
        "Model evaluation data or model file not found. "
        "Make sure you ran `train_aqi_model.py` to create "
        "`data/processed/aqi_model_eval.csv` and `models/aqi_model.pkl`."
    )
else:
    # Compute metrics
    y_true = eval_df["AQI_actual"]
    y_pred = eval_df["AQI_pred"]
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.metric("Test MAE", f"{mae:.2f}")
    with col_m2:
        st.metric("Test RMSE", f"{rmse:.2f}")
    with col_m3:
        mean_aqi = aqi_df["AQI"].mean() if aqi_df is not None else np.nan
        st.metric("Mean AQI (overall)", f"{mean_aqi:.1f}" if not np.isnan(mean_aqi) else "N/A")

    # Actual vs predicted over time
    st.markdown("### Actual vs predicted AQI over time")

    eval_long = eval_df.melt(
        id_vars="date",
        value_vars=["AQI_actual", "AQI_pred"],
        var_name="series",
        value_name="AQI",
    )

    fig_eval = px.line(
        eval_long,
        x="date",
        y="AQI",
        color="series",
        title="Actual vs predicted AQI on test period",
    )
    fig_eval.update_traces(mode="lines+markers")
    st.plotly_chart(fig_eval, use_container_width=True, key="deep_eval_line")

    # Actual vs predicted scatter
    st.markdown("### Actual vs predicted scatter")

    fig_scatter = px.scatter(
        eval_df,
        x="AQI_actual",
        y="AQI_pred",
        title="Actual vs predicted AQI",
        trendline="ols",
        labels={"AQI_actual": "Actual AQI", "AQI_pred": "Predicted AQI"},
    )
    st.plotly_chart(fig_scatter, use_container_width=True, key="deep_eval_scatter")

    # Residual analysis
    st.markdown("### Residual analysis")
    eval_df["residual"] = eval_df["AQI_actual"] - eval_df["AQI_pred"]

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        fig_res_hist = px.histogram(
            eval_df,
            x="residual",
            nbins=20,
            title="Residuals (actual minus predicted)",
        )
        st.plotly_chart(fig_res_hist, use_container_width=True, key="deep_res_hist")

    with col_r2:
        fig_res_time = px.line(
            eval_df,
            x="date",
            y="residual",
            title="Residuals over time",
        )
        fig_res_time.update_traces(mode="lines+markers")
        st.plotly_chart(fig_res_time, use_container_width=True, key="deep_res_time")

    st.markdown("### Sample of model predictions")
    st.dataframe(eval_df.tail(20), use_container_width=True)

# -------------------------------------------------------------------
# Section 3: Feature importance
# -------------------------------------------------------------------

st.markdown("## 3. Feature importance")

if aqi_model is None:
    st.info("AQI model not loaded, cannot show feature importances.")
else:
    # Feature order from train_aqi_model.py
    feature_cols = ["AQI_lag1", "AQI_lag2", "AQI_lag3", "temp", "humidity", "wind"]

    if hasattr(aqi_model, "feature_importances_"):
        importances = aqi_model.feature_importances_
        fi_df = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)

        fig_fi = px.bar(
            fi_df,
            x="feature",
            y="importance",
            title="Feature importances from Random Forest",
        )
        st.plotly_chart(fig_fi, use_container_width=True, key="deep_feature_importance")

        st.caption(
            "This chart shows which inputs the model relied on most for predicting AQI, "
            "such as recent AQI history and weather conditions."
        )
    else:
        st.info("Current model type does not expose feature_importances_.")

# -------------------------------------------------------------------
# Navigation
# -------------------------------------------------------------------

st.markdown("---")
st.page_link("streamlit_app.py", label="⬅ Back to Home")
