import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

# Paths relative to project root
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

st.set_page_config(page_title="Water Alerts Deep Dive", layout="wide")

st.title("Water Alerts Deep Dive")
st.caption("Analysis of drinking water advisories and disruptions.")

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

@st.cache_data
def load_water():
    # This should match what fetch_data.py saves: water_alerts.csv
    path = DATA_DIR / "water_alerts.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df

water_df = load_water()

# ---------------------------------------------------------
# Guard
# ---------------------------------------------------------

if water_df is None or water_df.empty:
    st.error("Water alerts dataset not found. Make sure data/processed/water_alerts.csv exists.")
else:
    # Clean column names
    cols = {c.lower(): c for c in water_df.columns}

    area_col = cols.get("area_name") or cols.get("ward") or None
    type_col = cols.get("alert_type") or None
    sev_col = cols.get("severity") or None
    date_col = cols.get("date") or None
    desc_col = cols.get("description") or None

    df = water_df.copy()

    rename_map = {}
    if area_col:
        rename_map[area_col] = "area_name"
    else:
        df["area_name"] = "Unknown"

    if type_col:
        rename_map[type_col] = "alert_type"
    else:
        df["alert_type"] = "General notice"

    if sev_col:
        rename_map[sev_col] = "severity"
    else:
        df["severity"] = "Moderate"

    if date_col:
        rename_map[date_col] = "date"
    else:
        df["date"] = pd.NaT

    if desc_col:
        rename_map[desc_col] = "description"

    df = df.rename(columns=rename_map)

    # Parse date if present
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])

    # Sidebar filters
    st.sidebar.markdown("### Water deep dive filters")
    area_filter = st.sidebar.text_input("Filter by area name", "")

    type_options = sorted(df["alert_type"].dropna().unique().tolist())
    alert_filter = st.sidebar.multiselect(
        "Filter by alert type",
        type_options,
        default=type_options,
    )

    df_view = df.copy()

    if area_filter.strip():
        df_view = df_view[df_view["area_name"].str.contains(area_filter, case=False, na=False)]

    df_view = df_view[df_view["alert_type"].isin(alert_filter)]

    if df_view.empty:
        st.warning("No results match the current filters.")
        st.stop()

    # ---------------------------------------------------------
    # Section 1: Summary metrics
    # ---------------------------------------------------------

    st.markdown("## 1. Summary statistics")

    total_alerts = len(df_view)
    most_common_type = df_view["alert_type"].value_counts().idxmax()
    most_common_area = df_view["area_name"].value_counts().idxmax()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total alerts", str(total_alerts))
    with col2:
        st.metric("Most common alert type", most_common_type)
    with col3:
        st.metric("Most affected area", most_common_area)

    # ---------------------------------------------------------
    # Section 2: Alerts over time
    # ---------------------------------------------------------

    st.markdown("## 2. Alerts over time")

    if "date" in df_view.columns:
        df_daily = df_view.groupby("date").size().reset_index(name="count")

        fig_timeline = px.area(
            df_daily,
            x="date",
            y="count",
            title="Timeline of water alerts",
            labels={"date": "Date", "count": "Number of alerts"},
        )
        st.plotly_chart(fig_timeline, use_container_width=True, key="water_timeline")
    else:
        st.info("No valid date column found for timeline plot.")

    # ---------------------------------------------------------
    # Section 3: Alerts by type and severity
    # ---------------------------------------------------------

    st.markdown("## 3. Alerts by type and severity")

    colA, colB = st.columns(2)

    with colA:
        type_counts = df_view["alert_type"].value_counts().reset_index()
        type_counts.columns = ["alert_type", "count"]
        fig_type = px.bar(
            type_counts,
            x="alert_type",
            y="count",
            title="Alerts by type",
            labels={"alert_type": "Alert type", "count": "Count"},
        )
        fig_type.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig_type, use_container_width=True, key="water_type_bar")

    with colB:
        sev_counts = df_view["severity"].value_counts().reset_index()
        sev_counts.columns = ["severity", "count"]
        fig_sev = px.bar(
            sev_counts,
            x="severity",
            y="count",
            title="Alerts by severity",
            labels={"severity": "Severity", "count": "Count"},
        )
        fig_sev.update_layout(xaxis_tickangle=-35)
        st.plotly_chart(fig_sev, use_container_width=True, key="water_severity_bar")

    # ---------------------------------------------------------
    # Section 4: Alerts by area
    # ---------------------------------------------------------

    st.markdown("## 4. Geographic distribution (by area)")

    area_counts = df_view.groupby("area_name").size().reset_index(name="count")
    fig_area = px.bar(
        area_counts.sort_values("count", ascending=False),
        x="area_name",
        y="count",
        title="Alerts by area",
        labels={"area_name": "Area", "count": "Number of alerts"},
    )
    fig_area.update_layout(xaxis_tickangle=-40)
    st.plotly_chart(fig_area, use_container_width=True, key="water_area_bar")

    # ---------------------------------------------------------
    # Section 5: Detailed table
    # ---------------------------------------------------------

    st.markdown("## 5. Detailed alert table")

    st.dataframe(
        df_view.sort_values("date", ascending=False).reset_index(drop=True),
        use_container_width=True,
    )

# Navigation
st.markdown("---")
st.page_link("streamlit_app.py", label="⬅ Back to Home")
