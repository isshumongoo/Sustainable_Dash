import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

# Paths relative to project root
BASE_DIR = Path(__file__).resolve().parents[2]  # app/pages -> app -> project root
DATA_DIR = BASE_DIR / "data" / "processed"

st.set_page_config(page_title="Waste and Recycling Deep Dive", layout="wide")

st.title("Waste and Recycling Deep Dive")
st.caption("Exploring recycling performance across areas.")

# -------------------------------------------------------------------
# Load data
# -------------------------------------------------------------------


@st.cache_data
def load_waste_data():
    waste_path = DATA_DIR / "waste_by_area.csv"
    if not waste_path.exists():
        return None
    df = pd.read_csv(waste_path)
    return df


waste_df = load_waste_data()

# -------------------------------------------------------------------
# Guard: check data
# -------------------------------------------------------------------

if waste_df is None or waste_df.empty:
    st.error(
        "Waste dataset not found or empty. "
        "Make sure `data/processed/waste_by_area.csv` exists."
    )
else:
    # Normalize column names just in case
    cols_lower = {c.lower(): c for c in waste_df.columns}
    area_col = cols_lower.get("area_name", None)
    recycle_col = cols_lower.get("recycle_rate", None)

    if area_col is None or recycle_col is None:
        st.error(
            "Expected columns `area_name` and `recycle_rate` in waste_by_area.csv. "
            f"Found columns: {list(waste_df.columns)}"
        )
    else:
        df = waste_df.rename(columns={area_col: "area_name", recycle_col: "recycle_rate"})

        # Clean recycle_rate if it comes in as string
        df["recycle_rate"] = pd.to_numeric(df["recycle_rate"], errors="coerce")
        df = df.dropna(subset=["recycle_rate"])

        # Optional simple sidebar filter
        st.sidebar.markdown("### Waste deep dive filters")
        area_filter = st.sidebar.text_input(
            "Filter by area name (optional)",
            value="",
            help="Type part of an area or ward name to filter the table and charts.",
        )

        df_view = df.copy()
        if area_filter.strip():
            df_view = df_view[df_view["area_name"].str.contains(area_filter, case=False, na=False)]

        # -------------------------------------------------------------------
        # Section 1 - Summary metrics
        # -------------------------------------------------------------------

        st.markdown("## 1. Summary of recycling performance")

        avg_rate = df_view["recycle_rate"].mean()
        best_row = df_view.loc[df_view["recycle_rate"].idxmax()]
        worst_row = df_view.loc[df_view["recycle_rate"].idxmin()]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average recycling rate", f"{avg_rate:.1f} %")
        with col2:
            st.metric(
                "Best area",
                f"{best_row['area_name']} ({best_row['recycle_rate']:.1f} %)",
            )
        with col3:
            st.metric(
                "Lowest area",
                f"{worst_row['area_name']} ({worst_row['recycle_rate']:.1f} %)",
            )

        # -------------------------------------------------------------------
        # Section 2 - Distribution and rankings
        # -------------------------------------------------------------------

        st.markdown("## 2. Distribution and rankings")

        col_left, col_right = st.columns(2)

        with col_left:
            fig_hist = px.histogram(
                df_view,
                x="recycle_rate",
                nbins=15,
                title="Distribution of recycling rates",
                labels={"recycle_rate": "Recycling rate (%)"},
            )
            st.plotly_chart(fig_hist, use_container_width=True, key="waste_hist")

        with col_right:
            # Top and bottom performers
            top_n = 10 if len(df_view) >= 10 else len(df_view)
            top = df_view.sort_values("recycle_rate", ascending=False).head(top_n)
            bottom = df_view.sort_values("recycle_rate", ascending=True).head(top_n)

            st.markdown("**Top performing areas**")
            fig_top = px.bar(
                top,
                x="area_name",
                y="recycle_rate",
                title=f"Top {top_n} areas by recycling rate",
                labels={"area_name": "Area", "recycle_rate": "Recycling rate (%)"},
            )
            fig_top.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_top, use_container_width=True, key="waste_top_bar")

            st.markdown("**Lowest performing areas**")
            fig_bottom = px.bar(
                bottom,
                x="area_name",
                y="recycle_rate",
                title=f"Bottom {top_n} areas by recycling rate",
                labels={"area_name": "Area", "recycle_rate": "Recycling rate (%)"},
            )
            fig_bottom.update_layout(xaxis_tickangle=-35)
            st.plotly_chart(fig_bottom, use_container_width=True, key="waste_bottom_bar")

        # -------------------------------------------------------------------
        # Section 3 - Detailed table
        # -------------------------------------------------------------------

        st.markdown("## 3. Detailed table")

        st.write(
            "You can sort this table to see which areas lead or lag in recycling rates, "
            "and apply the text filter on the left to focus on specific wards or neighborhoods."
        )

        st.dataframe(
            df_view.sort_values("recycle_rate", ascending=False).reset_index(drop=True),
            use_container_width=True,
        )

# -------------------------------------------------------------------
# Navigation
# -------------------------------------------------------------------

st.markdown("---")
st.page_link("streamlit_app.py", label="⬅ Back to Home")
