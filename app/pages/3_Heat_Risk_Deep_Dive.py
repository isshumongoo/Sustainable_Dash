import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

# Paths relative to project root
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed"

st.set_page_config(page_title="Heat and Urban Canopy Deep Dive", layout="wide")

st.title("Heat and Urban Canopy Deep Dive")
st.caption("Exploring the relationship between tree canopy and heat burden across areas.")

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

@st.cache_data
def load_canopy():
    path = DATA_DIR / "tree_canopy_by_area.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data
def load_heat_index():
    # Optional file. If not available, we infer a simple heat risk score.
    path = DATA_DIR / "heat_index_by_tract.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

canopy_df = load_canopy()
heat_df = load_heat_index()

# ---------------------------------------------------------
# Guard clauses
# ---------------------------------------------------------

if canopy_df is None or canopy_df.empty:
    st.error("Tree canopy dataset not found. Make sure data/processed/tree_canopy_by_area.csv exists.")
else:
    # Normalize columns
    cols = {c.lower(): c for c in canopy_df.columns}
    area_col = cols.get("area_name", None)
    canopy_col = cols.get("tree_canopy", None)

    if area_col is None or canopy_col is None:
        st.error(f"Expected columns area_name and tree_canopy. Found: {list(canopy_df.columns)}")
    else:
        canopy_df = canopy_df.rename(columns={area_col: "area_name", canopy_col: "tree_canopy"})
        canopy_df["tree_canopy"] = pd.to_numeric(canopy_df["tree_canopy"], errors="coerce")
        canopy_df = canopy_df.dropna(subset=["tree_canopy"])

        # Merge heat index if available
        if heat_df is not None and not heat_df.empty:
            heat_cols = {c.lower(): c for c in heat_df.columns}
            heat_area = (
                heat_cols.get("area_name")
                or heat_cols.get("tract")
                or list(heat_df.columns)[0]
            )
            heat_metric = heat_cols.get("heat_index", None)

            if heat_metric:
                heat_df = heat_df.rename(
                    columns={heat_area: "area_name", heat_metric: "heat_index"}
                )
                df = canopy_df.merge(
                    heat_df[["area_name", "heat_index"]],
                    on="area_name",
                    how="left",
                )
            else:
                df = canopy_df.copy()
                st.info("Could not find heat_index column in heat file. Only showing canopy analysis.")
        else:
            df = canopy_df.copy()
            st.info("Heat index file not found. Using a simple heat risk score based on canopy percent.")

        # 🔥 Ensure we always have a usable heat_index column
        if "heat_index" not in df.columns or df["heat_index"].isna().all():
            # Lower canopy → higher heat burden
            max_canopy = df["tree_canopy"].max()
            df["heat_index"] = max_canopy - df["tree_canopy"]


        # Sidebar filter
        st.sidebar.markdown("### Heat deep dive filters")
        area_filter = st.sidebar.text_input("Filter by area name (optional)", "")

        df_view = df.copy()
        if area_filter.strip():
            df_view = df_view[df_view["area_name"].str.contains(area_filter, case=False, na=False)]

        if df_view.empty:
            st.warning("No areas match the current filter. Clear the filter to see data.")
            st.stop()

        # ---------------------------------------------------------
        # Section 1: Summary metrics
        # ---------------------------------------------------------

        st.markdown("## 1. Summary statistics")

        avg_canopy = df_view["tree_canopy"].mean()
        avg_heat = df_view["heat_index"].mean()

        best_canopy = df_view.loc[df_view["tree_canopy"].idxmax()]
        worst_canopy = df_view.loc[df_view["tree_canopy"].idxmin()]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average tree canopy", f"{avg_canopy:.1f} %")
        with col2:
            st.metric(
                "Highest canopy",
                f"{best_canopy['area_name']} ({best_canopy['tree_canopy']:.1f} %)"
            )
        with col3:
            st.metric(
                "Lowest canopy",
                f"{worst_canopy['area_name']} ({worst_canopy['tree_canopy']:.1f} %)"
            )

        # ---------------------------------------------------------
        # Section 2: Canopy and heat rankings
        # ---------------------------------------------------------

        st.markdown("## 2. Areas with highest heat burden and lowest canopy")

        col_left, col_right = st.columns(2)

        with col_left:
            fig_low_can = px.bar(
                df_view.sort_values("tree_canopy").head(10),
                x="area_name",
                y="tree_canopy",
                title="Lowest 10 areas by tree canopy",
                labels={"tree_canopy": "Canopy (%)"},
            )
            fig_low_can.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig_low_can, use_container_width=True, key="low_canopy_bar")

        with col_right:
            fig_high_heat = px.bar(
                df_view.sort_values("heat_index", ascending=False).head(10),
                x="area_name",
                y="heat_index",
                title="Highest 10 areas by heat index",
                labels={"heat_index": "Heat index"},
            )
            fig_high_heat.update_layout(xaxis_tickangle=-40)
            st.plotly_chart(fig_high_heat, use_container_width=True, key="high_heat_bar")

        # ---------------------------------------------------------
        # Section 3: Relationship analysis
        # ---------------------------------------------------------

        st.markdown("## 3. Relationship between tree canopy and heat burden")

        fig_scatter = px.scatter(
            df_view,
            x="tree_canopy",
            y="heat_index",
            trendline="ols",
            title="Tree canopy vs heat index",
            labels={"tree_canopy": "Tree canopy (%)", "heat_index": "Heat burden"},
        )
        st.plotly_chart(fig_scatter, use_container_width=True, key="canopy_heat_scatter")

        st.write(
            "This chart helps illustrate the inverse relationship: areas with lower tree canopy tend to experience higher heat burden."
        )

        # ---------------------------------------------------------
        # Section 4: Full table
        # ---------------------------------------------------------

        st.markdown("## 4. Detailed table")

        st.dataframe(
            df_view.sort_values("tree_canopy", ascending=False).reset_index(drop=True),
            use_container_width=True,
        )

# Navigation
st.markdown("---")
st.page_link("streamlit_app.py", label="⬅ Back to Home")
