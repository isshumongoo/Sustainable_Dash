import streamlit as st

st.set_page_config(page_title="Methods and Data", layout="wide")

st.title("Methods and Data")

st.subheader("Data sources")
st.markdown(
    """
    - Air quality: EPA or AirNow historical AQI data for Washington, DC  
    - Weather: Hourly temperature, humidity, and wind from a public weather API  
    - Waste and recycling: DC Open Data on solid waste and recycling by ward or route  
    - Heat risk: Land surface temperature or heat index at tract level  
    - Demographics: Census ACS (income, age distribution)
    """
)

st.subheader("Data processing")
st.markdown(
    """
    - Cleaned date formats and standardized units  
    - Joined datasets where possible by date and geographic key (ward, ZIP, tract)  
    - Created lag features for AQI modeling (AQI at previous hours)  
    - Built summary tables for recycling and heat indicators
    """
)

st.subheader("Modeling")
st.markdown(
    """
    - Problem: predict short term AQI for the city  
    - Models: linear regression baseline and random forest regressor  
    - Evaluation: time based train test split, reporting MAE and RMSE  
    - Features: lagged AQI values plus temperature, humidity, and wind
    """
)

st.subheader("Limitations")
st.markdown(
    """
    - Data gaps and imperfect spatial coverage  
    - Correlation does not imply causation  
    - Predictions are approximate and not official health guidance
    """
)

st.markdown("---")
st.page_link("streamlit_app.py", label="⬅ Back to Home")
