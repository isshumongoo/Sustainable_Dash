# Environmental Conditions Dashboard  
## Applied Data Science Final Project — Washington, DC Environmental Analysis

---

# Overview

This project analyzes environmental conditions in Washington, DC and presents the results in a multi-page Streamlit dashboard.  
The dashboard includes:

- Air Quality (with a machine learning prediction model)  
- Waste and Recycling Performance  
- Urban Heat and Tree Canopy Coverage  
- Water Alerts and Service Disruptions  

The goal is to unify environmental indicators that are typically evaluated separately and provide a more complete picture of environmental health, equity, and sustainability in DC.

---

# Problem Statement

Air quality, heat exposure, recycling performance, and water safety each impact health and quality of life. These systems are often assessed independently, making it difficult to understand how environmental burdens overlap across neighborhoods.  
The project addresses this by creating a single dashboard that integrates these conditions and allows for deeper exploration and pattern detection.

---

# Why This Matters

- Poor air quality contributes to asthma and respiratory disease  
- Low recycling performance strains waste systems  
- Neighborhoods with low tree canopy experience higher heat exposure  
- Water advisories affect drinking, cooking, and sanitation  
- Environmental inequities often overlap in the same neighborhoods  

A combined analytical view helps highlight areas that may need additional support or resources.

---

# Approach and Methods

## Data Sources

- Air quality data from Open-Meteo and processed AQI files  
- Recycling and waste performance dataset by area  
- Tree canopy coverage merged with heat index or synthetic heat burden metric  
- Structured dataset of water alerts and disruptions  

## Tools and Libraries

- Python  
- Streamlit  
- Pandas and NumPy  
- Scikit-Learn (Random Forest for AQI predictions)  
- Plotly Express (interactive charts)  
- Requests (API calls)  
- Custom Streamlit theme using `config.toml`

## Pipeline Overview

1. Data extraction using `fetch_data.py`  
2. AQI model training using `train_aqi_model.py`  
3. Multi-page Streamlit UI for exploration  
4. Visualization of air, waste, heat, and water datasets  
5. Integration of machine learning predictions with historical trends  

---

# Results Summary

## Air Quality
- Clear daily and seasonal AQI patterns  
- Reasonable model performance predicting AQI from weather  
- Visualization of predicted vs actual AQI and residuals  

## Waste and Recycling
- Significant variation across neighborhoods  
- Identification of top and bottom performers  
- Supports targeted recycling outreach  

## Heat Risk
- Strong inverse correlation between tree canopy and heat burden  
- Highlights heat-vulnerable neighborhoods  

## Water Alerts
- Patterns in service disruptions and advisories  
- Identification of most affected areas  

---

# Project Structure

ADS_Final_Project/
│
├── app/
│ ├── streamlit_app.py
│ ├── .streamlit/
│ │ └── config.toml
│ └── pages/
│ ├── 1_Air_Quality_Deep_Dive.py
│ ├── 2_Waste_and_Recycling_Deep_Dive.py
│ ├── 3_Heat_Risk_Deep_Dive.py
│ ├── 4_Methods_and_Data.py
│ ├── 5_Water_Deep_Dive.py
│
├── data/
│ ├── processed/
│ └── raw/
│
├── models/
│ └── aqi_model.pkl
│
├── fetch_data.py
├── train_aqi_model.py
└── README.md

yaml
Copy code

---

# Running the Application

## Step 1 — Install dependencies
pip install -r requirements.txt

## Step 2 — Fetch data
python fetch_data.py

## Step 3 — Train the AQI model
python train_aqi_model.py

## Step 4 — Run the Streamlit dashboard
streamlit run app/streamlit_app.py

---

# Theme Configuration (`config.toml`)

[theme]
primaryColor = "#4C8C47"
backgroundColor = "#F6FAF4"
secondaryBackgroundColor = "#DAE5D0"
textColor = "#1F3B1C"
font = "sans serif"

---

# Challenges

- Missing or inconsistent API fields  
- Gaps in environmental datasets  
- Differences in geographic identifiers across sources  
- Debugging Streamlit caching and page linking  
- Creating a heat index proxy when data was incomplete  
- Designing a cohesive earth-tone UI  

---

# Final Impact

This dashboard provides a unified, interactive view of Washington DC’s environmental conditions.  
It highlights key patterns, identifies vulnerable neighborhoods, and demonstrates combined strengths in:

- Data wrangling  
- Machine learning  
- Visualization  
- Dashboard design  
- Environmental analytics  

This project supports more informed environmental planning and public health decision-making.

