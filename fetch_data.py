"""
Fetch environmental data for Washington DC from public APIs
and save both raw and processed CSVs.

APIs used:
- OpenAQ: air quality (PM2.5 as AQI proxy)
- Open-Meteo: hourly weather (temperature, humidity, windspeed)
- DC Open Data: waste and recycling
- DC Open Data: tree canopy
- EPA Water Quality Portal: water quality measurements

Note:
You may need to adjust field names once you inspect the actual JSON
responses from these APIs. This script is designed to be a clear starting point.
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta

# -------------------------------------------------------------------
# Paths and constants
# -------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Washington DC coordinates
DC_LAT = 38.9072
DC_LON = -77.0369

# Use the last 30 days so Open-Meteo is happy
END_DATE = date.today()
START_DATE = END_DATE - timedelta(days=30)

# Strings for API params
START_DATE_STR = START_DATE.isoformat()
END_DATE_STR = END_DATE.isoformat()

# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def safe_request(url: str, params: dict | None = None) -> requests.Response:
    """Perform a GET request with basic error handling."""
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp

# -------------------------------------------------------------------
# 1. Open-Meteo Air Quality - hourly US AQI for Washington DC
# -------------------------------------------------------------------

def fetch_open_meteo_air_quality(lat: float = DC_LAT, lon: float = DC_LON) -> pd.DataFrame:
    """
    Fetch hourly US AQI for the DC area from Open-Meteo's air quality API
    and aggregate to daily averages.

    This gives us a proper AQI value per hour, which we then average by day.
    """
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "us_aqi",
        "timezone": "America/New_York",
        "start_date": START_DATE_STR,
        "end_date": END_DATE_STR,
    }

    resp = safe_request(url, params=params)
    data = resp.json()

    hourly = data.get("hourly", {})
    time_list = hourly.get("time", [])
    aqi_list = hourly.get("us_aqi", [])

    if not time_list or not aqi_list:
        print("No AQI data returned from Open-Meteo air quality API")
        return pd.DataFrame()

    rows = []
    for t, aqi_value in zip(time_list, aqi_list):
        rows.append(
            {
                "datetime": pd.to_datetime(t),
                "AQI": aqi_value,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(RAW_DIR / "open_meteo_aqi_hourly_raw.csv", index=False)

    # Aggregate to daily averages
    df["date"] = df["datetime"].dt.date
    daily = (
        df.groupby("date", as_index=False)["AQI"]
        .mean()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily.to_csv(PROCESSED_DIR / "open_meteo_aqi_daily.csv", index=False)

    print(f"Saved Open-Meteo daily AQI to {PROCESSED_DIR / 'open_meteo_aqi_daily.csv'}")
    return daily



# -------------------------------------------------------------------
# 2. Open-Meteo - hourly temperature, humidity, windspeed
# -------------------------------------------------------------------

def fetch_open_meteo_hourly(lat: float = DC_LAT, lon: float = DC_LON) -> pd.DataFrame:
    """
    Fetch hourly weather data (temperature, humidity, windspeed) for the DC area
    from Open-Meteo and aggregate to daily averages.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m",
        "timezone": "America/New_York",
        "start_date": START_DATE_STR,
        "end_date": END_DATE_STR,
    }

    resp = safe_request(url, params=params)
    data = resp.json()

    hourly = data.get("hourly", {})
    time_list = hourly.get("time", [])
    temp_list = hourly.get("temperature_2m", [])
    hum_list = hourly.get("relativehumidity_2m", [])
    wind_list = hourly.get("windspeed_10m", [])

    if not time_list:
        print("No hourly data returned from Open-Meteo")
        return pd.DataFrame()

    rows = []
    for t, temp, hum, wind in zip(time_list, temp_list, hum_list, wind_list):
        rows.append(
            {
                "datetime": pd.to_datetime(t),
                "temp": temp,
                "humidity": hum,
                "wind": wind,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(RAW_DIR / "open_meteo_hourly_raw.csv", index=False)

    # Aggregate to daily averages
    df["date"] = df["datetime"].dt.date
    daily = (
        df.groupby("date", as_index=False)[["temp", "humidity", "wind"]]
        .mean()
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily.to_csv(PROCESSED_DIR / "open_meteo_daily.csv", index=False)

    print(f"Saved Open-Meteo daily averages to {PROCESSED_DIR / 'open_meteo_daily.csv'}")
    return daily


# -------------------------------------------------------------------
# 3. Build aqi_daily.csv by merging OpenAQ + Open-Meteo
# -------------------------------------------------------------------

def build_aqi_daily():
    """
    Merge OpenAQ daily PM2.5 (as AQI proxy) with Open-Meteo daily weather
    into a single dataset: aqi_daily.csv
    """

    aq_path = PROCESSED_DIR / "open_meteo_aqi_daily.csv"
    wx_path = PROCESSED_DIR / "open_meteo_daily.csv"

    if not aq_path.exists() or not wx_path.exists():
        raise FileNotFoundError("Missing processed OpenAQ or Open-Meteo daily CSVs")

    aq = pd.read_csv(aq_path, parse_dates=["date"])
    wx = pd.read_csv(wx_path, parse_dates=["date"])

    merged = pd.merge(aq, wx, on="date", how="inner")
    merged = merged.sort_values("date")

    merged.to_csv(PROCESSED_DIR / "aqi_daily.csv", index=False)
    print(f"Saved merged AQI dataset to {PROCESSED_DIR / 'aqi_daily.csv'}")
    return merged


# -------------------------------------------------------------------
# 4. DC Open Data - waste and recycling
# -------------------------------------------------------------------

def fetch_dc_waste() -> pd.DataFrame:
    """
    Try to fetch waste or recycling data from DC Open Data.
    If the API does not return JSON or fails, fall back to a small
    sample dataset so the dashboard still works.

    For the app, we want:
    - area_name
    - recycle_rate
    """
    url = "https://opendata.dc.gov/resource/jb5t-ybvk.json"
    params = {"$limit": 5000}

    try:
        resp = safe_request(url, params=params)
        try:
            data = resp.json()
        except ValueError:
            print("DC waste API did not return JSON. Using sample waste data instead.")
            raise RuntimeError("Non JSON response")
    except Exception as e:
        print(f"Could not fetch DC waste data from API: {e}")
        print("Creating sample waste_by_area.csv instead.")

        # Sample data you can mention in your report as placeholder
        df_sample = pd.DataFrame(
            {
                "area_name": ["Ward 1", "Ward 2", "Ward 3", "Ward 4"],
                "recycle_rate": [32.5, 41.2, 50.1, 28.7],
            }
        )
        df_sample.to_csv(PROCESSED_DIR / "waste_by_area.csv", index=False)
        print(f"Saved sample waste data to {PROCESSED_DIR / 'waste_by_area.csv'}")
        return df_sample

    # If JSON decoding worked, continue with processing
    df = pd.DataFrame(data)
    df.to_csv(RAW_DIR / "dc_waste_raw.csv", index=False)

    if df.empty:
        print("DC waste API returned no data. Falling back to sample data.")
        df_sample = pd.DataFrame(
            {
                "area_name": ["Ward 1", "Ward 2", "Ward 3", "Ward 4"],
                "recycle_rate": [32.5, 41.2, 50.1, 28.7],
            }
        )
        df_sample.to_csv(PROCESSED_DIR / "waste_by_area.csv", index=False)
        return df_sample

    # Try to map columns
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if "ward" in lc or "area" in lc or "neighborhood" in lc:
            col_map[c] = "area_name"
        if "recycle" in lc and ("rate" in lc or "percent" in lc or "pct" in lc):
            col_map[c] = "recycle_rate"

    df = df.rename(columns=col_map)

    keep_cols = [c for c in ["area_name", "recycle_rate"] if c in df.columns]
    if keep_cols:
        df = df[keep_cols]

    if df.empty or "area_name" not in df.columns or "recycle_rate" not in df.columns:
        print("Could not find expected columns in DC waste data. Using sample data instead.")
        df_sample = pd.DataFrame(
            {
                "area_name": ["Ward 1", "Ward 2", "Ward 3", "Ward 4"],
                "recycle_rate": [32.5, 41.2, 50.1, 28.7],
            }
        )
        df_sample.to_csv(PROCESSED_DIR / "waste_by_area.csv", index=False)
        return df_sample

    df.to_csv(PROCESSED_DIR / "waste_by_area.csv", index=False)
    print(f"Saved processed waste data to {PROCESSED_DIR / 'waste_by_area.csv'}")
    return df

# -------------------------------------------------------------------
# 5. DC Open Data - tree canopy (for heat risk)
# -------------------------------------------------------------------

def fetch_dc_tree_canopy() -> pd.DataFrame:
    """
    Try to fetch tree canopy data for DC from Open Data.
    If the API does not return JSON or fails, fall back to a small
    sample dataset so the dashboard still works.

    For the app, we want:
    - area_name
    - tree_canopy (percent)
    """
    url = "https://opendata.dc.gov/resource/somq-wq4j.json"
    params = {"$limit": 5000}

    try:
        resp = safe_request(url, params=params)
        try:
            data = resp.json()
        except ValueError:
            print("DC tree canopy API did not return JSON. Using sample canopy data instead.")
            raise RuntimeError("Non JSON response")
    except Exception as e:
        print(f"Could not fetch DC tree canopy data from API: {e}")
        print("Creating sample tree_canopy_by_area.csv instead.")

        df_sample = pd.DataFrame(
            {
                "area_name": ["Ward 1", "Ward 2", "Ward 3", "Ward 4"],
                "tree_canopy": [25.0, 32.5, 40.2, 18.7],
            }
        )
        df_sample.to_csv(PROCESSED_DIR / "tree_canopy_by_area.csv", index=False)
        print(f"Saved sample canopy data to {PROCESSED_DIR / 'tree_canopy_by_area.csv'}")
        return df_sample

    df = pd.DataFrame(data)
    df.to_csv(RAW_DIR / "dc_tree_canopy_raw.csv", index=False)

    if df.empty:
        print("DC tree canopy API returned no data. Falling back to sample data.")
        df_sample = pd.DataFrame(
            {
                "area_name": ["Ward 1", "Ward 2", "Ward 3", "Ward 4"],
                "tree_canopy": [25.0, 32.5, 40.2, 18.7],
            }
        )
        df_sample.to_csv(PROCESSED_DIR / "tree_canopy_by_area.csv", index=False)
        return df_sample

    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if "ward" in lc or "area" in lc or "neighborhood" in lc or "name" in lc:
            col_map[c] = "area_name"
        if "canopy" in lc and ("percent" in lc or "%" in lc or "pct" in lc or "cover" in lc):
            col_map[c] = "tree_canopy"

    df = df.rename(columns=col_map)
    keep_cols = [c for c in ["area_name", "tree_canopy"] if c in df.columns]
    if keep_cols:
        df = df[keep_cols]

    if df.empty or "area_name" not in df.columns or "tree_canopy" not in df.columns:
        print("Could not find expected columns in DC tree canopy data. Using sample data instead.")
        df_sample = pd.DataFrame(
            {
                "area_name": ["Ward 1", "Ward 2", "Ward 3", "Ward 4"],
                "tree_canopy": [25.0, 32.5, 40.2, 18.7],
            }
        )
        df_sample.to_csv(PROCESSED_DIR / "tree_canopy_by_area.csv", index=False)
        return df_sample

    df.to_csv(PROCESSED_DIR / "tree_canopy_by_area.csv", index=False)
    print(f"Saved processed canopy data to {PROCESSED_DIR / 'tree_canopy_by_area.csv'}")
    return df


# -------------------------------------------------------------------
# 6. EPA Water Quality Portal - DC water quality
# -------------------------------------------------------------------

def fetch_water_quality_dc() -> pd.DataFrame:
    """
    Fetch water quality results for DC from the Water Quality Portal.

    If the API fails or returns unexpected data, fall back to a small
    sample dataset so the dashboard still works.

    For the app, we want something like:
    - date
    - alert_type
    - description
    """
    url = "https://www.waterqualitydata.us/data/Result/search"
    params = {
        "statecode": "US:11",  # DC
        "mimeType": "json",
        "sorted": "yes",
        "sampleMedia": "Water",
        "startDateLo": START_DATE_STR,
        "startDateHi": END_DATE_STR,
    }

    try:
        resp = safe_request(url, params=params)
        try:
            data = resp.json()
        except ValueError:
            print("Water Quality Portal did not return JSON. Using sample water data instead.")
            raise RuntimeError("Non JSON response")
    except Exception as e:
        print(f"Could not fetch water quality data from API: {e}")
        print("Creating sample water_alerts.csv instead.")

        df_sample = pd.DataFrame(
            {
                "date": [pd.to_datetime("2024-01-05"), pd.to_datetime("2024-02-10")],
                "alert_type": ["Boil Water Advisory", "Maintenance"],
                "description": [
                    "Boil water advisory affecting parts of Ward 4.",
                    "Planned maintenance in Ward 7.",
                ],
            }
        )
        df_sample.to_csv(PROCESSED_DIR / "water_alerts.csv", index=False)
        print(f"Saved sample water data to {PROCESSED_DIR / 'water_alerts.csv'}")
        return df_sample

    if not isinstance(data, list):
        print("Unexpected WQP response format. Using sample data instead.")
        df_sample = pd.DataFrame(
            {
                "date": [pd.to_datetime("2024-01-05"), pd.to_datetime("2024-02-10")],
                "alert_type": ["Boil Water Advisory", "Maintenance"],
                "description": [
                    "Boil water advisory affecting parts of Ward 4.",
                    "Planned maintenance in Ward 7.",
                ],
            }
        )
        df_sample.to_csv(PROCESSED_DIR / "water_alerts.csv", index=False)
        return df_sample

    df = pd.DataFrame(data)
    df.to_csv(RAW_DIR / "water_quality_raw.csv", index=False)

    if df.empty:
        print("WQP returned no data. Using sample data instead.")
        df_sample = pd.DataFrame(
            {
                "date": [pd.to_datetime("2024-01-05"), pd.to_datetime("2024-02-10")],
                "alert_type": ["Boil Water Advisory", "Maintenance"],
                "description": [
                    "Boil water advisory affecting parts of Ward 4.",
                    "Planned maintenance in Ward 7.",
                ],
            }
        )
        df_sample.to_csv(PROCESSED_DIR / "water_alerts.csv", index=False)
        return df_sample

    # Try to map likely fields
    col_map = {}
    for c in df.columns:
        lc = c.lower()
        if "activitystartdate" in lc or ("date" in lc and "start" in lc):
            col_map[c] = "date"
        if "characteristicname" in lc or "resultname" in lc:
            col_map[c] = "alert_type"
        if "resultcomment" in lc or "remark" in lc or "description" in lc:
            col_map[c] = "description"

    df = df.rename(columns=col_map)

    keep_cols = [c for c in ["date", "alert_type", "description"] if c in df.columns]
    df = df[keep_cols] if keep_cols else df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df.empty or "date" not in df.columns or "alert_type" not in df.columns:
        print("Could not find expected columns in WQP data. Using sample data instead.")
        df_sample = pd.DataFrame(
            {
                "date": [pd.to_datetime("2024-01-05"), pd.to_datetime("2024-02-10")],
                "alert_type": ["Boil Water Advisory", "Maintenance"],
                "description": [
                    "Boil water advisory affecting parts of Ward 4.",
                    "Planned maintenance in Ward 7.",
                ],
            }
        )
        df_sample.to_csv(PROCESSED_DIR / "water_alerts.csv", index=False)
        return df_sample

    df.to_csv(PROCESSED_DIR / "water_alerts.csv", index=False)
    print(f"Saved processed water data to {PROCESSED_DIR / 'water_alerts.csv'}")
    return df

# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------

def main():
    print("Fetching Open-Meteo air quality (US AQI) daily...")
    fetch_open_meteo_air_quality()

    print("Fetching Open-Meteo hourly weather and aggregating...")
    fetch_open_meteo_hourly()

    print("Building merged aqi_daily dataset...")
    build_aqi_daily()

    print("Fetching DC waste and recycling data...")
    fetch_dc_waste()

    print("Fetching DC tree canopy data...")
    fetch_dc_tree_canopy()

    print("Fetching DC water quality data...")
    fetch_water_quality_dc()

    print("All fetch tasks complete.")


if __name__ == "__main__":
    main()
