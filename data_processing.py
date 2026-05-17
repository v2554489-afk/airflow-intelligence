import pandas as pd

def process_data(data):
    iaqi = data.get("iaqi", {})

    df = pd.DataFrame([{
        "AQI": data.get("aqi", 0),
        "PM2.5": iaqi.get("pm25", {}).get("v", 0),
        "PM10": iaqi.get("pm10", {}).get("v", 0),
        "NO2": iaqi.get("no2", {}).get("v", 0),
        "O3": iaqi.get("o3", {}).get("v", 0),
    }])

    return df