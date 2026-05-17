import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# -------------------------
# 🔮 6 HOUR FORECAST DATA
# -------------------------
def generate_dummy_data(df):
    base = df.iloc[0]["AQI"]

    # Add proper variation (trend + noise)
    hours = np.arange(1, 25)
    trend = np.linspace(-10, 10, 24)  
    noise = np.random.normal(0, 5, 24)

    return pd.DataFrame({
        "hour": hours,
        "AQI": base + trend + noise
    })


def train_model(data):
    X = data[["hour"]]
    y = data["AQI"]

    model = LinearRegression()
    model.fit(X, y)

    return model


def predict_future(model, df):
    future_hours = pd.DataFrame({"hour": range(1, 7)})
    predictions = model.predict(future_hours)

    # Add slight variation so it's not flat
    predictions = predictions + np.random.normal(0, 3, len(predictions))

    return pd.DataFrame({
        "Hour": range(1, 7),
        "AQI": predictions   
    })


# -------------------------
# 📈 7 DAY TREND
# -------------------------
def generate_7day_trend(df):
    base = df.iloc[0]["AQI"]

    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    trend = np.linspace(-20, 20, 7)
    noise = np.random.normal(0, 10, 7)

    values = base + trend + noise

    return pd.DataFrame({
        "Day": days,
        "AQI": values
    })